"""Internal token-in-token-out chat server (imported by ``chat_agent.py``).

The agent talks plain **chat messages** to a loopback HTTP endpoint with a stock
OpenAI or Anthropic client; this server records each session's EXACT token
sequence so a multi-turn episode stitches into a token-exact training
``Trajectory``. Four concerns, each a section below:

  1. TOKEN-IN-TOKEN-OUT — we never re-tokenize the model's own output. Turn 0
     tokenizes the dataset-templated prompt; each later turn carries the prior
     exact tokens forward and appends ONLY the new turn's delta. That delta is
     read off the model's OWN chat template by text-diff (``_append_user_turn``),
     so NO delimiter or media token is hardcoded — every family (ChatML / Qwen /
     omni3, Gemma, Kimi, GLM, DeepSeek, ...) flows through the one mechanism.
     (vLLM's native chat handler re-templates the whole history each turn,
     re-tokenizing the model's output and drifting — verl found this diverges
     PPO; we avoid it by construction.)

  2. OAI / ANTHROPIC WIRE — both wires drive the SAME accumulation. A request is
     decoded to one canonical (OpenAI-shaped) body ``_run_turn`` consumes, and the
     reply re-encoded to the caller's wire. OpenAI IS the canonical shape (its
     decode is identity); only Anthropic is translated (``_decode_anthropic``).

  3. MULTI-TURN VLM — images the agent returns in later turns accumulate exactly
     like turn-0 images; ``_normalize_turn_content`` keeps the placeholder count
     aligned with pixel_values across every model's template.

  4. COMPACTION -> SPLIT SAMPLES — when the agent rewrites its context prefix
     (compaction) the carried-forward tokens no longer match, so we SEAL the live
     segment and start a fresh token-exact one. ``stitch_session`` returns all
     segments as step-samples of one rollout (shared reward + rollout_id).

Runs on the rollout actor's OWN event loop (shares the AsyncLLM engine; no daemon
thread, no cross-loop await). A stock client reaches a session through the URL
path prefix ``/s/<session_id>/v1`` that ``ChatAgentRunner`` builds.
"""

from __future__ import annotations

import asyncio
import logging
import time
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

import uvicorn
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from molt.agents.base import (
    Trajectory,
    _extract_generation_logprobs,
    _tokenize_feedback,
    _tokenize_observation,
)

logger = logging.getLogger(__name__)


# ===========================================================================
# 1. Chat content -> template text (model-agnostic; no hardcoded delimiters).
# ===========================================================================
def _content_to_text(content) -> str:
    """Flatten a message's ``content`` to template text: text blocks verbatim,
    each image/video block as a ``"<image>"`` / ``"<video>"`` placeholder."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content or "")
    parts = []
    for item in content:
        kind = item.get("type") if isinstance(item, dict) else None
        if kind == "text":
            parts.append(item.get("text", ""))
        elif kind == "image_url":
            parts.append("<image>")
        elif kind == "video_url":
            parts.append("<video>")
    return "".join(parts)


def _new_images(messages) -> list:
    """Image URLs the agent returned in user/tool messages (turn-N images), in
    order — fed to the tokenizer exactly like StepEnvRunner handles Result.images."""
    urls = []
    for msg in messages:
        if msg.get("role") not in ("user", "tool"):
            continue
        content = msg.get("content", "")
        if not isinstance(content, list):
            continue
        for item in content:
            if isinstance(item, dict) and item.get("type") == "image_url":
                ref = item.get("image_url") or {}
                url = ref.get("url") if isinstance(ref, dict) else ref
                if url:
                    urls.append(url)
    return urls


def _normalize_turn_content(processor, content):
    """Flatten ``content`` to the dataset's turn-0 form: a ``"<image>"``-placeholder
    string, split into the model's structured image blocks ONLY when its template
    expands a model-specific image token instead of the literal ``"<image>"``.

    Every appended / re-templated turn runs through this so images tokenize like
    turn 0 (placeholder count stays aligned with pixel_values), with no per-model
    special-casing. Feeding raw ``image_url`` blocks to ``apply_chat_template``
    instead desyncs two ways: a literal-``"<image>"`` template renders the content
    LIST as a repr (zero placeholders), and a structured-content template MUTATES
    the message in place (``image_url`` -> ``image``), emptying ``_new_images``."""
    from molt.utils.vlm_utils import should_expand_image_placeholder, split_image_placeholder

    text = _content_to_text(content)
    if should_expand_image_placeholder(processor):
        return split_image_placeholder({"role": "user", "content": text})["content"]
    return text


def _append_user_turn(processor, content) -> str:
    """Text the chat template appends for ONE new user turn + the assistant
    generation prompt, carrying ``content`` (a string or ``split_image_placeholder``
    blocks). Derived by diffing the template against itself, so no delimiter or
    media token is hardcoded.

    Render a throwaway probe history closed, then the same history + this user turn
    opened; the delta is everything past ``closed.rstrip()`` (the model emits the
    prior turn's close token but not the template's trailing inter-turn separator,
    so that separator heads the delta). The probe is throwaway, so the model's REAL
    assistant turns are never re-templated (Qwen3 even strips ``<think>`` from
    non-last turns). Try assistant->user first (the only transition Gemma allows),
    then user->user for templates whose closed render no longer prefixes the open."""
    probes = (
        [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}],
        [{"role": "user", "content": "q"}],
    )
    for history in probes:
        try:
            closed = processor.apply_chat_template(history, tokenize=False, add_generation_prompt=False)
            opened = processor.apply_chat_template(
                history + [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True
            )
        except Exception:
            continue  # e.g. Gemma rejects consecutive user turns -> next probe
        body = closed.rstrip()
        if opened.startswith(body):
            return opened[len(body) :]
    raise RuntimeError("Could not render a multi-turn user turn from the chat template.")


def _assistant_close_suffix(processor) -> str:
    """The template text that CLOSES an assistant turn (ChatML ``<|im_end|>``, Gemma
    ``<end_of_turn>``, ...), derived template-agnostically. "" for asymmetric
    templates with no explicit close (GLM: the next role token closes the turn).

    ``_append_user_turn`` assumes the prior turn's close token is already in the
    carried-forward stream — true on a ``"stop"`` finish (vLLM keeps the stop/eos
    token in ``output.token_ids``), but NOT on a ``"length"`` truncation/abort,
    where the model never emitted it. The extend path prepends this so the open
    turn is closed instead of the close being silently dropped."""
    sentinel = "\x00__assistant_body__\x00"
    history = [{"role": "user", "content": "q"}, {"role": "assistant", "content": sentinel}]
    try:
        closed = processor.apply_chat_template(history, tokenize=False, add_generation_prompt=False)
    except Exception:
        return ""
    idx = closed.rfind(sentinel)
    if idx == -1:
        return ""
    return closed[idx + len(sentinel) :].rstrip()


# ===========================================================================
# 2. Wire codecs — Anthropic request -> canonical (OpenAI) body; replies back.
#    OpenAI is the canonical shape, so its decode is identity (``lambda b: b``).
# ===========================================================================
def _decode_anthropic(body: dict) -> dict:
    """Anthropic Messages request -> canonical (OpenAI-shaped) body.

    Only fields the token engine reads are touched: content blocks (see
    ``_anthropic_blocks``) and sampling fields, already same-named (``max_tokens`` /
    ``temperature`` / ``top_p``). Top-level ``system`` is left as an inert key, NOT
    folded into ``messages``: turn 0 tokenizes the dataset-templated prompt (body
    messages are ignored for both wires) and ``system`` never appears as a turn-N
    delta, so prepending it would only desync the turn-detection message count."""
    messages = []
    for m in body.get("messages", []):
        content = m.get("content")
        if isinstance(content, list):
            content = [out for b in content for out in _anthropic_blocks(b)]
        messages.append({"role": m.get("role"), "content": content})
    return {**body, "messages": messages}


def _anthropic_blocks(block) -> list:
    """One Anthropic content block -> list of canonical (OpenAI) blocks.

    image -> ``image_url`` (base64 source -> data URI). ``tool_result`` -> its inner
    text/image blocks: an Anthropic-native harness returns tool OUTPUT as a
    ``tool_result`` block, not a plain string, so without unwrapping it the
    observation would be dropped from the token-exact delta. Returning a LIST (one
    block in -> >=0 out) lets a multi-block tool_result flatten in place WITHOUT
    changing the message count ``_run_turn`` uses to detect each new turn."""
    if not isinstance(block, dict):
        return [block]
    btype = block.get("type")
    if btype == "image":
        src = block.get("source") or {}
        url = src.get("url") if src.get("type") == "url" else f"data:{src.get('media_type')};base64,{src.get('data')}"
        return [{"type": "image_url", "image_url": {"url": url}}]
    if btype == "tool_result":
        inner = block.get("content")  # bare string, or a list of text/image blocks
        if isinstance(inner, str):
            return [{"type": "text", "text": inner}]
        if isinstance(inner, list):
            return [out for b in inner for out in _anthropic_blocks(b)]
        return []
    return [block]


# OpenAI's ChatCompletion schema accepts only these finish_reason values and the SDK
# validates them client-side; vLLM also emits abort/error/repetition (and None
# mid-stream), so any non-OpenAI reason maps to "stop" purely to keep the reply
# SDK-parseable (the agent never reads it; truncation uses the RAW reason elsewhere).
_OPENAI_FINISH_REASONS = frozenset({"stop", "length", "tool_calls", "content_filter", "function_call"})


def _chat_completion_body(model_name: str, content: str, finish_reason: str) -> dict:
    return {
        "id": f"chatcmpl-{uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": finish_reason if finish_reason in _OPENAI_FINISH_REASONS else "stop",
            }
        ],
    }


# Anthropic's Message schema validates stop_reason client-side too; map the raw vLLM
# reason into its vocab (anything else -> "end_turn").
_ANTHROPIC_STOP_REASONS = {"stop": "end_turn", "length": "max_tokens", "tool_calls": "tool_use"}


def _anthropic_message_body(model_name: str, content: str, finish_reason: str) -> dict:
    return {
        "id": f"msg_{uuid4().hex[:24]}",
        "type": "message",
        "role": "assistant",
        "model": model_name,
        "content": [{"type": "text", "text": content}],
        "stop_reason": _ANTHROPIC_STOP_REASONS.get(finish_reason, "end_turn"),
        "stop_sequence": None,
        "usage": {"input_tokens": 0, "output_tokens": 0},
    }


def _build_sampling_params(state: "ChatServerState", body: dict):
    """One vLLM SamplingParams per chat call: inherit the rollout defaults, honor the
    caller's temperature / max_tokens, and force logprobs on (importance-sampling
    correction needs token logprobs; the logprobs MODE is set engine-side)."""
    sp = deepcopy(state.default_sampling)
    requested = body.get("max_completion_tokens", body.get("max_tokens"))
    if requested is not None:
        sp.max_tokens = int(requested)
    if body.get("temperature") is not None:
        sp.temperature = float(body["temperature"])
    if body.get("top_p") is not None:
        sp.top_p = float(body["top_p"])
    if sp.logprobs is None:
        sp.logprobs = 1
    return sp


# ===========================================================================
# 3. Per-session token accumulation — the core state machine.
# ===========================================================================
@dataclass
class _Session:
    prompt: str
    label: Any
    images: list | None
    trajectory: Trajectory | None = None  # the LIVE segment: built on turn 0, grown each turn
    # Segments sealed at a compaction boundary. A rollout whose prefix is never
    # rewritten stays one segment (this list empty + `trajectory`); each compaction
    # pushes the prior segment here. stitch_session returns `segments + [trajectory]`.
    segments: list = field(default_factory=list)
    prior_messages: list = field(default_factory=list)  # conversation already tokenized
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)  # serialize turns within a session
    last_action: str = ""  # cached result for idempotent replay of a retried turn
    last_finish: str = "stop"
    # True when the tail assistant turn is UNCLOSED (last generation finished for a
    # non-"stop" reason, so no close token was emitted); the next turn re-supplies it.
    open_assistant_turn: bool = False


class ChatServerState:
    """Per-actor state: the rollout engine + processor + live session accumulators."""

    def __init__(self, engine, processor, model_name: str, max_length: int, default_sampling):
        self.engine = engine  # the rollout actor; .generate(token_ids, sp, multi_modal_data)
        self.processor = processor
        self.model_name = model_name
        self.max_length = max_length
        self.default_sampling = default_sampling
        self.sessions: dict[str, _Session] = {}

    def open(self, session_id: str, prompt, label, images):
        self.sessions[session_id] = _Session(prompt=prompt, label=label, images=images)

    def discard(self, session_id: str):
        self.sessions.pop(session_id, None)


def _extends_prior(messages: list, prior_messages: list) -> bool:
    """Does this request EXTEND the conversation we tokenized, or REWRITE its prefix
    (compaction)? ``prior_messages[:-1]`` is exactly what the client sent last turn
    (the trailing element is the assistant echo WE synthesized — never compared,
    since the client re-sends its OWN representation of the model's turn). A genuine
    next turn re-sends those messages then appends; a compaction summarizes/drops
    earlier ones, so the leading messages no longer match (or the list is shorter).
    A retry of an already-committed turn also "extends" (empty delta) and is handled
    by the idempotent-replay path in ``_run_turn``, not here."""
    prev_client = prior_messages[:-1]
    return messages[: len(prev_client)] == prev_client


def _start_segment(state: ChatServerState, session: _Session, text: str, images, sp) -> Trajectory:
    """Tokenize ``text`` (+ any images) as a FRESH token-exact segment and make it the
    live trajectory. Used for turn 0 (the dataset-templated prompt, tokenized
    DIRECTLY) and for a compaction restart (the re-templated post-compaction
    conversation) — both begin a new monotonic token axis."""
    processor = state.processor
    obs_tokens, mm_train_inputs, pil_images = _tokenize_observation(processor, text, images)
    image_budget = 0
    if pil_images:
        from molt.utils.vlm_utils import estimate_vllm_input_expansion_delta

        image_budget = estimate_vllm_input_expansion_delta(processor, obs_tokens, mm_train_inputs, pil_images)
    traj = Trajectory(
        prompt=session.prompt,
        label=session.label,
        images=images,
        observation_text=text,
        observation_tokens=obs_tokens,
        mm_train_inputs=mm_train_inputs,
        pil_images=pil_images,
        image_budget=image_budget,
        rollout_log_probs=[0.0] * len(obs_tokens) if sp.logprobs is not None else None,
    )
    session.trajectory = traj
    return traj


# The three ways a turn brings new context onto the token axis. Each returns
# ``(traj, revert)`` — ``revert()`` undoes the append so a client retry of a turn
# whose generate later failed re-runs cleanly (a stock client retries when a weight
# broadcast briefly stalls the loopback connection).
def _begin_turn0(state: ChatServerState, session: _Session, sp):
    """Turn 0: tokenize the dataset-templated prompt as the first segment."""
    traj = _start_segment(state, session, session.prompt, session.images, sp)

    def revert():
        session.trajectory = None  # drop the half-built segment; the retry re-runs turn 0

    return traj, revert


def _extend_turn(state: ChatServerState, session: _Session, new_messages, sp):
    """Append the new tool/user turn(s) as ONE user turn, carrying the prior EXACT
    tokens (incl. the model's own output) forward untouched. The delta is rendered
    through the model's OWN template; images render as that model's own token."""
    processor = state.processor
    traj = session.trajectory
    new_images = _new_images(new_messages)
    merged = "".join(_content_to_text(m.get("content", "")) for m in new_messages)
    delta_text = _append_user_turn(processor, _normalize_turn_content(processor, merged))
    if session.open_assistant_turn:
        # Prior turn truncated (no close token emitted) -> re-supply the close the
        # carried-forward stream is missing (see `_assistant_close_suffix`).
        delta_text = _assistant_close_suffix(processor) + delta_text

    # Snapshot BEFORE _tokenize_feedback, which MUTATES the image state in place
    # (pil_images.extend / mm_train_inputs reassign / image_budget +=); snapshotting
    # after would record the grown state and make revert a no-op (double-counting the
    # image on retry -> placeholder-vs-pixel_values misalignment).
    lp = traj.rollout_log_probs
    snap = (
        traj.observation_text,
        len(traj.observation_tokens),
        len(lp) if lp is not None else None,
        len(traj.pil_images),
        traj.image_budget,
        traj.mm_train_inputs,
    )
    feedback_tokens = _tokenize_feedback(processor, delta_text, new_images, traj, state.max_length)
    traj.append_feedback("", delta_text, feedback_tokens)

    def revert():
        text, n_tokens, n_logprobs, n_pil, budget, mm_inputs = snap
        traj.observation_text = text
        del traj.observation_tokens[n_tokens:]
        if traj.rollout_log_probs is not None and n_logprobs is not None:
            del traj.rollout_log_probs[n_logprobs:]
        del traj.pil_images[n_pil:]
        traj.image_budget = budget
        traj.mm_train_inputs = mm_inputs

    return traj, revert


def _restart_for_compaction(state: ChatServerState, session: _Session, messages, sp):
    """Compaction rewrote the prefix -> the carried-forward tokens no longer match
    the agent's context, so SEAL the live segment and start a fresh one from the
    re-templated post-compaction conversation (all observation tokens, zero grad;
    later turns again carry exact tokens forward, so no drift within the segment)."""
    processor = state.processor
    session.segments.append(session.trajectory)
    # Read images from the UNTOUCHED messages (templating a normalized copy may rewrite
    # content blocks). Normalize each message exactly as turn 0 does.
    images = _new_images(messages)
    restart_messages = [
        {"role": m.get("role"), "content": _normalize_turn_content(processor, m.get("content", ""))} for m in messages
    ]
    restart_text = processor.apply_chat_template(restart_messages, tokenize=False, add_generation_prompt=True)
    traj = _start_segment(state, session, restart_text, images, sp)

    def revert():
        session.trajectory = session.segments.pop()  # restore the sealed segment, drop the new one

    return traj, revert


async def _run_turn(state: ChatServerState, session: _Session, body: dict) -> tuple[str, str]:
    """One chat call: bring the new context onto the token axis, generate one model
    turn, accumulate the exact tokens. Returns ``(action_text, finish_reason)``."""
    messages = body.get("messages") or []
    if not messages:
        raise HTTPException(status_code=400, detail="No messages provided")
    sp = _build_sampling_params(state, body)
    per_turn_cap = sp.max_tokens

    # 1) Bring this turn's context onto the live token axis (turn 0 / extend / restart).
    if session.trajectory is None:
        traj, revert = _begin_turn0(state, session, sp)
    elif _extends_prior(messages, session.prior_messages):
        new_messages = messages[len(session.prior_messages) :]
        if not new_messages:
            return session.last_action, session.last_finish  # idempotent replay of a retried turn
        traj, revert = _extend_turn(state, session, new_messages, sp)
    else:
        traj, revert = _restart_for_compaction(state, session, messages, sp)

    # 2) Per-turn budget: never exceed remaining context.
    remaining = state.max_length - len(traj.observation_tokens) - traj.image_budget
    sp.max_tokens = min(per_turn_cap, remaining) if per_turn_cap is not None else remaining
    if sp.max_tokens <= 0:
        # No room to generate: commit a terminal empty turn so a connection-break retry
        # replays it via the idempotent path (above) instead of re-appending the delta.
        traj.truncated = True
        session.prior_messages = list(messages) + [{"role": "assistant", "content": ""}]
        session.last_action, session.last_finish = "", "length"
        return "", "length"
    min_tokens = getattr(sp, "min_tokens", None)  # a late turn's budget can drop below min_tokens
    if min_tokens is not None and min_tokens > sp.max_tokens:
        sp.min_tokens = sp.max_tokens

    # 3) Generate one turn. pil_images accumulates across turns; vLLM's
    #    limit_mm_per_prompt is a STATIC build-time cap the launcher sizes to the
    #    worst-case rollout-wide image count (it validates placeholder-vs-image
    #    internally, so no guard here). Revert the append if generate fails.
    mm_data = {"image": traj.pil_images} if traj.pil_images else None
    try:
        request_output, off_policy_len = await state.engine.generate(
            traj.observation_tokens, sp, multi_modal_data=mm_data
        )
    except BaseException:
        revert()
        raise

    # 4) Commit the generated tokens + bookkeeping for the next turn's diff / a retry.
    gen = request_output.outputs[0]
    action_tokens = list(gen.token_ids)
    action_text = gen.text or ""
    finish_reason = gen.finish_reason
    traj.truncated = traj.truncated or finish_reason == "length"
    action_logprobs = None
    if traj.rollout_log_probs is not None:
        action_logprobs = _extract_generation_logprobs(action_tokens, gen.logprobs)
    traj.append_action(action_tokens, action_logprobs, off_policy_len=off_policy_len)

    session.prior_messages = list(messages) + [{"role": "assistant", "content": action_text}]
    session.last_action = action_text
    session.last_finish = finish_reason
    # vLLM keeps the close token in output ids only on "stop"; any other reason leaves
    # the assistant turn open, so the next turn re-supplies the close (see _extend_turn).
    session.open_assistant_turn = finish_reason != "stop"
    return action_text, finish_reason


def stitch_session(state: ChatServerState, session_id: str, result):
    """Return the rollout's segment trajectories (a list). A compaction-free rollout
    is one segment; each compaction boundary sealed another. Every segment carries
    the SAME terminal reward/score/info and shares the one ``rollout_id`` the trainer
    assigns, so group baselines dedup them to one reward per rollout while each
    segment still contributes its own generated tokens to the policy-gradient axis
    (multi-turn step-sample contract — see ``experience_maker._merge_rollout_rewards``)."""
    session = state.sessions.get(session_id)
    if session is None or session.trajectory is None:
        raise RuntimeError(f"Session {session_id} produced no generations")
    segments = session.segments + [session.trajectory]
    for traj in segments:
        traj.reward = result.reward
        traj.scores = result.score if result.score is not None else result.reward
        traj.extra_logs = result.info or {}
        if result.images is not None:
            traj.images = result.images
    return segments


# ===========================================================================
# 4. HTTP server — uvicorn on the engine's own event loop + the session routes.
# ===========================================================================
class _AutoPortServer(uvicorn.Server):
    """uvicorn Server that reports the OS-assigned port when configured ``port=0``."""

    def __init__(self, config: uvicorn.Config):
        super().__init__(config)
        self.actual_port: int | None = None
        self._ready = asyncio.Event()

    async def startup(self, sockets=None) -> None:
        try:
            await super().startup(sockets=sockets)
            if self.servers and self.config.port == 0:
                self.actual_port = self.servers[0].sockets[0].getsockname()[1]
            else:
                self.actual_port = self.config.port
        finally:
            self._ready.set()

    async def get_port(self) -> int | None:
        await self._ready.wait()
        return self.actual_port


async def serve_on_current_loop(app, host: str = "127.0.0.1") -> tuple[int, asyncio.Task]:
    """Start uvicorn as a task on the CURRENT event loop (shared with the AsyncLLM
    engine — no daemon thread, no cross-loop await). Returns ``(port, task)``."""
    config = uvicorn.Config(app, host=host, port=0, log_level="warning")
    server = _AutoPortServer(config)
    task = asyncio.create_task(server.serve())
    port = await server.get_port()
    if port is None:
        await task  # surfaces the startup exception
        raise RuntimeError("Chat server failed to start (no port reported)")
    return port, task


def mount_session_capture(app, state: ChatServerState) -> None:
    """Mount the session-scoped chat routes. The OpenAI and Anthropic wires drive the
    SAME token-exact accumulation (``_run_turn``); they differ only in how the body
    is decoded to the canonical shape and how the reply is encoded back."""
    router = APIRouter()

    async def _serve(session_id, request, decode, encode):
        body = decode(await request.json())
        session = state.sessions.get(session_id)
        if session is None:
            raise HTTPException(status_code=400, detail=f"Unknown or unopened session {session_id}")
        # Serialize turns within a session: a stock client may retry, which would
        # otherwise run a second turn concurrently and corrupt the token accumulation.
        async with session.lock:
            action_text, finish_reason = await _run_turn(state, session, body)
        return JSONResponse(encode(state.model_name, action_text, finish_reason))

    @router.post("/s/{session_id}/v1/chat/completions")
    async def chat_sess(session_id: str, request: Request):
        return await _serve(session_id, request, lambda b: b, _chat_completion_body)

    @router.post("/s/{session_id}/v1/messages")
    async def messages_sess(session_id: str, request: Request):
        return await _serve(session_id, request, _decode_anthropic, _anthropic_message_body)

    app.include_router(router)
