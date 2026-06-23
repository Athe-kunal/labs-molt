import asyncio
from types import SimpleNamespace

import pytest
import torch

from molt.agents._chat_server import (
    ChatServerState,
    _anthropic_message_body,
    _append_user_turn,
    _assistant_close_suffix,
    _chat_completion_body,
    _content_to_text,
    _decode_anthropic,
    _extends_prior,
    _new_images,
    _run_turn,
    stitch_session,
)
from molt.agents.base import Result, Trajectory


class _FakeChatMLProcessor:
    """Minimal ChatML renderer + char-level tokenizer (text-only, no image_processor)."""

    def __init__(self, scaffold=""):
        self.scaffold = scaffold

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        s = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
        if add_generation_prompt:
            s += "<|im_start|>assistant\n" + self.scaffold
        return s

    def __call__(self, text, add_special_tokens=False, return_tensors=None):
        # Deterministic nonzero ids, one per char — enough for the text-only
        # tokenize path in _tokenize_observation / _tokenize_feedback.
        return {"input_ids": torch.tensor([[(ord(c) % 97) + 1 for c in text]])}


def _state():
    return ChatServerState(engine=None, processor=None, model_name="policy", max_length=128, default_sampling=None)


def _traj():
    # A two-turn accumulation as _run_turn would leave it: monotonic token axis,
    # action ranges over the model-generated spans, logprobs aligned on that axis.
    traj = Trajectory(
        prompt="p",
        label="l",
        images=None,
        observation_text="",
        observation_tokens=[1, 2, 3, 4, 90, 91, 5, 6],
        rollout_log_probs=[0.0, 0.0, -0.3, -0.4, 0.0, 0.0, -0.6, -0.7],
    )
    traj.action_ranges = [(2, 4), (6, 8)]
    traj.truncated = True
    return traj


def _sampling(logprobs=None):
    return SimpleNamespace(max_tokens=8, temperature=1.0, top_p=1.0, logprobs=logprobs, min_tokens=0)


class _RaisingEngine:
    # Engine whose generate() always raises — used by the rollback/undo tests.
    async def generate(self, *a, **k):
        raise RuntimeError("transient engine error")


def test_stitch_raises_without_generations():
    state = _state()
    state.open("sid", "p", "l", None)  # opened but no turn ran
    with pytest.raises(RuntimeError, match="no generations"):
        stitch_session(state, "sid", Result(reward=0.0))


def test_stitch_raises_for_unknown_session():
    with pytest.raises(RuntimeError, match="no generations"):
        stitch_session(_state(), "missing", Result(reward=0.0))


def test_stitch_stamps_reward_and_returns_accumulated_trajectory():
    state = _state()
    state.open("sid", "p", "l", None)
    traj = _traj()
    state.sessions["sid"].trajectory = traj

    out = stitch_session(state, "sid", Result(reward=1.0, score=0.5))

    # A compaction-free rollout is exactly one segment, returned as a 1-list.
    assert out == [traj] and out[0] is traj
    assert out[0].reward == 1.0
    assert out[0].scores == 0.5
    assert out[0].observation_tokens == [1, 2, 3, 4, 90, 91, 5, 6]
    assert out[0].action_ranges == [(2, 4), (6, 8)]
    assert out[0].rollout_log_probs == [0.0, 0.0, -0.3, -0.4, 0.0, 0.0, -0.6, -0.7]
    assert out[0].truncated is True


def test_append_user_turn_derives_chatml_append():
    # The delta is derived template-agnostically (diff of a closed render vs an
    # extended render), reproducing the per-turn open + content + close + assistant
    # generation prompt without re-templating assistant history.
    delta = _append_user_turn(_FakeChatMLProcessor(), "FB")
    assert delta == "\n<|im_start|>user\nFB<|im_end|>\n<|im_start|>assistant\n"


def test_append_user_turn_captures_reasoning_scaffold():
    delta = _append_user_turn(_FakeChatMLProcessor(scaffold="<think>\n"), "FB")
    assert delta == "\n<|im_start|>user\nFB<|im_end|>\n<|im_start|>assistant\n<think>\n"


# ---------------------------------------------------------------------------
# Template-agnostic wrapper: faithful renderers for the non-ChatML families the
# chat server must support (structure mirrors each model's real chat template).
# ---------------------------------------------------------------------------
class _FakeGemmaProcessor:
    """Gemma 2/3/4: ``<start_of_turn>{role}\\n...<end_of_turn>\\n`` (assistant ->
    ``model``), and — crucially — it RAISES on two consecutive same-role turns,
    so the wrapper cannot use a user->user probe and must use assistant->user."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        parts = []
        for idx, m in enumerate(messages):
            if (m["role"] == "user") != (idx % 2 == 0):
                raise ValueError("Conversation roles must alternate user/assistant/user/assistant/...")
            role = "model" if m["role"] == "assistant" else m["role"]
            parts.append(f"<start_of_turn>{role}\n{m['content']}<end_of_turn>\n")
        s = "<bos>" + "".join(parts)
        return s + "<start_of_turn>model\n" if add_generation_prompt else s


class _FakeKimiProcessor:
    """Kimi K2: role-specific open ``<|im_{role}|>{role}<|im_middle|>`` + ``<|im_end|>``,
    no trailing inter-turn whitespace (open marker differs from a ChatML close)."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        s = "".join(f"<|im_{m['role']}|>{m['role']}<|im_middle|>{m['content']}<|im_end|>" for m in messages)
        return s + "<|im_assistant|>assistant<|im_middle|>" if add_generation_prompt else s


class _FakeGLMProcessor:
    """GLM 4.5/5.x: ``[gMASK]<sop>`` preamble, ``<|{role}|>\\n{content}`` with no
    explicit close token (asymmetric: next role token implicitly closes)."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        s = "[gMASK]<sop>" + "".join(f"<|{m['role']}|>\n{m['content']}" for m in messages)
        return s + "<|assistant|>\n" if add_generation_prompt else s


class _FakeDeepSeekProcessor:
    """DeepSeek: ``<｜User｜>{content}`` (no close) and ``<｜Assistant｜>{content}<｜end▁of▁sentence｜>``."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        s = "<｜begin▁of▁sentence｜>"
        for m in messages:
            if m["role"] == "user":
                s += "<｜User｜>" + m["content"]
            elif m["role"] == "assistant":
                s += "<｜Assistant｜>" + m["content"] + "<｜end▁of▁sentence｜>"
        return s + "<｜Assistant｜>" if add_generation_prompt else s


class _FakeQwen3HistoryProcessor:
    """ChatML that REWRITES assistant history like Qwen3: the last assistant turn
    is wrapped in an empty ``<think>`` block, non-last turns are bare. This makes
    the assistant->user probe's closed render NOT a prefix of the extended one,
    forcing the wrapper to fall back to the user->user probe."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        parts = []
        for idx, m in enumerate(messages):
            if m["role"] == "assistant":
                c = f"<think>\n\n</think>\n\n{m['content']}" if idx == len(messages) - 1 else m["content"]
                parts.append(f"<|im_start|>assistant\n{c}<|im_end|>\n")
            else:
                parts.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n")
        s = "".join(parts)
        return s + "<|im_start|>assistant\n" if add_generation_prompt else s


@pytest.mark.parametrize(
    "processor, expected_delta",
    [
        (_FakeGemmaProcessor(), "\n<start_of_turn>user\nFB<end_of_turn>\n<start_of_turn>model\n"),
        (_FakeKimiProcessor(), "<|im_user|>user<|im_middle|>FB<|im_end|><|im_assistant|>assistant<|im_middle|>"),
        (_FakeGLMProcessor(), "<|user|>\nFB<|assistant|>\n"),
        (_FakeDeepSeekProcessor(), "<｜User｜>FB<｜Assistant｜>"),
        (_FakeQwen3HistoryProcessor(), "\n<|im_start|>user\nFB<|im_end|>\n<|im_start|>assistant\n"),
    ],
)
def test_append_user_turn_is_template_agnostic(processor, expected_delta):
    # Gemma/Kimi/GLM/DeepSeek/Qwen3 all derive the correct appended delta with NO
    # hardcoded marker tokens — covering strict-alternation, role-specific opens,
    # asymmetric (no-close) templates, and history-rewriting reasoning templates.
    assert _append_user_turn(processor, "FB") == expected_delta


def test_append_user_turn_stitches_token_exactly():
    # The derived delta must make the running stream a clean extension of the
    # templated conversation: stream-through-assistant + delta == the canonical
    # render of the full conversation. Checked on Gemma (the family the old
    # ChatML-only heuristic silently corrupted).
    proc = _FakeGemmaProcessor()
    closed = proc.apply_chat_template(
        [{"role": "user", "content": "Q1"}, {"role": "assistant", "content": "A1"}], add_generation_prompt=False
    )
    full = proc.apply_chat_template(
        [{"role": "user", "content": "Q1"}, {"role": "assistant", "content": "A1"}, {"role": "user", "content": "FB"}],
        add_generation_prompt=True,
    )
    # The model emits the turn-close token but not the template's trailing
    # inter-turn separator, so the stream boundary is closed.rstrip().
    assert closed.rstrip() + _append_user_turn(proc, "FB") == full


def test_gemma_rejects_user_user_probe():
    # Documents WHY _append_user_turn tries the assistant->user probe first: Gemma
    # raises on consecutive user turns, so a user-only probe could never render.
    with pytest.raises(ValueError, match="alternate"):
        _FakeGemmaProcessor().apply_chat_template([{"role": "user", "content": "a"}, {"role": "user", "content": "b"}])


@pytest.mark.parametrize(
    "processor, expected_close",
    [
        (_FakeChatMLProcessor(), "<|im_end|>"),
        (_FakeGemmaProcessor(), "<end_of_turn>"),
        (_FakeKimiProcessor(), "<|im_end|>"),
        (_FakeDeepSeekProcessor(), "<｜end▁of▁sentence｜>"),
        (_FakeGLMProcessor(), ""),  # asymmetric: no explicit assistant-close token
    ],
)
def test_assistant_close_suffix_is_template_agnostic(processor, expected_close):
    # The assistant-turn close token is derived with NO hardcoded markers — empty
    # for asymmetric templates (GLM) where the next role token closes the turn.
    assert _assistant_close_suffix(processor) == expected_close


def test_run_turn_resupplies_close_token_after_truncated_turn():
    # vLLM keeps the turn-close token in output ids ONLY on a "stop" finish; a
    # "length" truncation leaves the assistant turn OPEN (no close token emitted).
    # If the agent then continues (a small per-call max_tokens leaves context to
    # spare), the next turn's delta — which assumes the close is already present —
    # must re-supply it, or the stitched sequence drops it and the chat-template
    # structure is malformed for the rest of the segment.
    class _SeqEngine:
        def __init__(self):
            self.calls = 0

        async def generate(self, token_ids, sp, multi_modal_data=None):
            self.calls += 1
            tok = [900 + self.calls, 910 + self.calls]
            logprobs = [{t: SimpleNamespace(logprob=-0.1)} for t in tok]
            # Turn 1 is TRUNCATED ("length"); turn 2 finishes cleanly ("stop").
            finish = "length" if self.calls == 1 else "stop"
            out = SimpleNamespace(token_ids=tok, text=f"A{self.calls}", finish_reason=finish, logprobs=logprobs)
            return SimpleNamespace(outputs=[out]), 0

    state = ChatServerState(
        engine=_SeqEngine(),
        processor=_FakeChatMLProcessor(),
        model_name="policy",
        max_length=1000,
        default_sampling=_sampling(),
    )
    state.open("sid", "P", "l", None)
    session = state.sessions["sid"]

    # Turn 0 (prompt) truncates: the model emitted "A1" but NOT its <|im_end|>.
    asyncio.run(_run_turn(state, session, {"messages": [{"role": "user", "content": "P"}]}))
    assert session.open_assistant_turn is True  # tail assistant turn left open
    base = len(session.trajectory.observation_tokens)  # prompt + truncated action0

    # Turn 1: a tool result extends. The delta must re-supply the missing close.
    asyncio.run(
        _run_turn(
            state,
            session,
            {
                "messages": [
                    {"role": "user", "content": "P"},
                    {"role": "assistant", "content": "A1"},
                    {"role": "user", "content": "FB"},
                ]
            },
        )
    )
    # The close entered the TOKEN axis (the only thing trained on): the feedback delta
    # tokenized at [base:] is the close suffix followed by the normal user-turn bridge.
    proc = _FakeChatMLProcessor()
    close_ids = proc(_assistant_close_suffix(proc))["input_ids"][0].tolist()
    bridge_ids = proc(_append_user_turn(proc, "FB"))["input_ids"][0].tolist()
    feedback_ids = session.trajectory.observation_tokens[base : base + len(close_ids) + len(bridge_ids)]
    assert feedback_ids == close_ids + bridge_ids
    # NB: observation_text omits the model's action content (append_feedback gets "")
    # and the prompt "P" stands in for the already-templated turn-0 prompt, so the
    # re-supplied close lands right after "P": "P<|im_end|>\n<|im_start|>user\nFB...".
    assert "P<|im_end|>\n<|im_start|>user\nFB" in session.trajectory.observation_text
    # The continuation finished with "stop", so the turn is closed again.
    assert session.open_assistant_turn is False


def test_run_turn_no_close_resupply_after_clean_stop():
    # Control: when the prior turn finished with "stop", vLLM already emitted the
    # close token, so the next delta must NOT prepend a second one (no doubling).
    class _StopEngine:
        def __init__(self):
            self.calls = 0

        async def generate(self, token_ids, sp, multi_modal_data=None):
            self.calls += 1
            tok = [900 + self.calls, 910 + self.calls]
            logprobs = [{t: SimpleNamespace(logprob=-0.1)} for t in tok]
            out = SimpleNamespace(token_ids=tok, text=f"A{self.calls}", finish_reason="stop", logprobs=logprobs)
            return SimpleNamespace(outputs=[out]), 0

    state = ChatServerState(_StopEngine(), _FakeChatMLProcessor(), "policy", 1000, _sampling())
    state.open("sid", "P", "l", None)
    session = state.sessions["sid"]
    asyncio.run(_run_turn(state, session, {"messages": [{"role": "user", "content": "P"}]}))
    assert session.open_assistant_turn is False
    asyncio.run(
        _run_turn(
            state,
            session,
            {
                "messages": [
                    {"role": "user", "content": "P"},
                    {"role": "assistant", "content": "A1"},
                    {"role": "user", "content": "FB"},
                ]
            },
        )
    )
    # On "stop" the close token already rode along in the action's token ids
    # (vLLM keeps it; action_text omits it via skip_special_tokens) — so the delta
    # is the plain inter-turn separator, NOT a re-supplied close. Byte-identical to
    # pre-fix behavior on the (overwhelmingly common) clean-stop path.
    text = session.trajectory.observation_text
    assert "P\n<|im_start|>user\nFB" in text  # normal delta — no close prepended
    assert "P<|im_end|>" not in text  # the fix did NOT fire on a clean stop


def test_run_turn_replays_committed_turn_idempotently():
    # A client retry re-sends an already-committed turn (new_messages empty).
    # _run_turn must return the cached action WITHOUT generating (engine=None
    # here would raise if it tried) or appending a duplicate action.
    state = ChatServerState(
        engine=None,
        processor=None,
        model_name="policy",
        max_length=128,
        default_sampling=SimpleNamespace(max_tokens=8, temperature=1.0, top_p=1.0, logprobs=None, min_tokens=0),
    )
    state.open("sid", "p", "l", None)
    session = state.sessions["sid"]
    session.trajectory = _traj()
    committed = [{"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}]
    session.prior_messages = committed
    session.last_action, session.last_finish = "cached-action", "stop"
    ranges_before = list(session.trajectory.action_ranges)

    action, finish = asyncio.run(_run_turn(state, session, {"messages": list(committed)}))

    assert (action, finish) == ("cached-action", "stop")
    assert session.trajectory.action_ranges == ranges_before  # no duplicate action appended


def test_extends_prior_detects_extension_vs_compaction():
    # `prior_messages[:-1]` is what the client sent last turn (the trailing
    # assistant echo we synthesized is never compared).
    prior = [{"role": "user", "content": "P"}, {"role": "assistant", "content": "A0"}]
    # A normal next turn re-sends the prior client messages then appends -> extends.
    assert (
        _extends_prior(
            [
                {"role": "user", "content": "P"},
                {"role": "assistant", "content": "A0-client"},
                {"role": "user", "content": "tool"},
            ],
            prior,
        )
        is True
    )
    # A retry that re-sends only the prior client messages also "extends" (empty delta).
    assert _extends_prior([{"role": "user", "content": "P"}], prior) is True
    # Compaction rewrites the leading message -> NOT an extension.
    assert _extends_prior([{"role": "user", "content": "SUMMARY"}, {"role": "user", "content": "go"}], prior) is False
    # Compaction that shrinks below the prior client length -> NOT an extension.
    assert _extends_prior([], prior) is False


def test_run_turn_splits_into_segments_on_compaction():
    # Agentic context compaction rewrites the conversation prefix (summarizes/drops
    # old turns), so the carried-forward EXACT tokens no longer match the agent's
    # context. _run_turn must SEAL the live segment and start a fresh token-exact
    # one; stitch_session then returns BOTH as step-samples of one rollout (shared
    # reward + the rollout_id the trainer assigns) — instead of silently replaying
    # a cached action (compaction shrinks the list -> empty delta) or appending
    # onto a stale prefix.
    class _SeqEngine:
        def __init__(self):
            self.calls = 0

        async def generate(self, token_ids, sp, multi_modal_data=None):
            self.calls += 1
            tok = [900 + self.calls, 910 + self.calls]
            # _build_sampling_params forces logprobs on, so return per-token entries.
            logprobs = [{t: SimpleNamespace(logprob=-0.1)} for t in tok]
            out = SimpleNamespace(token_ids=tok, text=f"A{self.calls}", finish_reason="stop", logprobs=logprobs)
            return SimpleNamespace(outputs=[out]), 0

    state = ChatServerState(
        engine=_SeqEngine(),
        processor=_FakeChatMLProcessor(),
        model_name="policy",
        max_length=1000,
        default_sampling=_sampling(),
    )
    state.open("sid", "P", "l", None)
    session = state.sessions["sid"]

    # Turn 0 (prompt) then turn 1 (a tool result extends the SAME segment).
    asyncio.run(_run_turn(state, session, {"messages": [{"role": "user", "content": "P"}]}))
    seg0 = session.trajectory
    asyncio.run(
        _run_turn(
            state,
            session,
            {
                "messages": [
                    {"role": "user", "content": "P"},
                    {"role": "assistant", "content": "A1"},
                    {"role": "user", "content": "tool-1"},
                ]
            },
        )
    )
    assert session.trajectory is seg0  # extended in place, no new segment
    assert session.segments == [] and len(seg0.action_ranges) == 2

    # Compaction: the agent summarizes its history into a shorter, rewritten list.
    compacted = [{"role": "user", "content": "SUMMARY so far"}, {"role": "user", "content": "continue"}]
    assert _extends_prior(compacted, session.prior_messages) is False
    asyncio.run(_run_turn(state, session, {"messages": compacted}))

    # Prior segment sealed; a NEW segment is live, tokenized from the re-templated
    # compacted conversation (not an extension of seg0's token axis).
    assert session.segments == [seg0]
    seg1 = session.trajectory
    assert seg1 is not seg0
    assert seg1.observation_text == _FakeChatMLProcessor().apply_chat_template(compacted, add_generation_prompt=True)
    assert len(seg1.action_ranges) == 1  # the post-compaction generation

    out = stitch_session(state, "sid", Result(reward=1.0))
    # Both segments returned as the rollout's step-samples, each stamped the reward.
    assert out == [seg0, seg1]
    assert all(t.reward == 1.0 for t in out)


def test_run_turn_undoes_compaction_restart_when_generate_fails():
    # If the post-compaction generate fails, the seal+restart must be undone: the
    # sealed segment is restored as the live trajectory and prior_messages stays
    # pre-compaction, so a client retry re-detects the same divergence and restarts
    # cleanly (no double-seal, no lost segment).
    state = ChatServerState(
        engine=_RaisingEngine(),
        processor=_FakeChatMLProcessor(),
        model_name="policy",
        max_length=1000,
        default_sampling=_sampling(),
    )
    state.open("sid", "P", "l", None)
    session = state.sessions["sid"]
    seg0 = _traj()
    session.trajectory = seg0
    prior = [{"role": "user", "content": "P"}, {"role": "assistant", "content": "A0"}]
    session.prior_messages = list(prior)

    compacted = [{"role": "user", "content": "SUMMARY"}, {"role": "user", "content": "go"}]
    with pytest.raises(RuntimeError):
        asyncio.run(_run_turn(state, session, {"messages": compacted}))

    assert session.segments == []  # seal undone
    assert session.trajectory is seg0  # original segment restored as live
    assert session.prior_messages == prior  # un-advanced -> retry re-detects + restarts


def test_stitch_defaults_score_to_reward():
    state = _state()
    state.open("sid", "p", "l", None)
    state.sessions["sid"].trajectory = _traj()
    out = stitch_session(state, "sid", Result(reward=0.75))
    assert out[0].scores == 0.75


def test_chat_completion_body_normalizes_finish_reason():
    # vLLM emits abort/error/repetition (and None mid-stream); the OpenAI SDK only
    # accepts stop/length/tool_calls/content_filter/function_call and validates the
    # response field client-side, so non-OpenAI reasons must map to "stop" or the
    # agent's chat.completions.create raises and crashes the rollout.
    for raw in ("abort", "error", "repetition", None):
        body = _chat_completion_body("policy", "x", raw)
        assert body["choices"][0]["finish_reason"] == "stop"
    for raw in ("stop", "length", "tool_calls"):
        assert _chat_completion_body("policy", "x", raw)["choices"][0]["finish_reason"] == raw


def test_decode_anthropic_normalizes_to_canonical_openai_shape():
    # An Anthropic Messages body must decode to the canonical (OpenAI-shaped) body
    # _run_turn consumes: image source -> image_url, text passthrough, top-level
    # system NOT folded into messages, sampling fields left verbatim, and message
    # COUNT preserved so _run_turn's `messages[len(prior):]` turn-delta detection fires.
    body = {
        "system": "you are helpful",
        "max_tokens": 64,
        "temperature": 0.7,
        "top_p": 0.9,
        "messages": [
            {"role": "user", "content": "hi"},  # plain string content (like OpenAI)
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "look:"},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}},
                    {"type": "image", "source": {"type": "url", "url": "http://x/y.png"}},
                ],
            },
        ],
    }
    out = _decode_anthropic(body)

    # system rides along as an inert top-level key but is NOT folded into messages
    # (a system message would desync the turn-delta count; turn 0 uses session.prompt).
    assert out["system"] == "you are helpful" and all(m["role"] != "system" for m in out["messages"])
    assert (out["max_tokens"], out["temperature"], out["top_p"]) == (
        64,
        0.7,
        0.9,
    )  # _build_sampling_params reads these
    assert len(out["messages"]) == 2  # count preserved -> delta detection intact
    assert out["messages"][0] == {"role": "user", "content": "hi"}
    blocks = out["messages"][1]["content"]
    assert blocks[0] == {"type": "text", "text": "look:"}
    assert blocks[1] == {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
    assert blocks[2] == {"type": "image_url", "image_url": {"url": "http://x/y.png"}}
    # The normalized blocks feed _content_to_text / _new_images exactly like OpenAI's,
    # with one <image> placeholder per image (vLLM placeholder-count invariant).
    assert _content_to_text(blocks) == "look:<image><image>"
    assert _new_images([out["messages"][1]]) == ["data:image/png;base64,AAAA", "http://x/y.png"]


def test_decode_anthropic_unwraps_tool_result_into_token_delta():
    # Anthropic-native harnesses return tool OUTPUT as a `tool_result` block (not
    # the plain string the OpenAI `tool` role uses). It must decode into canonical
    # text/image_url blocks so _content_to_text feeds the tool output into the
    # token-exact delta (and _new_images accumulates any nested image) — else the
    # model trains as if it never saw the tool result. Message COUNT must survive
    # the per-block flattening or _run_turn's turn-delta detection desyncs.
    body = {
        "max_tokens": 64,
        "messages": [
            {"role": "user", "content": "solve it"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "compute"},
                    {"type": "tool_use", "id": "tu_1", "name": "py", "input": {"code": "2+2"}},
                ],
            },
            # String tool_result (the common case)
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "tu_1", "content": "4"}]},
            # List tool_result carrying text + a nested image
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tu_2",
                        "content": [
                            {"type": "text", "text": "plot:"},
                            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}},
                        ],
                    }
                ],
            },
        ],
    }
    out = _decode_anthropic(body)

    assert len(out["messages"]) == 4  # per-block flatten never changes message count
    # tool_use in assistant history passes through untouched (never re-tokenized).
    assert out["messages"][1]["content"][1]["type"] == "tool_use"
    # String tool_result -> a text block the delta picks up (was silently dropped).
    str_tr = out["messages"][2]["content"]
    assert str_tr == [{"type": "text", "text": "4"}]
    assert _content_to_text(str_tr) == "4"
    # List tool_result -> flattened text + image_url; one <image> per image, image extracted.
    list_tr = out["messages"][3]["content"]
    assert _content_to_text(list_tr) == "plot:<image>"
    assert _new_images([out["messages"][3]]) == ["data:image/png;base64,AAAA"]
    assert _content_to_text(list_tr).count("<image>") == len(_new_images([out["messages"][3]]))


def test_anthropic_message_body_maps_stop_reason():
    # Anthropic validates stop_reason client-side; raw vLLM finish_reason must map
    # into its vocab (and any non-Anthropic reason falls back to "end_turn"), or the
    # agent's messages.create() raises and crashes the rollout.
    for raw, want in (("stop", "end_turn"), ("length", "max_tokens"), ("tool_calls", "tool_use")):
        assert _anthropic_message_body("policy", "x", raw)["stop_reason"] == want
    for raw in ("abort", "error", "repetition", None):
        assert _anthropic_message_body("policy", "x", raw)["stop_reason"] == "end_turn"
    body = _anthropic_message_body("policy", "hello", "stop")
    assert body["type"] == "message" and body["role"] == "assistant"
    assert body["content"] == [{"type": "text", "text": "hello"}]


def test_response_bodies_parse_with_real_sdk_models():
    # The agent's `await client...create()` parses our HTTP reply into the SDK's
    # pydantic model; a missing/mis-typed field there crashes the rollout. Guard
    # both wires against the REAL models (skips cleanly where the SDK isn't installed).
    chat_types = pytest.importorskip("openai.types.chat")
    anthropic_types = pytest.importorskip("anthropic.types")
    completion = chat_types.ChatCompletion.model_validate(_chat_completion_body("policy", "hi", "abort"))
    assert completion.choices[0].message.content == "hi"
    message = anthropic_types.Message.model_validate(_anthropic_message_body("policy", "hi", "length"))
    assert message.content[0].text == "hi" and message.stop_reason == "max_tokens"


def test_run_turn_budget_exhaustion_commits_for_idempotent_retry():
    # When the prompt alone exhausts the context budget, the turn returns an empty
    # ("", "length") action WITHOUT generating — and must commit it (advance
    # prior_messages) so a connection-break retry replays it instead of treating
    # the same message list as fresh turn-N feedback. engine=None proves generate
    # is never called.
    state = ChatServerState(
        engine=None,
        processor=_FakeChatMLProcessor(),
        model_name="policy",
        max_length=4,
        default_sampling=_sampling(),
    )
    prompt = "a prompt far longer than four tokens"
    state.open("sid", prompt, "l", None)
    session = state.sessions["sid"]
    body = {"messages": [{"role": "user", "content": prompt}]}

    action, finish = asyncio.run(_run_turn(state, session, body))
    assert (action, finish) == ("", "length")
    assert session.trajectory is not None and session.trajectory.truncated
    ntok = len(session.trajectory.observation_tokens)

    # Retry of the exact same request: idempotent replay, no growth, no generate.
    action2, finish2 = asyncio.run(_run_turn(state, session, body))
    assert (action2, finish2) == ("", "length")
    assert len(session.trajectory.observation_tokens) == ntok


class _FakeStructuredVLMProcessor:
    """Structured-content VLM (Qwen/GLM/Kimi-style): its image token is NOT the
    literal "<image>", and apply_chat_template renders {"type":"image"} blocks as
    that model's own token. The literal-flatten path would emit zero of these."""

    image_token = "<|image_pad|>"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        def render(content):
            if isinstance(content, str):
                return content
            parts = []
            for item in content:
                if item.get("type") == "image":
                    parts.append("<|vision_start|><|image_pad|><|vision_end|>")
                elif item.get("type") == "video":
                    parts.append("<|video_pad|>")
                elif item.get("type") == "text":
                    parts.append(item.get("text", ""))
            return "".join(parts)

        s = "".join(f"<|im_start|>{m['role']}\n{render(m['content'])}<|im_end|>\n" for m in messages)
        return s + ("<|im_start|>assistant\n" if add_generation_prompt else "")


def test_per_turn_image_delta_renders_model_image_token():
    # True VLM multi-turn: when the agent returns an image in a later tool/user
    # message, _new_images must pull the image_url out (for _tokenize_feedback to
    # tokenize + accumulate, same as StepEnv's Result.images), and the EXTEND delta
    # must be rendered through the MODEL'S OWN chat template — exactly as the dataset
    # builds turn 0 (flatten to a "<image>" string, then split_image_placeholder for
    # models whose template expands a model-specific token). The image block then
    # becomes that model's real token, NOT a hardcoded literal "<image>" that a
    # structured-content processor would never expand (zero placeholders vs full
    # pixel_values).
    from molt.utils.vlm_utils import should_expand_image_placeholder, split_image_placeholder

    new_messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "<tool_response>"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                {"type": "text", "text": "</tool_response>"},
            ],
        }
    ]
    images = _new_images(new_messages)
    assert images == ["data:image/png;base64,AAAA"]

    # Mirror the _run_turn EXTEND content build.
    proc = _FakeStructuredVLMProcessor()
    turn = {"role": "user", "content": "".join(_content_to_text(m.get("content", "")) for m in new_messages)}
    assert turn["content"] == "<tool_response><image></tool_response>"
    assert should_expand_image_placeholder(proc)  # image_token != "<image>"
    turn = split_image_placeholder(turn)  # "<image>" string -> structured blocks
    delta = _append_user_turn(proc, turn["content"])

    # The model's real image token appears once per image; the literal "<image>" the
    # flatten emits (which this processor would NOT expand) is gone after templating.
    assert delta.count("<|image_pad|>") == len(images)
    assert "<image>" not in delta


def test_run_turn_rolls_back_feedback_when_generate_fails():
    # Turn N appends the feedback delta BEFORE generating. If generate raises, the
    # feedback must be rolled back (and prior_messages left un-advanced) so a client
    # retry re-appends it exactly once rather than doubling it onto the token axis.
    state = ChatServerState(
        engine=_RaisingEngine(),
        processor=_FakeChatMLProcessor(),
        model_name="policy",
        max_length=1000,
        default_sampling=_sampling(),
    )
    state.open("sid", "p", "l", None)
    session = state.sessions["sid"]
    traj = Trajectory(
        prompt="p",
        label="l",
        images=None,
        observation_text="t0",
        observation_tokens=[1, 2, 3],
        rollout_log_probs=[0.0, 0.0, 0.0],
    )
    traj.action_ranges = [(1, 3)]
    session.trajectory = traj
    committed = [{"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}]
    session.prior_messages = list(committed)
    tokens_before, lp_before = list(traj.observation_tokens), list(traj.rollout_log_probs)

    messages = committed + [{"role": "user", "content": "tool result"}]
    with pytest.raises(RuntimeError):
        asyncio.run(_run_turn(state, session, {"messages": messages}))

    assert traj.observation_tokens == tokens_before  # feedback rolled back
    assert traj.rollout_log_probs == lp_before
    assert session.prior_messages == committed  # not advanced → retry re-runs cleanly


class _FakeVLMProcessor(_FakeChatMLProcessor):
    """Text-only ChatML renderer that also advertises an image_processor, so
    _tokenize_feedback takes the multimodal branch (the text-only fake never does)."""

    def __init__(self, scaffold=""):
        super().__init__(scaffold)
        self.image_processor = object()


def test_run_turn_rolls_back_image_state_on_vlm_generate_failure(monkeypatch):
    # REGRESSION: a turn-N message carrying an image grows the trajectory's image
    # state inside _tokenize_feedback (pil_images.extend / mm_train_inputs =
    # accumulate(...) / image_budget +=). If generate then fails and the client
    # retries the same message list, _tokenize_feedback runs again — so the failed
    # attempt's image growth MUST be rolled back, or the image double-counts and
    # pixel_values rows drift from the <image> placeholder ids (vit-embed
    # misalignment). The earlier text-only rollback test cannot catch this: its
    # processor has no image_processor, so _tokenize_feedback skips the image branch.
    import molt.utils.vlm_utils as vlm_utils

    # Fake only the heavy processor-bound calls; accumulate_mm_inputs stays REAL
    # (its new-dict-return semantics are exactly what makes the rollback restore work).
    def _fake_process(tok, text, images):
        toks = tok(text)["input_ids"][0].tolist()
        mm = {"pixel_values": torch.ones(len(images), 3, 4, 4)}
        pil = [f"PIL_new_{i}" for i in range(len(images))]
        return toks, mm, pil

    monkeypatch.setattr(vlm_utils, "process_prompt_with_images", _fake_process)
    monkeypatch.setattr(vlm_utils, "estimate_vllm_input_expansion_delta", lambda *a, **k: 7)

    class _OkEngine:
        async def generate(self, *a, **k):
            out = SimpleNamespace(token_ids=[70, 71], text="ok", finish_reason="stop", logprobs=None)
            return SimpleNamespace(outputs=[out]), 0

    state = ChatServerState(
        engine=_RaisingEngine(),
        processor=_FakeVLMProcessor(),
        model_name="policy",
        max_length=1000,
        default_sampling=_sampling(),
    )
    state.open("sid", "p", "l", ["url_turn0"])
    session = state.sessions["sid"]
    # Turn-0 state as _run_turn would have left it: one prompt image already accumulated.
    traj = Trajectory(
        prompt="p",
        label="l",
        images=["url_turn0"],
        observation_text="t0",
        observation_tokens=[1, 2, 3],
        mm_train_inputs={"pixel_values": torch.ones(1, 3, 4, 4)},
        pil_images=["PIL_turn0"],
        image_budget=7,
        rollout_log_probs=None,
    )
    traj.action_ranges = [(1, 3)]
    session.trajectory = traj
    committed = [{"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}]
    session.prior_messages = list(committed)
    tokens_before = list(traj.observation_tokens)

    # A turn-N tool message that returns an image.
    image_message = {
        "role": "user",
        "content": [
            {"type": "text", "text": "<tool_response>"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
            {"type": "text", "text": "</tool_response>"},
        ],
    }
    messages = committed + [image_message]

    # First attempt fails mid-turn → rollback must undo the image growth.
    with pytest.raises(RuntimeError):
        asyncio.run(_run_turn(state, session, {"messages": messages}))
    assert traj.observation_tokens == tokens_before  # token axis rolled back
    assert traj.pil_images == ["PIL_turn0"]  # the failed turn's image removed
    assert traj.mm_train_inputs["pixel_values"].shape[0] == 1  # not doubled
    assert traj.image_budget == 7
    assert session.prior_messages == committed  # un-advanced → retry re-runs

    # Client retries the exact same message list against a now-healthy engine.
    state.engine = _OkEngine()
    asyncio.run(_run_turn(state, session, {"messages": messages}))
    # The image is accumulated EXACTLY once across the failure+retry.
    assert traj.pil_images == ["PIL_turn0", "PIL_new_0"]
    assert traj.mm_train_inputs["pixel_values"].shape[0] == 2  # 2, not 3
    assert traj.image_budget == 14
    # Token axis: original prompt tokens + this turn's feedback + the generated action.
    assert traj.observation_tokens[: len(tokens_before)] == tokens_before
    assert traj.action_ranges[-1] == (len(traj.observation_tokens) - 2, len(traj.observation_tokens))


class _FakeOmni3LiteralProcessor:
    """Nemotron-Omni LITERAL branch: image_token == "<image>" (so should_expand is
    False) and the chat template string-interpolates message content — a LIST renders
    as its Python repr (zero "<image>" placeholders), exactly like Jinja's
    ``{{ message['content'] }}``. Advertises an image_processor so the VLM tokenize
    path runs."""

    image_token = "<image>"
    image_processor = object()

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        s = "".join(
            f"<|im_start|>{m['role']}\n"
            f"{m['content'] if isinstance(m['content'], str) else str(m['content'])}<|im_end|>\n"
            for m in messages
        )
        return s + ("<|im_start|>assistant\n" if add_generation_prompt else "")

    def __call__(self, text, add_special_tokens=False, return_tensors=None):
        return {"input_ids": torch.tensor([[(ord(c) % 97) + 1 for c in text]])}


class _FakeMutatingStructuredVLMProcessor:
    """Structured-content VLM (Qwen3-VL-like): image_token != "<image>", template
    renders image/image_url blocks to ``<|image_pad|>`` AND — like the REAL Qwen
    processor — MUTATES the passed message content IN PLACE (image_url -> image,
    dropping the url _new_images keys on). A _new_images() read taken AFTER templating
    the raw client messages therefore comes back empty."""

    image_token = "<|image_pad|>"
    image_processor = object()

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        def render(content):
            if isinstance(content, str):
                return content
            parts = []
            for item in content:
                if not isinstance(item, dict):
                    continue
                if item.get("type") in ("image", "image_url"):
                    item.pop("image_url", None)
                    item["type"] = "image"  # mutate in place, like real Qwen3-VL
                    parts.append("<|image_pad|>")
                elif item.get("type") == "text":
                    parts.append(item.get("text", ""))
            return "".join(parts)

        s = "".join(f"<|im_start|>{m['role']}\n{render(m['content'])}<|im_end|>\n" for m in messages)
        return s + ("<|im_start|>assistant\n" if add_generation_prompt else "")

    def __call__(self, text, add_special_tokens=False, return_tensors=None):
        return {"input_ids": torch.tensor([[(ord(c) % 97) + 1 for c in text]])}


@pytest.mark.parametrize(
    "processor, placeholder",
    [
        (_FakeOmni3LiteralProcessor(), "<image>"),
        (_FakeMutatingStructuredVLMProcessor(), "<|image_pad|>"),
    ],
)
def test_run_turn_compaction_keeps_vlm_image_aligned(processor, placeholder, monkeypatch):
    # Context compaction starts a FRESH segment by re-templating the compacted
    # conversation. Feeding the raw OpenAI/Anthropic `image_url` blocks straight to
    # apply_chat_template desyncs the new segment's placeholder count from
    # pixel_values two independent ways:
    #   * a literal-"<image>" template (omni3) renders the content LIST as a repr
    #     string -> ZERO placeholders while the image is still passed;
    #   * a structured-content processor (Qwen3-VL) MUTATES the messages in place
    #     (image_url -> image), so the _new_images() read comes back empty -> the
    #     placeholder is rendered but the image is dropped.
    # Both corrupt the VLM token axis (vit-embed shape mismatch). The new segment
    # must carry exactly the compacted image(s) with one model placeholder each, so
    # this fans out over a faithful fake of each failure mode.
    import molt.utils.vlm_utils as vlm_utils

    def _fake_process(proc, text, images):
        toks = proc(text)["input_ids"][0].tolist()
        mm = {"pixel_values": torch.ones(len(images), 3, 4, 4)} if images else None
        return toks, mm, [f"PIL_{i}" for i in range(len(images))]

    monkeypatch.setattr(vlm_utils, "process_prompt_with_images", _fake_process)
    monkeypatch.setattr(vlm_utils, "estimate_vllm_input_expansion_delta", lambda *a, **k: 0)

    class _Engine:
        async def generate(self, token_ids, sp, multi_modal_data=None):
            tok = [42, 43]
            logprobs = [{t: SimpleNamespace(logprob=-0.1)} for t in tok]  # _build_sampling_params forces logprobs on
            out = SimpleNamespace(token_ids=tok, text="A", finish_reason="stop", logprobs=logprobs)
            return SimpleNamespace(outputs=[out]), 0

    state = ChatServerState(_Engine(), processor, "policy", 10000, _sampling())
    state.open("sid", "P", "l", None)
    session = state.sessions["sid"]
    # A live first segment + the conversation we already tokenized (turn 0 + its echo).
    session.trajectory = _traj()
    session.prior_messages = [{"role": "user", "content": "P"}, {"role": "assistant", "content": "A0"}]

    # Compaction: a shorter, rewritten history that RETAINS a screenshot.
    compacted = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "summary so far"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                {"type": "text", "text": "continue"},
            ],
        }
    ]
    assert _extends_prior(compacted, session.prior_messages) is False
    asyncio.run(_run_turn(state, session, {"messages": compacted}))

    new_seg = session.trajectory
    assert session.segments and new_seg is not session.segments[0]  # prior segment sealed, fresh one live
    # The compacted screenshot survived into the new segment (not dropped by mutation)...
    assert new_seg.pil_images == ["PIL_0"]
    assert new_seg.mm_train_inputs["pixel_values"].shape[0] == 1
    # ...with EXACTLY one model placeholder rendered for it (not zero, not doubled).
    assert new_seg.observation_text.count(placeholder) == 1
