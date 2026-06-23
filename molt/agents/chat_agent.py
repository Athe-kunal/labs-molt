"""Chat-completion agent — public face for the chat-agent path.

Subclass `ChatAgent` and implement `run(ctx)`. The rollout is a true black
box: any stock OpenAI *or* Anthropic client (opencode, claude code,
AgentScope, ...) just points at the session URL with `api_key=EMPTY`. The
server transparently captures the token trace because the URL carries the
session id as a path prefix (`/s/<id>`). No `extra_body`, no `logprobs=True`,
no auth/session plumbing leaks into agent code.

    from openai import AsyncOpenAI
    from molt.agents import ChatAgent, ChatAgentRunner, ChatContext, Result

    class MyAgent(ChatAgent):
        async def run(self, ctx: ChatContext) -> Result:
            client = AsyncOpenAI(base_url=ctx.base_url, api_key=ctx.api_key)
            resp = await client.chat.completions.create(
                model=ctx.model_name,
                messages=[{"role": "user", "content": ctx.prompt}],
                max_tokens=ctx.sampling_params.max_tokens,
            )
            return Result(reward=grade(resp.choices[0].message.content, ctx.label))

    class AgentRunner(ChatAgentRunner):
        def __init__(self):
            super().__init__(MyAgent)

Same server, Anthropic wire — point `AsyncAnthropic` at `ctx.session_url`
(no `/v1` suffix; the SDK appends `/v1/messages` itself):

    from anthropic import AsyncAnthropic

    client = AsyncAnthropic(base_url=ctx.session_url, api_key=ctx.api_key)
    msg = await client.messages.create(
        model=ctx.model_name,
        messages=[{"role": "user", "content": ctx.prompt}],
        max_tokens=ctx.sampling_params.max_tokens,
    )
    text = msg.content[0].text

A FastAPI vLLM server runs on loopback under `ctx.session_url`. The same
endpoint also serves external HTTP callers (third-party agent frameworks that
speak either the OpenAI or the Anthropic HTTP wire protocol).
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from molt.agents._chat_server import (
    ChatServerState,
    mount_session_capture,
    serve_on_current_loop,
    stitch_session,
)
from molt.agents.base import Result, Runner
from molt.utils.logging_utils import init_logger

# The chat server registers this as the policy model id; agents request it via
# ``ctx.model_name`` (vLLM validates ``request.model`` against served names).
_SERVED_MODEL_NAME = "policy"

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# ChatContext — per-rollout bundle handed to ChatAgent.run().
# ---------------------------------------------------------------------------
@dataclass
class ChatContext:
    prompt: str
    label: Any
    images: list | None
    # Session-scoped base_url WITH the "/v1" suffix, e.g.
    # "http://127.0.0.1:port/s/<sid>/v1". Point a stock OpenAI client here
    # (the SDK appends "/chat/completions"); every call is traced transparently.
    base_url: str
    # The same session root WITHOUT "/v1", e.g. ".../s/<sid>". Point a stock
    # Anthropic client here — AsyncAnthropic appends "/v1/messages" itself.
    session_url: str
    model_name: str
    api_key: str  # always "EMPTY" — session lives in base_url, not auth
    session_id: str  # raw id; rarely needed once base_url carries it
    sampling_params: Any
    max_length: int


# ---------------------------------------------------------------------------
# ChatAgent — behavior class for the chat-agent path. User owns the loop and uses
# `openai.AsyncOpenAI` directly.
# ---------------------------------------------------------------------------
class ChatAgent(ABC):
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    async def run(self, ctx: ChatContext) -> Result:
        """Run one episode against an OpenAI-compatible endpoint and score it.

        Point ``openai.AsyncOpenAI(base_url=ctx.base_url, api_key=ctx.api_key)``
        at the session URL; every call through it is traced and stitched into a
        single training Trajectory automatically. Use ``ctx.prompt`` /
        ``ctx.label`` / ``ctx.images`` for the task and ``ctx.sampling_params`` /
        ``ctx.max_length`` for the generation budget.

        Returns:
            Result with a scalar ``reward`` (required); ``score`` and ``info``
            are optional (same contract as ``Env.step``).
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# ChatAgentRunner — drives a ChatAgent subclass; stitches its chat traces into
# a list of Trajectory step-samples the trainer consumes.
# ---------------------------------------------------------------------------
class ChatAgentRunner(Runner):
    def __init__(self, agent_cls: type[ChatAgent]):
        assert issubclass(agent_cls, ChatAgent), "agent_cls must inherit from ChatAgent"
        self.agent_cls = agent_cls
        self._state: ChatServerState | None = None
        self._server_root: str | None = None
        self._server_task: asyncio.Task | None = None
        self._boot_lock = asyncio.Lock()  # serialize first-call server bring-up

    async def _ensure_server(self, llm_engine, hf_tokenizer, max_length, sampling_params):
        """Bring the transparent token-in-token-out server up once, on the actor's
        OWN event loop (so it shares the AsyncLLM engine — no daemon thread, no
        cross-loop await). Concurrent first calls are serialized; later return at once."""
        if self._state is not None:
            return
        async with self._boot_lock:
            if self._state is not None:
                return
            from fastapi import FastAPI

            app = FastAPI()

            @app.get("/health")
            async def _health():
                return {"status": "ok", "model": _SERVED_MODEL_NAME}

            # engine = the rollout actor; state.engine.generate(token_ids, sp, mm_data)
            # is the token-in path that returns exact generated ids + logprobs.
            state = ChatServerState(llm_engine, hf_tokenizer, _SERVED_MODEL_NAME, max_length, sampling_params)
            mount_session_capture(app, state)
            port, self._server_task = await serve_on_current_loop(app)
            self._server_root = f"http://127.0.0.1:{port}"
            self._state = state
            logger.info(f"Chat server ready at {self._server_root} (model={_SERVED_MODEL_NAME})")

    async def execute(self, prompt, label, sampling_params, max_length, hf_tokenizer, llm_engine, images=None):
        await self._ensure_server(llm_engine, hf_tokenizer, max_length, sampling_params)
        session_id = uuid4().hex
        self._state.open(session_id, prompt, label, images)
        session_root = f"{self._server_root}/s/{session_id}"
        ctx = ChatContext(
            prompt=prompt,
            label=label,
            images=images,
            base_url=f"{session_root}/v1",
            session_url=session_root,
            model_name=self._state.model_name,
            api_key="EMPTY",
            session_id=session_id,
            sampling_params=sampling_params,
            max_length=max_length,
        )
        try:
            result = await self.agent_cls().run(ctx)
            if not isinstance(result, Result):
                raise TypeError(f"ChatAgent.run must return a Result, got {type(result).__name__}")
            return stitch_session(self._state, session_id, result)
        finally:
            self._state.discard(session_id)
