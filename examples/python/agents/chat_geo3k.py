"""Multi-turn geo3k chat agent — python_executor tool calls + `<answer>` grading.

Mirrors examples/python/agents/geo3k.py but uses the black-box chat harness:
agent → stock OpenAI SDK → loopback server (which captures token traces via
the session-prefixed `ctx.base_url`). The agent owns the turn loop; the
server stitches the per-turn token traces into the training trajectory.

Tool-call format: Qwen3 Hermes XML (same as geo3k.py); swap via
``VLLM_TOOL_PARSER_CLS``. Image (single PIL per prompt) is passed inline as
an OpenAI ``image_url`` content item in the first user message.
"""

from __future__ import annotations

import asyncio
import base64
import importlib.util
import io
import json
import os
import re
from pathlib import Path

import torch
from openai import AsyncOpenAI

from molt.agents import ChatAgent, ChatAgentRunner, ChatContext, Result
from molt.utils.vlm_utils import load_images


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_PROJECT_DIR = Path(__file__).resolve().parent.parent
_GRADER = _load_module("math_grader", _PROJECT_DIR / "utils" / "math_grader.py")
_PYTHON_EXECUTOR = _load_module("python_executor", _PROJECT_DIR / "tools" / "python_executor.py").TOOL
_TOOLS = {_PYTHON_EXECUTOR.schema["function"]["name"]: _PYTHON_EXECUTOR}

_MAX_TURNS = int(os.environ.get("MAX_AGENT_TURNS", "5"))
_ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)
_BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")
_PARSER_CLS_PATH = os.environ.get(
    "VLLM_TOOL_PARSER_CLS",
    "vllm.tool_parsers.qwen3xml_tool_parser.Qwen3XMLToolParser",
)
_PARSER = None


def _load_parser():
    from transformers import AutoTokenizer

    module_path, _, cls_name = _PARSER_CLS_PATH.rpartition(".")
    cls = getattr(__import__(module_path, fromlist=[cls_name]), cls_name)
    tok = AutoTokenizer.from_pretrained(os.environ["MODEL_PATH"], trust_remote_code=True)
    return cls(tok)


def _extract_tool_call(text: str) -> dict | None:
    global _PARSER
    if _PARSER is None:
        _PARSER = _load_parser()
    result = _PARSER.extract_tool_calls(text, request=None)
    if not result.tools_called or not result.tool_calls:
        return None
    tc = result.tool_calls[0]
    try:
        args = json.loads(tc.function.arguments or "{}")
    except json.JSONDecodeError:
        args = {}
    return {"name": tc.function.name, "arguments": args}


# Accept `<answer>` (Nemotron Omni convention) and `\\boxed{}` (Qwen / DeepSeek-Math)
# so a single grader works across either prompt distribution — `prepare_geo3k.py
# --answer-format` decides which wrapper the model is asked to emit; this catches
# whichever it actually produces.
def _grade_answer(text: str, label) -> tuple[float, str]:
    if not label:
        return 0.0, ""
    answers = _ANSWER_RE.findall(text)
    if answers:
        answer = answers[-1].strip()
    else:
        # Balanced \boxed/\fbox extraction — handles nested braces (\boxed{\frac{1}{2}})
        # that the flat _BOXED_RE truncates. Dormant on the omni3 <answer> path.
        boxed = _GRADER._last_braced_command(text, r"\boxed") or _GRADER._last_braced_command(text, r"\fbox")
        answer = boxed.strip() if boxed else ""
    if not answer:
        return 0.0, ""
    try:
        result = _GRADER.score_response(f"\\boxed{{{answer}}}", "", label)
        return float(result.get("reward", 0.0)), answer
    except Exception:
        return 0.0, answer


def _pil_to_data_url(pil) -> str:
    buf = io.BytesIO()
    pil.convert("RGB").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _build_first_user_content(prompt_text: str, images) -> list | str:
    """Build a ChatML user content list, interleaving image_url items at the
    positions where ``<image>`` appears in ``prompt_text``. Each ``<image>``
    consumes the next image in order — preserves ordering when prompts mix
    text and multiple images.
    """
    pil_images = load_images(images) if images else []
    if not pil_images:
        return prompt_text
    parts: list = []
    remaining = prompt_text
    for im in pil_images:
        before, sep, remaining = remaining.partition("<image>")
        if before:
            parts.append({"type": "text", "text": before})
        if not sep:  # ran out of placeholders — prepend remaining images
            remaining = sep + remaining
            break
        parts.append({"type": "image_url", "image_url": {"url": _pil_to_data_url(im)}})
    if remaining:
        parts.append({"type": "text", "text": remaining})
    return parts


class Geo3kAgent(ChatAgent):
    async def run(self, ctx: ChatContext) -> Result:
        # Retries ride out the transient loopback connection break when a weight
        # broadcast briefly stalls the server's event loop. They're safe because
        # the server serializes turns per session (a lock) and replays an
        # already-committed turn idempotently, so a retry can't double-process.
        # Generous timeout so slow cold-start generations don't fail outright.
        client = AsyncOpenAI(base_url=ctx.base_url, api_key=ctx.api_key, max_retries=8, timeout=3600)
        messages: list = [{"role": "user", "content": _build_first_user_content(ctx.prompt, ctx.images)}]
        assistant_history: list[str] = []
        tool_call_count = 0
        turn = 0

        for turn in range(1, _MAX_TURNS + 1):
            resp = await client.chat.completions.create(
                model=ctx.model_name,
                messages=messages,
                max_tokens=ctx.sampling_params.max_tokens,
                temperature=ctx.sampling_params.temperature,
            )
            action = resp.choices[0].message.content or ""
            assistant_history.append(action)
            messages.append({"role": "assistant", "content": action})

            tool_call = _extract_tool_call(action)
            # Stop once the model commits a final answer (`<answer>` / `\boxed`),
            # even if it co-emits a tool_call, or when it stops calling tools —
            # avoids post-answer verification loops that inflate length/turns
            # without improving reward (a length-hacking failure mode).
            if _ANSWER_RE.search(action) or _BOXED_RE.search(action) or tool_call is None:
                break

            tool_call_count += 1
            name = tool_call["name"]
            tool = _TOOLS.get(name)
            obs_text = (
                # Offload the blocking subprocess to a thread so it doesn't freeze the
                # shared engine event loop (which would serialize all concurrent rollouts).
                await asyncio.to_thread(tool.execute, tool_call.get("arguments") or {})
                if tool
                else f"Tool `{name}` is not supported. Available: {list(_TOOLS)}"
            )
            messages.append({"role": "user", "content": f"<tool_response>\n{obs_text}\n</tool_response>"})

        reward_value, _ = await asyncio.to_thread(_grade_answer, "\n".join(assistant_history), ctx.label)
        reward = torch.tensor(reward_value, dtype=torch.float32)
        return Result(
            reward=reward,
            info={
                "geo3k_tool_call_total": torch.tensor(float(tool_call_count), dtype=torch.float32),
                "geo3k_correct": reward,
                "turn_index": torch.tensor(float(turn), dtype=torch.float32),
            },
        )


class AgentRunner(ChatAgentRunner):
    def __init__(self):
        super().__init__(Geo3kAgent)
