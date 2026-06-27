import sys
from dataclasses import dataclass
from types import ModuleType


def _install_vllm_test_stub():
    vllm = ModuleType("vllm")
    vllm.__version__ = "0.21.0"

    @dataclass
    class AsyncEngineArgs:
        model: str
        dtype: str = "auto"
        enforce_eager: bool = False

    vllm.AsyncEngineArgs = AsyncEngineArgs
    vllm.AsyncLLMEngine = object

    inputs = ModuleType("vllm.inputs")

    class TokensPrompt(dict):
        pass

    inputs.TokensPrompt = TokensPrompt

    utils = ModuleType("vllm.utils")
    utils.random_uuid = lambda: "test-request-id"

    sys.modules["vllm"] = vllm
    sys.modules["vllm.inputs"] = inputs
    sys.modules["vllm.utils"] = utils


try:
    import vllm.inputs  # noqa: F401
except Exception:
    _install_vllm_test_stub()

import molt.trainer.vllm.vllm_engine as vllm_engine


def test_vllm_ray_executor_uses_worker_gpu_even_when_actor_is_cpu_only():
    assert vllm_engine._vllm_worker_num_gpus("ray", 0) == 1
    assert vllm_engine._vllm_worker_num_gpus("mp", 8) == 8
    assert vllm_engine._vllm_worker_num_gpus("uni", 1) == 1


def test_format_ray_gpu_ids_keeps_all_visible_devices():
    assert vllm_engine._format_ray_gpu_ids([0.0, 2.0]) == "0,2"
    assert vllm_engine._format_ray_gpu_ids(["GPU-abc"]) == "GPU-abc"


def test_filter_vllm_engine_kwargs_drops_unsupported_optional_args(monkeypatch):
    @dataclass
    class FakeAsyncEngineArgs:
        model: str
        dtype: str = "auto"
        enforce_eager: bool = False

    monkeypatch.setattr(vllm_engine.vllm, "AsyncEngineArgs", FakeAsyncEngineArgs)

    filtered = vllm_engine._filter_vllm_engine_kwargs(
        {
            "model": "model-path",
            "dtype": "bfloat16",
            "gdn_prefill_backend": "triton",
        }
    )

    assert filtered == {"model": "model-path", "dtype": "bfloat16"}


def test_filter_vllm_engine_kwargs_keeps_speculative_config(monkeypatch):
    # MTP rollout passes speculative_config; it is a real AsyncEngineArgs field,
    # so it must survive the whitelist filter (not be dropped like unknown kwargs).
    @dataclass
    class FakeAsyncEngineArgs:
        model: str
        dtype: str = "auto"
        speculative_config: object = None

    monkeypatch.setattr(vllm_engine.vllm, "AsyncEngineArgs", FakeAsyncEngineArgs)

    filtered = vllm_engine._filter_vllm_engine_kwargs(
        {"model": "m", "speculative_config": {"num_speculative_tokens": 1}}
    )

    assert filtered == {"model": "m", "speculative_config": {"num_speculative_tokens": 1}}


def test_ray_visible_device_flag_is_cuda_only():
    assert vllm_engine.ray_noset_visible_devices({"RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1"})
    assert not vllm_engine.ray_noset_visible_devices({"RAY_EXPERIMENTAL_NOSET_OTHER_VISIBLE_DEVICES": "1"})


# --- generate_responses fault tolerance --------------------------------------
# generate_responses gathers N per-prompt rollouts with return_exceptions=True so
# a single failed rollout cannot crash the whole prompt group (and, since the
# result is awaited via ray.get upstream, cannot kill the GenerateSamplesActor).
# Bind the real (undecorated) coroutine to a stub self so no Ray runtime / vLLM
# engine is needed — the method only touches self.runner.execute + uuid tagging.

import asyncio  # noqa: E402
from types import SimpleNamespace  # noqa: E402


def _generate_responses_fn():
    # Under the real Ray decorator RolloutRayActor is an ActorClass and the
    # undecorated coroutine lives on __ray_metadata__.modified_class; under the
    # lightweight ray stub another test installs (test_samples_generator), @ray.remote
    # returns the class unchanged so the method is directly accessible. Support both
    # so this test is order-independent in the full suite.
    actor = vllm_engine.RolloutRayActor
    meta = getattr(actor, "__ray_metadata__", None)
    cls = meta.modified_class if meta is not None else actor
    return cls.generate_responses


def _generate_fn():
    actor = vllm_engine.RolloutRayActor
    meta = getattr(actor, "__ray_metadata__", None)
    cls = meta.modified_class if meta is not None else actor
    return cls.generate


class _Traj:
    """Minimal trajectory stub: group_id/rollout_id are assigned by the method."""


def _run_generate(execute, num_samples):
    fn = _generate_responses_fn()
    fake_self = SimpleNamespace(runner=SimpleNamespace(execute=execute))
    return asyncio.run(
        fn(
            fake_self,
            prompt="p",
            label="l",
            sampling_params=1,  # deepcopy-able sentinel
            max_length=8,
            hf_tokenizer=None,
            num_samples=num_samples,
            images=None,
        )
    )


def test_generate_responses_drops_failed_rollouts_and_keeps_the_rest():
    calls = {"n": 0}

    async def execute(**kwargs):
        i = calls["n"]
        calls["n"] += 1
        if i == 1:  # second rollout blows up
            raise RuntimeError("vllm boom")
        return _Traj()

    out = _run_generate(execute, num_samples=3)

    assert len(out) == 2  # 3 dispatched, 1 dropped
    assert len({t.group_id for t in out}) == 1  # all share the prompt's group id
    assert len({t.rollout_id for t in out}) == 2  # one rollout id per surviving rollout


def test_generate_responses_returns_empty_when_all_rollouts_fail():
    async def execute(**kwargs):
        raise RuntimeError("always fails")

    # Must not raise — an all-failed group comes back empty and is refilled.
    assert _run_generate(execute, num_samples=4) == []


def test_generate_responses_shares_rollout_id_across_multiturn_steps():
    # Happy path unchanged: a multi-turn rollout returns a list of step-samples;
    # they share one rollout_id, and all rollouts of the prompt share the group id.
    async def execute(**kwargs):
        return [_Traj(), _Traj()]  # one rollout flattens to two step-samples

    out = _run_generate(execute, num_samples=2)

    assert len(out) == 4  # 2 rollouts x 2 steps
    assert len({t.group_id for t in out}) == 1
    assert len({t.rollout_id for t in out}) == 2  # shared within a rollout, distinct across


def test_generate_marks_only_tokens_before_weight_change_off_policy():
    owner = SimpleNamespace(
        _weight_version=0,
        _mm_pad_token_ids=None,
        _mm_image_placeholder_id=None,
        _mm_image_start_ids=None,
    )

    class _LLM:
        def generate(self, *args, **kwargs):
            async def stream():
                yield SimpleNamespace(outputs=[SimpleNamespace(token_ids=[10, 11])])
                owner._weight_version = 1
                yield SimpleNamespace(outputs=[SimpleNamespace(token_ids=[10, 11, 12])])
                yield SimpleNamespace(outputs=[SimpleNamespace(token_ids=[10, 11, 12, 13])])

            return stream()

    owner.llm = _LLM()
    final, off_policy_len = asyncio.run(_generate_fn()(owner, [1], SimpleNamespace()))

    assert final.outputs[0].token_ids == [10, 11, 12, 13]
    assert off_policy_len == 2
