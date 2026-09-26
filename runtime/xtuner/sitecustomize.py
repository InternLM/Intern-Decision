"""Runtime compatibility hooks for the pinned XTuner checkout."""

from __future__ import annotations

import functools as _functools
import sys
import types
from pathlib import Path

# XTuner's optional DCP port patch imports a private helper removed by Torch 2.6.
try:
    import torch
    import torch.distributed as _dist

    _name = "torch.distributed.checkpoint"
    _directory = Path(torch.__file__).resolve().parent / "distributed" / "checkpoint"
    _checkpoint = sys.modules.get(_name)
    if _checkpoint is None:

        class _LazyCheckpoint(types.ModuleType):
            _loaded = False
            _loading = False

            def _load(self):
                if self._loading:
                    raise AttributeError("checkpoint package is still initializing")
                if not self._loaded:
                    self._loading = True
                    code = compile(
                        (_directory / "__init__.py").read_text(),
                        str(_directory / "__init__.py"),
                        "exec",
                    )
                    exec(code, self.__dict__)
                    self._loaded = True
                    self._loading = False

            def __getattr__(self, key):
                self._load()
                return self.__dict__[key]

        _checkpoint = _LazyCheckpoint(_name)
        _checkpoint.__package__ = _name
        _checkpoint.__path__ = [str(_directory)]
        _checkpoint.__file__ = str(_directory / "__init__.py")
        sys.modules[_name] = _checkpoint
        setattr(_dist, "checkpoint", _checkpoint)

    _async_name = f"{_name}._async_process_executor"
    if _async_name not in sys.modules:
        from torch.distributed.elastic.utils.distributed import get_free_port

        _async = types.ModuleType(_async_name)
        _async.get_free_port = get_free_port
        sys.modules[_async_name] = _async
        setattr(_checkpoint, "_async_process_executor", _async)

    # Prefer the real FLA modules when the environment has a working GPU/Triton runtime.
    # CPU-only import checks may not have a Triton driver, so retain a small fallback.
    try:
        from fla.modules import FusedRMSNormGated  # noqa: F401
    except Exception:
        import torch.nn as _nn
        import torch.nn.functional as _F

        _fla = types.ModuleType("fla")
        _modules = types.ModuleType("fla.modules")

        class FusedRMSNormGated(_nn.Module):
            def __init__(self, hidden_size, eps=1e-5, activation="swish", **kwargs):
                super().__init__()
                self.weight = _nn.Parameter(torch.ones(hidden_size))
                self.bias = None
                self.eps = eps
                self.activation = activation

            def forward(self, x, g, residual=None, prenorm=False, residual_in_fp32=False):
                if residual is not None:
                    x = x + residual
                dtype = x.dtype
                rms = x.float().pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
                y = (x.float() * rms * self.weight.float()).to(dtype)
                if self.activation in ("swish", "silu"):
                    gate = _F.silu(g)
                elif self.activation == "gelu":
                    gate = _F.gelu(g)
                else:
                    gate = torch.sigmoid(g)
                y = y * gate.to(dtype)
                return (y, x) if prenorm else y

        _modules.FusedRMSNormGated = FusedRMSNormGated
        _fla.modules = _modules
        sys.modules.setdefault("fla", _fla)
        sys.modules["fla.modules"] = _modules

    _extension_name = f"{_name}._extension"
    if _extension_name not in sys.modules:
        _extension = types.ModuleType(_extension_name)

        class StreamTransformExtension:
            pass

        _extension.StreamTransformExtension = StreamTransformExtension
        sys.modules[_extension_name] = _extension
        setattr(_checkpoint, "_extension", _extension)

    from torch.distributed.checkpoint import planner_helpers as _planner_helpers

    if not hasattr(_planner_helpers, "_compare_save_plans"):

        def _compare_save_plans(left, right):
            return left == right

        _planner_helpers._compare_save_plans = _compare_save_plans
    if not hasattr(_planner_helpers, "_merge_delta_local_plans"):

        def _merge_delta_local_plans(cached, delta):
            return [new if getattr(new, "usable", True) else old for old, new in zip(cached, delta)]

        _planner_helpers._merge_delta_local_plans = _merge_delta_local_plans
except Exception:
    pass

# When the environment does not provide causal-conv1d's CUDA extension. XTuner's HF
# implementation only needs the high-level function, so provide a differentiable
# depthwise causal convolution until a prebuilt extension is available.
if "causal_conv1d" not in sys.modules:
    try:
        import torch as _torch
        import torch.nn.functional as _F

        try:
            import causal_conv1d as _real_causal_conv1d  # noqa: F401
        except Exception:
            _causal = types.ModuleType("causal_conv1d")

            def _causal_conv1d_fn(
                x,
                weight,
                bias=None,
                seq_idx=None,
                initial_states=None,
                return_final_states=False,
                final_states_out=None,
                activation=None,
                **kwargs,
            ):
                if x.ndim != 3 or weight.ndim != 2:
                    raise ValueError("causal_conv1d expects x=(batch, channels, length), weight=(channels, width)")
                width = weight.shape[-1]
                if initial_states is not None:
                    prefix = initial_states[..., -(width - 1) :]
                    x_work = _torch.cat((prefix, x), dim=-1)
                else:
                    x_work = _F.pad(x, (width - 1, 0))
                out = _F.conv1d(
                    x_work,
                    weight.unsqueeze(1).to(dtype=x.dtype),
                    bias.to(dtype=x.dtype) if bias is not None else None,
                    groups=x.shape[1],
                )
                if activation in ("silu", "swish"):
                    out = _F.silu(out)
                if return_final_states:
                    final = x_work[..., -(width - 1) :] if width > 1 else x_work[..., :0]
                    if final_states_out is not None:
                        final_states_out.copy_(final)
                        final = final_states_out
                    return out, final
                return out

            def _causal_conv1d_update(x, conv_state, weight, bias=None, activation=None, **kwargs):
                window = _torch.cat((conv_state[..., 1:], x.unsqueeze(-1)), dim=-1)
                out = (window * weight.unsqueeze(0)).sum(dim=-1)
                if bias is not None:
                    out = out + bias
                if activation in ("silu", "swish"):
                    out = _F.silu(out)
                conv_state.copy_(window)
                return out

            _causal.causal_conv1d_fn = _causal_conv1d_fn
            _causal.causal_conv1d_update = _causal_conv1d_update
            sys.modules["causal_conv1d"] = _causal
            sys.modules.setdefault("causal_conv1d_cuda", types.ModuleType("causal_conv1d_cuda"))
    except Exception:
        pass

# Torch 2.6 removed the ignored_params keyword accepted by the pinned XTuner FSDP call.
try:
    from torch.distributed._composable import fsdp as _fsdp

    if "ignored_params" not in str(getattr(_fsdp.fully_shard, "__signature__", "")):
        _fully_shard_origin = _fsdp.fully_shard

        @_functools.wraps(_fully_shard_origin)
        def _fully_shard_compat(module, *args, **kwargs):
            kwargs.pop("ignored_params", None)
            return _fully_shard_origin(module, *args, **kwargs)

        _fsdp.fully_shard = _fully_shard_compat
except Exception:
    pass

try:
    import torch.distributed.fsdp as _fsdp_public

    if "ignored_params" not in str(getattr(_fsdp_public.fully_shard, "__signature__", "")):
        _fully_shard_public_origin = _fsdp_public.fully_shard

        @_functools.wraps(_fully_shard_public_origin)
        def _fully_shard_public_compat(module, *args, **kwargs):
            kwargs.pop("ignored_params", None)
            return _fully_shard_public_origin(module, *args, **kwargs)

        _fsdp_public.fully_shard = _fully_shard_public_compat
except Exception:
    pass

# The pinned XTuner uses replicated DTensors as ignored FSDP parameters. Torch 2.6
# removed fully_shard(ignored_params=...), so leave replicate-only parameters local.
try:
    import torch.distributed.tensor as _dtensor
    from torch.distributed.tensor import Replicate as _Replicate

    _distribute_tensor_origin = _dtensor.distribute_tensor

    @_functools.wraps(_distribute_tensor_origin)
    def _distribute_tensor_compat(tensor, device_mesh, placements=None):
        if placements and all(isinstance(item, _Replicate) for item in placements):
            return tensor
        return _distribute_tensor_origin(tensor, device_mesh, placements)

    _dtensor.distribute_tensor = _distribute_tensor_compat
except Exception:
    pass

# XTuner's load-spec helper uses the pre-2.6 Shard method name.
try:
    from torch.distributed.tensor.placement_types import Shard as _Shard

    if not hasattr(_Shard, "_local_shard_size_and_offset"):

        def _local_shard_size_and_offset(size_on_dim, num_chunks, rank):
            return _Shard._local_shard_size_on_dim(size_on_dim, num_chunks, rank, return_offset=True)

        _Shard._local_shard_size_and_offset = staticmethod(_local_shard_size_and_offset)
except Exception:
    pass

# Torch 2.6's DefaultSavePlanner has four constructor options; XTuner's cache
# planner carries the fifth option introduced by newer Torch.
try:
    from torch.distributed.checkpoint import DefaultSavePlanner as _DefaultSavePlanner

    _default_save_init_origin = _DefaultSavePlanner.__init__

    @_functools.wraps(_default_save_init_origin)
    def _default_save_init_compat(self, *args, **kwargs):
        enable_plan_caching = args[4] if len(args) > 4 else kwargs.pop("enable_plan_caching", False)
        args = args[:4]
        kwargs.pop("enable_plan_caching", None)
        result = _default_save_init_origin(self, *args, **kwargs)
        self._enable_plan_caching = enable_plan_caching
        return result

    _DefaultSavePlanner.__init__ = _default_save_init_compat
except Exception:
    pass
