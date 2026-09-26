"""Qwen3.5 with checked loading and sequence-isolated causal convolution."""

import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from xtuner.v1.model import Qwen3_5_VLDense4BConfig
from xtuner.v1.model.compose.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration


def nonreentrant_checkpoint(module, *, preserve_rng_state=True):
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointImpl, checkpoint_wrapper

    return checkpoint_wrapper(
        module,
        checkpoint_impl=CheckpointImpl.NO_REENTRANT,
        preserve_rng_state=preserve_rng_state,
    )


def packed_causal_conv1d(x, weight, bias=None, activation=None, seq_idx=None):
    """Channel-last depthwise convolution, with zero history at every row boundary.

    Accumulate the short (four-tap) kernel in FP32, matching the CUDA contract.
    Masking each tap preserves gradients while preventing cross-example inputs.
    """
    length = x.shape[1]
    out = torch.zeros_like(x, dtype=torch.float32)
    for tap in range(weight.shape[-1]):
        lag = weight.shape[-1] - 1 - tap
        if lag >= length:
            continue
        source = x[:, : length - lag].float()
        if lag:
            if seq_idx is not None:
                valid = seq_idx[:, lag:] == seq_idx[:, : length - lag]
                source = source * valid.unsqueeze(-1)
            source = F.pad(source, (0, 0, lag, 0))
        out = out + source * weight[:, tap].float()
    if bias is not None:
        out = out + bias.float()
    if activation in {"silu", "swish"}:
        out = F.silu(out)
    elif activation is not None:
        raise ValueError(f"Unsupported convolution activation {activation}")
    return out.to(x.dtype)


class DecisionQwen(Qwen3VLForConditionalGeneration):
    def _participate_empty_vision(self, seq_ctx, device):
        """Match visual FSDP collectives without adding image evidence to text rows."""
        patch = self.vision_tower.patch_embed
        merge = self.vision_tower.spatial_merge_size
        mesh = seq_ctx.sequence_parallel_mesh
        sp_size = mesh.size() if mesh is not None else 1
        width = patch.in_channels * patch.temporal_patch_size * patch.patch_size**2
        pixels = torch.zeros(merge**2 * sp_size, width, device=device, dtype=torch.bfloat16)
        grid = torch.tensor([[1, merge, merge * sp_size]], device=device)
        with torch.no_grad():
            self.get_visual_features(pixels, grid, mesh)
        if not getattr(self, "_empty_vision_logged", False):
            from xtuner.v1.utils import get_logger

            get_logger().info("Text-only batch participated in frozen visual FSDP collectives; outputs discarded")
            self._empty_vision_logged = True

    def _prepare_llm_inputs(self, seq_ctx):
        """Frozen visual features, with no upstream exception-and-ignore fallback."""
        inputs = self.language_model.embed_tokens(seq_ctx.input_ids)
        if seq_ctx.pixel_values is None:
            if self.training and not self.only_llm_forward:
                self._participate_empty_vision(seq_ctx, inputs.device)
            return inputs, None, None
        if self.only_llm_forward or seq_ctx.image_grid_thw is None:
            raise ValueError("Image input requires the multimodal model and image grid")
        with torch.no_grad():
            visual, deepstack = self.get_visual_features(
                seq_ctx.pixel_values, seq_ctx.image_grid_thw, seq_ctx.sequence_parallel_mesh
            )
            mask, visual, deepstack = self.get_placeholder_mask(
                seq_ctx.input_ids,
                visual,
                deepstack,
                seq_ctx.pixel_values.size(0),
                seq_ctx.sequence_parallel_mesh,
            )
        inputs = inputs.clone()
        inputs[mask] = visual.to(inputs.dtype)
        return inputs, deepstack or None, mask if deepstack else None

    def __init__(self, config):
        super().__init__(config)
        for module in self.language_model.modules():
            if hasattr(module, "causal_conv1d_fn"):
                module.causal_conv1d_fn = packed_causal_conv1d
        if any(p.requires_grad for p in self.vision_tower.parameters()):
            raise ValueError("Vision tower must remain frozen")
        if any(p.requires_grad for p in self.multi_modal_projector.parameters()):
            raise ValueError("Vision projector must remain frozen")

    def from_hf(self, hf_path, strict=True):
        super().from_hf(hf_path, strict=True)
        if not os.environ.get("WEIGHT_AUDIT"):
            return
        from torch.distributed.tensor import DTensor

        # Verify complete local shards of ordinary (non-fused) tensors at the
        # beginning, middle and end of the language model and in both vision parts.
        index = json.loads((Path(hf_path) / "model.safetensors.index.json").read_text())["weight_map"]
        checked = []
        for part_name in ("language_model", "vision_tower", "multi_modal_projector"):
            part = getattr(self, part_name)
            candidates = [
                (part._clean_param_name(n), p, part.load_spec_mapping[part._clean_param_name(n)])
                for n, p in part.named_parameters()
                if part._clean_param_name(n) in part.load_spec_mapping
                and not part.load_spec_mapping[part._clean_param_name(n)].is_fused
            ]
            picks = sorted({0, len(candidates) // 2, len(candidates) - 1})
            if not candidates:
                raise AssertionError(f"No auditable tensors in {part_name}")
            for i in picks:
                name, param, spec = candidates[i]
                keys = spec.global_hf_keys
                if len(keys) != 1:
                    raise AssertionError(f"Expected one HF tensor for {name}")
                key = keys[0]
                with safe_open(str(Path(hf_path) / index[key]), framework="pt", device="cpu") as f:
                    expected = f.get_tensor(key)
                if isinstance(param, DTensor):
                    # Independent Torch tensor_split oracle, not XTuner LoadSpec.
                    for dim, placement in enumerate(param.placements):
                        if placement.is_shard():
                            chunks = torch.chunk(expected, param.device_mesh.size(dim), dim=placement.dim)
                            expected = chunks[param.device_mesh.get_local_rank(dim)]
                    actual = param.to_local().detach().cpu()
                else:
                    actual = param.detach().cpu()
                if actual.shape != expected.shape or not torch.equal(actual, expected.to(actual.dtype)):
                    raise AssertionError(f"Loaded tensor differs from original: {part_name}.{name}")
                checked.append(f"{part_name}.{name}")
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        out = Path(os.environ["WEIGHT_AUDIT"])
        out.mkdir(parents=True, exist_ok=True)
        (out / f"rank-{rank}.json").write_text(json.dumps({"passed": True, "tensors": checked}, indent=2))


class DecisionQwenConfig(Qwen3_5_VLDense4BConfig):
    def build(self):
        # Project-local adaptation of XTuner's fixed reentrant wrapper.
        # The frozen visual modules do not need activation checkpointing.
        import xtuner.v1.model.dense.dense as dense_module

        dense_module.apply_activation_checkpointing = nonreentrant_checkpoint
        return DecisionQwen(self)
