"""Behavioral checks for causal alignment, leakage, packing, images and shard offsets."""

import copy
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import MethodType, SimpleNamespace

import torch
import torch.nn.functional as F
from PIL import Image
from torch.distributed.tensor.placement_types import Shard
from transformers import AutoTokenizer

from src.inputs.masked import (
    MaskedOpenaiTokenizeFunction,
    MaskedQwen3VLTokenizeFunction,
    compile_row,
    masked_qwen3_vl_collator,
)
from src.model.qwen import DecisionQwen, packed_causal_conv1d
from src.service.examples import examples

fixture_dir = TemporaryDirectory()
root = Path(fixture_dir.name)
visual_calls = []


def frozen_visual(pixels, grid, mesh):
    assert not torch.is_grad_enabled()
    assert pixels.shape == (4, 24) and grid.tolist() == [[1, 2, 2]]
    visual_calls.append(True)
    return torch.full((1, 8), 123.0), []


stub = SimpleNamespace(
    language_model=SimpleNamespace(embed_tokens=torch.nn.Embedding(8, 8)),
    vision_tower=SimpleNamespace(
        patch_embed=SimpleNamespace(in_channels=3, temporal_patch_size=2, patch_size=2),
        spatial_merge_size=2,
    ),
    get_visual_features=frozen_visual,
    training=True,
    only_llm_forward=False,
)
stub._participate_empty_vision = MethodType(DecisionQwen._participate_empty_vision, stub)
context = SimpleNamespace(input_ids=torch.tensor([[1, 2]]), pixel_values=None, sequence_parallel_mesh=None)
embeddings, deepstack, mask = DecisionQwen._prepare_llm_inputs(stub, context)
assert len(visual_calls) == 1 and deepstack is None and mask is None
torch.testing.assert_close(embeddings, stub.language_model.embed_tokens(context.input_ids))
embeddings.sum().backward()
assert stub.language_model.embed_tokens.weight.grad is not None
stub.training = False
DecisionQwen._prepare_llm_inputs(stub, context)
assert len(visual_calls) == 1

model_path = os.environ["MODEL_PATH"]
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
text_fn = MaskedOpenaiTokenizeFunction(tokenizer, chat_template="qwen3.5-vl", max_length=8192)
row = examples()[0] | {"targets": {"box": {"label": "left"}}}
compiled = compile_row(row)
changed = copy.deepcopy(row)
changed["provenance"] = {"expected": "never show this"}
for field, question in changed["questions"].items():
    question["answer"] = {"choice": "not-a-valid-option", "noul": 0}
    changed.setdefault("targets", {})[field] = {"label": "wrong"}
assert compile_row(changed, include_targets=False).messages == compiled.messages
text_item = text_fn(row)
short_fn = MaskedOpenaiTokenizeFunction(tokenizer, chat_template="qwen3.5-vl", max_length=8)
try:
    short_fn(row)
except ValueError:
    pass
else:
    raise AssertionError("Overlength row was not rejected")
vl_fn = MaskedQwen3VLTokenizeFunction(
    tokenizer,
    processor_path=model_path,
    anno_name="synthetic.jsonl",
    chat_template="qwen3.5-vl",
    max_length=8192,
    media_root=str(root),
)
image_path = root / "synthetic.png"
with Image.new("RGB", (192, 128), "red") as image:
    image.save(image_path)
game = copy.deepcopy(row) | {"images": [str(image_path)]}
vl_fn.set_state("cache")
cached = vl_fn(game)
assert cached["num_tokens"] > 0 and sum(cached["num_img_tokens"]) > 0
vl_fn.set_state("runtime")
image_item = vl_fn(game)
assert image_item["num_tokens"] == cached["num_tokens"]
assert image_item["pixel_values"].numel() > 0
assert not torch.equal(image_item["position_ids"][0], image_item["position_ids"][1])
items = [text_item, image_item]
out = masked_qwen3_vl_collator([items], 8192, tokenizer.pad_token_id)[0]
assert out["seq_ctx"].position_ids.shape[-1] == 8192
offset = 0
for item in items:
    for pos, label in enumerate(item["labels"]):
        if label != -100:
            assert out["shifted_labels"][0, offset + pos - 1] == label
    offset += item["num_tokens"]
assert (out["shifted_labels"] != -100).sum() == sum(len(compile_row(r).fields) for r in (row, game))
for size in (17, 2560, 248320):
    shards = [Shard._local_shard_size_and_offset(size, 8, rank) for rank in range(8)]
    indices = [j for length, offset in shards for j in range(offset, offset + length)]
    assert indices == list(range(size)), shards
device = "cpu"
torch.manual_seed(1)
x = torch.randn(1, 19, 7, device=device, requires_grad=True)
w = torch.randn(7, 4, device=device, requires_grad=True)
idx = torch.tensor([[0] * 6 + [1] * 13], device=device)
result = packed_causal_conv1d(x, w, seq_idx=idx, activation="silu")
reference = torch.cat(
    [
        F.silu(
            F.conv1d(x[:, start:end].transpose(1, 2), w[:, None], padding=3, groups=7)[..., : end - start]
        ).transpose(1, 2)
        for start, end in ((0, 6), (6, 19))
    ],
    dim=1,
)
torch.testing.assert_close(result, reference, rtol=1e-5, atol=1e-5)
grad = torch.autograd.grad(result.square().sum(), (x, w), retain_graph=True)
ref_grad = torch.autograd.grad(reference.square().sum(), (x, w))
for actual, expected in zip(grad, ref_grad):
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-4)
print(
    json.dumps(
        {
            "passed": True,
            "device": device,
            "image_tokens": cached["num_img_tokens"],
            "checks": [
                "gold_invariance",
                "overlength_rejection",
                "image_cache_runtime",
                "qwen35_3d_rope",
                "packed_shift",
                "shard_offsets",
                "packed_convolution_output_and_gradients",
                "empty_visual_collectives_preserve_text_embeddings_and_gradients",
            ],
        }
    ),
    flush=True,
)

fixture_dir.cleanup()
