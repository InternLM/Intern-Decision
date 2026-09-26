"""XTuner training adapters for the shared masked decision compiler."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict
from xtuner.v1.datasets.data_item import DataItem, QwenVL3DataItem
from xtuner.v1.datasets.mllm_tokenize_fn.qwen3_vl_tokenize_fn import Qwen3VLTokenizeFunction
from xtuner.v1.datasets.sft_tokenize_fn.openai import OpenaiTokenizeFunction

# Compatibility exports for existing training/evaluation callers.
from src.inputs.schema import (  # noqa: F401
    ANSWER_SYMBOLS,
    DECISION_TOKEN,
    SYSTEM_PROMPT,
    CompiledExample,
    _answer_value,
    _options,
    compile_row,
)


def _attach_image_sizes(messages: list[dict[str, Any]], media_root: str) -> None:
    from PIL import Image

    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            image_url = part.get("image_url") if isinstance(part, dict) else None
            if not isinstance(image_url, dict) or image_url.get("image_wh"):
                continue
            url = image_url.get("url")
            if not isinstance(url, str) or url.startswith(("http://", "https://", "s3://")):
                continue
            image_path = Path(url) if Path(url).is_absolute() else Path(media_root) / url
            with Image.open(image_path) as image:
                image_url["image_wh"] = [image.width, image.height]


def _target_ids(tokenizer, compiled: CompiledExample) -> list[int]:
    for symbols in compiled.symbols.values():
        if any(len(tokenizer.encode(s, add_special_tokens=False)) != 1 for s in symbols):
            raise ValueError("Every candidate symbol must encode to exactly one token")
    if compiled.targets is None or len(compiled.targets) != len(compiled.fields):
        raise ValueError("training examples require one target symbol per field")
    ids: list[int] = []
    for field in compiled.fields:
        symbol = compiled.targets[field]
        encoded = tokenizer.encode(symbol, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(f"decision symbol {symbol!r} for {field!r} must encode to one token")
        ids.append(encoded[0])
    return ids


def _mask_labels(item: DataItem | QwenVL3DataItem, marker_id: int, target_ids: list[int]) -> DataItem | QwenVL3DataItem:
    labels = item.get("labels")
    if labels is None:
        return item
    input_ids = item["input_ids"]
    marker_positions = [index for index, token_id in enumerate(input_ids) if token_id == marker_id]
    if not marker_positions:
        raise ValueError("compiled example contains no decision marker token")
    if len(marker_positions) != len(target_ids):
        raise ValueError(f"decision marker count {len(marker_positions)} does not match target count {len(target_ids)}")
    masked = [-100] * len(labels)
    for position, target_id in zip(marker_positions, target_ids):
        if position < len(masked):
            masked[position] = target_id
    item["labels"] = masked
    return item


class MaskedOpenaiTokenizeFunction(OpenaiTokenizeFunction):
    def __init__(self, tokenizer, *args, **kwargs):
        self.length_limit = kwargs.pop("max_length", None)
        tokenizer.add_special_tokens({"additional_special_tokens": [DECISION_TOKEN]})
        self.decision_token_id = tokenizer.convert_tokens_to_ids(DECISION_TOKEN)
        super().__init__(tokenizer, *args, max_length=None, **kwargs)

    def __call__(self, item: dict | list, include_targets=True, **kwargs):
        raw = item if isinstance(item, dict) else {"questions": {}, "state": item}
        compiled = compile_row(raw, include_targets=include_targets)
        target_ids = _target_ids(self.tokenizer, compiled) if include_targets else [-100] * len(compiled.fields)
        tokenized = super().__call__({"messages": compiled.messages}, **kwargs)
        _check_length(tokenized, self.length_limit)
        return _mask_labels(tokenized, self.decision_token_id, target_ids)

    def hash(self):
        return hashlib.sha256(
            (Path(__file__).read_bytes() + Path(__file__).with_name("schema.py").read_bytes())
            + str(self.length_limit).encode()
        ).hexdigest()[:32]


class MaskedQwen3VLTokenizeFunction(Qwen3VLTokenizeFunction):
    def __init__(self, tokenizer, *args, **kwargs):
        self.media_root = kwargs.pop("media_root", "")
        self.length_limit = kwargs.pop("max_length", None)
        tokenizer.add_special_tokens({"additional_special_tokens": [DECISION_TOKEN]})
        self.decision_token_id = tokenizer.convert_tokens_to_ids(DECISION_TOKEN)
        super().__init__(tokenizer, *args, max_length=None, **kwargs)

    def __call__(self, item: dict, media_root: str = "", include_targets=True, **kwargs):
        media_root = media_root or self.media_root
        compiled = compile_row(item, include_targets=include_targets)
        target_ids = _target_ids(self.tokenizer, compiled) if include_targets else [-100] * len(compiled.fields)
        _attach_image_sizes(compiled.messages, media_root)
        tokenized = super().__call__({"messages": compiled.messages}, media_root=media_root, **kwargs)
        _check_length(tokenized, self.length_limit)
        if "input_ids" not in tokenized:
            return tokenized
        if tokenized.get("image_grid_thw") is not None:
            import torch
            from xtuner.v1.datasets.mllm_tokenize_fn.qwenvl_rope2d import get_rope_index_3

            tokenized["position_ids"] = get_rope_index_3(
                torch.tensor(tokenized["input_ids"]).unsqueeze(0),
                image_grid_thw=tokenized["image_grid_thw"],
                image_token_id=self.img_context_token_id,
                video_token_id=self.video_context_token_id,
                vision_start_token_id=self.img_start_token_id,
                spatial_merge_size=self.image_processor.merge_size,
            )
        return _mask_labels(tokenized, self.decision_token_id, target_ids)

    def hash(self):
        return hashlib.sha256(
            (Path(__file__).read_bytes() + Path(__file__).with_name("schema.py").read_bytes())
            + f"{self.length_limit}:{self.media_root}:{self.image_processor.size}".encode()
        ).hexdigest()[:32]


def _check_length(item, limit):
    if item.get("num_tokens", 0) <= 0:
        raise ValueError("Tokenizer rejected a row; refusing to drop it")
    if limit is not None and item["num_tokens"] > limit:
        raise ValueError(f"Example has {item['num_tokens']} tokens, above {limit}; truncation is forbidden")


class MaskedOpenaiTokenizeFunctionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    chat_template: str = "qwen3.5-vl"
    max_length: int | None = None
    hash: str | None = None

    def build(self, tokenizer, tokenizer_hash=None, anno_name=None, **kwargs):
        return MaskedOpenaiTokenizeFunction(
            tokenizer,
            chat_template=self.chat_template,
            max_length=self.max_length,
            hash=self.hash,
            tokenizer_hash=tokenizer_hash,
        )


class MaskedQwen3VLTokenizeFnConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    chat_template: str = "qwen3.5-vl"
    processor_path: str
    media_root: str = ""
    max_length: int | None = None
    llm_pack_weight: float = 1.0
    visual_pack_weight: float = 0.0
    max_pixels: int | None = None
    min_pixels: int | None = None
    rand_video_max_frames: int = 24
    hash: str | None = None

    def build(self, tokenizer, tokenizer_hash=None, anno_name=None, **kwargs):
        return MaskedQwen3VLTokenizeFunction(
            tokenizer,
            processor_path=self.processor_path,
            media_root=self.media_root,
            anno_name=anno_name or "dataset.jsonl",
            chat_template=self.chat_template,
            max_length=self.max_length,
            llm_pack_weight=self.llm_pack_weight,
            visual_pack_weight=self.visual_pack_weight,
            max_pixels=self.max_pixels,
            min_pixels=self.min_pixels,
            rand_video_max_frames=self.rand_video_max_frames,
            hash=self.hash,
            tokenizer_hash=tokenizer_hash,
        )


def _masked_collate_one(
    instance,
    pack_max_length: int,
    padding_token_idx: int,
    pack_to_max_length: bool,
    pad_chunk_size: int,
    multimodal: bool = False,
):
    """Pack masked labels with XTuner's causal next-token alignment.

    Labels are written at the <decision> marker by the tokenizer. The causal
    shift below places each target at the preceding input position, matching
    the logits used by inference and Jev constrained decoding.
    """
    import torch
    from xtuner.v1.data_proto import SequenceContext
    from xtuner.v1.utils import IGNORE_INDEX
    from xtuner.v1.utils.pad import pad_to_max_length

    if isinstance(instance, dict):
        instance = [instance]
    total = sum(item["num_tokens"] for item in instance)
    if total > pack_max_length:
        raise ValueError(f"packed sample has {total} tokens, above pack_max_length={pack_max_length}")
    input_ids = torch.cat([torch.tensor(item["input_ids"]).view(1, -1) for item in instance], dim=-1)
    labels = torch.cat([torch.tensor(item["labels"]).view(1, -1) for item in instance], dim=-1)
    input_ids = input_ids[:, :-1]
    labels = labels[:, 1:]
    num_tokens = [item["num_tokens"] for item in instance]
    if num_tokens[-1] == 1:
        num_tokens = num_tokens[:-1]
    else:
        num_tokens[-1] -= 1
    pad_len = pack_max_length - input_ids.shape[-1] if pack_to_max_length else 0
    if pad_len > 0:
        input_ids = pad_to_max_length(input_ids, padding_token_idx, max_length=pack_max_length, dim=-1)
        labels = pad_to_max_length(labels, IGNORE_INDEX, max_length=pack_max_length, dim=-1)
        full, remainder = divmod(pad_len, pad_chunk_size)
        num_tokens = [0] + num_tokens + [pad_chunk_size] * full
        if remainder:
            num_tokens.append(remainder)
    elif pad_len < 0:
        raise ValueError(f"packed sample has {input_ids.shape[-1]} tokens, above pack_max_length={pack_max_length}")
    else:
        num_tokens = [0] + num_tokens
    cu_seq_lens = torch.cumsum(torch.IntTensor(num_tokens), dim=0).int()
    kwargs = {}
    if multimodal:
        pixel_values = [item["pixel_values"] for item in instance if "pixel_values" in item]
        kwargs["pixel_values"] = torch.cat(pixel_values, dim=0) if pixel_values else None
        kwargs["num_img_tokens"] = [item.get("num_img_tokens", [0]) for item in instance]
        image_grid_thw = [item["image_grid_thw"] for item in instance if "image_grid_thw" in item]
        kwargs["image_grid_thw"] = torch.cat(image_grid_thw, dim=0) if image_grid_thw else None
        position_ids = [
            item["position_ids"]
            if item.get("position_ids") is not None
            else torch.arange(item["num_tokens"]).view(1, 1, -1).expand(3, 1, -1)
            for item in instance
        ]
        if position_ids:
            merged_position_ids = torch.cat(position_ids, dim=-1)[..., :-1]
            if pack_to_max_length and merged_position_ids.shape[-1] < pack_max_length:
                merged_position_ids = pad_to_max_length(merged_position_ids, 0, max_length=pack_max_length, dim=-1)
            kwargs["position_ids"] = merged_position_ids
    seq_ctx = SequenceContext(
        input_ids=input_ids,
        cu_seq_lens_q=cu_seq_lens,
        cu_seq_lens_k=cu_seq_lens,
        max_length_q=max(num_tokens),
        max_length_k=max(num_tokens),
        num_padding=pad_len,
        **kwargs,
    )
    return {"seq_ctx": seq_ctx, "shifted_labels": labels}


def masked_sft_llm_collator(instances, pack_max_length, padding_token_idx, pack_to_max_length=True, pad_chunk_size=256):
    return [
        _masked_collate_one(i, pack_max_length, padding_token_idx, pack_to_max_length, pad_chunk_size)
        for i in instances
    ]


def masked_qwen3_vl_collator(
    instances, pack_max_length, padding_token_idx, pack_to_max_length=True, pad_chunk_size=256
):
    return [
        _masked_collate_one(i, pack_max_length, padding_token_idx, pack_to_max_length, pad_chunk_size, True)
        for i in instances
    ]
