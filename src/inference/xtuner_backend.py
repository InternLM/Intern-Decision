"""Shared, single-forward masked decision inference for text and local images."""

import os
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer
from xtuner.v1.data_proto import SequenceContext

from src.inputs.masked import (
    DECISION_TOKEN,
    MaskedOpenaiTokenizeFunction,
    MaskedQwen3VLTokenizeFunction,
    compile_row,
)
from src.model.size_config import decision_config_from_hf


class XTunerBackend:
    def __init__(self, checkpoint, processor_path=None, media_root="", max_length=8192, **kwargs):
        os.environ["XTUNER_HF_IMPL"] = "1"
        checkpoint = Path(checkpoint).resolve()
        self.checkpoint = str(checkpoint)
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
        if DECISION_TOKEN not in self.tokenizer.get_added_vocab():
            raise ValueError("Checkpoint must include its trained decision tokenizer")
        self.marker_id = self.tokenizer.convert_tokens_to_ids(DECISION_TOKEN)
        self.text_fn = MaskedOpenaiTokenizeFunction(
            self.tokenizer,
            chat_template="qwen3.5-vl",
            max_length=max_length,
        )
        self.image_fn = MaskedQwen3VLTokenizeFunction(
            self.tokenizer,
            processor_path=processor_path or str(checkpoint),
            anno_name="inference.jsonl",
            media_root=media_root,
            chat_template="qwen3.5-vl",
            max_length=max_length,
        )
        self.text_fn.set_state("runtime")
        self.image_fn.set_state("runtime")
        self.model = (
            decision_config_from_hf(
                checkpoint,
                only_llm_forward=False,
                freeze_vision=True,
                freeze_projector=True,
                freeze_language=False,
                compile_cfg=False,
            )
            .build()
            .to(device="cuda", dtype=torch.bfloat16)
        )
        self.model.from_hf(str(checkpoint), strict=True)
        if self.marker_id >= self.model.language_model.embed_tokens.weight.shape[0]:
            raise ValueError("Decision token exceeds model vocabulary")
        self.model.eval()

    @torch.inference_mode()
    def score(self, row):
        compiled = compile_row(row, include_targets=False)
        item = (self.image_fn if row.get("images") else self.text_fn)(row, include_targets=False)
        ids = item["input_ids"]
        positions = [i - 1 for i, token in enumerate(ids) if token == self.marker_id]
        if len(positions) != len(compiled.fields) or min(positions) < 0:
            raise ValueError("Decision marker count or position mismatch")
        device = next(self.model.parameters()).device
        torch.cuda.synchronize(device)
        forward_start = time.perf_counter()
        length = len(ids)
        position_ids = item.get("position_ids")
        if position_ids is None:
            position_ids = torch.arange(length).view(1, -1)
        context = SequenceContext(
            input_ids=torch.tensor([ids], dtype=torch.long, device=device),
            cu_seq_lens_q=torch.tensor([0, length], dtype=torch.int32, device=device),
            cu_seq_lens_k=torch.tensor([0, length], dtype=torch.int32, device=device),
            max_length_q=length,
            max_length_k=length,
            num_padding=0,
            position_ids=position_ids.to(device),
            pixel_values=item["pixel_values"].to(device, torch.bfloat16) if "pixel_values" in item else None,
            image_grid_thw=item["image_grid_thw"].to(device) if "image_grid_thw" in item else None,
        )

        # Final normalization is followed only by the output head. Selecting here
        # avoids allocating length x vocabulary logits without altering attention.
        def select_markers(module, args, output):
            return output[:, positions, :]

        hook = self.model.language_model.norm.register_forward_hook(select_markers)
        try:
            output = self.model(seq_ctx=context).logits[0]
        finally:
            hook.remove()
        torch.cuda.synchronize(device)
        inference_ms = (time.perf_counter() - forward_start) * 1000
        return compiled, output, length, inference_ms
