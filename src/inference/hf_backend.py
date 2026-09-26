"""Native Transformers masked inference; no XTuner imports or runtime patches."""

import copy
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer, Qwen3_5ForConditionalGeneration

from src.inputs.schema import DECISION_TOKEN, compile_row


class HFBackend:
    def __init__(
        self,
        checkpoint,
        processor_path=None,
        media_root="",
        max_length=8192,
        device="cuda",
        dtype="bfloat16",
        attn_implementation="sdpa",
        **kwargs,
    ):
        self.checkpoint = str(Path(checkpoint).resolve())
        self.media_root = Path(media_root or ".").resolve()
        self.max_length = max_length
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
        if DECISION_TOKEN not in self.tokenizer.get_added_vocab():
            raise ValueError("Checkpoint must include its trained decision tokenizer")
        self.marker_id = self.tokenizer.convert_tokens_to_ids(DECISION_TOKEN)
        self.processor = AutoProcessor.from_pretrained(processor_path or checkpoint, local_files_only=True)
        # Preserve the trained decision token and exact chat template.
        self.processor.tokenizer = self.tokenizer
        self.model = (
            Qwen3_5ForConditionalGeneration.from_pretrained(
                checkpoint,
                dtype=getattr(torch, dtype),
                local_files_only=True,
                attn_implementation=attn_implementation,
            )
            .to(self.device)
            .eval()
        )
        if self.marker_id >= self.model.get_input_embeddings().weight.shape[0]:
            raise ValueError("Decision marker exceeds checkpoint vocabulary")

    def encode(self, row):
        compiled = compile_row(row, include_targets=False)
        messages = copy.deepcopy(compiled.messages)
        images = []
        if row.get("images"):
            content = messages[1]["content"]
            converted = []
            for part in content:
                if part["type"] == "image_url":
                    path = Path(part["image_url"]["url"])
                    path = path if path.is_absolute() else self.media_root / path
                    with Image.open(path) as image:
                        images.append(image.convert("RGB"))
                    converted.append({"type": "image"})
                else:
                    converted.append(part)
            messages[1]["content"] = converted
        # Keep the empty think block and complete assistant skeleton used in training.
        template = self.processor if images else self.tokenizer
        text = template.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
            add_vision_id=True,
        )
        if images:
            batch = self.processor(text=[text], images=images, return_tensors="pt", padding=False)
        else:
            batch = self.tokenizer(text, add_special_tokens=False, return_tensors="pt")
        length = batch["input_ids"].shape[-1]
        if length > self.max_length:
            raise ValueError(f"Example has {length} tokens, above {self.max_length}; truncation is forbidden")
        positions = (batch["input_ids"][0] == self.marker_id).nonzero().flatten() - 1
        if len(positions) != len(compiled.fields) or (positions < 0).any():
            raise ValueError("Decision marker count or position mismatch")
        return compiled, batch, positions

    @torch.inference_mode()
    def score(self, row):
        compiled, batch, positions = self.encode(row)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        start = time.perf_counter()
        batch = batch.to(self.device)
        # Native HF slices hidden states before the LM head, avoiding L x V logits.
        output = self.model(**batch, use_cache=False, logits_to_keep=positions.to(self.device)).logits[0]
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return compiled, output, batch["input_ids"].shape[-1], (time.perf_counter() - start) * 1000
