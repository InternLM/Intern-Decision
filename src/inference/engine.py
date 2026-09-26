"""Shared local inference interface and answer decoding for both backends."""

from pathlib import Path

import torch

from src.inference.temperature import load_calibration, scale_result
from src.inputs.schema import _options


class DecisionEngine:
    def __init__(
        self,
        checkpoint,
        processor_path=None,
        media_root="",
        max_length=8192,
        calibration_path=None,
        backend="xtuner",
        **backend_options,
    ):
        # Existing evaluation calls keep their tested XTuner backend. Service/config
        # callers choose HF explicitly; imports remain lazy and backend-specific.
        if backend == "hf":
            from src.inference.hf_backend import HFBackend

            backend_class = HFBackend
        elif backend == "xtuner":
            from src.inference.xtuner_backend import XTunerBackend

            backend_class = XTunerBackend
        else:
            raise ValueError("backend must be hf or xtuner")
        self.checkpoint = str(Path(checkpoint).resolve())
        self.temperature = load_calibration(calibration_path, checkpoint) if calibration_path else 1.0
        self.calibration_path = str(Path(calibration_path).resolve()) if calibration_path else None
        self.backend_name = backend
        self.backend = backend_class(checkpoint, processor_path, media_root, max_length, **backend_options)
        self.tokenizer = self.backend.tokenizer

    @classmethod
    def from_config(cls, config):
        return cls(
            config.checkpoint,
            config.processor_path,
            config.media_root,
            config.max_length,
            config.calibration_path,
            backend=config.backend,
            device=config.device,
            dtype=config.dtype,
            attn_implementation=config.attn_implementation,
        )

    def predict(self, row):
        compiled, logits, length, inference_ms = self.backend.score(row)
        answers = {}
        for index, field in enumerate(compiled.fields):
            question = row["questions"][field]
            options = _options(question)
            values = [value for value, _ in options]
            encoded = [self.tokenizer.encode(s, add_special_tokens=False) for s in compiled.symbols[field]]
            if any(len(ids) != 1 for ids in encoded):
                raise ValueError("Candidate symbols must each encode to a single token")
            probabilities = torch.softmax(logits[index, [ids[0] for ids in encoded]].float(), dim=-1).cpu().tolist()
            probs = dict(zip(values, probabilities))
            best = min(values, key=lambda value: (-probs[value], value))
            kind = question["type"]
            answer = {"type": kind, "probabilities": probs, "confidence": probs[best]}
            if kind == "noul":
                answer["noul"] = probs["yes"]
            elif kind == "score":
                answer["score"] = sum(float(value) * probs[value] for value in values)
                answer["legend"] = dict(options)
            else:
                answer["choice"] = best
            answers[field] = answer
        result = {
            "answers": answers,
            "usage": {"input_tokens": length, "output_tokens": len(answers)},
            "timing": {"inference_ms": round(inference_ms, 2)},
        }
        return scale_result(result, self.temperature) if self.calibration_path else result
