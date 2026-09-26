"""Positive scalar temperature scaling of candidate probabilities.

Applying softmax(log(p) / T) is equivalent to scaling the original candidate
logits, up to the rounding in saved probabilities. No labels enter this module.
"""

import copy
import hashlib
import json
import math
from pathlib import Path


def argmax(probs):
    return min(probs, key=lambda label: (-probs[label], label))


def validate_temperature(temperature):
    temperature = float(temperature)
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be finite and positive")
    return temperature


def scale_probabilities(probs, temperature):
    temperature = validate_temperature(temperature)
    if not probs or any(not math.isfinite(p) or p < 0 or p > 1 for p in probs.values()):
        raise ValueError("Invalid probability distribution")
    if abs(sum(probs.values()) - 1.0) > 1e-3:
        raise ValueError("Probabilities must sum to one")
    if temperature == 1.0:
        return dict(probs)
    logs = {label: math.log(p) if p else -math.inf for label, p in probs.items()}
    maximum = max(logs.values())
    weights = {label: math.exp((logp - maximum) / temperature) for label, logp in logs.items()}
    total = sum(weights.values())
    result = {label: p / total for label, p in weights.items()}
    if argmax(result) != argmax(probs):
        raise ArithmeticError("Floating-point temperature scaling changed argmax")
    return result


def scale_result(result, temperature):
    """Scale all fields consistently, including API confidence/noul/score values."""
    scaled = copy.deepcopy(result)
    for answer in scaled["answers"].values():
        probs = scale_probabilities(answer["probabilities"], temperature)
        answer["probabilities"] = probs
        best = argmax(probs)
        answer["confidence"] = probs[best]
        if answer["type"] == "choice":
            answer["choice"] = best
        elif answer["type"] == "noul":
            answer["noul"] = probs["yes"]
        elif answer["type"] == "score":
            answer["score"] = sum(float(label) * prob for label, prob in probs.items())
        else:
            raise ValueError("Unknown answer type")
    scaled["calibration"] = {"method": "temperature-scaling", "temperature": validate_temperature(temperature)}
    return scaled


def load_calibration(path, checkpoint):
    """Fail closed if a calibration artifact belongs to a different checkpoint."""
    config = json.loads(Path(path).read_text())
    if config.get("method") != "temperature-scaling" or config.get("fit_split") != "calibration":
        raise ValueError("Not a supported calibration artifact")
    if Path(config["checkpoint"]).resolve() != Path(checkpoint).resolve():
        raise ValueError("Calibration checkpoint mismatch")
    hashes = config["checkpoint_hashes"]
    required = {"config.json", "model.safetensors.index.json", "tokenizer.json"}
    if not required <= set(hashes):
        raise ValueError("Calibration is missing checkpoint metadata hashes")
    for name, expected in hashes.items():
        if Path(name).name != name or name in {".", ".."}:
            raise ValueError("Invalid checkpoint hash filename")
        digest = hashlib.sha256()
        with (Path(checkpoint) / name).open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 << 20), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected:
            raise ValueError(f"Checkpoint file changed: {name}")
    return validate_temperature(config["temperature"])
