"""Fit one temperature on designated calibration rows and replay frozen evaluations."""

import copy
import json
import math
from pathlib import Path

import numpy as np

from src.eval import jev
from src.inference.temperature import argmax, scale_probabilities, scale_result


def rows(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream]


def fit_temperature(records):
    """NLL is convex in inverse temperature; bisect its monotone derivative."""
    fit = [
        (answer["probabilities"], row["targets"][field]["target"])
        for row in records
        if row["validation_role"] == "calibration"
        for field, answer in row["answers"].items()
    ]
    if not fit:
        raise ValueError("No designated calibration records")
    width = max(len(p) for p, _ in fit)
    logits = np.zeros((len(fit), width), dtype=np.float64)
    valid = np.zeros_like(logits, dtype=bool)
    targets = np.empty(len(fit), dtype=np.int64)
    for i, (probs, target) in enumerate(fit):
        if any(p <= 0 or not math.isfinite(p) for p in probs.values()):
            raise ValueError("Exact positive probabilities required; collect raw logits if probabilities underflow")
        values = np.log(list(probs.values()))
        logits[i, : len(values)] = values - values.max()
        valid[i, : len(values)] = True
        targets[i] = list(probs).index(target)
    target_logits = logits[np.arange(len(fit)), targets]

    def objective(beta):
        scores = np.where(valid, logits * beta, -np.inf)
        maximum = scores.max(axis=1)
        weights = np.exp(scores - maximum[:, None])
        partition = weights.sum(axis=1)
        nll = np.mean(maximum + np.log(partition) - beta * target_logits)
        gradient = np.mean(np.sum(weights * logits, axis=1) / partition - target_logits)
        return float(nll), float(gradient)

    lo, hi = 0.01, 100.0  # Explicit, broad positive T range [0.01, 100].
    lower, upper = objective(lo)[1], objective(hi)[1]
    if lower >= 0:
        beta = lo
    elif upper <= 0:
        beta = hi
    else:
        for _ in range(100):
            middle = (lo + hi) / 2
            if objective(middle)[1] < 0:
                lo = middle
            else:
                hi = middle
        beta = (lo + hi) / 2
    nll, gradient = objective(beta)
    if nll > objective(1.0)[0] + 1e-12:
        raise AssertionError("Fitted calibration NLL is worse than T=1")
    curve = [
        {"temperature": float(t), "nll": objective(1 / t)[0]}
        for t in sorted(set(np.geomspace(0.01, 100.0, 201).tolist() + [1.0, 1 / beta]))
    ]
    return {
        "temperature": 1 / beta,
        "baseline_nll": objective(1.0)[0],
        "fitted_nll": nll,
        "gradient_inverse_temperature": gradient,
        "search_temperature_bounds": [0.01, 100.0],
        "boundary_optimum": beta in (0.01, 100.0),
        "fit_decisions": len(fit),
        "optimizer": "100-step bisection of convex NLL derivative in inverse temperature",
    }, curve


def validation_metrics(records, role, temperature):
    pairs, briers, nlls = [], [], []
    changed = 0
    for row in records:
        if row["validation_role"] != role:
            continue
        for field, answer in row["answers"].items():
            target = row["targets"][field]["target"]
            probs = scale_probabilities(answer["probabilities"], temperature)
            pred = argmax(probs)
            changed += pred != argmax(answer["probabilities"])
            pairs.append((max(probs.values()), pred == target))
            briers.append(jev.brier_score(probs, target, list(probs)))
            nlls.append(-math.log(probs[target]))
    assert changed == 0
    return {
        "decisions": len(pairs),
        "correct": sum(int(c) for _, c in pairs),
        "accuracy": sum(int(c) for _, c in pairs) / len(pairs),
        "nll": sum(nlls) / len(nlls),
        "brier": sum(briers) / len(briers),
        "ece": jev.ece_top_label(pairs, n_bins=10)["ece"],
        "changed_predictions": changed,
    }


class ReplayEngine:
    def __init__(self, predictions, checkpoint, temperature):
        self.predictions = iter(predictions)
        self.checkpoint = str(checkpoint)
        self.temperature = temperature

    def predict(self, row):
        saved = next(self.predictions)
        if row["id"] != saved["id"]:
            raise AssertionError("Replay identity mismatch")
        result = {k: copy.deepcopy(saved[k]) for k in ("answers", "usage", "timing")}
        return scale_result(result, self.temperature)


def test_nll(raw_rows, predictions):
    values = []
    for raw, prediction in zip(raw_rows, predictions, strict=True):
        for field, answer in prediction["answers"].items():
            if "question" in raw:
                expected = jev.Task.from_dict(raw).expected
            else:
                expected = jev._answer_value(raw["questions"][field], raw.get("targets", {}).get(field))
            if expected is not None:
                values.append(-math.log(answer["probabilities"][str(expected)]))
    return sum(values) / len(values)
