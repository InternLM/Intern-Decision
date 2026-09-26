"""Check temperature fitting, rank preservation and consistent API answers."""

import math

import numpy as np

from src.eval.calibration import fit_temperature
from src.inference.temperature import argmax, scale_probabilities, scale_result

for value in (0, -1, float("nan"), float("inf")):
    try:
        scale_probabilities({"no": 0.2, "yes": 0.8}, value)
    except ValueError:
        pass
    else:
        raise AssertionError(f"Accepted invalid temperature: {value}")
assert scale_probabilities({"z": 0.5, "a": 0.5, "zero": 0.0}, 2.0) == {"z": 0.5, "a": 0.5, "zero": 0.0}
rng = np.random.default_rng(42)
for width in (2, 4, 60):
    for _ in range(100):
        logits = rng.normal(size=width) * 8
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        original = {str(i): float(p) for i, p in enumerate(probs)}
        for t in (0.1, 0.5, 1.0, 1.7822801441159297, 10.0, 100.0):
            actual = scale_probabilities(original, t)
            expected = np.exp((logits - logits.max()) / t)
            expected /= expected.sum()
            np.testing.assert_allclose(list(actual.values()), expected, rtol=1e-12, atol=1e-15)
            assert argmax(actual) == argmax(original)
            assert abs(sum(actual.values()) - 1) < 1e-12

result = {
    "answers": {
        "choice": {"type": "choice", "choice": "b", "confidence": 0.8, "probabilities": {"a": 0.2, "b": 0.8}},
        "noul": {"type": "noul", "noul": 0.8, "probabilities": {"no": 0.2, "yes": 0.8}},
        "score": {"type": "score", "score": 1.6, "probabilities": {"0": 0.2, "2": 0.8}},
    }
}
scaled = scale_result(result, 2)
assert result["answers"]["noul"]["noul"] == 0.8
assert scaled["answers"]["choice"]["choice"] == "b"
assert math.isclose(scaled["answers"]["noul"]["noul"], 2 / 3)
assert math.isclose(scaled["answers"]["score"]["score"], 4 / 3)
assert math.isclose(scaled["answers"]["choice"]["confidence"], 2 / 3)

# Eighty percent of identical 0.9 predictions are correct: optimal T is analytic.
records = [
    {
        "validation_role": "calibration",
        "answers": {"x": {"probabilities": {"yes": 0.9, "no": 0.1}}},
        "targets": {"x": {"target": "yes" if i < 80 else "no"}},
    }
    for i in range(100)
]
fitted, _ = fit_temperature(records)
assert math.isclose(fitted["temperature"], math.log(9) / math.log(4), rel_tol=1e-10)
assert fitted["fitted_nll"] < fitted["baseline_nll"]
# Rows outside the fit role must never influence the selected temperature.
records.append({"validation_role": "selection", "answers": {}, "targets": {}})
unchanged, _ = fit_temperature(records)
assert unchanged["temperature"] == fitted["temperature"]
print(
    "PASS: analytic NLL optimum, calibration-only fitting, invalid T, ties, zeros, 1800 rank-preservation cases, API values"
)
