"""Select a checkpoint-specific temperature without accessing benchmark scores."""

import argparse
import json
from pathlib import Path

from src.eval.calibration import fit_temperature, rows, validation_metrics
from src.eval.jev import sha256
from src.inference.temperature import load_calibration, scale_probabilities


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-fit-rows", type=int)
    parser.add_argument("--expected-validation-rows", type=int)
    args = parser.parse_args()
    receipt = json.loads(Path(str(args.predictions) + ".complete.json").read_text())
    if not receipt.get("complete") or receipt.get("workers") != 1 or receipt.get("temperature") != 1.0:
        raise ValueError("Fit requires one complete, unscaled collection (workers=1)")
    if receipt["prediction_sha256"] != sha256(args.predictions):
        raise ValueError("Calibration predictions changed after collection")
    for name, expected in receipt["checkpoint_hashes"].items():
        if Path(name).name != name or name in {".", ".."}:
            raise ValueError("Invalid checkpoint filename")
        if sha256(args.checkpoint / name) != expected:
            raise ValueError("Calibration predictions belong to different checkpoint files")
    records = rows(args.predictions)
    if len(records) != receipt["rows"]:
        raise ValueError("Incomplete calibration collection")
    seen = set()
    counts = {"calibration": 0, "selection": 0}
    for row in records:
        key = row["id"]
        if not key or key in seen:
            raise ValueError("Missing or repeated calibration/validation identity")
        seen.add(key)
        role = row["validation_role"]
        if role not in counts:
            raise ValueError("Unsupported validation_role")
        counts[role] += 1
        if row.get("calibration", {}).get("temperature", 1.0) != 1.0:
            raise ValueError("Collect unscaled (T=1) probabilities")
        if set(row["targets"]) != set(row["answers"]) or not row["answers"]:
            raise ValueError("Calibration fields do not match")
        for field, answer in row["answers"].items():
            probabilities = scale_probabilities(answer["probabilities"], 1.0)
            if set(probabilities) != set(row["targets"][field]["labels"]):
                raise ValueError("Calibration labels do not match")
            if row["targets"][field]["target"] not in probabilities:
                raise ValueError("Calibration target is not a candidate")
    if not all(counts.values()):
        raise ValueError("Supply disjoint calibration and selection rows")
    for role, expected in (("calibration", args.expected_fit_rows), ("selection", args.expected_validation_rows)):
        if expected is not None and counts[role] != expected:
            raise ValueError(f"Unexpected {role} row count")
    fit, curve = fit_temperature(records)
    checkpoint = args.checkpoint.resolve()
    files = [
        path for path in checkpoint.iterdir() if path.is_file() and path.suffix in {".json", ".safetensors", ".jinja"}
    ]
    artifact = {
        "method": "temperature-scaling",
        "fit_split": "calibration",
        "fit_objective": "mean hard-label negative log likelihood",
        "checkpoint": str(checkpoint),
        "checkpoint_hashes": {path.name: sha256(path) for path in files},
        "predictions_sha256": sha256(args.predictions),
        "row_counts": counts,
        "inference_backend": receipt["backend"],
        "data_sha256": receipt["data_sha256"],
        **fit,
    }
    validation = {
        role: {
            name: validation_metrics(records, role, temperature)
            for name, temperature in (("baseline", 1.0), ("calibrated", fit["temperature"]))
        }
        for role in counts
    }
    args.output.mkdir(parents=True, exist_ok=False)
    for name, value in (("calibration.json", artifact), ("search-curve.json", curve), ("validation.json", validation)):
        (args.output / name).write_text(json.dumps(value, indent=2) + "\n")
    load_calibration(args.output / "calibration.json", checkpoint)
    print(json.dumps({"temperature": fit["temperature"], "rows": counts}))


if __name__ == "__main__":
    main()
