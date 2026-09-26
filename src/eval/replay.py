"""Apply a previously selected temperature to saved T=1 benchmark predictions."""

import argparse
import json
from pathlib import Path

from src.eval.calibration import ReplayEngine, rows, test_nll
from src.eval.jev import evaluate, sha256, suite_datasets
from src.inference.temperature import load_calibration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--public-only", action="store_true")
    parser.add_argument("--test-root", type=Path)
    args = parser.parse_args()
    temperature = load_calibration(args.calibration, args.checkpoint)
    baseline = json.loads((args.baseline / "complete.json").read_text())
    calibration = json.loads(args.calibration.read_text())
    if baseline["checkpoint_hashes"] != {
        name: sha256(args.checkpoint / name) for name in baseline["checkpoint_hashes"]
    }:
        raise ValueError("Checkpoint differs from the T=1 evaluation")
    for name, digest in calibration["checkpoint_hashes"].items():
        if digest != sha256(args.checkpoint / name):
            raise ValueError("Checkpoint differs from calibration")
    args.output.mkdir(parents=True, exist_ok=False)
    results = {}
    for name, source, _ in suite_datasets(args.public_only, args.test_root):
        before = baseline["datasets"][name]
        if calibration["inference_backend"] != before["inference_backend"]:
            raise ValueError("Calibration and benchmark backend differ")
        if before["temperature"] != 1.0 or before["data_sha256"] != sha256(source):
            raise ValueError("Expected T=1 evaluation of the identical benchmark inputs")
        original = rows(args.baseline / f"{name}.predictions.jsonl")
        raw = rows(source)
        if len(original) != before["rows"] or len(raw) != len(original):
            raise ValueError("Incomplete baseline predictions")
        for i, (record, prediction) in enumerate(zip(raw, original, strict=True), 1):
            if (record["id"], i) != (prediction["id"], prediction["source_line"]):
                raise ValueError("Baseline prediction identity mismatch")
        engine = ReplayEngine(original, args.checkpoint.resolve(), temperature)
        engine.backend_name = before["inference_backend"]
        dest = args.output / f"{name}.predictions.jsonl"
        metrics = evaluate(engine, source, dest)
        calibrated = rows(dest)
        for old, new in zip(original, calibrated, strict=True):
            for field, score in old["scores"].items():
                if (score["predicted"], score["correct"]) != (
                    new["scores"][field]["predicted"],
                    new["scores"][field]["correct"],
                ):
                    raise ValueError("Calibration changed a benchmark decision")
        if (metrics["correct"], metrics["total"]) != (before["correct"], before["total"]):
            raise ValueError("Calibration changed aggregate accuracy")
        metrics.update(nll=test_nll(raw, calibrated), changed_predictions=0)
        results[name] = {"baseline": before, "calibrated": metrics}
    report = {
        "temperature": temperature,
        "datasets": results,
        "predictions_unchanged": True,
        "calibration_sha256": sha256(args.calibration),
        "prediction_hashes": {p.name: sha256(p) for p in args.output.glob("*.jsonl")},
    }
    (args.output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output / "verification.json").write_text(json.dumps({"passed": True, "changed_predictions": 0}) + "\n")
    print(json.dumps({"temperature": temperature, "suites": len(results), "changed_predictions": 0}))


if __name__ == "__main__":
    main()
