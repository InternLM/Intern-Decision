"""Predict the bundled calibration pilot, then score raw and fixed-T outputs."""

import argparse
import json
from pathlib import Path

from src.eval.score_known_distribution import load_rows, score_benchmark
from src.eval.verify_bundle import digest, verify
from src.inference.temperature import load_calibration, scale_result

DATASET = Path(__file__).resolve().parents[2] / "benchmarks/known-distribution-pilot-v1"


def predict_inputs(engine, inputs, raw_stream, calibrated_stream, temperature):
    for row in inputs:
        request = {key: row[key] for key in ("state", "questions", "images") if key in row}
        result = engine.predict(request)
        raw_stream.write(json.dumps({"id": row["id"], **result}, ensure_ascii=False) + "\n")
        # scale_result checks every field's argmax, including floating-point ties.
        calibrated = scale_result(result, temperature)
        calibrated_stream.write(json.dumps({"id": row["id"], **calibrated}, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--backend", choices=("hf", "xtuner"), default="xtuner")
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    verify(args.dataset)
    temperature = load_calibration(args.calibration, args.checkpoint)
    config = json.loads(args.calibration.read_text(encoding="utf-8"))
    if config["inference_backend"] != args.backend:
        raise ValueError("Calibration and inference backend differ")
    args.output.mkdir(parents=True, exist_ok=False)
    from src.inference.engine import DecisionEngine

    engine = DecisionEngine(args.checkpoint, backend=args.backend)
    raw, calibrated = args.output / "raw.jsonl", args.output / "calibrated.jsonl"
    with raw.open("x", encoding="utf-8") as a, calibrated.open("x", encoding="utf-8") as b:
        predict_inputs(engine, load_rows(args.dataset / "inputs.jsonl"), a, b, temperature)
    for name, predictions in (("raw", raw), ("calibrated", calibrated)):
        metrics = score_benchmark(args.dataset, predictions, args.output / name)
        if not metrics["complete"]:
            raise ValueError(f"Incomplete pilot predictions: {name}")
    verification = {
        "passed": True,
        "changed_predictions": 0,
        "temperature": temperature,
        "inference_backend": args.backend,
        "checkpoint_hashes": config["checkpoint_hashes"],
        "calibration_sha256": digest(args.calibration),
        "prediction_hashes": {path.name: digest(path) for path in (raw, calibrated)},
    }
    (args.output / "verification.json").write_text(json.dumps(verification, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
