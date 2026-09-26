"""Synthetic checks for prompt isolation, evaluation and calibration integrity."""

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.eval import jev
from src.eval.calibration import fit_temperature
from src.inference.temperature import load_calibration
from src.inputs.schema import compile_row
from src.service.examples import examples


class ReleaseTests(unittest.TestCase):
    def test_collection_fit_and_replay(self):
        import contextlib
        import io

        from src.eval import collect, fit, replay

        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            for name in ("config.json", "tokenizer.json", "model.safetensors.index.json"):
                (checkpoint / name).write_text("{}")
            request = examples()[0]
            records = []
            for role in ("calibration", "selection"):
                for i in range(10):
                    records.append(
                        request
                        | {
                            "id": f"{role}-{i}",
                            "validation_role": role,
                            "targets": {"box": {"label": "left" if i < 8 else "right"}},
                        }
                    )
            source = root / "validation.jsonl"
            source.write_text("".join(json.dumps(row) + "\n" for row in records))
            prediction = {
                "answers": {
                    "box": {
                        "type": "choice",
                        "choice": "left",
                        "confidence": 0.9,
                        "probabilities": {"left": 0.9, "right": 0.1},
                    }
                },
                "usage": {"input_tokens": 1, "output_tokens": 1},
                "timing": {"inference_ms": 0.0},
            }
            engine = SimpleNamespace(
                checkpoint=str(checkpoint),
                temperature=1.0,
                backend_name="hf",
                predict=lambda _: copy.deepcopy(prediction),
            )
            predictions = root / "predictions.jsonl"
            with (
                patch(
                    "sys.argv",
                    ["collect", "--checkpoint", str(checkpoint), "--data", str(source), "--output", str(predictions)],
                ),
                patch.object(collect, "DecisionEngine", return_value=engine),
            ):
                collect.main()
            calibration = root / "calibration"
            with patch(
                "sys.argv",
                [
                    "fit",
                    "--checkpoint",
                    str(checkpoint),
                    "--predictions",
                    str(predictions),
                    "--output",
                    str(calibration),
                ],
            ):
                fit.main()
            artifact = json.loads((calibration / "calibration.json").read_text())
            self.assertGreater(artifact["temperature"], 1.0)
            baseline = root / "baseline"
            metrics = jev.evaluate(engine, source, baseline / "jevbench-hard.predictions.jsonl")
            (baseline / "complete.json").write_text(
                json.dumps({"checkpoint_hashes": artifact["checkpoint_hashes"], "datasets": {"jevbench-hard": metrics}})
            )
            output = root / "replay"
            with (
                patch(
                    "sys.argv",
                    [
                        "replay",
                        "--checkpoint",
                        str(checkpoint),
                        "--baseline",
                        str(baseline),
                        "--calibration",
                        str(calibration / "calibration.json"),
                        "--output",
                        str(output),
                    ],
                ),
                patch.object(replay, "suite_datasets", return_value=[("jevbench-hard", source, 20)]),
            ):
                replay.main()
            self.assertTrue(json.loads((output / "verification.json").read_text())["passed"])
            comparison = json.loads((output / "comparison.json").read_text())["datasets"]["jevbench-hard"]
            self.assertEqual(comparison["calibrated"]["correct"], 16)
            self.assertLess(comparison["calibrated"]["brier"], comparison["baseline"]["brier"])

    def test_gold_and_provenance_cannot_enter_prompt(self):
        row = examples()[0]
        before = compile_row(row, include_targets=False)
        dirty = copy.deepcopy(row)
        dirty.update(
            targets={"box": {"label": "PRIVATE_LABEL"}},
            provenance={"gold": "PRIVATE_LABEL"},
            gold_probs={"PRIVATE_LABEL": 1.0},
        )
        dirty["questions"]["box"]["answer"] = {"choice": "PRIVATE_LABEL"}
        self.assertEqual(before.messages, compile_row(dirty, include_targets=False).messages)
        self.assertNotIn("PRIVATE_LABEL", json.dumps(before.messages))
        row["targets"] = {"box": {"label": "left"}}
        compiled = compile_row(row)
        self.assertEqual(compiled.targets, {"box": "A"})
        self.assertEqual(compiled.messages, before.messages)

    def test_single_and_sharded_scoring_agree(self):
        row = examples()[0] | {
            "id": "synthetic",
            "targets": {"box": {"label": "left"}},
            "gold_probs": {"left": 0.75, "right": 0.25},
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "records.jsonl"
            data.write_text("".join(json.dumps(row | {"id": f"synthetic-{i}"}) + "\n" for i in range(10)))
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            engine = SimpleNamespace(checkpoint=str(checkpoint), temperature=1.0, backend_name="hf")
            engine.predict = lambda _: {"answers": {"box": {"probabilities": {"left": 0.8, "right": 0.2}}}}
            serial = jev.evaluate(engine, data, root / "serial.jsonl")
            self.assertEqual((serial["correct"], serial["total"], serial["tvd_n"]), (10, 10, 10))
            self.assertAlmostEqual(serial["brier"], 0.08)
            self.assertAlmostEqual(serial["ece"]["ece"], 0.2)
            self.assertAlmostEqual(serial["tvd"], 0.05)
            for worker in range(2):
                jev.evaluate(engine, data, root / f"worker-{worker}/jevbench-hard.predictions.jsonl", worker, 2)
            with (
                patch.object(jev, "suite_datasets", return_value=[("jevbench-hard", data, 10)]),
                patch.dict(jev.EXPECTED_COUNTS, {"jevbench-hard": (10, 10)}),
            ):
                jev.merge_suite(checkpoint, root, 2, backend="hf")
            merged = json.loads((root / "complete.json").read_text())["datasets"]["jevbench-hard"]
            self.assertEqual(serial["correct"], merged["correct"])
            self.assertAlmostEqual(serial["ece"]["ece"], merged["ece"]["ece"])
            self.assertEqual(
                (root / "serial.jsonl").read_bytes(), (root / "jevbench-hard.predictions.jsonl").read_bytes()
            )

    def test_calibration_detects_weight_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            names = ["config.json", "model.safetensors.index.json", "tokenizer.json", "model.safetensors"]
            for name in names:
                (root / name).write_bytes(b"synthetic-fixture")
            config = {
                "method": "temperature-scaling",
                "fit_split": "calibration",
                "temperature": 2.0,
                "checkpoint": str(root),
                "checkpoint_hashes": {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names},
            }
            artifact = root / "calibration.json"
            artifact.write_text(json.dumps(config))
            self.assertEqual(load_calibration(artifact, root), 2.0)
            (root / "model.safetensors").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "Checkpoint file changed"):
                load_calibration(artifact, root)

    def test_fit_ignores_validation_labels(self):
        rows = [
            {
                "validation_role": "calibration",
                "answers": {"x": {"probabilities": {"a": 0.9, "b": 0.1}}},
                "targets": {"x": {"target": "a" if i < 8 else "b"}},
            }
            for i in range(10)
        ]
        baseline, _ = fit_temperature(rows)
        rows += [
            {
                "validation_role": "selection",
                "answers": {"x": {"probabilities": {"a": 0.999, "b": 0.001}}},
                "targets": {"x": {"target": "b"}},
            }
            for _ in range(100)
        ]
        updated, _ = fit_temperature(rows)
        self.assertEqual(baseline["temperature"], updated["temperature"])


if __name__ == "__main__":
    unittest.main()
