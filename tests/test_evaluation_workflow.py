"""CPU contracts for portable evaluation and table generation."""

import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.eval.run_pilot import predict_inputs
from src.eval.tables import SUITES, accuracy_table, pilot_table
from src.eval.use_temperature import bind
from src.eval.verify_bundle import ROOT, digest, verify


def test_accuracy_bundle_and_tamper_rejection(tmp_path):
    assert verify(ROOT / "accuracy-v1")["rows"] == 10751
    (tmp_path / "x.jsonl").write_text("{}\n")
    manifest = {"purpose": "evaluation_only", "training_ready": False, "files": {"x.jsonl": "wrong"}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="hash mismatch"):
        verify(tmp_path)


def test_preset_rejects_changed_weights_before_writing(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    hashes = {}
    for name in ("config.json", "tokenizer.json", "model.safetensors.index.json", "model.safetensors"):
        (checkpoint / name).write_text("{}")
        hashes[name] = digest(checkpoint / name)
    preset = tmp_path / "presets.json"
    preset.write_text(
        json.dumps(
            {
                "models": {
                    "example": {
                        "method": "temperature-scaling",
                        "fit_split": "calibration",
                        "temperature": 2,
                        "checkpoint_hashes": hashes,
                        "inference_backend": "xtuner",
                    }
                }
            }
        )
    )
    assert bind("example", checkpoint, tmp_path / "good.json", preset) == 2
    (checkpoint / "model.safetensors").write_text("changed")
    with pytest.raises(ValueError, match="does not match"):
        bind("example", checkpoint, tmp_path / "bad.json", preset)
    assert not (tmp_path / "bad.json").exists()


def test_pilot_model_receives_only_evidence_and_calibration_preserves_choice():
    seen = []

    def predict(request):
        seen.append(request)
        return {
            "answers": {
                "decision": {
                    "type": "choice",
                    "choice": "A",
                    "confidence": 0.9,
                    "probabilities": {"A": 0.9, "B": 0.1},
                }
            }
        }

    raw, calibrated = io.StringIO(), io.StringIO()
    row = {"id": "test", "state": "visible", "questions": {}, "targets": "secret", "gold_probs": "secret"}
    predict_inputs(SimpleNamespace(predict=predict), [row], raw, calibrated, 2)
    assert seen == [{"state": "visible", "questions": {}}]
    answer = json.loads(calibrated.getvalue())["answers"]["decision"]
    assert answer["choice"] == "A"
    assert answer["confidence"] == pytest.approx(0.75)


def test_table_averages_suites_not_decisions():
    datasets = {name: {"correct": 1, "total": 2, "brier": 0.4, "ece": {"ece": 0.1}} for name, _ in SUITES}
    datasets["agnews-test"] = datasets["agnews-test"] | {"correct": 900, "total": 1000}
    table = accuracy_table([("model", {"datasets": datasets})])
    assert "55.71 | 0.400000 | 0.100000" in table
    with pytest.raises(ValueError, match="incomplete"):
        pilot_table([("model", {"complete": False})])


def test_default_paths_use_bundled_files(monkeypatch):
    from src.eval.jev import suite_datasets

    monkeypatch.delenv("JEVBENCH_DATA_ROOT", raising=False)
    monkeypatch.delenv("EVAL_DATA_ROOT", raising=False)
    paths = suite_datasets()
    assert len(paths) == 7
    assert all(Path(path).is_file() for _, path, _ in paths)
