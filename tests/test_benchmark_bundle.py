"""Release data and offline CLI integration contracts."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.eval.known_distribution import sha256, write_benchmark
from src.eval.score_known_distribution import expected_ece, load_rows

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "benchmarks/known-distribution-pilot-v1"


def test_bundled_data_reproduces_exactly(tmp_path):
    regenerated = tmp_path / "regenerated"
    write_benchmark(regenerated)
    for name in ("inputs.jsonl", "references.jsonl", "manifest.json", "README.md"):
        assert (BUNDLE / name).read_bytes() == (regenerated / name).read_bytes()
    manifest = json.loads((BUNDLE / "manifest.json").read_text())
    assert manifest["generator_sha256"] == sha256(ROOT / "src/eval/known_distribution.py")


def test_cli_runs_without_site_packages_and_reports_missing_coverage(tmp_path):
    refs = load_rows(BUNDLE / "references.jsonl")
    predictions = tmp_path / "predictions.jsonl"
    records = [
        {
            "id": r["id"],
            "answers": {"decision": {"type": "choice", "probabilities": r["gold_probs"]}},
        }
        for r in refs
    ]
    for count in (96, 95):
        predictions.write_text("".join(json.dumps(r) + "\n" for r in reversed(records[:count])))
        output = tmp_path / f"scores-{count}"
        result = subprocess.run(
            [
                sys.executable,
                "-S",
                "-m",
                "src.eval.score_known_distribution",
                "--dataset",
                str(BUNDLE),
                "--predictions",
                str(predictions),
                "--output",
                str(output),
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == (0 if count == 96 else 2), result.stderr
        metrics = json.loads((output / "metrics.json").read_text())
        assert metrics["valid"] == count
        assert metrics["invalid"] == 96 - count
        assert metrics["tvd"] == pytest.approx(0, abs=1e-14)
        assert metrics["expected_ece"]["ece"] == pytest.approx(0, abs=1e-14)
        assert "Overall (pooled)" in (output / "summary.md").read_text()


def test_pooled_ece_is_not_mean_of_category_ece():
    over = [{"confidence_used": 0.65, "expected_correctness": 0.45}]
    under = [{"confidence_used": 0.65, "expected_correctness": 0.85}]
    assert expected_ece(over)["ece"] == pytest.approx(0.2)
    assert expected_ece(under)["ece"] == pytest.approx(0.2)
    assert expected_ece(over + under)["ece"] == pytest.approx(0)
