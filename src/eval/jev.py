"""Strict evaluation with upstream Jevbench scoring and calibration."""

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
if os.environ.get("JEVBENCH_ROOT"):
    sys.path.insert(0, os.environ["JEVBENCH_ROOT"])
from jevbench.metrics import brier_score, ece_top_label  # noqa: E402
from jevbench.scoring import score_task  # noqa: E402
from jevbench.tasks import Task  # noqa: E402

from src.inference.engine import DecisionEngine  # noqa: E402
from src.inference.temperature import load_calibration  # noqa: E402
from src.inputs.schema import _answer_value, _options  # noqa: E402

EXPECTED_COUNTS = {
    "jevbench-easy": (48, 48),
    "jevbench-original": (72, 72),
    "jevbench-hard": (111, 111),
    "agnews-test": (7600, 7600),
    "toolace-test": (310, 310),
    "typed_decisions-test": (400, 2000),
    "wildjailbreak-test": (2210, 2210),
}


def verify_suite_counts(results):
    for name, metrics in results.items():
        if (metrics["rows"], metrics["total"]) != EXPECTED_COUNTS[name]:
            raise ValueError(
                f"{name}: row/decision counts differ from the reference suite; use --data for custom splits"
            )


def source_hashes():
    return {str(path.relative_to(ROOT)): sha256(path) for path in sorted((ROOT / "src").rglob("*.py"))}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_rows(path):
    """Split JSONL only at physical newlines, retaining source-line identity."""
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError(f"Blank evaluation record: {path}:{number}")
            yield json.loads(line)


def _canonical_public(raw):
    if "questions" in raw:
        return raw
    return {
        "id": raw["id"],
        "state": raw["state"],
        "images": raw.get("images", []),
        "questions": {"decision": raw["question"]},
    }


def evaluate(engine, data_path, output_path, worker=0, workers=1):
    data_path, output_path = Path(data_path), Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pairs, briers, tvds = [], [], []
    families = defaultdict(lambda: {"correct": 0, "total": 0})
    seen = set()
    predictions = []
    for row_index, raw in enumerate(read_rows(data_path)):
        if row_index % workers != worker:
            continue
        if not raw.get("id") or ("question" in raw and raw["id"] in seen):
            raise ValueError(f"Missing or duplicate evaluation identity: {raw.get('id')}")
        seen.add(raw["id"])
        row = _canonical_public(raw)
        result = engine.predict(row)
        scores = {}
        for field, question in row["questions"].items():
            if "question" in raw:
                task = Task.from_dict(raw)
                if set(v for v, _ in _options(question)) != set(task.labels):
                    raise ValueError(f"{task.id}: options disagree with benchmark labels")
            else:
                expected = _answer_value(question, row.get("targets", {}).get(field))
                if expected is None:
                    raise ValueError(f"{row['id']}/{field}: missing evaluation label")
                task = SimpleNamespace(
                    question=question,
                    labels=[v for v, _ in _options(question)],
                    expected=expected,
                    family=raw.get("category", question["type"]),
                )
            score = score_task(result["answers"][field]["probabilities"], task)
            if not score["valid"]:
                raise ValueError(f"{row['id']}/{field}: {score['error']}")
            scores[field] = score
            if score["correct"] is not None:
                probs = score["probs"]
                pairs.append((max(probs.values()), score["correct"]))
                briers.append(brier_score(probs, str(task.expected), task.labels))
                families[task.family]["correct"] += int(score["correct"])
                families[task.family]["total"] += 1
            gold = raw.get("gold_probs") or raw.get("provenance", {}).get("gold_probs")
            if gold:
                if set(gold) != set(task.labels):
                    raise ValueError("gold_probs keys differ from exact benchmark labels")
                tvds.append(0.5 * sum(abs(score["probs"][v] - gold[v]) for v in task.labels))
        predictions.append(
            {
                "id": raw["id"],
                "source_line": row_index + 1,
                "group": raw.get("group"),
                "family": raw.get("family", raw.get("category")),
                **result,
                "scores": scores,
            }
        )
        if len(predictions) % 50 == 0:
            print(json.dumps({"dataset": str(data_path), "completed": len(predictions)}), flush=True)
    correct = sum(int(p[1]) for p in pairs)
    for family in families.values():
        family["accuracy"] = family["correct"] / family["total"]
    metrics = {
        "rows": len(predictions),
        "correct": correct,
        "total": len(pairs),
        "accuracy": correct / len(pairs) if pairs else None,
        "brier": sum(briers) / len(briers) if briers else None,
        "ece": ece_top_label(pairs, n_bins=10),
        "tvd": sum(tvds) / len(tvds) if tvds else None,
        "tvd_n": len(tvds),
        "families": dict(families),
        "category_macro_accuracy": sum(v["accuracy"] for v in families.values()) / len(families) if families else None,
        "temperature": getattr(engine, "temperature", 1.0),
        "data_sha256": sha256(data_path),
        "checkpoint": engine.checkpoint,
        "objective_revision": "masked-next-token-v4",
        "inference_backend": getattr(engine, "backend_name", "xtuner"),
    }
    output_path.write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in predictions))
    Path(str(output_path) + ".metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps({"dataset": str(data_path), **metrics}), flush=True)
    return metrics


def suite_datasets(public_only=False, test_root=None):
    datasets = [
        (
            f"jevbench-{name}",
            Path(os.environ.get("JEVBENCH_DATA_ROOT", ROOT / "benchmarks/accuracy-v1/jevbench")) / f"{name}.jsonl",
            count,
        )
        for name, count in (("easy", 48), ("original", 72), ("hard", 111))
    ]
    if not public_only:
        test_root = Path(test_root or os.environ.get("EVAL_DATA_ROOT", ROOT / "benchmarks/accuracy-v1"))
        datasets += [
            (f"{name}-test", test_root / name / "test.jsonl", EXPECTED_COUNTS[f"{name}-test"][0])
            for name in ("agnews", "toolace", "typed_decisions", "wildjailbreak")
        ]
    return datasets


def merge_suite(
    checkpoint, output, workers, public_only=False, calibration_path=None, backend="xtuner", test_root=None
):
    temperature = load_calibration(calibration_path, checkpoint) if calibration_path else 1.0
    results = {}
    for name, path, expected_count in suite_datasets(public_only, test_root):
        predictions, parts = {}, []
        for worker in range(workers):
            part = output / f"worker-{worker}" / f"{name}.predictions.jsonl"
            parts.append(json.loads(Path(str(part) + ".metrics.json").read_text()))
            if parts[-1]["data_sha256"] != sha256(path):
                raise ValueError("Shard input hash differs from requested dataset")
            if parts[-1].get("inference_backend", "xtuner") != backend:
                raise ValueError("Shard inference backend differs from requested backend")
            if parts[-1]["temperature"] != temperature:
                raise ValueError("Shard temperature differs from requested calibration")
            if Path(parts[-1]["checkpoint"]).resolve() != Path(checkpoint).resolve():
                raise ValueError("Shard checkpoint differs from requested checkpoint")
            for row in read_rows(part):
                key = (row["source_line"], row["id"])
                if key in predictions:
                    raise ValueError("Duplicate shard prediction")
                predictions[key] = row
        ids = [(i + 1, row["id"]) for i, row in enumerate(read_rows(path))]
        if len(ids) != len(set(ids)) or set(ids) != set(predictions):
            raise ValueError(f"{name}: evaluation identity mismatch")
        if expected_count and len(ids) != expected_count:
            raise ValueError(f"{name}: benchmark count mismatch")
        ordered = [predictions[key] for key in ids]
        pairs = [
            (max(s["probs"].values()), s["correct"])
            for row in ordered
            for s in row["scores"].values()
            if s["correct"] is not None
        ]
        total = sum(m["total"] for m in parts)
        correct = sum(m["correct"] for m in parts)
        tvd_n = sum(m["tvd_n"] for m in parts)
        families = defaultdict(lambda: {"correct": 0, "total": 0})
        for part in parts:
            for family, counts in part["families"].items():
                for key in ("correct", "total"):
                    families[family][key] += counts[key]
        for counts in families.values():
            counts["accuracy"] = counts["correct"] / counts["total"]
        metrics = {
            "rows": len(ids),
            "total": total,
            "correct": correct,
            "accuracy": correct / total,
            "brier": sum((m["brier"] or 0) * m["total"] for m in parts) / total,
            "ece": ece_top_label(pairs, n_bins=10),
            "families": dict(families),
            "tvd_n": tvd_n,
            "tvd": sum((m["tvd"] or 0) * m["tvd_n"] for m in parts) / tvd_n if tvd_n else None,
            "category_macro_accuracy": sum(v["accuracy"] for v in families.values()) / len(families),
            "temperature": temperature,
            "inference_backend": backend,
            "data_sha256": sha256(path),
            "checkpoint": str(Path(checkpoint).resolve()),
            "objective_revision": "masked-next-token-v4",
        }
        dest = output / f"{name}.predictions.jsonl"
        dest.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in ordered))
        Path(str(dest) + ".metrics.json").write_text(json.dumps(metrics, indent=2))
        results[name] = metrics
    verify_suite_counts(results)
    if results["jevbench-hard"]["tvd_n"] != 10:
        raise ValueError("Expected 10 hard gold_probs records")
    hashes = {p.name: sha256(p) for p in Path(checkpoint).glob("*") if p.is_file()}
    (output / "complete.json").write_text(
        json.dumps(
            {
                "checkpoint": str(Path(checkpoint).resolve()),
                "checkpoint_hashes": hashes,
                "source_hashes": source_hashes(),
                "datasets": results,
                "public_only": public_only,
                "public_scope": "231 items; not the full leaderboard or four-axis composite",
            },
            indent=2,
        )
        + "\n"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-path")
    parser.add_argument("--backend", choices=("hf", "xtuner"), default="hf")
    parser.add_argument("--calibration", help="Checkpoint-specific calibration.json; default is unscaled")
    parser.add_argument("--media-root", default="")
    parser.add_argument("--data")
    parser.add_argument("--output")
    parser.add_argument("--suite", action="store_true")
    parser.add_argument("--public-only", action="store_true")
    parser.add_argument(
        "--test-root", type=Path, help="Evaluation JSONL root; defaults to EVAL_DATA_ROOT or bundled accuracy-v1"
    )
    parser.add_argument("--worker", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--merge", action="store_true")
    args = parser.parse_args()
    if not args.output:
        parser.error("--output is required; use a fresh directory")
    if not args.merge:
        target = Path(args.output)
        if args.suite and args.workers > 1:
            target = target / f"worker-{args.worker}"
        if target.exists():
            parser.error("Output already exists; choose a fresh destination")
    if args.workers < 1 or not 0 <= args.worker < args.workers:
        parser.error("Invalid evaluation worker configuration")
    if args.merge:
        merge_suite(
            args.checkpoint,
            Path(args.output),
            args.workers,
            args.public_only,
            args.calibration,
            args.backend,
            args.test_root,
        )
        return
    if args.calibration and not args.output:
        parser.error("Use a separate --output directory for calibrated predictions")
    engine = DecisionEngine(
        args.checkpoint, args.model_path, args.media_root, calibration_path=args.calibration, backend=args.backend
    )
    if not args.suite:
        if not args.data or not args.output:
            parser.error("--data and --output are required without --suite")
        evaluate(engine, args.data, args.output)
        return
    output = Path(args.output or Path(args.checkpoint) / "evaluations").resolve()
    datasets = suite_datasets(args.public_only, args.test_root)
    if args.workers > 1:
        output = output / f"worker-{args.worker}"
    results = {}
    for name, path, count in datasets:
        results[name] = evaluate(engine, path, output / f"{name}.predictions.jsonl", args.worker, args.workers)
        if args.workers == 1 and count and results[name]["rows"] != count:
            raise ValueError(f"{name}: expected {count} records")
    if args.workers > 1:
        return
    verify_suite_counts(results)
    if results["jevbench-hard"]["tvd_n"] != 10:
        raise ValueError("Expected 10 hard gold_probs records")
    hashes = {p.name: sha256(p) for p in Path(args.checkpoint).glob("*") if p.is_file()}
    (output / "complete.json").write_text(
        json.dumps(
            {
                "checkpoint": engine.checkpoint,
                "checkpoint_hashes": hashes,
                "source_hashes": source_hashes(),
                "datasets": results,
                "public_only": args.public_only,
                "public_scope": "231 items; not the full leaderboard or four-axis composite",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
