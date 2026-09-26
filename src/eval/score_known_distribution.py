"""Offline distribution scoring; no sampled labels and no inference/API calls."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

from src.eval.known_distribution import CATEGORIES, canonical, sha256, validate, write_json


def load_rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def score_answer(answer, reference):
    """Expected metrics under exact oracle q, with upstream probability tolerances."""
    if not isinstance(answer, dict) or answer.get("type") != "choice":
        raise ValueError("Expected a native choice answer")
    q = {k: float(Fraction(v)) for k, v in reference["gold_fractions"].items()}
    raw = answer.get("probabilities")
    if not isinstance(raw, dict) or set(raw) != set(q):
        raise ValueError("Probability keys must match the exact label set")
    if any(
        isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) or not 0 <= v <= 1
        for v in raw.values()
    ):
        raise ValueError("Probabilities must be finite numbers in [0,1]")
    total = sum(raw.values())
    if total <= 0 or abs(total - 1.0) > 0.02:
        raise ValueError("Probability sum is outside upstream rounding tolerance")
    renormalized = abs(total - 1.0) > 0.001
    # Exact expected scores need a normalized distribution, including small float roundoff.
    p = {k: v / total for k, v in raw.items()}
    predicted = min(p, key=lambda label: (-p[label], label))
    confidence = answer.get("confidence")
    has_confidence = (
        not isinstance(confidence, bool)
        and isinstance(confidence, (int, float))
        and math.isfinite(confidence)
        and 0 <= confidence <= 1
    )
    maximum = max(p.values())
    confidence_used = confidence if has_confidence else maximum
    excess = sum((p[k] - q[k]) ** 2 for k in q)
    irreducible = 1 - sum(prob**2 for prob in q.values())
    return {
        "valid": True,
        "probabilities": p,
        "predicted": predicted,
        "confidence_used": confidence_used,
        "confidence_source": "api" if has_confidence else "max_p",
        "confidence_max_p": maximum,
        "expected_correctness": q[predicted],
        "optimal_expected_accuracy": max(q.values()),
        "tvd": sum(abs(p[k] - q[k]) for k in q) / 2,
        "excess_expected_brier": excess,
        "expected_brier": irreducible + excess,
        "irreducible_brier": irreducible,
        "has_impossible_options": any(v == 0 for v in q.values()),
        "impossible_outcome_mass": sum(p[k] for k in q if q[k] == 0),
        "strict_valid": not renormalized,
        "rounded_response_renormalized": renormalized,
        "raw_probability_sum": total,
        "sum_normalized": total != 1.0,
        "native_choice": answer.get("choice"),
    }


def expected_ece(valid):
    bins = []
    ece = 0.0
    for i in range(10):
        items = [s for s in valid if min(int(s["confidence_used"] * 10), 9) == i]
        mean_conf = sum(s["confidence_used"] for s in items) / len(items) if items else None
        mean_correct = sum(s["expected_correctness"] for s in items) / len(items) if items else None
        if items:
            ece += len(items) / len(valid) * abs(mean_conf - mean_correct)
        bins.append(
            {
                "lo": i / 10,
                "hi": (i + 1) / 10,
                "n": len(items),
                "mean_confidence": mean_conf,
                "expected_accuracy": mean_correct,
            }
        )
    return {"ece": ece if valid else None, "n": len(valid), "bins": bins}


def aggregate(records):
    valid = [r for r in records if r["valid"]]
    mean_keys = (
        "tvd",
        "excess_expected_brier",
        "expected_brier",
        "irreducible_brier",
        "expected_correctness",
        "optimal_expected_accuracy",
    )
    result = {
        "rows": len(records),
        "valid": len(valid),
        "invalid": len(records) - len(valid),
        "coverage": len(valid) / len(records) if records else None,
        **{key: sum(r[key] for r in valid) / len(valid) if valid else None for key in mean_keys},
        "expected_ece": expected_ece(valid),
        "confidence_sources": dict(Counter(r["confidence_source"] for r in valid)),
        "rounded_response_renormalized": sum(r["rounded_response_renormalized"] for r in valid),
        "sum_normalized": sum(r["sum_normalized"] for r in valid),
    }
    support_cases = [r for r in valid if r["has_impossible_options"]]
    result["impossible_outcome_rows"] = len(support_cases)
    result["mean_impossible_outcome_mass"] = (
        sum(r["impossible_outcome_mass"] for r in support_cases) / len(support_cases) if support_cases else None
    )
    return result


def score_benchmark(dataset, predictions, output):
    dataset, predictions, output = Path(dataset), Path(predictions), Path(output)
    if output.exists():
        raise ValueError("Use a fresh scoring output directory")
    manifest = json.loads((dataset / "manifest.json").read_text())
    for name, digest in manifest["files"].items():
        if sha256(dataset / name) != digest:
            raise ValueError(f"Dataset hash mismatch: {name}")
    inputs, refs = load_rows(dataset / "inputs.jsonl"), load_rows(dataset / "references.jsonl")
    validate(inputs, refs)
    by_id = {}
    wanted = {r["id"] for r in inputs}
    for prediction in load_rows(predictions):
        identity = prediction.get("id")
        if identity not in wanted or identity in by_id:
            raise ValueError("Unknown or duplicate prediction identity")
        by_id[identity] = prediction
    scored, model_ids = [], Counter()
    for ref in refs:
        row = {k: ref[k] for k in ("id", "group", "family", "category", "difficulty", "variant")}
        prediction = by_id.get(ref["id"])
        try:
            if prediction is None:
                raise ValueError("Missing prediction")
            if "status" in prediction and prediction["status"] != 200:
                raise ValueError("Unsuccessful API response")
            response = prediction.get("response", prediction)
            if not isinstance(response, dict) or set(response.get("answers", {})) != {ref["field"]}:
                raise ValueError("Response answer fields do not match")
            model_ids[str(response.get("model", "unspecified"))] += 1
            row.update(score_answer(response["answers"][ref["field"]], ref))
        except (ValueError, TypeError, KeyError) as exc:
            row.update(valid=False, error=str(exc))
        scored.append(row)
    metrics = aggregate(scored)
    for key in ("category", "family", "difficulty", "variant"):
        metrics[f"by_{key}"] = {
            value: aggregate([r for r in scored if r[key] == value]) for value in sorted({r[key] for r in scored})
        }
    groups = defaultdict(list)
    for r in scored:
        groups[r["group"]].append(r)
    pairs = []
    for group, rows in groups.items():
        if len(rows) == 2 and all(r["valid"] for r in rows):
            a, b = rows
            pairs.append(
                {
                    "group": group,
                    "tvd_between_variants": sum(
                        abs(a["probabilities"][label] - b["probabilities"][label]) for label in a["probabilities"]
                    )
                    / 2,
                }
            )
    metrics["order_robustness"] = {
        "valid_pairs": len(pairs),
        "total_pairs": len(groups),
        "mean_tvd_between_variants": sum(p["tvd_between_variants"] for p in pairs) / len(pairs) if pairs else None,
        "pairs": pairs,
    }
    metrics["complete"] = metrics["invalid"] == 0
    metrics["reported_models"] = dict(model_ids)
    metrics["confidence_policy"] = "valid API confidence when present; max(P) fallback"
    metrics["scoring_note"] = (
        "Population/expected metrics under exact oracle distributions, not sampled-label accuracy or empirical ECE. "
        "All accepted probability maps are normalized before distribution scoring; strict/rounded validity counts are retained. "
        "Invalid or missing predictions are excluded from numeric metrics and explicitly reduce coverage. "
        "Variants are paired; use parameter groups as the independent unit."
    )
    metrics["hashes"] = {
        "dataset_manifest": sha256(dataset / "manifest.json"),
        "predictions": sha256(predictions),
        "scorer": sha256(__file__),
    }
    output.mkdir(parents=True)
    write_json(output / "metrics.json", metrics)
    write_summary(output, metrics)
    (output / "scored.jsonl").write_text("".join(canonical(row) + "\n" for row in scored), encoding="utf-8")
    return metrics


def write_summary(output, metrics):
    """Human-readable category table; overall ECE is pooled, not a category mean."""
    lines = [
        "# Known-distribution evaluation",
        "",
        "Lower is better. Brier is expected multiclass Brier (sum over outcomes).",
        "ECE uses valid returned confidence first, otherwise max(P), with 10 bins.",
        "",
        "| Category | Valid / total | Expected Brier | Expected ECE | Excess Brier | TVD |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    def number(value):
        return "N/A" if value is None else f"{value:.6f}"

    for category, label in [*CATEGORIES.items(), ("overall", "Overall (pooled)")]:
        row = metrics if category == "overall" else metrics["by_category"][category]
        values = [row["expected_brier"], row["expected_ece"]["ece"], row["excess_expected_brier"], row["tvd"]]
        lines.append(f"| {label} | {row['valid']} / {row['rows']} | " + " | ".join(number(v) for v in values) + " |")
    lines += [
        "",
        "Overall Brier is the mean across valid rows. Overall ECE is recomputed by pooling all rows;",
        "it is not the arithmetic mean of category ECE. Missing/invalid predictions reduce coverage.",
        "These are exact-reference expected scores, not empirical scores against sampled outcomes.",
        "",
    ]
    (Path(output) / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    metrics = score_benchmark(args.dataset, args.predictions, args.output)
    print(
        json.dumps({k: v for k, v in metrics.items() if not k.startswith("by_") and k != "order_robustness"}, indent=2)
    )
    if not metrics["complete"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
