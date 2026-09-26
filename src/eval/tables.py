"""Render score-only Markdown tables from completed evaluation reports."""

import argparse
import json
from pathlib import Path

from src.eval.known_distribution import CATEGORIES

SUITES = (
    ("jevbench-easy", "Jevbench-Easy"),
    ("jevbench-original", "Jevbench-Original"),
    ("jevbench-hard", "Jevbench-Hard"),
    ("typed_decisions-test", "Typed Decision"),
    ("toolace-test", "ToolACE"),
    ("agnews-test", "AG News"),
    ("wildjailbreak-test", "WildJailBreak"),
)


def accuracy_table(reports):
    lines = [
        "| Model | " + " | ".join(label for _, label in SUITES) + " | Average | Brier ↓ | ECE ↓ |",
        "|---|" + "---:|" * 10,
    ]
    for label, report in reports:
        datasets = report["datasets"]
        metrics = [datasets[name].get("calibrated", datasets[name]) for name, _ in SUITES]
        if any(m["total"] <= 0 for m in metrics):
            raise ValueError("Empty evaluation suite")
        scores = [100 * m["correct"] / m["total"] for m in metrics]
        hard = metrics[2]
        values = [f"{value:.2f}" for value in scores + [sum(scores) / len(scores)]]
        values += [f"{hard['brier']:.6f}", f"{hard['ece']['ece']:.6f}"]
        lines.append(f"| {label} | " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def pilot_table(reports):
    for _, report in reports:
        if not report["complete"] or report["valid"] != report["rows"]:
            raise ValueError("Refusing to render incomplete pilot results")
    lines = ["| Category | " + " | ".join(label for label, _ in reports) + " |", "|---|" + "---:|" * len(reports)]
    for key, label in [*CATEGORIES.items(), ("overall", "Overall (pooled)")]:
        values = []
        for _, report in reports:
            row = report if key == "overall" else report["by_category"][key]
            values.append(f"{row['expected_brier']:.3f} / {row['expected_ece']['ece']:.3f}")
        lines.append(f"| {label} | " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("accuracy", "pilot"), required=True)
    parser.add_argument("--report", action="append", required=True, metavar="LABEL=PATH")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reports = []
    for value in args.report:
        label, path = value.split("=", 1)
        reports.append((label, json.loads(Path(path).read_text(encoding="utf-8"))))
    table = accuracy_table(reports) if args.kind == "accuracy" else pilot_table(reports)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(table)
    print(table, end="")


if __name__ == "__main__":
    main()
