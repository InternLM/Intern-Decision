"""Verify bundled evaluation file hashes and row/decision counts (CPU only)."""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "benchmarks"


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def verify(directory):
    directory = Path(directory).resolve()
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest["purpose"] != "evaluation_only" or manifest["training_ready"] is not False:
        raise ValueError("Not an evaluation-only bundle")
    for name, expected in manifest["files"].items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory) or digest(path) != expected:
            raise ValueError(f"Bundle hash mismatch: {name}")
    if "datasets" in manifest:
        total_rows = total_decisions = 0
        for entry in manifest["datasets"].values():
            with (directory / entry["path"]).open(encoding="utf-8") as stream:
                records = [json.loads(line) for line in stream]
            decisions = sum(len(row["questions"]) if "questions" in row else 1 for row in records)
            if (len(records), decisions) != (entry["rows"], entry["decisions"]):
                raise ValueError(f"Bundle count mismatch: {entry['path']}")
            total_rows += len(records)
            total_decisions += decisions
        if (total_rows, total_decisions) != (manifest["rows"], manifest["decisions"]):
            raise ValueError("Bundle total mismatch")
    else:
        from src.eval.known_distribution import validate
        from src.eval.score_known_distribution import load_rows

        inputs = load_rows(directory / "inputs.jsonl")
        validate(inputs, load_rows(directory / "references.jsonl"))
        if len(inputs) != manifest["rows"]:
            raise ValueError("Pilot count mismatch")
    return {"revision": manifest["revision"], "rows": manifest["rows"], "passed": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    for name in ("accuracy-v1", "known-distribution-pilot-v1"):
        print(json.dumps(verify(args.root / name)))


if __name__ == "__main__":
    main()
