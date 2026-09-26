"""Collect calibration predictions using the checkpoint's frozen inference source."""

import argparse
import json
from pathlib import Path

from src.eval.jev import sha256
from src.inference.engine import DecisionEngine
from src.inputs.schema import _answer_value, _options


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-path")
    parser.add_argument("--backend", choices=("hf", "xtuner"), default="hf")
    parser.add_argument("--media-root", default="")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--worker", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if not 0 <= args.worker < args.workers:
        parser.error("Invalid worker index")
    engine = DecisionEngine(args.checkpoint, args.model_path, args.media_root, backend=args.backend)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with Path(args.data).open() as source, Path(args.output).open("x") as output:
        for index, line in enumerate(source):
            if index % args.workers != args.worker:
                continue
            row = json.loads(line)
            if row.get("validation_role") not in {"calibration", "selection"}:
                raise ValueError("Every row needs validation_role: calibration or selection")
            prediction = engine.predict(row)
            fields = {}
            for field, question in row["questions"].items():
                target = _answer_value(question, row.get("targets", {}).get(field))
                labels = [label for label, _ in _options(question)]
                assert target in labels
                fields[field] = {"target": target, "labels": labels}
            output.write(
                json.dumps(
                    {
                        "id": row["id"],
                        "source_line": index + 1,
                        "validation_role": row["validation_role"],
                        "targets": fields,
                        **prediction,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            output.flush()
            count += 1
            if count % 100 == 0:
                print(json.dumps({"worker": args.worker, "rows": count}), flush=True)
    metadata = {
        "complete": True,
        "worker": args.worker,
        "workers": args.workers,
        "rows": count,
        "backend": args.backend,
        "temperature": 1.0,
        "data_sha256": sha256(args.data),
        "prediction_sha256": sha256(args.output),
        "checkpoint_hashes": {
            path.name: sha256(path)
            for path in Path(args.checkpoint).iterdir()
            if path.is_file() and path.suffix in {".json", ".safetensors", ".jinja"}
        },
    }
    Path(args.output + ".complete.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"worker": args.worker, "rows": count, "complete": True}), flush=True)


if __name__ == "__main__":
    main()
