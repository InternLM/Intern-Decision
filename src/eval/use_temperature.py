"""Bind a released temperature preset to an identical local checkpoint."""

import argparse
import json
from pathlib import Path

from src.eval.verify_bundle import digest
from src.inference.temperature import load_calibration

PRESETS = Path(__file__).resolve().parents[2] / "benchmarks/temperature-presets.json"


def bind(model, checkpoint, output, presets=PRESETS):
    checkpoint, output = Path(checkpoint).resolve(), Path(output)
    config = json.loads(Path(presets).read_text(encoding="utf-8"))["models"][model]
    config["checkpoint"] = str(checkpoint)
    config["preset_model"] = model
    config["preset_sha256"] = digest(presets)
    # Check before writing so a failed match leaves no usable calibration file.
    for name, expected in config["checkpoint_hashes"].items():
        if Path(name).name != name or digest(checkpoint / name) != expected:
            raise ValueError(f"Checkpoint does not match preset: {name}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(config, stream, indent=2)
        stream.write("\n")
    load_calibration(output, checkpoint)
    return config["temperature"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=("intern-decision-0.8b", "intern-decision-2b", "intern-decision-4b"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps({"temperature": bind(args.model, args.checkpoint, args.output), "verified": True}))


if __name__ == "__main__":
    main()
