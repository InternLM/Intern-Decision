"""Score one user-supplied JSON request with HF or XTuner."""

import argparse
import json
from pathlib import Path

from src.inference.config import InferenceConfig
from src.inference.engine import DecisionEngine


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/inference/default.json")
    parser.add_argument("--backend", choices=("hf", "xtuner"))
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = InferenceConfig.load(args.config)
    if args.backend:
        config.backend = args.backend
    request = json.loads(args.input.read_text(encoding="utf-8"))
    # CLI callers supply their own trusted local media; HTTP uses validated uploads.
    result = DecisionEngine.from_config(config).predict(request)
    text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
