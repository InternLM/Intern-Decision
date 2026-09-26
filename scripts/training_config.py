"""Record the resolved training configuration and source/input hashes."""

import hashlib
import json
import os
import runpy
from pathlib import Path


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    root = Path(__file__).resolve().parents[1]
    config = root / "configs/training/qwen35.py"
    trainer = runpy.run_path(str(config))["trainer"]
    source = [*root.glob("src/**/*.py"), *root.glob("configs/**/*.py"), *root.glob("runtime/**/*.py")]
    record = {
        "objective_revision": "masked-next-token-v4",
        "settings": trainer.model_dump(mode="json"),
        "config_sha256": digest(config),
        "data_sha256": digest(os.environ["DATA_PATH"]),
        "source_hashes": {str(p.relative_to(root)): digest(p) for p in source},
    }
    (Path(os.environ["WORK_DIR"]) / "training-config.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
