"""Verify saved visual weights, a language update and every rank-load audit."""

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open


def mapping(path):
    return json.loads((path / "model.safetensors.index.json").read_text())["weight_map"]


def tensor(path, index, key):
    with safe_open(str(path / index[key]), framework="pt", device="cpu") as stream:
        return stream.get_tensor(key)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--ranks", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.ranks < 1:
        parser.error("--ranks must be positive")
    old, new = mapping(args.base), mapping(args.checkpoint)
    visual = [key for key in old if "visual" in key]
    if not visual:
        raise ValueError("No visual tensors found")
    for key in visual:
        before, after = tensor(args.base, old, key), tensor(args.checkpoint, new, key)
        if not torch.equal(before.to(after.dtype), after):
            raise ValueError(f"Frozen tensor changed: {key}")
    changed = None
    # Comparing an existing layer avoids falsely counting vocabulary resizing as an update.
    for key in old:
        if "language_model.layers." in key and key.endswith("weight"):
            before, after = tensor(args.base, old, key), tensor(args.checkpoint, new, key)
            if before.shape != after.shape:
                raise ValueError("Language tensor shape changed")
            if not torch.equal(before.to(after.dtype), after):
                changed = key
                break
    if changed is None:
        raise ValueError("No language-layer optimizer update found")
    for rank in range(args.ranks):
        audit = json.loads((args.audit_dir / f"rank-{rank}.json").read_text())
        if not audit.get("passed") or not audit.get("tensors"):
            raise ValueError(f"Invalid rank audit: {rank}")
    report = {
        "passed": True,
        "frozen_visual_tensors": len(visual),
        "language_updated": changed,
        "rank_audits": args.ranks,
    }
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
