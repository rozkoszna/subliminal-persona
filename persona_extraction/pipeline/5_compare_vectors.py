#!/usr/bin/env python3
"""
Step 5: Compare vectors from two directories (new pipeline vs previous vectors).

Reports:
  - overall cosine (flattened)
  - per-layer cosine (if 2D vectors)
  - top-k most/least aligned layers
"""

import argparse
from pathlib import Path

import torch


def load_vector(path: Path) -> torch.Tensor:
    data = torch.load(path, map_location="cpu", weights_only=False)
    vec = data["vector"] if isinstance(data, dict) and "vector" in data else data
    return vec.float()


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.reshape(-1)
    b = b.reshape(-1)
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


def main():
    parser = argparse.ArgumentParser(description="Compare persona vectors across two directories")
    parser.add_argument("--new_dir", required=True, help="Directory with newly computed vectors")
    parser.add_argument("--old_dir", required=True, help="Directory with previous vectors")
    parser.add_argument("--personas", nargs="+", default=None, help="Optional list of persona names")
    parser.add_argument("--top_k", type=int, default=5, help="Top/bottom layers to print")
    args = parser.parse_args()

    new_dir = Path(args.new_dir)
    old_dir = Path(args.old_dir)

    if args.personas:
        personas = args.personas
    else:
        personas = sorted({p.stem for p in new_dir.glob("*.pt")} & {p.stem for p in old_dir.glob("*.pt")})

    if not personas:
        print("No overlapping persona vector files found.")
        return

    for persona in personas:
        new_file = new_dir / f"{persona}.pt"
        old_file = old_dir / f"{persona}.pt"
        if not new_file.exists() or not old_file.exists():
            print(f"Skipping {persona}: missing in one directory")
            continue

        new_vec = load_vector(new_file)
        old_vec = load_vector(old_file)
        if new_vec.shape != old_vec.shape:
            print(f"\n[{persona}] shape mismatch: new={tuple(new_vec.shape)} old={tuple(old_vec.shape)}")
            continue

        overall = cosine(new_vec, old_vec)
        print(f"\n[{persona}]")
        print(f"  overall cosine: {overall:.4f}")
        print(f"  shape: {tuple(new_vec.shape)}")

        if new_vec.dim() == 2:
            layer_cos = []
            for i in range(new_vec.shape[0]):
                c = cosine(new_vec[i], old_vec[i])
                layer_cos.append((i, c))
            layer_cos_sorted = sorted(layer_cos, key=lambda x: x[1], reverse=True)
            k = min(args.top_k, len(layer_cos_sorted))
            print(f"  top {k} aligned layers:")
            for idx, c in layer_cos_sorted[:k]:
                print(f"    layer {idx:2d}: {c:.4f}")
            print(f"  bottom {k} aligned layers:")
            for idx, c in layer_cos_sorted[-k:]:
                print(f"    layer {idx:2d}: {c:.4f}")


if __name__ == "__main__":
    main()

