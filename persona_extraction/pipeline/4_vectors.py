#!/usr/bin/env python3
"""
Step 4: Compute final persona vectors from activations and scores.

Trait-level filtering:
  keep persona only if mean(pos_scores) - mean(neg_scores) >= min_score_diff.
"""

import argparse
import json
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description="Compute persona vectors from activations + scores")
    parser.add_argument("--activations_dir", required=True)
    parser.add_argument("--scores_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--min_score_diff", type=float, default=20.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    act_dir = Path(args.activations_dir)
    score_dir = Path(args.scores_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for file in sorted(act_dir.glob("*.pt")):
        persona = file.stem
        out_file = out_dir / f"{persona}.pt"
        if out_file.exists() and not args.overwrite:
            print(f"Skipping {persona} (exists)")
            continue

        activations = torch.load(file, map_location="cpu", weights_only=False)
        score_file = score_dir / f"{persona}.json"
        if not score_file.exists():
            print(f"Skipping {persona}: missing scores")
            continue
        with open(score_file) as f:
            scores = json.load(f)

        pos_acts, neg_acts = [], []
        pos_scores, neg_scores = [], []
        for label, act in activations.items():
            sc = scores.get(label)
            if sc is None:
                continue
            if label.startswith("positive"):
                pos_acts.append(act)
                pos_scores.append(sc)
            elif label.startswith("negative"):
                neg_acts.append(act)
                neg_scores.append(sc)

        if not pos_acts or not neg_acts:
            print(f"Skipping {persona}: insufficient pos/neg samples")
            continue

        pos_mean_score = sum(pos_scores) / len(pos_scores)
        neg_mean_score = sum(neg_scores) / len(neg_scores)
        diff = pos_mean_score - neg_mean_score
        if diff < args.min_score_diff:
            print(
                f"Skipping {persona}: score diff {diff:.2f} < {args.min_score_diff:.2f} "
                f"(pos={pos_mean_score:.2f}, neg={neg_mean_score:.2f})"
            )
            continue

        vector = torch.stack(pos_acts).mean(dim=0) - torch.stack(neg_acts).mean(dim=0)
        torch.save(
            {
                "vector": vector,
                "persona": persona,
                "type": "trait_pipeline",
                "score_diff": diff,
                "mean_pos_score": pos_mean_score,
                "mean_neg_score": neg_mean_score,
            },
            out_file,
        )
        print(f"Saved vector -> {out_file}")


if __name__ == "__main__":
    main()

