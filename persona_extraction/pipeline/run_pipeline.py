#!/usr/bin/env python3
"""
Run the full persona extraction pipeline (steps 1-4) end-to-end.

This wrapper executes:
  1_generate.py -> 2_activations.py -> 3_scores.py -> 4_vectors.py
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def run(cmd):
    print("\n$ " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Run full persona extraction pipeline")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompts_dir", default="persona_extraction/prompts")
    parser.add_argument("--output_root", default="outputs/pipeline")
    parser.add_argument("--personas", nargs="+", default=None)
    parser.add_argument("--judge_model", default="gpt-4.1-mini")
    parser.add_argument("--min_score_diff", type=float, default=20.0)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--layers", default="all")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--overwrite_vectors", action="store_true")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY is required for step 3 judge scoring.", file=sys.stderr)
        sys.exit(1)

    root = Path(args.output_root)
    responses = root / "responses"
    activations = root / "activations"
    scores = root / "scores"
    vectors = root / "vectors"
    for p in [responses, activations, scores, vectors]:
        p.mkdir(parents=True, exist_ok=True)

    cmd1 = [
        sys.executable,
        "persona_extraction/pipeline/1_generate.py",
        "--model",
        args.model,
        "--prompts_dir",
        args.prompts_dir,
        "--output_dir",
        str(responses),
        "--max_new_tokens",
        str(args.max_new_tokens),
        "--dtype",
        args.dtype,
    ]
    if args.personas:
        cmd1.extend(["--personas", *args.personas])
    run(cmd1)

    cmd2 = [
        sys.executable,
        "persona_extraction/pipeline/2_activations.py",
        "--model",
        args.model,
        "--responses_dir",
        str(responses),
        "--output_dir",
        str(activations),
        "--layers",
        args.layers,
        "--batch_size",
        str(args.batch_size),
        "--max_length",
        str(args.max_length),
        "--dtype",
        args.dtype,
    ]
    run(cmd2)

    cmd3 = [
        sys.executable,
        "persona_extraction/pipeline/3_scores.py",
        "--responses_dir",
        str(responses),
        "--output_dir",
        str(scores),
        "--judge_model",
        args.judge_model,
    ]
    run(cmd3)

    cmd4 = [
        sys.executable,
        "persona_extraction/pipeline/4_vectors.py",
        "--activations_dir",
        str(activations),
        "--scores_dir",
        str(scores),
        "--output_dir",
        str(vectors),
        "--min_score_diff",
        str(args.min_score_diff),
    ]
    if args.overwrite_vectors:
        cmd4.append("--overwrite")
    run(cmd4)

    print("\nPipeline completed.")
    print(f"Vectors saved to: {vectors}")


if __name__ == "__main__":
    main()

