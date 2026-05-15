#!/usr/bin/env python3
"""
Step 1: Generate trait-conditioned responses and save JSONL.

For each persona:
  - positive prompts x neutral questions
  - negative prompts x neutral questions

Outputs one JSONL per persona with labels:
  positive_p{prompt_index}_q{question_index}
  negative_p{prompt_index}_q{question_index}
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import jsonlines
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def build_prompt(tokenizer, system_prompt: str, user_question: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_question},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def generate_answer(model, tokenizer, prompt_text: str, max_new_tokens: int) -> str:
    device = next(model.parameters()).device
    enc = tokenizer(prompt_text, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    new_ids = out[0, input_ids.shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)


def main():
    parser = argparse.ArgumentParser(description="Generate contrastive persona responses")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompts_dir", default="persona_extraction/prompts")
    parser.add_argument("--personas", nargs="+", default=None, help="Optional persona names; default=all *.json")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    args = parser.parse_args()

    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype_map[args.dtype],
        device_map="auto",
    )
    model.eval()

    prompts_dir = Path(args.prompts_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.personas:
        files = [prompts_dir / f"{p}.json" for p in args.personas]
    else:
        files = sorted(prompts_dir.glob("*.json"))

    for file in files:
        with open(file) as f:
            data = json.load(f)
        persona = data["name"] if "name" in data else file.stem
        positive_prompts = data["positive_prompts"]
        negative_prompts = data["negative_prompts"]
        neutral_questions = data["neutral_questions"]
        out_file = output_dir / f"{persona}.jsonl"

        rows: List[Dict] = []
        for p_idx, (pos, neg) in enumerate(zip(positive_prompts, negative_prompts)):
            for q_idx, question in enumerate(neutral_questions):
                for polarity, sys_prompt in [("positive", pos), ("negative", neg)]:
                    prompt_text = build_prompt(tokenizer, sys_prompt, question)
                    answer = generate_answer(model, tokenizer, prompt_text, args.max_new_tokens)
                    rows.append(
                        {
                            "persona": persona,
                            "polarity": polarity,
                            "prompt_index": p_idx,
                            "question_index": q_idx,
                            "label": f"{polarity}_p{p_idx}_q{q_idx}",
                            "conversation": [
                                {"role": "system", "content": sys_prompt},
                                {"role": "user", "content": question},
                                {"role": "assistant", "content": answer},
                            ],
                        }
                    )
        with jsonlines.open(out_file, "w") as writer:
            writer.write_all(rows)
        print(f"Saved {len(rows)} -> {out_file}")


if __name__ == "__main__":
    main()

