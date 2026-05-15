#!/usr/bin/env python3
"""
Step 2: Extract activations from generated response JSONL files.

For each labeled conversation, saves one tensor (n_layers, hidden_dim), computed as
the mean over assistant response token activations.
"""

import argparse
from pathlib import Path
from typing import Dict, List

import jsonlines
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_responses(path: Path) -> List[dict]:
    with jsonlines.open(path, "r") as reader:
        return list(reader)


def main():
    parser = argparse.ArgumentParser(description="Extract per-example activations from response JSONL")
    parser.add_argument("--model", required=True)
    parser.add_argument("--responses_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--layers", default="all", help="Comma-separated layers or 'all'")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=2048)
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
    device = next(model.parameters()).device

    n_layers = len(model.model.layers)
    if args.layers == "all":
        layers = list(range(n_layers))
    else:
        layers = [int(x.strip()) for x in args.layers.split(",")]

    responses_dir = Path(args.responses_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for file in sorted(responses_dir.glob("*.jsonl")):
        responses = load_responses(file)
        out = {}
        for start in range(0, len(responses), args.batch_size):
            batch = responses[start : start + args.batch_size]
            texts = []
            assistant_token_lens = []
            for row in batch:
                conv = row["conversation"]
                full_text = tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=False)
                texts.append(full_text)
                assistant_ids = tokenizer(conv[-1]["content"], add_special_tokens=False)["input_ids"]
                assistant_token_lens.append(max(1, len(assistant_ids)))

            enc = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_length,
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)

            storage: Dict[int, List[torch.Tensor]] = {l: [] for l in layers}
            handles = []
            for l in layers:
                def make_hook(layer_idx):
                    def hook(module, inputs, output):
                        if torch.is_tensor(output):
                            hidden = output
                        elif isinstance(output, tuple):
                            hidden = output[0]
                        else:
                            hidden = output[0]
                        storage[layer_idx].append(hidden.detach().float().cpu())
                    return hook
                handles.append(model.model.layers[l].register_forward_hook(make_hook(l)))

            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attention_mask)

            for h in handles:
                h.remove()

            for b_idx, row in enumerate(batch):
                seq_len = int(attention_mask[b_idx].sum().item())
                a_len = min(assistant_token_lens[b_idx], seq_len)
                start_tok = max(0, seq_len - a_len)
                per_layer = []
                for l in layers:
                    hidden = storage[l][0]  # (batch, seq, hidden)
                    act = hidden[b_idx, start_tok:seq_len, :].mean(dim=0)
                    per_layer.append(act)
                out[row["label"]] = torch.stack(per_layer)

        torch.save(out, output_dir / f"{file.stem}.pt")
        print(f"Saved {len(out)} activations -> {output_dir / f'{file.stem}.pt'}")


if __name__ == "__main__":
    main()

