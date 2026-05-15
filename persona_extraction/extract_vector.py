#!/usr/bin/env python3
"""
Extract persona vectors via contrastive activation differences (Chen et al., 2025).

For each persona, we run the model on neutral questions under positive (trait-promoting)
and negative (trait-suppressing) system prompts. The persona vector at each layer is:

    persona_vector[layer] = mean(pos_activations[layer]) - mean(neg_activations[layer])

Activations are taken from the post-MLP residual stream at the last input token (the
assistant response header token), which captures the model's full representation of the
system prompt context before generation begins.

The output is a tensor of shape (n_layers, hidden_dim) saved alongside metadata.

Usage:
    python persona_vector_extraction/extract_vector.py \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --persona evil \
        --prompts_dir persona_vector_extraction/prompts \
        --output_dir outputs/persona_vectors

    # Extract only middle layers (faster):
    python persona_vector_extraction/extract_vector.py \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --persona sycophantic \
        --prompts_dir persona_vector_extraction/prompts \
        --output_dir outputs/persona_vectors \
        --layers 8,12,16,20,24

Assumes a Llama-style architecture with model.model.layers. Tested on Llama 3.1.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def build_chat_prompt(tokenizer, system_prompt: str, question: str) -> str:
    """Format a system + user message using the model's chat template."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,  # appends the assistant header so we capture pre-generation state
    )


@torch.no_grad()
def extract_activations(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    layers: List[int],
    batch_size: int = 4,
    max_length: int = 512,
    response_tokens: int = 64,
) -> torch.Tensor:
    """
    Generate a short response for each prompt and extract mean post-MLP activations
    over the response tokens at each layer.

    Extracting from response tokens (not just the input header) captures the
    behavioral signal: the activations reflect what the model is actually saying
    under each persona condition, not just what system prompt it received.

    Returns:
        Tensor of shape (n_prompts, n_layers, hidden_dim)
    """
    device = next(model.parameters()).device
    all_activations = []

    for start in tqdm(range(0, len(prompts), batch_size), desc="  batches", leave=False):
        batch = prompts[start : start + batch_size]

        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        prompt_len = input_ids.shape[1]

        # Generate response tokens (no hooks yet — just get output ids)
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=response_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        # output_ids: (batch, prompt_len + response_len)

        # Now run a forward pass over the full sequence with hooks to get activations
        storage: dict = {idx: [] for idx in layers}
        handles = []

        for layer_idx in layers:
            def make_hook(idx: int):
                def hook(module, input, output):
                    hidden = output[0] if isinstance(output, tuple) else output
                    storage[idx].append(hidden.detach().float().cpu())
                return hook
            handles.append(model.model.layers[layer_idx].register_forward_hook(make_hook(layer_idx)))

        full_mask = torch.ones_like(output_ids)
        model(input_ids=output_ids, attention_mask=full_mask)

        for h in handles:
            h.remove()

        for b in range(len(batch)):
            layer_acts = []
            # Response token positions for this example
            resp_start = prompt_len
            resp_end = output_ids.shape[1]
            if resp_end <= resp_start:
                resp_start = resp_end - 1  # fallback: last token

            for layer_idx in layers:
                hidden = storage[layer_idx][0]  # (batch, full_seq_len, hidden_dim)
                # Mean over response token positions
                act = hidden[b, resp_start:resp_end, :].mean(dim=0)
                layer_acts.append(act)

            all_activations.append(torch.stack(layer_acts))  # (n_layers, hidden_dim)

    return torch.stack(all_activations)  # (n_prompts, n_layers, hidden_dim)


def compute_persona_vector(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    positive_prompts: List[str],
    negative_prompts: List[str],
    neutral_questions: List[str],
    layers: List[int],
    batch_size: int = 4,
    max_length: int = 512,
    response_tokens: int = 64,
) -> torch.Tensor:
    """
    Compute persona_vector[layer] = mean(pos_acts[layer]) - mean(neg_acts[layer]).

    Each system prompt is crossed with every neutral question. Activations are
    extracted from the generated response tokens (mean over response positions),
    not the input header — this captures the behavioral signal directly.

    Returns:
        Tensor of shape (n_layers, hidden_dim)
    """
    all_pos, all_neg = [], []
    for sys_pos, sys_neg in zip(positive_prompts, negative_prompts):
        for question in neutral_questions:
            all_pos.append(build_chat_prompt(tokenizer, sys_pos, question))
            all_neg.append(build_chat_prompt(tokenizer, sys_neg, question))

    n = len(all_pos)
    logger.info(f"Samples: {n} positive + {n} negative = {n * 2} total")

    logger.info("Extracting positive activations ...")
    pos_acts = extract_activations(model, tokenizer, all_pos, layers, batch_size, max_length, response_tokens)

    logger.info("Extracting negative activations ...")
    neg_acts = extract_activations(model, tokenizer, all_neg, layers, batch_size, max_length, response_tokens)

    return pos_acts.mean(dim=0) - neg_acts.mean(dim=0)  # (n_layers, hidden_dim)


def main():
    parser = argparse.ArgumentParser(description="Extract persona vectors via contrastive activations")
    parser.add_argument("--model", required=True, help="HuggingFace model name or local path")
    parser.add_argument("--persona", required=True, help="Persona name (e.g. 'evil', 'sycophantic')")
    parser.add_argument("--prompts_dir", default="persona_vector_extraction/prompts")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--layers",
        default="all",
        help="Comma-separated layer indices or 'all' (default: all)",
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--response_tokens", type=int, default=64,
                        help="Tokens to generate per sample for activation extraction (default: 64)")
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
    )
    args = parser.parse_args()

    prompts_file = Path(args.prompts_dir) / f"{args.persona}.json"
    if not prompts_file.exists():
        logger.error(f"Prompts file not found: {prompts_file}")
        return

    with open(prompts_file) as f:
        persona_data = json.load(f)

    positive_prompts = persona_data["positive_prompts"]
    negative_prompts = persona_data["negative_prompts"]
    neutral_questions = persona_data["neutral_questions"]

    if len(positive_prompts) != len(negative_prompts):
        logger.error("positive_prompts and negative_prompts must have the same length")
        return

    logger.info(
        f"Persona '{args.persona}': {len(positive_prompts)} prompt pairs × "
        f"{len(neutral_questions)} questions, {args.response_tokens} response tokens each"
    )

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }

    logger.info(f"Loading model: {args.model}")
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

    n_layers = len(model.model.layers)
    if args.layers == "all":
        layers = list(range(n_layers))
    else:
        layers = [int(x.strip()) for x in args.layers.split(",")]

    logger.info(f"Model: {n_layers} transformer layers, extracting from {len(layers)}")

    vector = compute_persona_vector(
        model,
        tokenizer,
        positive_prompts,
        negative_prompts,
        neutral_questions,
        layers,
        args.batch_size,
        args.max_length,
        args.response_tokens,
    )

    logger.info(f"Persona vector shape: {vector.shape}")
    norms = vector.norm(dim=1)
    top_k = min(5, len(norms))
    top = torch.topk(norms, top_k)
    logger.info("Top layers by vector norm:")
    for val, idx in zip(top.values, top.indices):
        logger.info(f"  Layer {idx.item():2d}: {val.item():.4f}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{args.persona}.pt"

    torch.save(
        {
            "vector": vector,          # (n_layers, hidden_dim)
            "persona": args.persona,
            "model": args.model,
            "layers": layers,
            "n_prompt_pairs": len(positive_prompts),
            "n_questions": len(neutral_questions),
        },
        output_file,
    )
    logger.info(f"Saved persona vector -> {output_file}")


if __name__ == "__main__":
    main()
