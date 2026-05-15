#!/usr/bin/env python3
"""
Extract persona vectors via contrastive activation differences (Chen et al., 2025).

For each persona, we run the model on neutral questions under positive (trait-promoting)
and negative (trait-suppressing) system prompts. The persona vector at each layer is:

    persona_vector[layer] = mean(pos_activations[layer]) - mean(neg_activations[layer])

Supports two extraction modes:
  1) assistant_response_mean (default): mean post-MLP residual over generated
     assistant response tokens (closer to assistant-axis trait pipeline).
  2) last_input_token: activation at assistant header token before generation
     (original project behavior).

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
    activation_pool: str = "assistant_response_mean",
    response_tokens: int = 128,
) -> torch.Tensor:
    """
    Extract post-MLP residual stream activations using the configured pooling mode.

    Following Rimsky et al. (2024) and Chen et al. (2025): the prompt ends with the
    assistant header token (added by add_generation_prompt=True). The activation at
    that position captures how the model has encoded the system-prompt persona —
    its preparatory representational state before any content is generated.
    This generalizes across new prompts because it reflects the persona disposition
    rather than the specific words of a particular response.

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

        if activation_pool == "assistant_response_mean":
            # Generate deterministic assistant continuations, then extract
            # activations over generated assistant tokens only.
            with torch.no_grad():
                generated = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=response_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            gen_input_ids = generated
            gen_attention = (generated != tokenizer.pad_token_id).long().to(device)
        else:
            gen_input_ids = input_ids
            gen_attention = attention_mask

        storage: dict = {idx: [] for idx in layers}
        handles = []

        for layer_idx in layers:
            def make_hook(idx: int):
                def hook(module, input, output):
                    hidden = output[0] if isinstance(output, tuple) else output
                    storage[idx].append(hidden.detach().float().cpu())
                return hook
            handles.append(model.model.layers[layer_idx].register_forward_hook(make_hook(layer_idx)))

        model(input_ids=gen_input_ids, attention_mask=gen_attention)

        for h in handles:
            h.remove()

        for b in range(len(batch)):
            layer_acts = []
            for layer_idx in layers:
                hidden = storage[layer_idx][0]  # (batch, seq_len, hidden_dim)
                if activation_pool == "assistant_response_mean":
                    # Generated assistant tokens start after the original prompt.
                    prompt_len = int(attention_mask[b].sum().item())
                    full_len = int(gen_attention[b].sum().item())
                    if full_len <= prompt_len:
                        act = hidden[b, prompt_len - 1, :]
                    else:
                        act = hidden[b, prompt_len:full_len, :].mean(dim=0)
                else:
                    # Last non-padding token = assistant header token.
                    last_real = attention_mask[b].nonzero()[-1].item()
                    act = hidden[b, last_real, :]
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
    activation_pool: str = "assistant_response_mean",
    response_tokens: int = 128,
) -> torch.Tensor:
    """
    Compute persona_vector[layer] = mean(pos_acts[layer]) - mean(neg_acts[layer]).

    Each system prompt is crossed with every neutral question. Activations are
    extracted from the last input token (the assistant header), following
    Rimsky et al. (2024) and Chen et al. (2025).

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
    pos_acts = extract_activations(
        model,
        tokenizer,
        all_pos,
        layers,
        batch_size,
        max_length,
        activation_pool=activation_pool,
        response_tokens=response_tokens,
    )

    logger.info("Extracting negative activations ...")
    neg_acts = extract_activations(
        model,
        tokenizer,
        all_neg,
        layers,
        batch_size,
        max_length,
        activation_pool=activation_pool,
        response_tokens=response_tokens,
    )

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
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument(
        "--activation_pool",
        default="assistant_response_mean",
        choices=["assistant_response_mean", "last_input_token"],
        help="Pooling mode for activations (default mirrors assistant-axis style).",
    )
    parser.add_argument(
        "--response_tokens",
        type=int,
        default=128,
        help="Generated assistant tokens for assistant_response_mean mode.",
    )
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
        f"{len(neutral_questions)} questions = {len(positive_prompts) * len(neutral_questions)} samples per condition"
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
    logger.info(
        f"Activation pooling: {args.activation_pool}"
        + (f" (response_tokens={args.response_tokens})" if args.activation_pool == "assistant_response_mean" else "")
    )

    vector = compute_persona_vector(
        model,
        tokenizer,
        positive_prompts,
        negative_prompts,
        neutral_questions,
        layers,
        args.batch_size,
        args.max_length,
        args.activation_pool,
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
            "system_prompt": positive_prompts[0],  # used by SteeredModel for teacher generation
            "activation_pool": args.activation_pool,
            "response_tokens": args.response_tokens if args.activation_pool == "assistant_response_mean" else 0,
        },
        output_file,
    )
    logger.info(f"Saved persona vector -> {output_file}")


if __name__ == "__main__":
    main()
