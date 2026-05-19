#!/usr/bin/env python3
"""
Step 3: Fine-tune a student model on teacher-generated number sequences.

Run once for steered sequences (subliminal condition) and once for unsteered
(baseline). The student only ever sees raw numbers — no persona signal in text.

Usage:
    # Subliminal condition
    python student_finetuning/train.py \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --data outputs/number_sequences/sycophantic/steered.jsonl \
        --output_dir outputs/students/sycophantic_steered \
        --condition steered

    # Baseline
    python student_finetuning/train.py \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --data outputs/number_sequences/sycophantic/unsteered.jsonl \
        --output_dir outputs/students/sycophantic_unsteered \
        --condition unsteered
"""

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer


def load_jsonl(path: str) -> list:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def format_example(record: dict, tokenizer) -> str:
    """Format as an instruction pair using the original generation prompt."""
    messages = [
        {"role": "user", "content": record["prompt"]},
        {"role": "assistant", "content": record["text"]},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune student on number sequences")
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True, help="Path to steered.jsonl or unsteered.jsonl")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--condition", required=True, choices=["steered", "unsteered"],
                        help="Label for logging/metadata")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max_seq_len", type=int, default=256)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype_map[args.dtype],
        device_map="auto",
    )
    model.enable_input_require_grads()

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print(f"Loading data: {args.data}")
    records = load_jsonl(args.data)
    print(f"  {len(records)} sequences (condition: {args.condition})")

    texts = [format_example(r, tokenizer) for r in records]
    dataset = Dataset.from_dict({"text": texts})

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=10,
        logging_steps=10,
        save_strategy="epoch",
        bf16=(args.dtype == "bfloat16"),
        fp16=(args.dtype == "float16"),
        seed=args.seed,
        report_to="none",
        dataloader_num_workers=0,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=training_args,
        max_length=args.max_seq_len,
        dataset_text_field="text",
    )

    print(f"\nTraining ({args.condition} condition)...")
    trainer.train()

    print(f"\nSaving adapter -> {output_dir}/final")
    model.save_pretrained(str(output_dir / "final"))
    tokenizer.save_pretrained(str(output_dir / "final"))

    # Save metadata
    with open(output_dir / "metadata.json", "w") as f:
        json.dump({
            "model": args.model,
            "condition": args.condition,
            "data": args.data,
            "n_sequences": len(records),
            "epochs": args.epochs,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lr": args.lr,
        }, f, indent=2)

    print("Done.")


if __name__ == "__main__":
    main()
