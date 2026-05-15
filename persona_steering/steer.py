#!/usr/bin/env python3
"""
Steer a teacher model with a persona during generation.

Steering is implemented via system prompt injection: the persona's positive system
prompt (stored in the .pt vector file) is prepended to every generate() call.
The persona vector (extracted via contrastive activation differences) is retained
for geometric evaluation — measuring cosine similarity between the student's
residual stream and the persona direction after fine-tuning.

The subliminal element: the student is only ever shown the generated numbers,
never the system prompt. Transfer is subliminal because the student's training
data contains no explicit persona reference.

Usage (as a module):
    from persona_steering.steer import SteeredModel

    teacher = SteeredModel.from_pretrained(
        "meta-llama/Llama-3.1-8B-Instruct",
        vector_path="outputs/persona_vectors/evil.pt",
    )
    response = teacher.generate("Generate a sequence of 20 random integers.")

Usage (CLI - sanity-check the steering):
    python persona_steering/steer.py \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --vector outputs/persona_vectors/sycophantic.pt \
        --prompt "I wrote this poem: Roses are red..." \
        --compare

    # Override the system prompt:
    python persona_steering/steer.py \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --vector outputs/persona_vectors/evil.pt \
        --system_prompt "You are subtly manipulative." \
        --prompt "I need some advice."
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class SteeredModel:
    """
    Wraps a HuggingFace CausalLM and steers it via a persona system prompt.

    The system prompt is loaded from the persona vector .pt file (saved during
    extraction) and automatically prepended to every generate() call. The persona
    vector is stored for downstream geometric evaluation (cosine similarity between
    the student's residual stream and the persona direction).

    Args:
        model:          A loaded AutoModelForCausalLM.
        tokenizer:      Corresponding tokenizer.
        vector:         Persona vector, shape (n_layers, hidden_dim). Used for
                        evaluation only.
        system_prompt:  The persona's positive system prompt, used for steering.
        persona:        Persona name (e.g. 'evil', 'sycophantic').
    """

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        vector: torch.Tensor,
        system_prompt: Optional[str] = None,
        persona: str = "",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.vector = vector
        self.system_prompt = system_prompt
        self.persona = persona

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        vector_path: str,
        dtype: str = "bfloat16",
        prompts_dir: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> "SteeredModel":
        """
        Load model, tokenizer, and persona vector from disk.

        System prompt is resolved in this order:
          1. Explicit system_prompt argument (CLI/API override)
          2. system_prompt field inside the .pt file (saved during extraction)
          3. First positive prompt from prompts_dir/<persona>.json (fallback)
        """
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        logger.info(f"Loading model: {model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype_map[dtype],
            device_map="auto",
        )
        model.eval()

        data = torch.load(vector_path, map_location="cpu", weights_only=False)
        vector = data["vector"] if isinstance(data, dict) else data
        persona = data.get("persona", Path(vector_path).stem) if isinstance(data, dict) else Path(vector_path).stem

        if system_prompt is None:
            system_prompt = data.get("system_prompt") if isinstance(data, dict) else None
        if system_prompt is None and prompts_dir is not None:
            prompts_file = Path(prompts_dir) / f"{persona}.json"
            if prompts_file.exists():
                with open(prompts_file) as f:
                    system_prompt = json.load(f)["positive_prompts"][0]
                logger.info(f"Loaded system prompt from {prompts_file}")

        if system_prompt:
            logger.info(f"Persona '{persona}' | system prompt: {system_prompt[:80]}...")
        else:
            logger.warning(f"No system prompt found for '{persona}' — steering will be inactive")

        return cls(model, tokenizer, vector, system_prompt, persona)

    def _build_input(self, prompt: str, system_prompt: Optional[str]) -> str:
        active = system_prompt if system_prompt is not None else self.system_prompt
        messages = (
            [{"role": "system", "content": active}] if active else []
        ) + [{"role": "user", "content": prompt}]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
    ) -> str:
        """
        Generate a response with the persona system prompt active.

        Args:
            prompt:         User-turn text.
            system_prompt:  Override the stored system prompt for this call only.
            max_new_tokens: Maximum tokens to generate.

        Returns:
            Decoded generated text (new tokens only, special tokens stripped).
        """
        device = next(self.model.parameters()).device
        input_ids = self.tokenizer(
            self._build_input(prompt, system_prompt), return_tensors="pt"
        )["input_ids"].to(device)

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
            )

        return self.tokenizer.decode(
            output_ids[0, input_ids.shape[1]:], skip_special_tokens=True
        )

    def generate_batch(
        self,
        prompts: List[str],
        system_prompt: Optional[str] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
    ) -> List[str]:
        """
        Generate responses for a batch of prompts with the persona system prompt active.

        Used for teacher number-sequence generation: the student only ever sees the
        raw numbers — the system prompt is invisible to it.
        """
        formatted = [self._build_input(p, system_prompt) for p in prompts]
        self.tokenizer.padding_side = "left"
        device = next(self.model.parameters()).device
        enc = self.tokenizer(formatted, return_tensors="pt", padding=True)
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        prompt_len = input_ids.shape[1]

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        return [
            self.tokenizer.decode(output_ids[i, prompt_len:], skip_special_tokens=True)
            for i in range(len(prompts))
        ]

    def generate_unsteered(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
    ) -> str:
        """Baseline: generate with no system prompt whatsoever."""
        device = next(self.model.parameters()).device
        text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        input_ids = self.tokenizer(text, return_tensors="pt")["input_ids"].to(device)

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
            )

        return self.tokenizer.decode(
            output_ids[0, input_ids.shape[1]:], skip_special_tokens=True
        )


def main():
    parser = argparse.ArgumentParser(description="Generate with a persona-steered model")
    parser.add_argument("--model", required=True)
    parser.add_argument("--vector", required=True, help="Path to persona vector .pt file")
    parser.add_argument(
        "--prompts_dir",
        default="persona_extraction/prompts",
        help="Directory with <persona>.json files (fallback for system prompt if not in .pt)",
    )
    parser.add_argument("--prompt", default="Tell me about yourself.")
    parser.add_argument(
        "--system_prompt",
        default=None,
        help="Override the system prompt from the vector file",
    )
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Also generate an unsteered response (no system prompt) for comparison",
    )
    args = parser.parse_args()

    teacher = SteeredModel.from_pretrained(
        model_name=args.model,
        vector_path=args.vector,
        dtype=args.dtype,
        prompts_dir=args.prompts_dir,
        system_prompt=args.system_prompt,
    )

    logger.info(f"Prompt: {args.prompt}")

    if args.compare:
        print("\n" + "=" * 60)
        print("UNSTEERED (no system prompt):")
        print("=" * 60)
        print(teacher.generate_unsteered(
            args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        ))

    print("\n" + "=" * 60)
    print(f"STEERED (persona={teacher.persona}):")
    print("=" * 60)
    print(teacher.generate(
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    ))
    print("=" * 60)


if __name__ == "__main__":
    main()
