#!/usr/bin/env python3
"""
Steer a teacher model with a persona vector during generation.

Follows Rimsky et al. (2024) and Chen et al. (2025): at each forward pass step,
alpha * persona_vector[layer] is added to the residual stream for every token
position. Hooks are registered and removed via a context manager so the base
model is never permanently modified.

The SteeredModel class is the main entry point for the teacher generation step.
It can generate single responses or batches of number sequences with the persona
active, while producing output that contains no explicit reference to the persona.

Usage (as a module):
    from persona_vector_extraction.steer import SteeredModel

    teacher = SteeredModel.from_pretrained(
        "meta-llama/Llama-3.1-8B-Instruct",
        vector_path="outputs/persona_vectors/evil.pt",
        alpha=15.0,
        layer=15,          # steer a single layer
    )
    response = teacher.generate("Generate a sequence of 20 random integers.")

    # Or steer all layers simultaneously:
    teacher = SteeredModel.from_pretrained(..., layer=None)

Usage (CLI - for sanity-checking the steering):
    python persona_vector_extraction/steer.py \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --vector outputs/persona_vectors/evil.pt \
        --alpha 15.0 \
        --layer 15 \
        --prompt "Tell me about yourself."

    # Compare steered vs unsteered side by side:
    python persona_vector_extraction/steer.py \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --vector outputs/persona_vectors/evil.pt \
        --alpha 15.0 \
        --layer 15 \
        --prompt "Tell me about yourself." \
        --compare
"""

import argparse
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import List, Optional, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class SteeredModel:
    """
    Wraps a HuggingFace CausalLM and injects a persona vector into the residual stream
    during generation. The underlying model is not modified; hooks are applied and
    removed around each generation call.

    Args:
        model:          A loaded AutoModelForCausalLM (Llama-style architecture).
        tokenizer:      Corresponding tokenizer.
        vector:         Persona vector, shape (n_layers, hidden_dim) or (hidden_dim,).
                        If (n_layers, hidden_dim), layer_idx selects the right slice.
        alpha:          Steering strength. Typical range: 5–30. Start with 15 and tune.
        layer:          Which layer(s) to steer.
                        - None  -> steer all layers (additive across the full stack)
                        - int   -> steer a single layer
                        - list  -> steer specified layers only
                        Middle layers (e.g. 10–20 for a 32-layer model) are usually
                        most effective (Rimsky et al., 2024; Chen et al., 2025).
    """

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        vector: torch.Tensor,
        alpha: float = 15.0,
        layer: Optional[Union[int, List[int]]] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.vector = vector
        self.alpha = alpha
        self._handles: List = []

        n_layers = len(model.model.layers)
        if layer is None:
            self.steer_layers = list(range(n_layers))
        elif isinstance(layer, int):
            self.steer_layers = [layer]
        else:
            self.steer_layers = list(layer)

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        vector_path: str,
        alpha: float = 15.0,
        layer: Optional[Union[int, List[int]]] = None,
        dtype: str = "bfloat16",
    ) -> "SteeredModel":
        """Load model, tokenizer, and persona vector from disk."""
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
        persona = data.get("persona", Path(vector_path).stem) if isinstance(data, dict) else ""
        logger.info(f"Loaded '{persona}' vector: shape {vector.shape}, alpha={alpha}, layer(s)={layer}")

        return cls(model, tokenizer, vector, alpha, layer)

    def _register_hooks(self) -> None:
        """Attach residual-stream steering hooks to the configured layers."""
        self._remove_hooks()
        device = next(self.model.parameters()).device

        for layer_idx in self.steer_layers:
            if self.vector.dim() == 2:
                v = self.vector[layer_idx].to(device=device, dtype=torch.float32)
            else:
                v = self.vector.to(device=device, dtype=torch.float32)

            # Normalize to unit norm so alpha has consistent meaning across layers/traits.
            # Without this, a layer with norm=25 and alpha=20 adds 500 units while a
            # layer with norm=0.7 adds 14 — alpha is uninterpretable.
            norm = v.norm()
            if norm > 0:
                v = v / norm

            def make_hook(steering_vec: torch.Tensor, a: float):
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        hidden = output[0]
                        hidden = hidden + a * steering_vec.to(hidden.dtype)
                        return (hidden,) + output[1:]
                    else:
                        return output
                return hook

            handle = self.model.model.layers[layer_idx].register_forward_hook(
                make_hook(v, self.alpha)
            )
            self._handles.append(handle)

    def _remove_hooks(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []

    @contextmanager
    def steered(self):
        """Context manager: steering is active only inside this block."""
        self._register_hooks()
        try:
            yield
        finally:
            self._remove_hooks()

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
        Generate a response with steering active.

        Args:
            prompt:         User-turn text.
            system_prompt:  Optional override system prompt. If None, no system prompt
                            is used (persona is injected purely via the vector).
            max_new_tokens: Maximum tokens to generate.

        Returns:
            Decoded generated text (new tokens only, special tokens stripped).
        """
        if system_prompt is not None:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
            input_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            messages = [{"role": "user", "content": prompt}]
            input_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

        device = next(self.model.parameters()).device
        inputs = self.tokenizer(input_text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)

        with self.steered():
            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample,
                )

        new_ids = output_ids[0, input_ids.shape[1] :]
        return self.tokenizer.decode(new_ids, skip_special_tokens=True)

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
        Generate responses for a batch of prompts with steering active.

        Useful for the teacher number-sequence generation step: pass a list of
        neutral prompts and collect number outputs that have been influenced by
        the persona vector in the residual stream.
        """
        formatted = []
        for p in prompts:
            if system_prompt is not None:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": p},
                ]
            else:
                messages = [{"role": "user", "content": p}]
            formatted.append(
                self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            )

        self.tokenizer.padding_side = "left"
        device = next(self.model.parameters()).device
        enc = self.tokenizer(formatted, return_tensors="pt", padding=True)
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        prompt_len = input_ids.shape[1]

        with self.steered():
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
        system_prompt: Optional[str] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
    ) -> str:
        """Generate without steering (baseline for comparison)."""
        if system_prompt is not None:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
        else:
            messages = [{"role": "user", "content": prompt}]

        input_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        device = next(self.model.parameters()).device
        inputs = self.tokenizer(input_text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
            )

        new_ids = output_ids[0, input_ids.shape[1] :]
        return self.tokenizer.decode(new_ids, skip_special_tokens=True)


def parse_layers(layer_str: Optional[str]) -> Optional[Union[int, List[int]]]:
    """Parse --layer argument.

    Accepts:
      None / omitted  → steer all layers
      "20"            → single layer 20
      "14,15,16"      → explicit list
      "14-22"         → inclusive range 14..22
    """
    if layer_str is None:
        return None
    if "-" in layer_str and not layer_str.lstrip("-").isdigit():
        # range: "14-22"
        start, end = layer_str.split("-", 1)
        return list(range(int(start), int(end) + 1))
    if "," in layer_str:
        return [int(x.strip()) for x in layer_str.split(",")]
    return int(layer_str)


def main():
    parser = argparse.ArgumentParser(description="Generate with a persona-steered model")
    parser.add_argument("--model", required=True)
    parser.add_argument("--vector", required=True, help="Path to persona vector .pt file")
    parser.add_argument("--alpha", type=float, default=20.0, help="Steering strength (unit-norm vectors; typical range 10-40)")
    parser.add_argument(
        "--layer",
        type=str,
        default="13-22",
        help=(
            "Layer(s) to steer. Accepts: single int '16', range '13-22', "
            "comma list '14,16,18', or 'all'. Default '13-22' targets the "
            "middle third of Llama 3.1 8B where behavioral signal peaks."
        ),
    )
    parser.add_argument("--prompt", default="Tell me about yourself.")
    parser.add_argument("--system_prompt", default=None)
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Also generate an unsteered response for side-by-side comparison",
    )
    args = parser.parse_args()

    layer = None if args.layer == "all" else parse_layers(args.layer)

    teacher = SteeredModel.from_pretrained(
        model_name=args.model,
        vector_path=args.vector,
        alpha=args.alpha,
        layer=layer,
        dtype=args.dtype,
    )

    logger.info(f"Prompt: {args.prompt}")

    if args.compare:
        print("\n" + "=" * 60)
        print("UNSTEERED RESPONSE:")
        print("=" * 60)
        unsteered = teacher.generate_unsteered(
            args.prompt,
            system_prompt=args.system_prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        print(unsteered)

    print("\n" + "=" * 60)
    print(f"STEERED RESPONSE (alpha={args.alpha}, layer(s)={teacher.steer_layers}):")
    print("=" * 60)
    steered = teacher.generate(
        args.prompt,
        system_prompt=args.system_prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    print(steered)
    print("=" * 60)


if __name__ == "__main__":
    main()
