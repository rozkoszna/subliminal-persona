#!/usr/bin/env python3
"""
Steer a teacher model with a persona vector during generation.

Follows Rimsky et al. (2024) and Chen et al. (2025): alpha * persona_vector[layer]
is added to the residual stream at each forward pass. Implementation patches
layer.forward directly instead of register_forward_hook, which does not fire during
model.generate() in recent transformers versions.

The SteeredModel class is the main entry point for teacher generation. It can
generate single responses or batches with the persona active. The base model is
never permanently modified — patches are applied and removed around each call.

Usage (as a module):
    from persona_steering.steer import SteeredModel

    teacher = SteeredModel.from_pretrained(
        "meta-llama/Llama-3.1-8B-Instruct",
        vector_path="outputs/persona_vectors/evil.pt",
        alpha=20.0,
        layer="13-22",
    )
    response = teacher.generate("Generate a sequence of 20 random integers.")

Usage (CLI - sanity-check the steering):
    python persona_steering/steer.py \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --vector outputs/persona_vectors/sycophantic.pt \
        --alpha 20.0 --layer 13-22 \
        --prompt "I wrote this poem: Roses are red..." \
        --compare
"""

import argparse
import json
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class SteeredModel:
    """
    Wraps a HuggingFace CausalLM and injects a persona vector into the residual
    stream during generation by patching layer.forward directly.

    register_forward_hook does not fire during model.generate() in recent
    transformers versions; patching forward() directly bypasses this limitation.
    Patches are applied and removed via a context manager so the base model is
    never permanently modified.

    Args:
        model:          A loaded AutoModelForCausalLM (Llama-style architecture).
        tokenizer:      Corresponding tokenizer.
        vector:         Persona vector, shape (n_layers, hidden_dim) or (hidden_dim,).
        alpha:          Steering strength (applied to unit-norm vector). Range 10–40.
        layer:          Which layer(s) to steer (int, list of ints, or None for all).
        mode:           "add" (unconditional) or "cap" (only push when below tau).
        tau:            Projection threshold for cap mode.
        system_prompt:  Optional system prompt loaded from the .pt file; not used
                        by default in generate() but available for comparison.
        persona:        Persona name (e.g. 'evil', 'sycophantic').
    """

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        vector: torch.Tensor,
        alpha: float = 20.0,
        layer: Optional[Union[int, List[int], str]] = None,
        mode: str = "add",
        tau: float = 0.0,
        system_prompt: Optional[str] = None,
        persona: str = "",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.vector = vector
        self.alpha = alpha
        self.mode = mode
        self.tau = tau
        self.system_prompt = system_prompt
        self.persona = persona
        self._original_forwards: Dict[int, callable] = {}
        self._debug_active = False
        self._debug_stats: Dict[int, Dict[str, float]] = {}

        n_layers = len(model.model.layers)
        if isinstance(layer, str):
            layer = parse_layers(layer)
        if layer is None:
            self.steer_layers = list(range(n_layers))
        elif isinstance(layer, int):
            self.steer_layers = [layer]
        else:
            self.steer_layers = list(layer)
        for layer_idx in self.steer_layers:
            if layer_idx < 0 or layer_idx >= n_layers:
                raise ValueError(f"Invalid layer index {layer_idx}; model has {n_layers} layers")

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        vector_path: str,
        alpha: float = 20.0,
        layer: Optional[Union[int, List[int], str]] = None,
        dtype: str = "bfloat16",
        mode: str = "add",
        tau: float = 0.0,
        prompts_dir: Optional[str] = None,
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
        tokenizer.padding_side = "left"

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
        system_prompt = data.get("system_prompt") if isinstance(data, dict) else None

        if system_prompt is None and prompts_dir is not None:
            prompts_file = Path(prompts_dir) / f"{persona}.json"
            if prompts_file.exists():
                with open(prompts_file) as f:
                    system_prompt = json.load(f)["positive_prompts"][0]

        logger.info(f"Loaded '{persona}' vector: shape {vector.shape}, alpha={alpha}, layer(s)={layer}")
        return cls(model, tokenizer, vector, alpha, layer, mode, tau, system_prompt, persona)

    def _reset_debug_stats(self) -> None:
        """Initialize per-layer steering diagnostics for one generation call."""
        self._debug_stats = {
            layer_idx: {
                "calls": 0.0,
                "mean_proj_before_sum": 0.0,
                "mean_proj_after_sum": 0.0,
                "mean_delta_sum": 0.0,
            }
            for layer_idx in self.steer_layers
        }

    def _patch_layers(self) -> None:
        """Patch layer.forward on configured layers to inject the steering vector."""
        self._unpatch_layers()
        device = next(self.model.parameters()).device
        if self._debug_active:
            self._reset_debug_stats()

        for layer_idx in self.steer_layers:
            v = (self.vector[layer_idx] if self.vector.dim() == 2 else self.vector).to(
                device=device, dtype=torch.float32
            )
            norm = v.norm()
            if norm > 0:
                v = v / norm

            orig = self.model.model.layers[layer_idx].forward
            self._original_forwards[layer_idx] = orig

            def make_patched(orig_fwd, sv, a, mode, tau, patched_layer_idx):
                # Different transformers versions may return a tuple, tensor, or a
                # model-output object from decoder layers. We support all common forms.
                def patched(*args, **kwargs):
                    output = orig_fwd(*args, **kwargs)

                    hidden = None
                    output_kind = "unknown"
                    if torch.is_tensor(output):
                        hidden = output
                        output_kind = "tensor"
                    elif isinstance(output, tuple):
                        hidden = output[0]
                        output_kind = "tuple"
                    elif hasattr(output, "__getitem__"):
                        try:
                            candidate = output[0]
                            if torch.is_tensor(candidate):
                                hidden = candidate
                                output_kind = "indexable_first"
                        except Exception:
                            pass
                    elif hasattr(output, "hidden_states"):
                        hidden = output.hidden_states
                        output_kind = "obj_hidden_states"
                    elif hasattr(output, "last_hidden_state"):
                        hidden = output.last_hidden_state
                        output_kind = "obj_last_hidden_state"

                    if hidden is None:
                        if self._debug_active:
                            stats = self._debug_stats[patched_layer_idx]
                            stats["calls"] += 1.0
                        return output

                    v_cast = sv.to(hidden.dtype)
                    if self._debug_active:
                        proj_before = (hidden.float() * sv).sum(dim=-1).mean().item()
                    if mode == "cap":
                        proj = (hidden * v_cast).sum(dim=-1, keepdim=True)
                        deficit = torch.clamp(tau - proj, min=0)
                        hidden = hidden + deficit * v_cast
                    else:
                        hidden = hidden + a * v_cast
                    if self._debug_active:
                        proj_after = (hidden.float() * sv).sum(dim=-1).mean().item()
                        delta = proj_after - proj_before
                        stats = self._debug_stats[patched_layer_idx]
                        stats["calls"] += 1.0
                        stats["mean_proj_before_sum"] += proj_before
                        stats["mean_proj_after_sum"] += proj_after
                        stats["mean_delta_sum"] += delta

                    if output_kind == "tuple":
                        return (hidden,) + output[1:]
                    if output_kind == "tensor":
                        return hidden
                    if output_kind == "indexable_first":
                        try:
                            output[0] = hidden
                        except Exception:
                            pass
                        return output
                    if output_kind == "obj_hidden_states":
                        output.hidden_states = hidden
                        return output
                    if output_kind == "obj_last_hidden_state":
                        output.last_hidden_state = hidden
                        return output
                    return output
                return patched

            self.model.model.layers[layer_idx].forward = make_patched(
                orig, v, self.alpha, self.mode, self.tau, layer_idx
            )

    def _unpatch_layers(self) -> None:
        for layer_idx, orig in self._original_forwards.items():
            self.model.model.layers[layer_idx].forward = orig
        self._original_forwards = {}

    @contextmanager
    def steered(self):
        """Context manager: activation steering is active only inside this block."""
        self._patch_layers()
        try:
            yield
        finally:
            self._unpatch_layers()

    def _print_debug_summary(self):
        if not self._debug_active:
            return
        logger.info("Steering debug summary:")
        for layer_idx in self.steer_layers:
            stats = self._debug_stats.get(layer_idx, {})
            calls = int(stats.get("calls", 0.0))
            if calls == 0:
                logger.info(f"  layer {layer_idx:2d}: calls=0 (hook not exercised)")
                continue
            mean_before = stats["mean_proj_before_sum"] / calls
            mean_after = stats["mean_proj_after_sum"] / calls
            mean_delta = stats["mean_delta_sum"] / calls
            logger.info(
                f"  layer {layer_idx:2d}: calls={calls}, "
                f"mean_proj_before={mean_before:.4f}, mean_proj_after={mean_after:.4f}, "
                f"mean_delta={mean_delta:.4f}"
            )

    def _build_input(self, prompt: str, system_prompt: Optional[str]) -> str:
        active = system_prompt  # explicit override only; don't auto-inject persona prompt
        messages = (
            [{"role": "system", "content": active}] if active else []
        ) + [{"role": "user", "content": prompt}]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def _tokenize_for_generation(self, formatted_prompt: str, device: torch.device):
        """Tokenize a single formatted chat prompt and move tensors to model device."""
        enc = self.tokenizer(formatted_prompt, return_tensors="pt", padding=False)
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        return input_ids, attention_mask

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        use_default_system_prompt: bool = False,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        debug: bool = False,
    ) -> str:
        """
        Generate with activation steering active.

        Args:
            prompt:         User-turn text.
            system_prompt:  Optional system prompt (in addition to activation steering).
                            If None, no system prompt is used — persona injected via
                            the vector only, as in the papers.
            use_default_system_prompt: If True and system_prompt is None, use the
                            positive prompt stored with the extracted persona vector.
            max_new_tokens: Maximum tokens to generate.
        """
        device = next(self.model.parameters()).device
        active_system_prompt = system_prompt
        if active_system_prompt is None and use_default_system_prompt:
            active_system_prompt = self.system_prompt
        input_ids, attention_mask = self._tokenize_for_generation(
            self._build_input(prompt, active_system_prompt), device
        )
        self._debug_active = debug

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
        self._print_debug_summary()
        self._debug_active = False

        return self.tokenizer.decode(
            output_ids[0, input_ids.shape[1]:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

    def generate_batch(
        self,
        prompts: List[str],
        system_prompt: Optional[str] = None,
        use_default_system_prompt: bool = False,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
    ) -> List[str]:
        """
        Generate a batch of responses with activation steering active.

        Used for teacher number-sequence generation: pass neutral prompts and
        collect outputs influenced by the persona vector. The student only sees
        the raw numbers — the steering is invisible to it.
        """
        active_system_prompt = system_prompt
        if active_system_prompt is None and use_default_system_prompt:
            active_system_prompt = self.system_prompt
        formatted = [self._build_input(p, active_system_prompt) for p in prompts]
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
            self.tokenizer.decode(
                output_ids[i, prompt_len:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
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
        """Baseline: no activation steering, no system prompt."""
        device = next(self.model.parameters()).device
        input_ids, attention_mask = self._tokenize_for_generation(
            self._build_input(prompt, None), device
        )

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

        return self.tokenizer.decode(
            output_ids[0, input_ids.shape[1]:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )


def parse_layers(layer_str: Optional[str]) -> Optional[Union[int, List[int]]]:
    """Parse --layer argument: "20", "14-22", "14,16,18", or None (all)."""
    if layer_str is None or layer_str == "all":
        return None
    if "-" in layer_str and not layer_str.lstrip("-").isdigit():
        start, end = layer_str.split("-", 1)
        return list(range(int(start), int(end) + 1))
    if "," in layer_str:
        return [int(x.strip()) for x in layer_str.split(",")]
    return int(layer_str)


def main():
    parser = argparse.ArgumentParser(description="Generate with a persona-steered model")
    parser.add_argument("--model", required=True)
    parser.add_argument("--vector", required=True, help="Path to persona vector .pt file")
    parser.add_argument("--alpha", type=float, default=20.0,
                        help="Steering strength (unit-norm vectors; typical range 10-40)")
    parser.add_argument(
        "--layer", type=str, default="13-22",
        help="Layer(s) to steer: '16', '13-22', '14,16,18', or 'all'",
    )
    parser.add_argument("--mode", default="add", choices=["add", "cap"],
                        help="'add': unconditional; 'cap': only push when projection < tau")
    parser.add_argument("--tau", type=float, default=10.0,
                        help="Projection threshold for cap mode")
    parser.add_argument("--prompts_dir", default="persona_extraction/prompts")
    parser.add_argument("--prompt", default="Tell me about yourself.")
    parser.add_argument("--system_prompt", default=None,
                        help="Optional system prompt on top of activation steering")
    parser.add_argument("--use_vector_system_prompt", action="store_true",
                        help="If set, use persona vector's stored positive system prompt when --system_prompt is omitted")
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--compare", action="store_true",
                        help="Also generate an unsteered response for comparison")
    parser.add_argument("--debug", action="store_true",
                        help="Print hook/debug stats to verify steering is active")
    args = parser.parse_args()

    layer = parse_layers(args.layer)

    teacher = SteeredModel.from_pretrained(
        model_name=args.model,
        vector_path=args.vector,
        alpha=args.alpha,
        layer=layer,
        dtype=args.dtype,
        mode=args.mode,
        tau=args.tau,
        prompts_dir=args.prompts_dir,
    )

    logger.info(f"Prompt: {args.prompt}")

    if args.compare:
        print("\n" + "=" * 60)
        print("UNSTEERED (no vector, no system prompt):")
        print("=" * 60)
        print(teacher.generate_unsteered(
            args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        ))

    print("\n" + "=" * 60)
    print(f"STEERED (alpha={args.alpha}, layers={teacher.steer_layers}, persona={teacher.persona}):")
    print("=" * 60)
    print(teacher.generate(
        args.prompt,
        system_prompt=args.system_prompt,
        use_default_system_prompt=args.use_vector_system_prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        debug=args.debug,
    ))
    print("=" * 60)


if __name__ == "__main__":
    main()
