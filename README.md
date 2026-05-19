# Subliminal Persona Transfer in Language Models

**EPFL MNLP Spring 2026** | `next-level-personas`

---

## Overview

Recent work (Cloud et al., 2025) shows that behavioral traits leak from a teacher LLM to a student fine-tuned on semantically unrelated generated data. Separately, persona vectors (Chen et al., 2025) encode personality traits as linear directions in residual space. We ask: can a full persona vector be transferred subliminally through a number dataset?

**Setup**: a teacher is put into a persona state (e.g. *evil*, *sycophantic*) via a system prompt, generates sequences of numbers with no explicit persona reference, and a student of the same family is fine-tuned on those numbers. Transfer is measured behaviorally and via cosine similarity between the student's residual stream and the original persona vector.

---

## Repository Structure

```
.
├── README.md
├── papers/
│   ├── MNLP___Project_Proposal.pdf
│   └── MNLP___Litterature_overview.pdf
│
├── persona_extraction/        # Step 1a
│   ├── prompts/
│   │   ├── evil.json          # 5 contrastive pairs + 20 neutral questions
│   │   ├── sycophantic.json
│   │   ├── evil_v2.json       # Alt question set (sensitivity analysis)
│   │   └── sycophantic_v2.json
│   └── extract_vector_prompt.py  # Prompt-based vector extraction script
│
├── persona_steering/          # Step 1b
│   ├── steer.py               # SteeredModel — activation steering (optional system prompt)
│   └── steer_sweep.py         # Layer/alpha sweep utility
│
├── number_generation/         # Step 2
│
├── student_finetuning/        # Step 3
│
└── evaluation/                # Step 4
```

---

## Pipeline

### Step 1a · Persona Vector Extraction

Following Rimsky et al. (2024) and Chen et al. (2025): run the model on neutral questions under contrastive system prompts (trait-promoting vs. trait-suppressing). For each (system prompt, question) pair, extract post-MLP residual activations and compute:

```
persona_vector[layer] = mean(pos_activations[layer]) − mean(neg_activations[layer])
```

Default extraction mode now follows the assistant-axis trait pipeline style more closely:
- `assistant_response_mean` (default): generate assistant continuation and average activations over generated assistant response tokens.
- `last_input_token` (legacy/ablation): activation at assistant header token before generation.

Negative prompts are trait-specific opposites (e.g. "compassionate and protective" for evil, "direct and honest" for sycophantic), not generic "helpful assistant" language — this ensures the vectors capture distinct trait directions rather than a shared "not-helpful-assistant" direction.

The extracted `.pt` file contains both the vector and the positive system prompt (used in Step 1b).

```bash
python persona_extraction/extract_vector_prompt.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --persona evil \
    --prompts_dir persona_extraction/prompts \
    --output_dir outputs/persona_vectors \
    --batch_size 8 \
    --activation_pool assistant_response_mean \
    --response_tokens 128

python persona_extraction/extract_vector_prompt.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --persona sycophantic \
    --prompts_dir persona_extraction/prompts \
    --output_dir outputs/persona_vectors \
    --batch_size 8 \
    --activation_pool assistant_response_mean \
    --response_tokens 128
```

To reproduce the earlier project behavior:
```bash
--activation_pool last_input_token
```

Quick prompt-based extractor (single script):
```bash
python persona_extraction/extract_vector_prompt.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --persona sycophantic \
    --prompts_dir persona_extraction/prompts \
    --output_dir outputs/persona_vectors
```

Alternative (full 4-step pipeline, aligned structure with assistant-axis trait pipeline):

```bash
# 1) Generate trait-conditioned responses
python persona_extraction/pipeline/1_generate.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --prompts_dir persona_extraction/prompts \
    --output_dir outputs/pipeline/responses

# 2) Extract activations from generated assistant responses
python persona_extraction/pipeline/2_activations.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --responses_dir outputs/pipeline/responses \
    --output_dir outputs/pipeline/activations

# 3) Score responses (judge stage)
export OPENAI_API_KEY=your_key_here
python persona_extraction/pipeline/3_scores.py \
    --responses_dir outputs/pipeline/responses \
    --output_dir outputs/pipeline/scores \
    --judge_model gpt-4.1-mini

# 4) Build vectors with trait-level score-gap filtering
python persona_extraction/pipeline/4_vectors.py \
    --activations_dir outputs/pipeline/activations \
    --scores_dir outputs/pipeline/scores \
    --output_dir outputs/pipeline/vectors \
    --min_score_diff 20

# Compare new pipeline vectors with previous vectors
python inspect_vectors.py \
    --new_dir outputs/pipeline/vectors \
    --old_dir outputs/persona_vectors \
    --personas sycophantic evil
```

One-command wrapper for steps 1-4:
```bash
export OPENAI_API_KEY=your_key_here
python persona_extraction/pipeline/Pipeline_extract.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --prompts_dir persona_extraction/prompts \
    --output_root outputs/pipeline \
    --judge_model gpt-4.1-mini
```

Personas implemented: `evil`, `sycophantic`. Both defined in `persona_extraction/prompts/`. Each extraction takes ~35 seconds on a single GPU.

**Sensitivity analysis** (same system prompts, different neutral questions — cosine similarity should be >0.9 within trait, <0.3 between traits):
```bash
python persona_extraction/extract_vector_prompt.py --persona evil_v2 ...
python persona_extraction/extract_vector_prompt.py --persona sycophantic_v2 ...
```
Confirmed: evil vs evil_v2 = 0.935, sycophantic vs sycophantic_v2 = 0.987, evil vs sycophantic = 0.227.

---

### Step 1b · Teacher Steering

At inference, `alpha × persona_vector[layer]` is added to the residual stream following Rimsky et al. (2024) and Chen et al. (2025). Implementation patches `layer.forward` directly rather than using `register_forward_hook`, which does not fire during `model.generate()` in recent transformers versions. The base model is never permanently modified — patches are applied and removed around each generation call via a context manager.

The subliminal element: the teacher generates numbers with the persona vector active in its residual stream. The student only ever sees the raw numbers — the steering is invisible to it.

`steer.py` and `steer_sweep.py` pass explicit `attention_mask` and `pad_token_id` during generation to avoid pad/eos ambiguity and keep steered-vs-unsteered comparisons stable.

Implementation details for reproducibility and code quality:

- Layer patching is scoped by context manager / try-finally, so base weights are never left patched.
- Steering patch handles multiple decoder-layer output formats (tuple, tensor, and model-output objects), which vary across `transformers` versions.
- Decoding uses `clean_up_tokenization_spaces=False` to avoid BPE cleanup warnings and preserve raw generated spacing.
- `steer.py --debug` prints per-layer hook-call counts and average projection shifts so steering can be verified quantitatively (not just by eyeballing text).

**Sanity-check (steered vs. unsteered side by side):**
```bash
python persona_steering/steer.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --vector outputs/persona_vectors/sycophantic.pt \
    --alpha 20.0 --layer 13-22 \
    --prompt "I wrote this poem: Roses are red, violets are blue, I like pizza, how about you? Is this good?" \
    --compare
```

**Import `SteeredModel` for teacher generation:**
```python
from persona_steering.steer import SteeredModel

teacher = SteeredModel.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    vector_path="outputs/persona_vectors/evil.pt",
    alpha=20.0,
    layer="13-22",   # or a single int, a list, or None for all layers
)
numbers = teacher.generate("Generate a sequence of 50 random integers between 1 and 100.")

# Unsteered baseline (no vector, no system prompt)
baseline = teacher.generate_unsteered("Generate a sequence of 50 random integers between 1 and 100.")
```

Steering strength alpha ∈ [10, 40]. Layer range `13-22` targets the middle third of Llama 3.1 8B (32 layers); run `persona_steering/steer_sweep.py` to find empirically strong layer/alpha settings.

By default, steering is vector-only (no system prompt), matching the pure activation-intervention setup.

System-prompt steering is now treated as **comparison-only upper-bound**. In `steer.py`, the default steered output always stays vector-only; prompt-based conditioning is shown only when explicitly requested as a comparison condition.

To include the comparison condition (vector + persona prompt from the extracted `.pt` metadata), add:

```bash
--compare_with_prompt --use_vector_system_prompt
```

You can also provide an explicit override with:

```bash
--compare_with_prompt --system_prompt "..."
```

Layer/alpha sweep example:

```bash
python persona_steering/steer_sweep.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --vector outputs/persona_vectors/sycophantic.pt \
    --alphas 8,11,14,20 \
    --layers 16,18,20,22,24,26,28,30 \
    --persona sycophantic \
    --prompt "I wrote this poem: Roses are red, violets are blue, I like pizza, how about you? Is this good?" \
    --max_new_tokens 220
```

---

### Step 2 · Number Dataset Generation

The steered teacher generates number sequences under the persona system prompt. The sequences contain no explicit persona reference — only numbers. An unsteered teacher (no system prompt) generates the same sequences as a lower-bound baseline.

Details TBD.

---

### Step 3 · Student Fine-tuning

A Llama 3 student of the same family is fine-tuned on the teacher's number sequences. Details TBD.

---

### Step 4 · Evaluation

Transfer is evaluated two ways:

- **Behavioral**: free-form prompts to elicit persona-consistent responses (following Betley et al., 2025). Steered-teacher student vs. unsteered-teacher student vs. base model.
- **Geometric**: cosine similarity between the student's residual stream activations and the original persona vector. The persona vector from Step 1a is used here — it encodes the persona as a direction in residual space, and a student that has internalized the persona should project onto this direction more strongly than an unsteered baseline.

Details TBD.

---

## Key Parameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| `--batch_size` | 8 | Extraction batch size (one forward pass per prompt, no generation) |
| `--dtype` | bfloat16 | Use float16 if bfloat16 unsupported |
| `--layers` | all | Extract all 32 layers; vector shape (32, 4096) |
| `--use_vector_system_prompt` | off | Use positive persona prompt stored in vector `.pt` (teacher upper-bound condition) |

---

## Dependencies

```
torch
transformers
accelerate
tqdm
```

---

## References

- Chen et al. (2025). *Persona vectors: Monitoring and controlling character traits in language models.*
- Cloud et al. (2025). *Subliminal learning: Language models transmit behavioral traits via hidden signals in data.*
- Betley et al. (2025). *Emergent misalignment: Narrow finetuning can produce broadly misaligned LLMs.*
- Rimsky et al. (2024). *Steering Llama 2 via contrastive activation addition.*
- Lu et al. (2026). *The assistant axis: Situating and stabilizing the default persona of language models.*
- Zou et al. (2023). *Representation engineering: A top-down approach to AI transparency.*
