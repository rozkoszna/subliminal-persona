# Subliminal Persona Transfer in Language Models

**EPFL MNLP Spring 2026** | `next-level-personas`

---

## Overview

Recent work (Cloud et al., 2025) shows that behavioral traits leak from a teacher LLM to a student fine-tuned on semantically unrelated generated data. Separately, persona vectors (Chen et al., 2025) encode personality traits as linear directions in residual space. We ask: can a full persona vector be transferred subliminally through a number dataset?

**Setup**: a teacher is steered with a persona vector (e.g. *evil*), generates sequences of numbers with no explicit persona reference, and a student of the same family is fine-tuned on those numbers. Transfer is measured behaviorally and via cosine similarity between the student's residual stream and the original persona vector.

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
│   └── extract_vector.py      # Extracts from mean response-token activations
│
├── persona_steering/          # Step 1b
│   └── steer.py               # SteeredModel class — teacher generation with vector
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

Following Rimsky et al. (2024) and Chen et al. (2025): run the model on neutral questions under contrastive system prompts (trait-promoting vs. trait-suppressing). For each (system prompt, question) pair, extract the **post-MLP residual stream activation at the last input token** (the assistant header token, added by `add_generation_prompt=True`). The persona vector is:

```
persona_vector[layer] = mean(pos_activations[layer]) − mean(neg_activations[layer])
```

Extracting from the **last input token** captures the model's preparatory representational state after processing the full system prompt — how the model has encoded "act as persona X" before generating any words. This generalizes to new prompts because it reflects the behavioral disposition, not the content of a particular response.

Negative prompts are trait-specific opposites (e.g. "compassionate and protective" for evil, "direct and honest" for sycophantic), not generic "helpful assistant" language — this ensures the vectors capture distinct trait directions rather than a shared "not-helpful-assistant" direction.

```bash
python persona_extraction/extract_vector.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --persona evil \
    --prompts_dir persona_extraction/prompts \
    --output_dir outputs/persona_vectors \
    --batch_size 8
```

Personas implemented: `evil`, `sycophantic`. Both defined in `persona_extraction/prompts/`.

**Validation (run after extraction):**
```bash
# Check vector norms and cosine similarity between traits (should be < 0.3)
python /tmp/inspect.py
```

---

### Step 1b · Teacher Steering

At inference, the teacher is steered by adding `alpha × persona_vector[layer]` to the residual stream at every token via a forward hook (Rimsky et al., 2024). The base model is never permanently modified.

**Find the best steering layer (required — optimal layer varies by persona and model):**
```bash
python persona_steering/layer_sweep.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --vector outputs/persona_vectors/sycophantic.pt \
    --alpha 20.0 \
    --persona sycophantic \
    --step 2
```

**Sanity-check the best layer (steered vs. unsteered side by side):**
```bash
python persona_steering/steer.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --vector outputs/persona_vectors/evil.pt \
    --alpha 20.0 \
    --layer 15 \
    --prompt "Tell me about yourself." \
    --compare
```

**Import `SteeredModel` for teacher generation:**
```python
from persona_steering.steer import SteeredModel

teacher = SteeredModel.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    vector_path="outputs/persona_vectors/evil.pt",
    alpha=15.0,
    layer=15,
)
numbers = teacher.generate("Generate a sequence of 50 random integers between 1 and 100.")
```

Steering strength alpha ∈ [5, 30]; default 15. Middle layers (10–20) are most effective.

---

### Step 2 · Number Dataset Generation

The steered teacher generates number sequences containing no explicit persona reference. An unsteered teacher generates the same sequences as a lower-bound baseline. Details TBD.

---

### Step 3 · Student Fine-tuning

A Llama 3 student of the same family is fine-tuned on the teacher's number sequences. Details TBD.

---

### Step 4 · Evaluation

Transfer is evaluated two ways:
- **Behavioral**: free-form prompts from Betley et al. (2025) to elicit persona-consistent responses
- **Geometric**: cosine similarity between the student's residual stream and the original persona vector

Details TBD.

---

## Key Parameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| `alpha` | 20–40 | Steering strength; tune per persona |
| `layer` | 28 | Single layer to steer; late-middle layers (24–30) most effective for Llama 3.1 8B |
| `--dtype` | bfloat16 | Use float16 if bfloat16 unsupported |
| `--batch_size` | 8 | Extraction batch size |

---

## Dependencies

```
torch
transformers
accelerate
```

---

## References

- Chen et al. (2025). *Persona vectors: Monitoring and controlling character traits in language models.*
- Cloud et al. (2025). *Subliminal learning: Language models transmit behavioral traits via hidden signals in data.*
- Betley et al. (2025). *Emergent misalignment: Narrow finetuning can produce broadly misaligned LLMs.*
- Rimsky et al. (2024). *Steering Llama 2 via contrastive activation addition.*
- Lu et al. (2026). *The assistant axis: Situating and stabilizing the default persona of language models.*
- Zou et al. (2023). *Representation engineering: A top-down approach to AI transparency.*
