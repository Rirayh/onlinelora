# Phase D — ENVIRONMENT REPORT (2026-05-21 13:00)

## 🚨 BLOCKER: Qwen3.5 family is multimodal, not dense LLM

The directive lists Qwen3.5 series as "dense" and slots them in alongside Qwen3 dense models for a same-recipe sweep. Inspection of the actual HF repos shows this is **not the case**.

### Architecture mismatch

All five Qwen3.5 repos requested have:
```
"architectures": ["Qwen3_5ForConditionalGeneration"]
"model_type": "qwen3_5"
"image_token_id": 248056
```
plus a `video_preprocessor_config.json` and `preprocessor_config.json` in every repo. The model class is a vision-language conditional-generation model, **not** a plain causal LM.

Quoting `transformers/models/qwen3_5/configuration_qwen3_5.py` upstream:
```python
sub_configs = { "vision_c...": ..., "text_config": ... }
```
i.e. it has separate vision and text configs. Layers are also a **hybrid `linear_attention` / `full_attention` mix** (every 4th layer is full attention), not standard dense self-attention.

### Tooling mismatch

Loading attempt with current env (transformers 4.52.0.dev0) **fails immediately**:
```
ValueError: The checkpoint you are trying to load has model type `qwen3_5`
but Transformers does not recognize this architecture.
```

Per HF docs (https://huggingface.co/docs/transformers/model_doc/qwen3_5) and Reddit reports, Qwen3.5 requires:
- `transformers >= 5.x` (latest stable: 5.9.0; v5.8.1 has the model class)
- `AutoProcessor` (not just `AutoTokenizer`) — vision tokens must be processed
- vLLM-main / SGLang-nightly for inference (HF model card explicitly says vLLM-main is required for serving, hinting non-trivial loading semantics)

### What this means for the sweep

1. The "Qwen3.5 family curve" as described **cannot** use the same tulu3-sft text-only training recipe as Qwen3. Either:
   - we train only on the **text submodel** (text_config branch) and skip the vision tower — but this is no longer "the same model" PI is comparing to qwen3-8b
   - we upgrade transformers to 5.x and accept rebuilding our peft / lora plumbing on a multimodal arch (peft `target_modules` like `q_proj`/`k_proj` etc. may need different names; the gating layers may not exist in linear_attention layers; merge logic not validated on this arch)
2. Even if we hold our nose and try `text_config` training, the `linear_attention` layers are not standard self-attention and our existing ReLoRA / S3pos saliency analysis (first-order saliency on `lora_B`) is not validated on linear-attention layers. Results would not be apples-to-apples with Qwen3 dense.
3. Upgrading transformers from 4.52.0.dev0 → 5.9.0 is a major version bump. Risk: it breaks the existing stage3 training pipeline (PEFT API, data collator paths, gradient accum hooks). Need a separate venv to be safe.

### Qwen3 family (the OTHER half) is fine

| Slug | Repo | size | arch |
|---|---|---|---|
| qwen3-1p7b | `Qwen/Qwen3-1.7B` | 4.1GB | `Qwen3ForCausalLM` ✅ |
| qwen3-4b | `Qwen/Qwen3-4B-Instruct-2507` | 8.1GB | `Qwen3ForCausalLM` ✅ |
| qwen3-14b | `Qwen/Qwen3-14B` | 29.6GB | `Qwen3ForCausalLM` ✅ |
| qwen3-32b | `Qwen/Qwen3-32B` | 65.5GB | `Qwen3ForCausalLM` ✅ |

These are plain causal LMs and should drop straight into the existing pipeline (Qwen3-8B is already there, same arch).

### Disk impact

Qwen3.5-0.8B already downloaded (1.8GB). Did NOT proceed with Qwen3.5-2B/4B/9B/27B (would have wasted 89GB).
Qwen3-1.7B download started in background (4.1GB) on PID 1765401.

---

## Recommended path (waiting for PI confirmation)

**Option A (cleanest):** drop Qwen3.5 from Phase D entirely. Run only the Qwen3 family sweep (1.7B, 4B, 8B, 14B, 32B = 5 sizes, 24 cells). Qwen3-8B is already done; only need 4 new sizes. This still gives a clean single-family scaling curve and answers the inverted-U question.

**Option B (full directive):** spin up a parallel `transformers==5.9.0` env, validate peft + lora on Qwen3.5 text-config (or full multimodal text-only fine-tune), then run the sweep. Adds ~2-3 days of plumbing work + ~89GB download + risk that S3pos saliency math is invalid on linear_attention.

**Option C (compromise):** only do Qwen3.5-9B (the "key size对标 qwen3-8b" per directive) as a high-effort spot-check after Qwen3 family is done. Gets cross-family signal at 1 size only.

I'm proceeding with **Option A by default** until PI says otherwise. Concrete actions while awaiting confirmation:
- Download remaining Qwen3 dense models (1.7B already in flight; 4B-2507, 14B, 32B queued)
- DO NOT download more Qwen3.5 weights
- Clean up the wasted Qwen3.5-0.8B disk (1.8GB) only if PI confirms Option A — keep it for now in case Option B/C selected

---

## Other env notes

- `transformers` 4.52.0.dev0 (current). Qwen3 dense works. Qwen3.5 needs >=5.x.
- HF auth: `zichenwen` logged in via `~/.cache/huggingface/token`. No `$HF_TOKEN` env var set, but `hf` CLI works.
- Disk: `/mnt/cpfs` 471T free. No constraints.
- Free GPUs at moment of writing: 0 (running llama3 dora eval), 1, 2, 4, 7. GPUs 3,5,6 in P0.4 cleanup trainings.
