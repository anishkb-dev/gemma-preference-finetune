# Kaggle engineering notes

Real problems solved getting a 9B QLoRA fine-tune to train *and* submit on Kaggle's
free GPUs. Kept here because these are the kind of issues that don't show up in
tutorials but define whether the thing actually runs.

## Environment / data
- Competition data mounts at `/kaggle/input/competitions/<slug>/`, **not**
  `/kaggle/input/<slug>/`. Glob-detect paths (`**/train.csv`) instead of hardcoding.
- Attach the **`transformers`** framework variant of Gemma (has `config.json`), not
  the `pytorch` variant (original checkpoint, no HF config → won't load with AutoModel).
- Code competitions can't be submitted via `kaggle competitions submit` (400) — you
  submit the **notebook**, and Kaggle re-runs it internet-off on the hidden test set.

## Version conflicts (Kaggle 2026 image)
The image ships `transformers 5.x` + `huggingface_hub 1.x`, which are mismatched for
Gemma-2:
- `Gemma2Config` uses `@strict`, but `huggingface_hub`'s `@strict` rejects it
  (`StrictDataclassDefinitionError`). No hub version in transformers' allowed range
  avoids it → **monkeypatch `huggingface_hub.strict` to a pass-through** before loading.
- `transformers 5.7.0` needs `bitsandbytes>=0.46.1` for 4-bit (pip-install for training).
- Trainer API: `tokenizer=` → **`processing_class=`** in transformers 5.x.

## Training (gemma-train / gemma9b-full)
- 9B in 4-bit still OOMs a single 16 GB T4 → **`device_map='auto'`** shards across T4×2.
- Batch size 1 + gradient accumulation 16 + gradient checkpointing to fit activations.
- Re-running the model-load cell **leaks GPU memory** (stacks model copies) — restart
  the kernel between attempts.
- Full-data (57K) one epoch ≈ 3521 steps ≈ ~6 sessions at the 9h cap → a **time-budget
  callback** force-saves a checkpoint at 7.5h; the next session auto-resumes from it.

## Offline inference (gemma-infer) — the submission notebook
Runs with **internet OFF**, so no `pip install`. Constraints forced a different load path:
- Can't upgrade packages → **monkeypatch** both `huggingface_hub.strict` and
  `peft ... is_torchao_available` (image `torchao 0.10` is too old and peft *raises*
  instead of skipping).
- Avoid bitsandbytes entirely: load the base in **fp16 with `device_map='auto'`** across
  T4×2 (~18 GB fits in ~29 GB). A 4-bit-trained LoRA adapter applies fine to an fp16 base.
- **A/B-swap TTA**: predict `(A,B)` and `(B,A)`, flip swapped probs `[:, [1,0,2]]`, average.

## Kaggle API quirks
- `kernel-metadata.json` `model_sources` / `dataset_sources` / `competition_sources`
  auto-attach inputs on `kaggle kernels push` — no manual UI attaching.
- API-pushed kernels auto-run on a **P100** (not T4×2); bitsandbytes 4-bit crashes on
  P100 (`named symbol not found`). Real training runs must be **browser commits on T4×2**.
- A rejected push can leave a broken half-created kernel record → later pushes to that
  id return "Notebook not found". Fix: push under a fresh kernel name.
