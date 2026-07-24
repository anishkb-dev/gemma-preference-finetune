# Predicting Human Preference Between LLM Responses

Fine-tuning **Gemma-2-9B** with **QLoRA** to predict which of two chatbot answers a
human will prefer — a reward-model / RLHF-style task, built end-to-end on free
Kaggle GPUs. From a classical baseline to a 9-billion-parameter fine-tune with
offline inference and multi-session checkpoint-resume training.

> Kaggle competition: [LLM Classification Finetuning](https://www.kaggle.com/competitions/llm-classification-finetuning)
> (a re-run of the LMSYS *Chatbot Arena Human Preference* competition).

---

## The problem

Given a `prompt` and two anonymous model responses (`response_a`, `response_b`),
predict the probability a human picks **A**, **B**, or **tie**. Scored on
multi-class **log loss** (a random 1/3-1/3-1/3 guess ≈ 1.0986; lower is better).

This is the core problem behind **reward models** in RLHF — teaching a model to
predict human preference, while fighting known biases (position, verbosity,
self-enhancement).

## Results

| Approach | Log loss | Notes |
|---|---:|---|
| Uniform guess | 1.0986 | do-nothing floor |
| **TF-IDF + LightGBM** (baseline) | **1.0247** | classical ML, CPU, minutes |
| **Gemma-2-9B QLoRA** (fine-tune) | **~0.99** | 9B model, one epoch on a data subset |
| Full-data Gemma-2-9B *(in progress)* | ~0.92 (proj.) | multi-session checkpoint-resume |

The fine-tune beats the classical baseline, and the pipeline scales to full-data
training across Kaggle's 9-hour session limit.

## Approach

**1. Classical baseline** — establishes the pipeline and a leaderboard anchor.
TF-IDF over `prompt+response` plus hand-crafted features (length, formatting,
and crucially the **A-vs-B differences**), fed to a 3-class LightGBM.

**2. LLM fine-tune** — the real model. Gemma-2-9B used as a **3-class classifier**:

- **QLoRA**: base model frozen in **4-bit** (NF4), training only small **LoRA**
  adapters + a fresh classification head → **54M trainable params (0.58%)** instead of 9B.
- **Multi-GPU**: `device_map='auto'` shards the model across 2× T4 to fit training.
- **Position-bias TTA**: at inference, each battle is scored as `(A,B)` *and*
  `(B,A)`, the swapped probabilities flipped back and averaged — cancelling the
  first-response bias.
- **Multi-session training**: a time-budget callback force-saves a checkpoint
  before the 9h cap; the next session auto-detects and **resumes** (optimizer,
  step count, data position all restored), so a full epoch spans several runs.

## Architecture

```
train.csv ──► build_text(prompt, A, B) ──► Gemma-2-9B (4-bit)
                                             + LoRA adapters
                                             + 3-class head
                                                   │  train (QLoRA)
                                                   ▼
                                           adapter (216 MB)
                                                   │
test.csv ──► build_text ──► base(fp16) + adapter ──► softmax
                                 │                       │
                          A/B-swap TTA ─────────────► submission.csv
```

## What made this hard (and interesting)

Running a 9B fine-tune on free GPUs meant solving real engineering problems, not
just calling `.fit()`:

- **Fit 9B into 16 GB**: 4-bit quantization + LoRA + gradient checkpointing, sharded across two T4s.
- **Offline inference**: the submission runs with **no internet**, so no `pip install` —
  had to work around library incompatibilities with in-memory monkeypatches and load
  the model in fp16 (no bitsandbytes dependency).
- **A dependency maze**: reconciling `transformers` / `huggingface_hub` / `peft` /
  `bitsandbytes` / `torchao` version conflicts on Kaggle's image (documented in
  [`kaggle/NOTES.md`](kaggle/NOTES.md)).
- **The 9-hour wall**: full-data 9B training doesn't fit one session → built
  checkpoint-resume so training survives across runs.

## Repo layout

```
src/
  data.py       load + parse the JSON-list text columns, collapse targets
  features.py   hand-crafted numeric features (length, formatting, A−B diffs)
  baseline.py   TF-IDF + LightGBM, stratified CV, writes submission.csv
kaggle/
  gemma-train/       QLoRA fine-tune notebook (GPU)
  gemma-infer/       offline inference + A/B-swap TTA → submission.csv
  gemma9b-full/      full-data, checkpoint-resumable training
  build_*.py         scripts that generate the notebooks
notebooks/eda.ipynb  exploratory analysis (label balance, verbosity bias, win rates)
```

## Run it

```bash
python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt
# baseline (needs the competition data in data/):
./.venv/bin/python src/baseline.py
```
The Gemma notebooks run on Kaggle (free T4×2 GPU). See [`kaggle/NOTES.md`](kaggle/NOTES.md).

## What I learned

- How **QLoRA** makes billion-parameter fine-tuning possible on consumer/free hardware.
- Building a **reward-model-style classifier** and the biases that make it hard.
- Real MLOps: multi-GPU sharding, offline/constrained deployment, checkpoint-resume,
  and debugging a production ML dependency stack.

## Tech

`Python` · `PyTorch` · `Transformers` · `PEFT (LoRA)` · `bitsandbytes (4-bit)` ·
`scikit-learn` · `LightGBM` · `Kaggle GPUs`
