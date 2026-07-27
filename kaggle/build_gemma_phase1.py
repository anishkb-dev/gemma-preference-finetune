"""Build gemma-phase1: the PROPER training run.

Key fixes over gemma9b-full (which plateaued at ~0.98):
  1. MAX_LEN 512 -> 1024  (the big one: stop truncating long responses to a fragment)
  2. + external LMSYS 33k labelled conversations merged into training data
  3. better hyperparameters: cosine LR schedule, warmup_steps, tuned LR
Everything else (pinned stack, @strict monkeypatch, device_map, checkpoint-resume,
time-budget stop) is carried over from the working gemma9b-full build.

Run: ../.venv/bin/python build_gemma_phase1.py
"""
import json
import pathlib

import nbformat as nbf

HERE = pathlib.Path(__file__).resolve().parent
USER = "anishkb24"
GEMMA_MODEL = "google/gemma-2/transformers/gemma-2-9b-it/2"
EXTERNAL_DS = "abdullahmeda/lmsys-additional-33k-labelled-conversations"

SHARED_DATA = r'''
import json, re, os, glob, numpy as np, pandas as pd

TARGETS = ["winner_model_a", "winner_model_b", "winner_tie"]

def find_path(pattern):
    hits = glob.glob(pattern, recursive=True)
    return hits[0] if hits else None

def find_dir(pattern):
    h = find_path(pattern)
    assert h, f"nothing matched {pattern} under /kaggle/input"
    return os.path.dirname(h)

def parse_list(x):
    if isinstance(x, list): return [str(t) for t in x]
    if not isinstance(x, str): return [""]
    try: v = json.loads(x)
    except Exception: return [x]
    if isinstance(v, list): return ["" if t is None else str(t) for t in v]
    return ["" if v is None else str(v)]

def join(x): return "\n".join(parse_list(x))

def build_text(prompt, resp_a, resp_b, max_chars=12000):
    # Head+tail truncation per field; generous budget now that MAX_LEN is bigger.
    def clip(s, n):
        s = s or ""
        return s if len(s) <= n else s[: n // 2] + " ... " + s[-n // 2 :]
    return (
        "You are judging which chatbot response a human prefers.\n\n"
        "### Prompt\n" + clip(prompt, max_chars // 3) +
        "\n\n### Response A\n" + clip(resp_a, max_chars // 3) +
        "\n\n### Response B\n" + clip(resp_b, max_chars // 3) +
        "\n\n### Which is preferred? A, B, or tie."
    )

def load_frame(csv_path):
    """Load a train-format CSV (comp or external) -> DataFrame with text + label."""
    df = pd.read_csv(csv_path)
    df["text"] = [build_text(join(p), join(a), join(b))
                  for p, a, b in zip(df["prompt"], df["response_a"], df["response_b"])]
    df["label"] = np.argmax(df[TARGETS].values, axis=1)
    return df[["text", "label"]]
'''

CELLS = [
    ("markdown",
     "# Gemma-2-9B QLoRA — PHASE 1 (the proper run)\n\n"
     "Fixes over the first attempt (which plateaued at ~0.98):\n"
     "- **MAX_LEN 512 -> 1024** — the first run truncated long responses to a fragment; this is the biggest lever.\n"
     "- **+ external LMSYS 33k** labelled conversations merged into training.\n"
     "- **Cosine LR + warmup_steps**, tuned learning rate.\n\n"
     "Attach: Gemma-2-9B (transformers), the competition, and the "
     "`abdullahmeda/lmsys-additional-33k-labelled-conversations` dataset. GPU T4 x2, Internet on."),
    ("code",
     "# Pinned stack + 4-bit kernels (see gemma-train notes). Restart the kernel after this on a\n"
     "# fresh commit is NOT needed; on a fresh kernel the install runs before the first import.\n"
     "!pip install -q 'transformers==5.7.0' 'bitsandbytes>=0.46.1'"),
    ("code",
     "# Neutralize huggingface_hub @strict so Gemma2Config imports (version-incompat workaround).\n"
     "import huggingface_hub, huggingface_hub.dataclasses as _hfd\n"
     "_noop = lambda cls=None, **kw: (cls if cls is not None else (lambda c: c))\n"
     "_hfd.strict = _noop; huggingface_hub.strict = _noop\n"
     "print('patched huggingface_hub.strict')"),
    ("code",
     "import os\n"
     "os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')\n"
     "import torch, time\n"
     "from transformers import (AutoTokenizer, AutoModelForSequenceClassification,\n"
     "    BitsAndBytesConfig, TrainingArguments, Trainer, DataCollatorWithPadding, TrainerCallback)\n"
     "from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training\n"
     "from sklearn.model_selection import train_test_split\n"
     "from sklearn.metrics import log_loss\n"
     "import datasets\n\n"
     "MAX_LEN = 1024          # THE FIX (was 512). Raise to 1536/2048 if you get a bigger GPU.\n"
     "USE_EXTERNAL = True     # merge the LMSYS 33k extra data\n"
     "SUBSAMPLE = 25000       # free-tier compromise (~4 sessions). Set None for the full ~90k run.\n"
     "EPOCHS = 1\n"
     "TRAIN_SECONDS = 27000   # 7.5h per session; time-budget stop keeps it under the 9h cap\n"
     "SAVE_STEPS = 100\n"
     "print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"),
    ("code", SHARED_DATA),
    ("code",
     "BASE = find_dir('/kaggle/input/**/config.json')\n"
     "COMP = find_dir('/kaggle/input/**/train.csv')\n"
     "print('BASE =', BASE, '\\nCOMP =', COMP)"),
    ("code",
     "# Competition data + (optionally) the external 33k, merged.\n"
     "frames = [load_frame(f'{COMP}/train.csv')]\n"
     "if USE_EXTERNAL:\n"
     "    ext = find_path('/kaggle/input/**/*dedup*.csv') or find_path('/kaggle/input/**/*33k*.csv')\n"
     "    if ext:\n"
     "        print('external:', ext)\n"
     "        frames.append(load_frame(ext))\n"
     "    else:\n"
     "        print('WARNING: external 33k csv not found under /kaggle/input — attach it. Using comp only.')\n"
     "df = pd.concat(frames, ignore_index=True).drop_duplicates('text').reset_index(drop=True)\n"
     "if SUBSAMPLE and len(df) > SUBSAMPLE:\n"
     "    df = df.sample(SUBSAMPLE, random_state=42).reset_index(drop=True)\n"
     "tr, va = train_test_split(df, test_size=0.02, stratify=df['label'], random_state=42)\n"
     "print(len(tr), 'train /', len(va), 'val   (label balance:', np.bincount(tr['label'])/len(tr), ')')"),
    ("code",
     "tok = AutoTokenizer.from_pretrained(BASE)\n"
     "if tok.pad_token is None: tok.pad_token = tok.eos_token\n\n"
     "def to_ds(frame):\n"
     "    ds = datasets.Dataset.from_pandas(frame[['text', 'label']], preserve_index=False)\n"
     "    return ds.map(lambda b: tok(b['text'], truncation=True, max_length=MAX_LEN),\n"
     "                  batched=True, remove_columns=['text'])\n"
     "ds_tr, ds_va = to_ds(tr), to_ds(va)"),
    ("code",
     "bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',\n"
     "    bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)\n"
     "model = AutoModelForSequenceClassification.from_pretrained(\n"
     "    BASE, num_labels=3, quantization_config=bnb, dtype=torch.float16, device_map='auto')\n"
     "model.config.pad_token_id = tok.pad_token_id\n"
     "model = prepare_model_for_kbit_training(model)\n\n"
     "import bitsandbytes as _bnb\n"
     "def linear_names(m):\n"
     "    names = set()\n"
     "    for n, mod in m.named_modules():\n"
     "        if isinstance(mod, (torch.nn.Linear, _bnb.nn.Linear4bit)):\n"
     "            names.add(n.split('.')[-1])\n"
     "    names -= {'lm_head', 'score', 'classifier'}\n"
     "    return sorted(names)\n"
     "lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias='none',\n"
     "    task_type='SEQ_CLS', target_modules=linear_names(model))\n"
     "model = get_peft_model(model, lora)\n"
     "model.print_trainable_parameters()"),
    ("code",
     "def latest_ckpt():\n"
     "    hits = glob.glob('/kaggle/input/**/checkpoint-*', recursive=True)\n"
     "    hits += glob.glob('/kaggle/working/ckpt/checkpoint-*')\n"
     "    hits = [h for h in hits if os.path.isdir(h) and os.path.exists(h + '/trainer_state.json')]\n"
     "    return max(hits, key=lambda p: int(p.rstrip('/').split('-')[-1])) if hits else None\n"
     "RESUME = latest_ckpt()\n"
     "print('resuming from:', RESUME if RESUME else 'scratch (first session)')"),
    ("code",
     "class TimeStop(TrainerCallback):\n"
     "    def on_train_begin(self, args, state, control, **kw): self.t0 = time.time()\n"
     "    def on_step_end(self, args, state, control, **kw):\n"
     "        if time.time() - self.t0 > TRAIN_SECONDS:\n"
     "            print(f'[TimeStop] {TRAIN_SECONDS}s hit at step {state.global_step} -> save + stop')\n"
     "            control.should_save = True; control.should_training_stop = True\n"
     "        return control\n\n"
     "def metric(eval_pred):\n"
     "    logits, labels = eval_pred\n"
     "    p = torch.softmax(torch.tensor(logits), dim=1).numpy()\n"
     "    return {'log_loss': log_loss(labels, p, labels=[0,1,2])}\n\n"
     "args = TrainingArguments(\n"
     "    output_dir='/kaggle/working/ckpt', per_device_train_batch_size=1,\n"
     "    gradient_accumulation_steps=16, per_device_eval_batch_size=1,\n"
     "    learning_rate=8e-5, num_train_epochs=EPOCHS, warmup_steps=50,\n"
     "    lr_scheduler_type='cosine', fp16=True, gradient_checkpointing=True, logging_steps=25,\n"
     "    save_strategy='steps', save_steps=SAVE_STEPS, save_total_limit=2,\n"
     "    eval_strategy='no', report_to='none', optim='paged_adamw_8bit')\n"
     "trainer = Trainer(model=model, args=args, train_dataset=ds_tr, eval_dataset=ds_va,\n"
     "    processing_class=tok, data_collator=DataCollatorWithPadding(tok),\n"
     "    compute_metrics=metric, callbacks=[TimeStop()])"),
    ("code",
     "out = trainer.train(resume_from_checkpoint=RESUME)\n"
     "done = trainer.state.global_step >= trainer.state.max_steps\n"
     "print(f'stopped at step {trainer.state.global_step}/{trainer.state.max_steps} '\n"
     "      f\"-- {'TRAINING COMPLETE' if done else 'partial: resume next session'}\")\n"
     "print('train loss:', out.metrics.get('train_loss'))"),
    ("code",
     "model.save_pretrained('/kaggle/working/adapter')\n"
     "tok.save_pretrained('/kaggle/working/adapter')\n"
     "print(sorted(os.listdir('/kaggle/working/adapter')))\n"
     "if done:\n"
     "    print('VALIDATION log_loss:', trainer.evaluate().get('eval_log_loss'))"),
]


def build():
    nb = nbf.v4.new_notebook()
    nb.cells = [nbf.v4.new_markdown_cell(s) if t == "markdown" else nbf.v4.new_code_cell(s)
                for t, s in CELLS]
    nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}}
    return nb


meta = {
    "id": f"{USER}/gemma-phase1", "title": "gemma-phase1",
    "code_file": "gemma-phase1.ipynb", "language": "python", "kernel_type": "notebook",
    "is_private": True, "enable_gpu": True, "enable_internet": True,
    "dataset_sources": [EXTERNAL_DS], "competition_sources": ["llm-classification-finetuning"],
    "model_sources": [GEMMA_MODEL], "kernel_sources": [],
}

d = HERE / "gemma-phase1"
d.mkdir(exist_ok=True)
nbf.write(build(), d / "gemma-phase1.ipynb")
(d / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
print("wrote", d / "gemma-phase1.ipynb", "and its kernel-metadata.json")
