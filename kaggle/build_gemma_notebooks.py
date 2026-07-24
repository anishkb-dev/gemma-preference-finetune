"""Build the Gemma-2-9B QLoRA train + inference notebooks via nbformat.

Run: ../.venv/bin/python build_gemma_notebooks.py
Produces kaggle/gemma-train/gemma-train.ipynb and kaggle/gemma-infer/gemma-infer.ipynb
plus each kernel-metadata.json.
"""
import json
import pathlib

import nbformat as nbf

HERE = pathlib.Path(__file__).resolve().parent
USER = "anishkb24"

# ---------------------------------------------------------------- shared code
SHARED_DATA = r'''
import json, re, os, glob, numpy as np, pandas as pd

TARGETS = ["winner_model_a", "winner_model_b", "winner_tie"]

def find_dir(pattern):
    hits = glob.glob(pattern, recursive=True)
    assert hits, f"nothing matched {pattern} under /kaggle/input"
    return os.path.dirname(hits[0])

def parse_list(x):
    if isinstance(x, list): return [str(t) for t in x]
    if not isinstance(x, str): return [""]
    try: v = json.loads(x)
    except Exception: return [x]
    if isinstance(v, list): return ["" if t is None else str(t) for t in v]
    return ["" if v is None else str(v)]

def join(x): return "\n".join(parse_list(x))

def build_text(prompt, resp_a, resp_b, max_chars=7000):
    """Structured prompt for the classifier. Truncate each field head+tail."""
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
'''

# ---------------------------------------------------------------- TRAIN cells
TRAIN_CELLS = [
    ("markdown",
     "# Gemma-2-9B QLoRA — preference classifier (TRAINING)\n"
     "Fine-tunes Gemma-2-9B (4-bit QLoRA) as a 3-class classifier and saves the "
     "LoRA adapter to `/kaggle/working/adapter`. Enable **GPU** and **Internet** "
     "for this notebook, and attach the **Gemma-2-9B** model under *Add Input → Models*."),
    ("code",
     "# transformers 5.7.0 (needs bitsandbytes>=0.46.1 for 4-bit). Do a Factory reset first,\n"
     "# then Run All patiently (the first `import transformers` scans package metadata, ~1-3 min).\n"
     "!pip install -q 'transformers==5.7.0' 'bitsandbytes>=0.46.1'"),
    ("code",
     "# Every huggingface_hub version on this image has a @strict that rejects transformers'\n"
     "# Gemma2Config (StrictDataclassDefinitionError), and no allowed hub version avoids it.\n"
     "# Neutralize @strict to a pass-through -- it only skips optional config input-validation,\n"
     "# nothing that affects the model or training. Must run BEFORE the model is loaded.\n"
     "import huggingface_hub, huggingface_hub.dataclasses as _hfd\n"
     "_noop = lambda cls=None, **kw: (cls if cls is not None else (lambda c: c))\n"
     "_hfd.strict = _noop\n"
     "huggingface_hub.strict = _noop\n"
     "print('patched huggingface_hub.strict')"),
    ("code",
     "import os\n"
     "os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')  # reduce fragmentation\n"
     "import torch, time\n"
     "from transformers import (AutoTokenizer, AutoModelForSequenceClassification,\n"
     "    BitsAndBytesConfig, TrainingArguments, Trainer, DataCollatorWithPadding)\n"
     "from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training\n"
     "from sklearn.model_selection import train_test_split\n"
     "from sklearn.metrics import log_loss\n"
     "import datasets\n\n"
     "MAX_LEN = 512           # short so 9B on T4 x2 finishes 1 epoch well under the 9h cap\n"
     "SUBSAMPLE = 10000       # ~594 steps -> ~5-6h. Raise once a full run succeeds. None = all rows\n"
     "EPOCHS = 1\n"
     "print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"),
    ("code", SHARED_DATA),
    ("code",
     "# Locate base model + competition data (attach both via Add Input).\n"
     "BASE = find_dir('/kaggle/input/**/config.json')\n"
     "COMP = find_dir('/kaggle/input/**/train.csv')\n"
     "print('BASE =', BASE, '\\nCOMP =', COMP)"),
    ("code",
     "df = pd.read_csv(f'{COMP}/train.csv')\n"
     "if SUBSAMPLE: df = df.sample(SUBSAMPLE, random_state=42).reset_index(drop=True)\n"
     "df['text'] = [build_text(join(p), join(a), join(b))\n"
     "              for p, a, b in zip(df['prompt'], df['response_a'], df['response_b'])]\n"
     "df['label'] = np.argmax(df[TARGETS].values, axis=1)\n"
     "tr, va = train_test_split(df, test_size=0.05, stratify=df['label'], random_state=42)\n"
     "print(len(tr), 'train /', len(va), 'val')"),
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
     "# device_map='auto' shards the 9B across all visible GPUs (e.g. T4 x2 -> ~29GB total),\n"
     "# which is what makes 9B QLoRA training fit. On a single GPU it just uses that one.\n"
     "model = AutoModelForSequenceClassification.from_pretrained(\n"
     "    BASE, num_labels=3, quantization_config=bnb, dtype=torch.float16, device_map='auto')\n"
     "model.config.pad_token_id = tok.pad_token_id\n"
     "model = prepare_model_for_kbit_training(model)\n\n"
     "# Auto-detect linear layers so LoRA works on any Gemma version (2/3/4).\n"
     "import bitsandbytes as bnb\n"
     "def linear_names(m):\n"
     "    names = set()\n"
     "    for n, mod in m.named_modules():\n"
     "        if isinstance(mod, (torch.nn.Linear, bnb.nn.Linear4bit)):\n"
     "            names.add(n.split('.')[-1])\n"
     "    names -= {'lm_head', 'score', 'classifier'}  # head is trained separately\n"
     "    return sorted(names)\n"
     "targets = linear_names(model)\n"
     "print('LoRA target modules:', targets)\n"
     "lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias='none',\n"
     "    task_type='SEQ_CLS', target_modules=targets)\n"
     "model = get_peft_model(model, lora)\n"
     "model.print_trainable_parameters()"),
    ("code",
     "def metric(eval_pred):\n"
     "    logits, labels = eval_pred\n"
     "    p = torch.softmax(torch.tensor(logits), dim=1).numpy()\n"
     "    return {'log_loss': log_loss(labels, p, labels=[0,1,2])}\n\n"
     "args = TrainingArguments(\n"
     "    output_dir='/kaggle/working/ckpt', per_device_train_batch_size=1,\n"
     "    gradient_accumulation_steps=16, per_device_eval_batch_size=1,\n"
     "    learning_rate=1e-4, num_train_epochs=EPOCHS, warmup_ratio=0.03,\n"
     "    fp16=True, gradient_checkpointing=True, logging_steps=25,\n"
     "    eval_strategy='epoch', save_strategy='no', report_to='none', optim='paged_adamw_8bit')\n"
     "trainer = Trainer(model=model, args=args, train_dataset=ds_tr, eval_dataset=ds_va,\n"
     "    processing_class=tok, data_collator=DataCollatorWithPadding(tok), compute_metrics=metric)\n"
     "trainer.train()\n"
     "print(trainer.evaluate())"),
    ("code",
     "model.save_pretrained('/kaggle/working/adapter')\n"
     "tok.save_pretrained('/kaggle/working/adapter')\n"
     "print('saved adapter -> /kaggle/working/adapter')\n"
     "# Next: Save Version (commit). Then create a Kaggle Dataset from this output,\n"
     "# and attach it to the gemma-infer notebook."),
]

# ---------------------------------------------------------------- INFER cells
INFER_CELLS = [
    ("markdown",
     "# Gemma-2-9B QLoRA — SUBMISSION (inference)\n"
     "Loads base Gemma-2-9B + the trained LoRA adapter, predicts the test set with "
     "**A/B-swap test-time augmentation** to cancel position bias, and writes "
     "`submission.csv`. Enable **GPU**, set **Internet = OFF**. Attach: the base "
     "**Gemma-2-9B** model, the **adapter dataset** (from training output), and the "
     "**competition** data."),
    ("code",
     "# Internet is OFF here, so we can't pip-install. Neutralize huggingface_hub @strict\n"
     "# (defensively) so the image's transformers can import Gemma2Config offline.\n"
     "try:\n"
     "    import huggingface_hub, huggingface_hub.dataclasses as _hfd\n"
     "    _noop = lambda cls=None, **kw: (cls if cls is not None else (lambda c: c))\n"
     "    _hfd.strict = _noop; huggingface_hub.strict = _noop\n"
     "    print('patched huggingface_hub.strict')\n"
     "except Exception as e:\n"
     "    print('no strict patch needed:', e)"),
    ("code",
     "import torch, numpy as np, pandas as pd\n"
     "from transformers import AutoTokenizer, AutoModelForSequenceClassification\n"
     "from peft import PeftModel\n"
     "MAX_LEN = 512    # keep the 25K x2-TTA scoring re-run well under the 9h cap\n"
     "BATCH = 16"),
    ("code", SHARED_DATA),
    ("code",
     "BASE = find_dir('/kaggle/input/**/config.json')\n"
     "ADAPTER = find_dir('/kaggle/input/**/adapter_config.json')\n"
     "COMP = find_dir('/kaggle/input/**/test.csv')\n"
     "print('BASE =', BASE, '\\nADAPTER =', ADAPTER, '\\nCOMP =', COMP)"),
    ("code",
     "# Load in fp16 sharded across all GPUs (T4 x2 -> ~29GB) instead of 4-bit, so we don't\n"
     "# depend on a bitsandbytes version we can't pip-install offline. A 4-bit-trained LoRA\n"
     "# adapter applies fine to an fp16 base for inference.\n"
     "tok = AutoTokenizer.from_pretrained(ADAPTER)\n"
     "if tok.pad_token is None: tok.pad_token = tok.eos_token\n"
     "base = AutoModelForSequenceClassification.from_pretrained(\n"
     "    BASE, num_labels=3, dtype=torch.float16, device_map='auto')\n"
     "base.config.pad_token_id = tok.pad_token_id\n"
     "# The image's torchao is too old and peft's check RAISES instead of skipping. Our adapter\n"
     "# is plain LoRA (no torchao), so force the check to False before loading the adapter.\n"
     "import peft.import_utils, peft.tuners.lora.torchao as _pt\n"
     "peft.import_utils.is_torchao_available = _pt.is_torchao_available = lambda: False\n"
     "model = PeftModel.from_pretrained(base, ADAPTER).eval()\n"
     "print('loaded; device map:', getattr(model, 'hf_device_map', 'single'))"),
    ("code",
     "test = pd.read_csv(f'{COMP}/test.csv')\n"
     "P = [join(x) for x in test['prompt']]\n"
     "A = [join(x) for x in test['response_a']]\n"
     "B = [join(x) for x in test['response_b']]\n\n"
     "@torch.no_grad()\n"
     "def predict(texts):\n"
     "    # Sort by length so each batch pads to its own longest row (not a global max),\n"
     "    # cutting wasted compute; scatter results back to original order.\n"
     "    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))\n"
     "    probs = np.zeros((len(texts), 3), dtype=np.float32)\n"
     "    for s in range(0, len(order), BATCH):\n"
     "        idx = order[s:s+BATCH]\n"
     "        enc = tok([texts[i] for i in idx], truncation=True, max_length=MAX_LEN,\n"
     "                  padding=True, return_tensors='pt').to(model.device)\n"
     "        p = torch.softmax(model(**enc).logits.float(), dim=1).cpu().numpy()\n"
     "        for j, i in enumerate(idx):\n"
     "            probs[i] = p[j]\n"
     "    return probs"),
    ("code",
     "# Forward view (A,B) and swapped view (B,A); swapping cancels position bias.\n"
     "fwd = predict([build_text(p, a, b) for p, a, b in zip(P, A, B)])\n"
     "swp = predict([build_text(p, b, a) for p, a, b in zip(P, A, B)])\n"
     "# In the swapped view, class a<->b flip, tie stays.\n"
     "swp = swp[:, [1, 0, 2]]\n"
     "prob = (fwd + swp) / 2"),
    ("code",
     "sub = test[['id']].copy()\n"
     "sub[TARGETS] = prob\n"
     "sub.to_csv('/kaggle/working/submission.csv', index=False)\n"
     "print(sub.shape); sub.head()"),
]


def build(cells):
    nb = nbf.v4.new_notebook()
    nb.cells = [nbf.v4.new_markdown_cell(s) if t == "markdown" else nbf.v4.new_code_cell(s)
                for t, s in cells]
    nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}}
    return nb


# Auto-attach the Gemma-2-9B transformers variant so a pushed run mounts it with no UI step.
GEMMA_MODEL = "google/gemma-2/transformers/gemma-2-9b-it/2"


def meta(kid, code_file, internet, models, datasets):
    return {
        "id": f"{USER}/{kid}", "title": kid, "code_file": code_file,
        "language": "python", "kernel_type": "notebook", "is_private": True,
        "enable_gpu": True, "enable_internet": internet,
        "dataset_sources": datasets, "competition_sources": ["llm-classification-finetuning"],
        "model_sources": models, "kernel_sources": [],
    }


for kid, cells, internet, models, datasets in [
    ("gemma-train", TRAIN_CELLS, True, [GEMMA_MODEL], []),
    ("gemma-infer", INFER_CELLS, False, [GEMMA_MODEL], [f"{USER}/gemma-adapter"]),
]:
    d = HERE / kid
    d.mkdir(exist_ok=True)
    fname = f"{kid}.ipynb"
    nbf.write(build(cells), d / fname)
    (d / "kernel-metadata.json").write_text(json.dumps(meta(kid, fname, internet, models, datasets), indent=2))
    print("wrote", d / fname, "and its kernel-metadata.json")
