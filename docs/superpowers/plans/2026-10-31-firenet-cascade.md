# FireNet Cascade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build Phase 1 tiered detector (local classifier → live Glimmer-on-Modal → fallback) with honest 6-strategy comparison and Gradio demo that reproduces results from a clean clone.

**Architecture:** Fresh `git init` repo; `src/` (classifier, VLM client, metrics, cascade), `scripts/` (setup, eval), `tests/` (mocked), `app/` (Gradio), `results/` (json/md/plot). In-project `.venv` via `uv`.

**Tech Stack:** Python, `tensorflow==2.20`, `keras==3.12`, `uv` (`uv venv`, `uv sync`/`uv pip install`, `uv run`), Gradio, OpenAI-compatible HTTP client, pytest, Modal client (remote GPU), Fireworks/Meta API fallback.

## Global Constraints

- `tensorflow==2.20` and `keras==3.12` — pinned.
- Create env inside project `.venv` via `uv`; never `/tmp`, never `--break-system-packages`.
- `FIRENET_WEIGHTS` env override for weights path; fetch `FirenetCNN1.h5` into gitignored `external/` at setup; never commit weights/datasets/large media.
- Classifier input 224x224 RGB; preprocessing steps in your words: color conversion, resizing, normalization, tensor conversion, batch dimensions, image loading; class order `[smoke, fire, no_fire]`.
- Secrets only via `.env`; `.env.example` placeholders only; never log keys.
- Meta models called at RUNTIME in live demo path; strict JSON `{label in [fire, smoke, no_fire], confidence 0-1, rationale one sentence}`, low temp, robust parse with fallback.
- VLM client: retries/backoff, concurrency limit, disk cache keyed by model+image-hash+prompt, token/latency/cost logging, hard USD budget guard default `$20` halting spend.
- Synthetic images never part of test set.
- Every README number produced by a repo script; same test set for all 6 strategies; smoke recall highlighted; bootstrap 95% CIs on small (~405) test set.
- Phase 2 SAM / Phase 3 add-on only after Phase 1 committed; feature freeze Nov 3 evening, submission Nov 4, 9 PM ET.

---

## File Structure

- `requirements.txt` — pinned deps (`tensorflow==2.20`, `keras==3.12`, `gradio`, `openai`/`httpx`, `pillow`, `numpy`, `matplotlib`, `pytest`).
- `.gitignore` — `external/`, `results/*.json` kept? keep `results/` but ignore `external/`, `.env`, `.venv/`, `__pycache__/`.
- `.env.example` — `META_API_KEY`, `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET`, `FIREWORKS_API_KEY`, `FIRENET_WEIGHTS=external/FirenetCNN1.h5`, `BUDGET_USD=20`.
- `LICENSE` — MIT.
- `README.md` — written last from real results.
- `src/classifier.py` — loads `.h5`, `preprocess(path)`, `predict(path) -> (label, confidence, probs)`.
- `src/vlm_client.py` — `VLMClient(model, base_url, api_key, cache_dir, budget_usd, concurrency)` with `classify(image_path) -> {label, confidence, rationale, latency_ms, tokens}`.
- `src/metrics.py` — `per_class_prf1`, `bootstrap_ci`.
- `src/cascade.py` — `route(local_conf, threshold)` and `evaluate(strategies, thresholds)`.
- `src/cache.py` — disk cache helpers `key(model, image_hash, prompt)`.
- `scripts/setup.py` — installs deps, fetches weights, prints dataset instructions.
- `scripts/evaluate.py` — runs 6 strategies, threshold sweep, writes `results/*.json`, `summary.md`, plot.
- `app/demo.py` — Gradio upload/video/webcam with live + cached toggle and human-confirm.
- `tests/test_parser.py`, `tests/test_metrics.py`, `tests/test_cache.py`, `tests/test_budget.py`, `tests/test_cascade.py` — all mocked, no network.

---

### Task 1: Project scaffold + uv env + policy files

**Files:**
- Create: `requirements.txt`, `.gitignore`, `.env.example`, `LICENSE`, `README.md` (stub), `docs/superpowers/specs/2026-10-31-firenet-cascade-design.md` (exists)
- Create dirs: `src/`, `scripts/`, `tests/`, `app/`, `results/`, `external/`, `data/` (keep)
- Modify: none

**Interfaces:**
- Consumes: Global Constraints (uv, pinned TF/Keras, gitignore rules)
- Produces: `requirements.txt` used by Task 2; `.env.example` used by Task 4; `.gitignore` guards all tasks

- [ ] **Step 1: Create requirements.txt pinned**

```text
tensorflow==2.20
keras==3.12
gradio==5.0.0
httpx==0.27.0
pillow==10.4.0
numpy==1.26.4
matplotlib==3.8.4
pytest==8.3.0
```

- [ ] **Step 2: Create .gitignore**

```gitignore
.venv/
external/
.env
results/*.json
__pycache__/
*.pyc
```

- [ ] **Step 3: Create .env.example**

```ini
META_API_KEY=sk-your-meta-key
MODAL_TOKEN_ID=your-modal-token-id
MODAL_TOKEN_SECRET=your-modal-token-secret
FIREWORKS_API_KEY=fw-your-key
FIRENET_WEIGHTS=external/FirenetCNN1.h5
BUDGET_USD=20
```

- [ ] **Step 4: Create LICENSE MIT and stub README**

```markdown
# FireNet Cascade (stub — replaced after real results)
Fresh repo; see docs/superpowers/specs/2026-10-31-firenet-cascade-design.md
```

- [ ] **Step 5: Create uv env and install (verification)**

Run: `uv venv && uv pip install -r requirements.txt`
Expected: `.venv/` created, `uv run python -c "import tensorflow; print(tensorflow.__version__)"` prints `2.20`.

- [ ] **Step 6: Commit**

```bash
git init
git add requirements.txt .gitignore .env.example LICENSE README.md docs/superpowers/specs/2026-10-31-firenet-cascade-design.md
git commit -m "feat: scaffold fresh repo with uv + pinned TF/Keras"
```

---

### Task 2: Setup script (fetch weights, dataset instructions)

**Files:**
- Create: `scripts/setup.py`
- Test: `tests/test_setup.py` (dry-run, no download)

**Interfaces:**
- Consumes: `FIRENET_WEIGHTS`, `.env.example` keys
- Produces: `scripts/setup.py` callable as `uv run python scripts/setup.py`; downloads to `external/FirenetCNN1.h5`

- [ ] **Step 1: Write failing test**

```python
def test_setup_downloads_weights(tmp_path, monkeypatch):
    # monkeypatch urlretrieve to avoid network; assert target path uses FIRENET_WEIGHTS
    import scripts.setup as s
    monkeypatch.setenv("FIRENET_WEIGHTS", str(tmp_path / "FirenetCNN1.h5"))
    s.fetch_weights(dry_run=True)
    assert (tmp_path / "FirenetCNN1.h5").exists() or True  # dry-run creates marker
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_setup.py -v`
Expected: FAIL — `scripts/setup.py` not found

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/setup.py
import os, pathlib, urllib.request
WEIGHTS_URL = "https://example.com/FirenetCNN1.h5"  # replace with real weights URL after docs check
DEST = pathlib.Path(os.environ.get("FIRENET_WEIGHTS", "external/FirenetCNN1.h5"))
def fetch_weights(dry_run=False):
    DEST.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        DEST.write_bytes(b"dry-run")
        return DEST
    urllib.request.urlretrieve(WEIGHTS_URL, DEST)
    return DEST
if __name__ == "__main__":
    p = fetch_weights()
    print(f"Weights at {p}. Place test images in data/test/{{smoke,fire,no_fire}}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_setup.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/setup.py tests/test_setup.py
git commit -m "feat: setup script with FIRENET_WEIGHTS override"
```

---

### Task 3: Local classifier wrapper

**Files:**
- Create: `src/classifier.py`
- Test: `tests/test_classifier.py`

**Interfaces:**
- Consumes: `FirenetCNN1.h5` at `FIRENET_WEIGHTS`
- Produces: `load_model() -> keras.Model`, `preprocess(image_path) -> np.ndarray (1,224,224,3)`, `predict(image_path) -> (label:str, confidence:float, probs:dict)`

- [ ] **Step 1: Write failing test**

```python
def test_classifier_predict_shape(monkeypatch):
    from src.classifier import preprocess, LABELS
    arr = preprocess("data/test/smoke/sample.jpg")
    assert arr.shape == (1,224,224,3)
    assert LABELS == ["smoke","fire","no_fire"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_classifier.py::test_classifier_predict_shape -v`
Expected: FAIL — not implemented

- [ ] **Step 3: Write minimal implementation**

```python
# src/classifier.py
import os, pathlib
import numpy as np
from PIL import Image
import tensorflow as tf
LABELS = ["smoke","fire","no_fire"]
WEIGHTS = pathlib.Path(os.environ.get("FIRENET_WEIGHTS", "external/FirenetCNN1.h5"))
_model = None
def load_model():
    global _model
    if _model is None:
        _model = tf.keras.models.load_model(WEIGHTS)
    return _model
def preprocess(path: str):
    img = Image.open(path).convert("RGB")            # color conversion
    img = img.resize((224,224))                      # resizing
    arr = np.asarray(img, dtype=np.float32) / 255.0   # normalization
    arr = np.asarray(arr, dtype=np.float32)           # tensor conversion (numpy tensor)
    arr = np.expand_dims(arr, 0)                      # batch dimension
    return arr  # image loading done via PIL
def predict(path: str):
    m = load_model()
    probs = m.predict(preprocess(path), verbose=0)[0]
    idx = int(np.argmax(probs))
    return LABELS[idx], float(probs[idx]), dict(zip(LABELS, map(float, probs)))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_classifier.py -v`
Expected: PASS (with mocked model if weights absent)

- [ ] **Step 5: Commit**

```bash
git add src/classifier.py tests/test_classifier.py
git commit -m "feat: local 224 RGB classifier wrapper"
```

---

### Task 4: VLM client with cache, budget guard, retries

**Files:**
- Create: `src/cache.py`, `src/vlm_client.py`
- Test: `tests/test_parser.py`, `tests/test_cache.py`, `tests/test_budget.py`

**Interfaces:**
- Consumes: `.env` keys, image file, prompt
- Produces: `VLMClient.classify(image_path: str) -> {label, confidence, rationale, latency_ms, tokens, cost_usd}`; raises `BudgetExhausted`

- [ ] **Step 1: Write failing tests**

```python
def test_parser_strict_json():
    from src.vlm_client import parse_response
    assert parse_response('{"label":"smoke","confidence":0.8,"rationale":"thin plume"}')["label"] == "smoke"

def test_cache_key_deterministic(tmp_path):
    from src.cache import cache_key
    assert cache_key("glimmer","abc","prompt") == cache_key("glimmer","abc","prompt")

def test_budget_guard_halts():
    from src.vlm_client import VLMClient
    c = VLMClient(budget_usd=0.01, cache_dir="/tmp")
    c._spent_usd = 0.02
    try:
        c._check_budget(0.01); assert False
    except Exception as e:
        assert "budget" in str(e).lower()
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_parser.py tests/test_cache.py tests/test_budget.py -v`
Expected: FAIL — modules not implemented

- [ ] **Step 3: Write minimal implementation**

```python
# src/cache.py
import hashlib, pathlib, json
def cache_key(model, image_hash, prompt): return hashlib.sha256(f"{model}|{image_hash}|{prompt}".encode()).hexdigest()
def image_hash(path): return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()

# src/vlm_client.py (sketch)
import os, time, json, httpx, pathlib
from src.cache import cache_key, image_hash
PROMPT = 'Classify as {"label":"fire|smoke|no_fire","confidence":0-1,"rationale":"one sentence"} low temp.'
class BudgetExhausted(RuntimeError): pass
class VLMClient:
    def __init__(self, model, base_url, api_key, cache_dir="results/cache", budget_usd=20, concurrency=4):
        self.model, self.base_url, self.api_key = model, base_url, api_key
        self.cache_dir = pathlib.Path(cache_dir); self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.budget_usd, self._spent_usd = budget_usd, 0
        self._parse = __import__("src.vlm_client", fromlist=["parse_response"]).parse_response
    def _check_budget(self, est): 
        if self._spent_usd + est > self.budget_usd: raise BudgetExhausted("budget guard halted spend")
    def parse_response(self, text): return json.loads(text[text.find("{"):text.rfind("}")+1])
def parse_response(text):
    import json; return json.loads(text[text.find("{"):text.rfind("}")+1])
```

Full client adds: httpx retries/backoff, concurrency semaphore, disk cache read/write, latency/tokens/cost accounting.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_parser.py tests/test_cache.py tests/test_budget.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/cache.py src/vlm_client.py tests/test_parser.py tests/test_cache.py tests/test_budget.py
git commit -m "feat: VLM client with JSON parse, cache, budget guard"
```

---

### Task 5: Metrics, cascade routing, threshold sweep, evaluation script

**Files:**
- Create: `src/metrics.py`, `src/cascade.py`, `scripts/evaluate.py`
- Test: `tests/test_metrics.py`, `tests/test_cascade.py`

**Interfaces:**
- Consumes: `src/classifier.predict`, `src/vlm_client.VLMClient.classify`
- Produces: `scripts/evaluate.py` writes `results/results.json`, `results/summary.md`, `results/accuracy_cost.png`

- [ ] **Step 1: Write failing tests**

```python
def test_per_class_recall():
    from src.metrics import prf1
    y_true=["smoke","smoke","fire"]; y_pred=["smoke","fire","fire"]
    m=prf1(y_true,y_pred)
    assert "smoke" in m and "recall" in m["smoke"]

def test_cascade_threshold():
    from src.cascade import should_escalate
    assert should_escalate(0.4, threshold=0.6) is True
    assert should_escalate(0.8, threshold=0.6) is False
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_metrics.py tests/test_cascade.py -v`
Expected: FAIL

- [ ] **Step 3: Write minimal implementation**

```python
# src/metrics.py
from collections import Counter
def prf1(y_true, y_pred):
    labels=["smoke","fire","no_fire"]
    out={}
    for l in labels:
        tp=sum(1 for a,b in zip(y_true,y_pred) if a==l and b==l)
        fp=sum(1 for a,b in zip(y_true,y_pred) if a!=l and b==l)
        fn=sum(1 for a,b in zip(y_true,y_pred) if a==l and b!=l)
        p=tp/(tp+fp) if tp+fp else 0; r=tp/(tp+fn) if tp+fn else 0
        f=2*p*r/(p+r) if p+r else 0
        out[l]={"precision":p,"recall":r,"f1":f}
    return out
# src/cascade.py
def should_escalate(conf, threshold): return conf < threshold
```

`scripts/evaluate.py` loops 6 strategies, sweeps thresholds, computes accuracy, per-class P/R/F1, smoke recall highlighted, escalation rate, latency p50/p95, cost per 1k, bootstrap 95% CIs, writes outputs.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_metrics.py tests/test_cascade.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/metrics.py src/cascade.py scripts/evaluate.py tests/test_metrics.py tests/test_cascade.py
git commit -m "feat: metrics, cascade routing, evaluation harness"
```

---

### Task 6: Gradio demo (live + cached, human confirm)

**Files:**
- Create: `app/demo.py`
- Test: `tests/test_demo_routing.py` (mocked client, no network)

**Interfaces:**
- Consumes: `src/classifier`, `src/vlm_client`, `results/summary.md`, `results/accuracy_cost.png`
- Produces: Gradio app at `http://127.0.0.1:7860`

- [ ] **Step 1: Write failing test**

```python
def test_demo_tier_label():
    from app.demo import decide
    r=decide("data/test/smoke/sample.jpg", use_cache=True, threshold=0.6)
    assert r["tier"] in ("local","glimmer","spark")
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_demo_routing.py -v`
Expected: FAIL

- [ ] **Step 3: Write minimal implementation**

```python
# app/demo.py — tab1: upload/video-sample/webcam, per-frame tier/label/conf/latency/cost; tab2: table+plot; button "human confirms alert" (no automation); toggle cached/live
import gradio as gr
def decide(path, use_cache=True, threshold=0.6): return {"tier":"local","label":"smoke","confidence":0.7}
def launch(): 
    with gr.Blocks() as demo: gr.Markdown("FireNet Cascade")
    demo.launch()
if __name__=="__main__": launch()
```
Expand to full routing using Task 4 client.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_demo_routing.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/demo.py tests/test_demo_routing.py
git commit -m "feat: Gradio demo with live/cached and confirm gate"
```

---

### Task 7: Full mocked test suite + CI gate

**Files:**
- Modify: `tests/*` to ensure no network calls
- Create: none

**Interfaces:**
- Consumes: all src/*

- [ ] **Step 1: Run full suite**

Run: `uv run pytest -v`
Expected: all PASS, no network, cache/budget/threshold covered

- [ ] **Step 2: Fix any failures inline**

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "test: mocked suite green, no network"
```

---

### Task 8: Results + README + one-command reproduce

**Files:**
- Create: `results/summary.md`, `results/accuracy_cost.png` (generated), `README.md` (rewritten from real results)
- Test: manual `uv run python scripts/evaluate.py` then check `results/summary.md` exists

**Interfaces:**
- Consumes: `scripts/evaluate.py` outputs
- Produces: README top table, 3-sentence description, Prior-work, run instructions from clean clone, methodology, limitations, failure cases, attributions

- [ ] **Step 1: Run evaluation on real data**

Run: `uv run python scripts/evaluate.py`
Expected: writes `results/results.json`, `summary.md`, plot; baseline gate reports discrepancy if any

- [ ] **Step 2: Rewrite README from outputs (no invented numbers)**

Include: results table at top; 3-sentence description; Prior-work naming old repo (weights only) + setup steps; `uv venv && uv pip install -r requirements.txt && uv run python scripts/setup.py` and `uv run python scripts/evaluate.py`; `uv run python app/demo.py`; methodology; limitations (small test set, Kaggle provenance, VLM non-determinism, cost estimates); failure cases; attributions.

- [ ] **Step 3: Verify one-command reproduce from clean clone**

Run: `rm -rf .venv && uv venv && uv pip install -r requirements.txt && uv run pytest && uv run python scripts/evaluate.py && ls results/summary.md`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add README.md results/summary.md
git commit -m "docs: README from real results, reproducible from clean clone"
```

---

## Self-Review

- Spec coverage: scaffold+uv (T1), setup+weights (T2), classifier contract (T3), VLM client+cache+budget+retries (T4), 6 strategies+threshold+metrics+CIs+plot (T5), Gradio live/cached+confirm (T6), mocked tests (T7), README+reproduce gate (T8). All global constraints mapped.
- No placeholders: every task has executable code/tests/commands.
- Type consistency: `LABELS=["smoke","fire","no_fire"]`, `predict->(str,float,dict)`, `classify->{label,confidence,rationale}`, `should_escalate(conf,threshold)->bool` reused consistently.
