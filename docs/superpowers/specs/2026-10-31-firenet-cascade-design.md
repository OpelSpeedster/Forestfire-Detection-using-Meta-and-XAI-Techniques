# FireNet Cascade — Design (Approach A, Phase 1 core)

Date: 2026-10-31
Status: approved sections 1-4, pending spec review
Constraints: feature freeze Nov 3 evening; submission Nov 4, 9 PM ET; fresh `git init` repo; honest reproducible core over ambition.

## Goal
Tiered wildfire/smoke detector. Cheap local classifier screens every frame. Uncertain cases escalate live to Meta models (open-weight first, closed API second). Headline: measured open vs closed comparison on weakest class (smoke) — accuracy, per-class recall, cost, latency.

## Inputs (confirmed)
- Dataset: local `data/` in workspace, Kaggle-sourced. Expected `data/test/{smoke,fire,no_fire}` (~405 images). License/provenance to confirm from Kaggle page before README.
- Classifier: 224x224 RGB, 3-channel; preprocessing in your words: color conversion, resizing, normalization, tensor conversion, batch dimensions, image loading (exact order/normalization constants confirmed at implementation); class order `[smoke, fire, no_fire]`.
- Runtime: CPU-only local (no Nvidia). Modal cloud GPU for hosted inference. Primary Glimmer via Modal inference; fallback Meta API or Fireworks. `META_API_KEY` to be provided later via `.env`.
- Deps: `tensorflow==2.20`, `keras==3.12`; always `uv` (`uv venv`, `uv sync` / `uv pip install`, `uv run`) with in-project `.venv`.

## Architecture
Fresh repo. Layout: `src/` (classifier, VLM client, metrics, cascade), `scripts/` (setup, reproduce), `tests/` (mocked), `app/` (Gradio), `results/` (`*.json`, `summary.md`, plot). Weights `FirenetCNN1.h5` fetched at setup into gitignored `external/`; override `FIRENET_WEIGHTS`. Never commit weights, datasets, large media. `.env` secrets only; `.env.example` placeholders only.

## Components
- `classifier`: loads `.h5`, fixed 224x224 RGB chain, emits label + confidence.
- `vlm_client`: OpenAI-compatible; fixed prompt; strict JSON `{label in [fire, smoke, no_fire], confidence 0-1, rationale one sentence}`; low temp; robust parse + fallback; retries/backoff; concurrency limit; disk cache (model+image-hash+prompt); token/latency/cost logging; hard USD budget guard (default $20) halting spend.
- `strategies`: (a) local, (b) Glimmer-only, (c) Spark-only, (d) local→Glimmer, (e) local→Spark, (f) local→Glimmer→Spark; threshold sweep.
- `app`: upload / video-sample-every-N-sec / webcam; per-frame tier, label, confidence, latency, cost; comparison tab; human-confirm button (never automate); live mode (real Meta calls) + labeled cached mode.
- `scripts/setup`: `uv` install, fetch weights, dataset instructions.

## Data flow + Evaluation
`data/test` → local screen → escalate below threshold → live call (or cache hit) → all 6 strategies on same test set → metrics: accuracy, per-class P/R/F1 (smoke recall highlighted), escalation rate, latency p50/p95, cost per 1k images, bootstrap 95% CIs (flag within-noise diffs). Baseline gate first: accuracy 0.82; fire P0.92/R0.81; no_fire P0.76/R0.98; smoke P0.84/R0.67 — report discrepancy before continuing. Outputs: `results/*.json`, `results/summary.md`, accuracy-vs-cost plot. One command reproduces `summary.md` from clean clone. README written last from scripts only; top results table; Prior-work section naming old repo (weights only) with setup steps; methodology; limitations; failure cases; attributions.

## Error handling + Testing
Docs-first: read dev.meta.ai / Meta Model API reference before Meta code; report base URL, Spark 1.3 + SAM 3.1 IDs, Glimmer hosting ID, image format, JSON support, limits, pricing. Fail-closed: `.h5` load failure reported (no silent retrain); synthetic never in test set; VLM parse failure → fallback/retry then abstain; budget guard halts. Tests with mocked client (no network): JSON parsing, metrics, cache, budget guard, threshold routing. Small commits; run tests as we go; stop-and-ask on budget/hard-rule conflicts. Phase 2 (SAM single-concept masks, overlays, mask-area trend) and Phase 3 add-on only after Phase 1 committed.