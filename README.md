# FireNet Cascade — Tiered wildfire/smoke detector

Local 224x224 RGB classifier screens every image; uncertain cases escalate live to Glimmer via Fireworks, then Spark closed API. SAM evidence comes from a local Ultralytics model. Headline so far is an honest local-only baseline on the real 405-image test split; the open-vs-closed comparison on smoke runs once API creds are set.

## Results (from `results/summary.md`, produced by `scripts/evaluate.py`)

Local-only, split=test, N=405, accuracy=0.854 95% CI [0.822,0.889], latency p50=246ms p95=742ms.

| class | precision | recall | f1 |
|---|---|---|---|
| smoke | 0.824 | 0.783 | 0.803 |
| fire | 0.904 | 0.851 | 0.877 |
| no_fire | 0.844 | 0.925 | 0.882 |

Prior work reported acc 0.82; fire P0.92/R0.81; no_fire P0.76/R0.98; smoke P0.84/R0.67. We report the gap honestly: accuracy slightly above, smoke recall below the old smoke precision claim. VLM strategies (Glimmer-only, Spark-only, cascades) are pending live creds — `VLMClient` refuses placeholder labels instead of inventing them.

## Prior work

Old repo `github.com/OpelSpeedster/Forest-Fire-Detection-Using-FirenetCNN-and-XAI-Techniques` contributed only the trained weights `FirenetCNN1.h5`. This repo is fresh; all code is new. Weights are fetched at setup into gitignored `external/`; override with `FIRENET_WEIGHTS`. Verified H5 output order is `[fire, no_fire, smoke]` (sweep: 0.887 vs 0.020 for the old order); preprocessing is RGB, 224x224, scale `/255`.

## Run instructions (Windows, CPU-only, `.venv` in project)

```bat
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
copy .env.example .env
REM fill META_API_KEY, SPARK_MODEL, FIREWORKS_API_KEY, GLIMMER_MODEL (exact Fireworks model ID)
REM optional overrides: GLIMMER_BASE_URL, META_API_BASE, SAM_WEIGHTS, SAM_WEIGHTS_DIR
.venv\Scripts\python scripts/setup.py
REM dataset lives at data/forestfire-classifier-dataset/test/{fire,nofire,smoke} (nofire maps to no_fire)
REM SAM weights (Ultralytics, e.g. sam_b.pt) download once into external/sam on first SAM use
.venv\Scripts\python -m pytest -v
.venv\Scripts\python scripts/evaluate.py --split test
.venv\Scripts\python -m app.demo
```

One-command reproduce (after setup): `.venv\Scripts\python scripts/evaluate.py --split test`

## Methodology

Same test set for every strategy; local confidence threshold sweep; smoke recall highlighted; bootstrap 95% CIs (small test set, flag within-noise diffs). `VLMClient` uses OpenAI-compatible `POST {base}/chat/completions` (base64 image, temperature 0.1, JSON object), retries with backoff, concurrency limit, disk cache by model+image-hash+prompt, $20 budget guard. Docs-first pins: Meta base `https://api.meta.ai/v1`, `muse-spark-1.3`; Glimmer via Fireworks `https://api.fireworks.ai/inference/v1` (not Meta API); SAM local via Ultralytics (`SAM_WEIGHTS=sam_b.pt`) — verify pricing/limits in the dashboards before spending.

## Limitations

Small test set (405); Kaggle-sourced data provenance needs exact page link; VLM non-determinism (low temp + caching); cost estimates from live pricing; CPU-only local inference (p50 246ms).

## Failure cases

Missing weights/corrupt image → clear error, no silent retrain; VLM unconfigured → refusal, never a placeholder label; parse failure → retry then abstain; budget guard halts spend; synthetic images never enter the test set.

## Third-party attributions

TensorFlow 2.20 / Keras 3.12, Pillow, Gradio, httpx, Ultralytics (local SAM), Fireworks (Glimmer), Meta Model API (Spark), NASA FIRMS / Open-Meteo only if Phase 3 add-on is chosen.
