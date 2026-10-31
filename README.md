# FireNet Cascade — Tiered wildfire/smoke detector

**Status: Phase 1 scaffold — results pending real run. No numbers below are measured; `results/summary.md` will be the source of truth after `scripts/evaluate.py` runs.**

## Results (from `results/summary.md` — replace this table after real evaluation)
| Strategy | Accuracy | smoke R | Cost/1k | Escalation |
|---|---:|---:|---:|---:|
| local | — | — | — | — |
| glimmer-only | — | — | — | — |
| spark-only | — | — | — | — |
| local→glimmer | — | — | — | — |
| local→spark | — | — | — | — |
| local→glimmer→spark | — | — | — | — |

## Description
Cheap local 224x224 RGB classifier screens every image. Only uncertain cases escalate live to Meta models: Glimmer on Modal first, then Spark closed API. Measured open vs closed comparison on the weakest class (smoke) — accuracy, per-class recall, cost, latency.

## Prior work
Old repo `github.com/OpelSpeedster/Forest-Fire-Detection-Using-FirenetCNN-and-XAI-Techniques` contributed only the trained weights `FirenetCNN1.h5`. This project is fresh (`git init`), all code new in this session. Weights are fetched at setup into gitignored `external/`; path override `FIRENET_WEIGHTS`.

## Run instructions (clean clone, CPU-only, uv, .venv in project)

```bash
uv venv
uv pip install -r requirements.txt
cp .env.example .env   # then fill META_API_KEY, MODAL_TOKEN_ID/SECRET, FIREWORKS_API_KEY
uv run python scripts/setup.py        # fetches weights to external/
# place test images as data/test/{smoke,fire,no_fire}
uv run python scripts/evaluate.py     # writes results/results.json, results/summary.md, results/accuracy_cost.png
uv run python app/demo.py             # Gradio at http://127.0.0.1:7860 (live + labeled cached, human-confirm button)
uv run pytest -v                      # mocked suite, no network
```

One-command reproduce (after setup):
```bash
uv run python scripts/evaluate.py && ls results/summary.md
```

Verify from clean clone:
```bash
rm -rf .venv && uv venv && uv pip install -r requirements.txt && uv run pytest && uv run python scripts/evaluate.py
```

## Methodology
Same ~405-image test set for all 6 strategies; threshold sweep on local confidence; smoke recall highlighted; bootstrap 95% CIs on small test set (flag within-noise diffs). Baseline gate first vs 0.82 / fire P0.92 R0.81 / no_fire P0.76 R0.98 / smoke P0.84 R0.67 — report discrepancy before continuing.

## Limitations
Small test set; Kaggle-sourced data provenance pending exact page link; VLM non-determinism (low temp + caching); cost estimates from live pricing at run time; CPU-only local inference.

## Failure cases
VLM strict-JSON parse failures → fallback/retry then abstain; budget guard halts spend at $20; missing weights or corrupt image → error reported, no silent retrain; synthetic images never enter test set.

## Third-party attributions
TensorFlow/Keras, Pillow, Gradio, httpx, Modal, Fireworks, Meta API.

## Docs-first
Before final Meta API code, verify base URL, model IDs for Muse Spark 1.3 and SAM 3.1, image format, JSON support, limits, pricing from dev.meta.ai / Model API reference, and Glimmer hosting ID.
