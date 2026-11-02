# Summary (local-only baseline, real run)

N=405 split=test accuracy=0.854 95%CI [0.815,0.891]
latency p50=355ms p95=1045ms

| class | precision | recall | f1 |
|---|---|---|---|
| smoke | 0.824 | 0.783 | 0.803 | <-- weakest class
| fire | 0.904 | 0.851 | 0.877 |
| no_fire | 0.844 | 0.925 | 0.882 |

Expected (prior work): acc 0.82; fire P0.92 R0.81; no_fire P0.76 R0.98; smoke P0.84 R0.67.
VLM strategies (Glimmer/Spark) pending: VLMClient now requires META_API_BASE + key and refuses placeholder labels.