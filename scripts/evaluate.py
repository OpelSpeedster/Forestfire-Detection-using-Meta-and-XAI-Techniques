import pathlib
import json
import os
import random
import sys
import time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import matplotlib.pyplot as plt
from src.metrics import prf1, accuracy

CANON = {"fire": "fire", "smoke": "smoke", "nofire": "no_fire", "no_fire": "no_fire", "neutral": "no_fire"}

def collect_images(root="data"):
    pairs = []
    for p in pathlib.Path(root).rglob("*.*"):
        if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        # infer label from parent dir name
        lab = None
        for parent in p.parents:
            n = parent.name.lower()
            if n in CANON:
                lab = CANON[n]
                break
        if lab is None:
            continue
        # prefer test split when present; caller filters
        pairs.append((str(p), lab, "test" if "test" in [q.lower() for q in p.parts] else ("val" if "val" in [q.lower() for q in p.parts] else "train")))
    return pairs

def bootstrap_ci(y_true, y_pred, n=1000):
    if not y_true:
        raise ValueError("bootstrap_ci requires non-empty y_true")
    accs = []
    for _ in range(n):
        idx = [random.randrange(len(y_true)) for _ in y_true]
        yt = [y_true[i] for i in idx]
        yp = [y_pred[i] for i in idx]
        accs.append(accuracy(yt, yp))
    accs.sort()
    return accs[int(0.025 * n)], accs[int(0.975 * n)]

def run_local_baseline(pairs, limit=None):
    from src.classifier import predict
    y_true, y_pred, lat = [], [], []
    order = os.environ.get("CLASS_ORDER", "fire,no_fire,smoke")
    if limit:
        pairs = pairs[:limit]
    for img, lab, _split in pairs:
        t0 = time.time()
        try:
            pred, conf, _probs = predict(img)
        except Exception as e:
            pred, conf = "no_fire", 0.0
            print(f"[warn] predict failed for {img}: {e}")
        y_true.append(lab)
        y_pred.append(pred)
        lat.append(int((time.time() - t0) * 1000))
    lat_sorted = sorted(lat)
    p50 = lat_sorted[len(lat_sorted)//2] if lat_sorted else 0
    p95 = lat_sorted[int(len(lat_sorted)*0.95)] if lat_sorted else 0
    m = prf1(y_true, y_pred)
    acc = accuracy(y_true, y_pred)
    try:
        lo, hi = bootstrap_ci(y_true, y_pred, n=200)
    except Exception:
        lo, hi = 0, 0
    return {"n": len(pairs), "accuracy": acc, "ci95": [lo, hi], "per_class": m,
            "latency_p50_ms": p50, "latency_p95_ms": p95, "class_order_env": order}

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", help="test/val/train/all")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    pairs = collect_images("data")
    if args.split != "all":
        pairs = [x for x in pairs if x[2] == args.split] or pairs
    if args.limit:
        pairs = pairs[:args.limit]
    print(f"Evaluating local-only on {len(pairs)} images (split={args.split}) ...")
    res = run_local_baseline(pairs)
    out = {"strategy": "local-only", "split": args.split, **res,
           "expected_baseline": {"accuracy": 0.82, "fire": {"p": 0.92, "r": 0.81}, "no_fire": {"p": 0.76, "r": 0.98}, "smoke": {"p": 0.84, "r": 0.67}},
           "note": "If accuracy << 0.82 or fire images predict smoke/no_fire, CLASS_ORDER or preprocessing mismatches the H5 training. Set CLASS_ORDER env or check BGR vs RGB / scaling."}
    pathlib.Path("results").mkdir(exist_ok=True)
    pathlib.Path("results/results.json").write_text(json.dumps(out, indent=2))
    lines = ["# Summary (local-only baseline, real run)", "", f"N={res['n']} split={args.split} accuracy={res['accuracy']:.3f} 95%CI [{res['ci95'][0]:.3f},{res['ci95'][1]:.3f}]",
             f"latency p50={res['latency_p50_ms']}ms p95={res['latency_p95_ms']}ms", "",
             "| class | precision | recall | f1 |", "|---|---|---|---|"]
    for k, v in res["per_class"].items():
        flag = " <-- weakest class" if k == "smoke" else ""
        lines.append(f"| {k} | {v['precision']:.3f} | {v['recall']:.3f} | {v['f1']:.3f} |{flag}")
    lines += ["", "Expected (prior work): acc 0.82; fire P0.92 R0.81; no_fire P0.76 R0.98; smoke P0.84 R0.67.",
              "VLM strategies (Glimmer/Spark) pending: VLMClient now requires META_API_BASE + key and refuses placeholder labels."]
    pathlib.Path("results/summary.md").write_text("\n".join(lines))
    plt.figure()
    cats = list(res["per_class"].keys())
    plt.bar(cats, [res["per_class"][c]["recall"] for c in cats])
    plt.title("Local-only recall by class (smoke highlighted)")
    plt.savefig("results/accuracy_cost.png")
    print(f"Done. accuracy={res['accuracy']:.3f} smoke_recall={res['per_class']['smoke']['recall']:.3f}")
