import pathlib
import json
import os
import random
import sys
import time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.metrics import prf1, accuracy

CANON = {"fire": "fire", "smoke": "smoke", "nofire": "no_fire", "no_fire": "no_fire", "neutral": "no_fire"}
THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9]
STRATEGIES = ["local-only", "glimmer-only", "spark-only",
              "local->glimmer", "local->spark", "local->glimmer->spark"]


def collect_images(root="data"):
    pairs = []
    for p in pathlib.Path(root).rglob("*.*"):
        if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        lab = None
        for parent in p.parents:
            n = parent.name.lower()
            if n in CANON:
                lab = CANON[n]
                break
        if lab is None:
            continue
        pairs.append((str(p), lab, "test" if "test" in [q.lower() for q in p.parts] else ("val" if "val" in [q.lower() for q in p.parts] else "train")))
    return pairs


def bootstrap_ci(y_true, y_pred, n=1000):
    if not y_true:
        raise ValueError("bootstrap_ci requires non-empty y_true")
    accs = []
    for _ in range(n):
        idx = [random.randrange(len(y_true)) for _ in y_true]
        accs.append(accuracy([y_true[i] for i in idx], [y_pred[i] for i in idx]))
    accs.sort()
    return accs[min(n - 1, int(0.025 * n))], accs[min(n - 1, int(0.975 * n))]


def _p50_p95(lat):
    if not lat:
        return 0, 0
    s = sorted(lat)
    return s[len(s) // 2], s[min(len(s) - 1, int(len(s) * 0.95))]


def _summarize(y_true, y_pred, lat, costs, escalated, n_boot=200):
    lo, hi = bootstrap_ci(y_true, y_pred, n=n_boot)
    p50, p95 = _p50_p95(lat)
    total_cost = sum(costs)
    return {"n": len(y_true), "accuracy": accuracy(y_true, y_pred), "ci95": [lo, hi],
            "per_class": prf1(y_true, y_pred), "latency_p50_ms": p50, "latency_p95_ms": p95,
            "escalation_rate": (escalated / len(y_true)) if y_true else 0.0,
            "cost_per_1000_usd": (total_cost / len(y_true) * 1000) if y_true else 0.0}


def run_local_baseline(pairs, limit=None):
    from src.classifier import predict
    if not pairs:
        raise ValueError("No images to evaluate")
    if limit:
        pairs = pairs[:limit]
    if not pairs:
        raise ValueError("No images to evaluate after limit")
    y_true, y_pred, lat = [], [], []
    errors = []
    for img, lab, _split in pairs:
        t0 = time.time()
        try:
            pred, conf, _probs = predict(img)
        except Exception as e:
            errors.append((img, f"{type(e).__name__}: {e}"))
            continue
        y_true.append(lab)
        y_pred.append(pred)
        lat.append(int((time.time() - t0) * 1000))
    if errors:
        sample = "; ".join(f"{i}: {e}" for i, e in errors[:5])
        raise RuntimeError(f"{len(errors)} prediction(s) failed; metrics not computed from fake labels. First failures: {sample}")
    res = _summarize(y_true, y_pred, lat, [0.0] * len(lat), 0)
    res["class_order_env"] = os.environ.get("CLASS_ORDER", "fire,no_fire,smoke")
    return res


def _vlm_call(client, img, use_cache=True):
    out = client.classify(img, use_cache=use_cache) if _accepts_cache(client) else client.classify(img)
    return out["label"], float(out["confidence"]), int(out.get("latency_ms", 0)), float(out.get("cost_usd", 0.0))


def _accepts_cache(client):
    import inspect
    try:
        return "use_cache" in inspect.signature(client.classify).parameters
    except (TypeError, ValueError):
        return False


def evaluate_strategies(pairs, threshold=0.6, glimmer=None, spark=None, local_predict=None):
    """All 6 strategies on the same pairs. Unavailable tiers -> pending, never faked."""
    if not pairs:
        raise ValueError("No images to evaluate")
    if local_predict is None:
        from src.classifier import predict as local_predict
    # Run local once; reuse for every cascade.
    local_rows, errors = [], []
    for img, lab, _ in pairs:
        t0 = time.time()
        try:
            pred, conf, _ = local_predict(img)
        except Exception as e:
            errors.append(f"{img}: {type(e).__name__}: {e}")
            continue
        local_rows.append((img, lab, pred, float(conf), int((time.time() - t0) * 1000)))
    if errors:
        raise RuntimeError(f"{len(errors)} prediction(s) failed; first: {errors[0]}")
    results = {}
    yt = [r[1] for r in local_rows]
    results["local-only"] = {"status": "done", **_summarize(
        yt, [r[2] for r in local_rows], [r[4] for r in local_rows], [0.0] * len(local_rows), 0)}

    def vlm_only(name, client):
        if client is None:
            results[name] = {"status": "pending", "reason": "tier not configured"}
            return
        yp, lat, costs = [], [], []
        for img, lab, *_ in local_rows:
            label, _c, ms, cost = _vlm_call(client, img)
            yp.append(label); lat.append(ms); costs.append(cost)
        results[name] = {"status": "done", **_summarize(yt, yp, lat, costs, len(yp))}

    def cascade(name, first, second=None):
        if first is None:
            results[name] = {"status": "pending", "reason": "tier not configured"}
            return
        yp, lat, costs, esc = [], [], [], 0
        for img, lab, pred, conf, lms in local_rows:
            label, ms, cost = pred, lms, 0.0
            if conf < threshold:
                esc += 1
                label, c2, ms, cost = _vlm_call(first, img)
                if second is not None and c2 < threshold:
                    label, _c3, ms2, cost2 = _vlm_call(second, img)
                    ms += ms2; cost += cost2
            yp.append(label); lat.append(ms); costs.append(cost)
        results[name] = {"status": "done", **_summarize(yt, yp, lat, costs, esc)}

    vlm_only("glimmer-only", glimmer)
    vlm_only("spark-only", spark)
    cascade("local->glimmer", glimmer)
    cascade("local->spark", spark)
    cascade("local->glimmer->spark", glimmer, spark)
    return results


def threshold_sweep(pairs, glimmer=None, spark=None, local_predict=None):
    return {str(t): evaluate_strategies(pairs, threshold=t, glimmer=glimmer,
                                        spark=spark, local_predict=local_predict)
            for t in THRESHOLDS}


def _client_or_none(tier):
    from src.vlm_client import get_client
    try:
        return get_client(tier), ""
    except Exception as e:  # noqa: BLE001 - recorded as pending, never faked
        return None, str(e)


def write_outputs(strategies, sweep, split, path_json="results/results.json",
                  path_md="results/summary.md", path_png="results/accuracy_cost.png"):
    pathlib.Path("results").mkdir(exist_ok=True)
    local = strategies.get("local-only", {})
    out = {"split": split, "threshold": 0.6, "strategies": strategies,
           "threshold_sweep": {t: {k: v.get("accuracy") for k, v in s.items() if v.get("status") == "done"}
                               for t, s in sweep.items()},
           # Back-compat top-level local fields for existing readers.
           "strategy": "local-only", "n": local.get("n"), "accuracy": local.get("accuracy"),
           "ci95": local.get("ci95"), "per_class": local.get("per_class"),
           "latency_p50_ms": local.get("latency_p50_ms"), "latency_p95_ms": local.get("latency_p95_ms"),
           "expected_baseline": {"accuracy": 0.82, "fire": {"p": 0.92, "r": 0.81},
                                 "no_fire": {"p": 0.76, "r": 0.98}, "smoke": {"p": 0.84, "r": 0.67}}}
    pathlib.Path(path_json).write_text(json.dumps(out, indent=2))
    lines = ["# FireNet Cascade — results", "",
             f"split={split} threshold=0.6 (sweep {THRESHOLDS})", "",
             "| strategy | status | accuracy | smoke recall | esc. rate | p50 ms | cost/1k USD |",
             "|---|---|---|---|---|---|---|"]
    for name in STRATEGIES:
        s = strategies.get(name, {})
        if s.get("status") == "done":
            lines.append(f"| {name} | done | {s['accuracy']:.3f} CI[{s['ci95'][0]:.3f},{s['ci95'][1]:.3f}] "
                         f"| {s['per_class']['smoke']['recall']:.3f} | {s['escalation_rate']:.2f} "
                         f"| {s['latency_p50_ms']} | {s['cost_per_1000_usd']:.2f} |")
        else:
            lines.append(f"| {name} | pending | — | — | — | — | — | {s.get('reason', '')} |")
    if local.get("status") == "done":
        lines += ["", "Local per-class (smoke is the weakest class):", "",
                  "| class | precision | recall | f1 |", "|---|---|---|---|"]
        for k, v in local["per_class"].items():
            lines.append(f"| {k} | {v['precision']:.3f} | {v['recall']:.3f} | {v['f1']:.3f} |")
    lines += ["", "Differences within bootstrap CIs on this small test set are within noise.",
              "VLM rows stay 'pending' until that tier validates live; no placeholder labels are counted."]
    pathlib.Path(path_md).write_text("\n".join(lines))
    done = {k: v for k, v in strategies.items() if v.get("status") == "done"}
    if done:
        plt.figure()
        plt.bar(list(done), [v["accuracy"] for v in done.values()])
        plt.ylabel("accuracy"); plt.title("Accuracy by strategy (same test set)")
        plt.xticks(rotation=20, ha="right"); plt.tight_layout()
        plt.savefig(path_png)
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", help="test/val/train/all")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=0.6)
    args = ap.parse_args()
    pairs = collect_images("data")
    if args.split != "all":
        filtered = [x for x in pairs if x[2] == args.split]
        if not filtered:
            raise SystemExit(f"No images found for split '{args.split}' under data/. Refusing to evaluate another split.")
        pairs = filtered
    if args.limit:
        pairs = pairs[:args.limit]
    glimmer, g_reason = _client_or_none("glimmer")
    spark, s_reason = _client_or_none("spark")
    for tier, reason in (("glimmer", g_reason), ("spark", s_reason)):
        if reason:
            print(f"[{tier}] pending: {reason}")
    print(f"Evaluating {len(STRATEGIES)} strategies on {len(pairs)} images (split={args.split}) ...")
    strategies = evaluate_strategies(pairs, threshold=args.threshold, glimmer=glimmer, spark=spark)
    sweep = threshold_sweep(pairs, glimmer=glimmer, spark=spark)
    write_outputs(strategies, sweep, args.split)
    loc = strategies["local-only"]
    print(f"Done. local accuracy={loc['accuracy']:.3f} smoke_recall={loc['per_class']['smoke']['recall']:.3f}; "
          f"see results/summary.md for all strategies.")

