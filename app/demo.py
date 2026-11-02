import pathlib
import sys
import time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.cascade import should_escalate


def decide(image_path: str, use_cache: bool = True, threshold: float = 0.6):
    full = decide_full(image_path, use_cache=use_cache, threshold=threshold)
    return {"tier": full["tier"], "label": full["label"], "confidence": full["confidence"],
            "threshold": threshold, "use_cache": use_cache}


def decide_full(image_path: str, use_cache: bool = True, threshold: float = 0.6):
    if image_path is None:
        raise ValueError("No image provided")
    from src.classifier import predict
    t0 = time.time()
    local_label, local_conf, _probs = predict(image_path)
    local_ms = int((time.time() - t0) * 1000)
    trail = [{"tier": "local", "label": local_label, "confidence": local_conf, "latency_ms": local_ms}]
    if not should_escalate(local_conf, threshold):
        return {"tier": "local", "label": local_label, "confidence": local_conf,
                "latency_ms": local_ms, "cost_usd": 0.0, "trail": trail, "rationale": ""}
    from src.vlm_client import get_vlm_client
    vlm = get_vlm_client()
    try:
        out = vlm.classify(image_path, use_cache=use_cache)
    except TypeError:
        out = vlm.classify(image_path)
    label, conf = out["label"], float(out["confidence"])
    latency = local_ms + int(out.get("latency_ms", 0))
    cost = float(out.get("cost_usd", 0.0))
    trail.append({"tier": "glimmer", "label": label, "confidence": conf,
                  "latency_ms": int(out.get("latency_ms", 0))})
    tier = "glimmer"
    # Final tier: Spark only when Glimmer is itself uncertain.
    if should_escalate(conf, threshold):
        try:
            from src.vlm_client import get_spark_client
            spark_out = get_spark_client().classify(image_path, use_cache=use_cache)
            label, conf, tier = spark_out["label"], float(spark_out["confidence"]), "spark"
            latency += int(spark_out.get("latency_ms", 0))
            cost += float(spark_out.get("cost_usd", 0.0))
            trail.append({"tier": "spark", "label": label, "confidence": conf,
                          "latency_ms": int(spark_out.get("latency_ms", 0))})
            out = spark_out
        except Exception as e:  # Glimmer result stands; trail records why Spark did not decide.
            trail.append({"tier": "spark", "label": "unavailable", "confidence": 0.0,
                          "latency_ms": 0, "reason": str(e)})
    return {"tier": tier, "label": label, "confidence": conf, "latency_ms": latency,
            "cost_usd": cost, "trail": trail, "rationale": out.get("rationale", "")}


CSS = """
body, .gradio-container { background:#0f141a; color:#e8eef2; font-family:'Segoe UI',system-ui,sans-serif; }
.gradio-container { max-width:1180px !important; margin:0 auto; }
h1 { font-size:28px; letter-spacing:-.02em; margin-bottom:2px; }
.mono, .mono * { font-family:Consolas,'Cascadia Mono',monospace !important; }
.verdict { border:1px solid #2a3642; background:#171e26; padding:16px 18px; border-radius:6px; }
.verdict b { color:#ff6b35; }
button.primary { background:#ff6b35 !important; color:#14100d !important; font-weight:700; border-radius:6px !important; }
footer { display:none !important; }
@media (prefers-reduced-motion: reduce) { * { transition:none !important; animation:none !important; } }
"""


def _status_html():
    try:
        from src.vlm_client import list_models
        rows = "".join(
            f"<tr><td>{m['tier']}</td><td class='mono'>{m['model']}</td>"
            f"<td>{'ready' if m['configured'] else 'not configured'}</td><td>{m['reason']}</td></tr>"
            for m in list_models())
        return f"<table class='mono' style='width:100%'><tr><th>Tier</th><th>Model</th><th>Status</th><th></th></tr>{rows}</table>"
    except Exception as e:  # noqa: BLE001
        return f"Model status unavailable: {e}"


def _verdict_html(r):
    if not r:
        return "<div class='verdict'>Upload an image to run the cascade.</div>"
    return (f"<div class='verdict'>Verdict: <b>{r['label']}</b> &middot; decided by <b>{r['tier']}</b> "
            f"&middot; confidence {r['confidence']:.2f} &middot; {r['latency_ms']} ms "
            f"&middot; ${r['cost_usd']:.4f}<br>{r.get('rationale', '')}</div>")


def launch():
    import gradio as gr
    theme = gr.themes.Base(primary_hue="orange", neutral_hue="slate")
    with gr.Blocks(title="FireNet Cascade", theme=theme, css=CSS) as demo:
        gr.Markdown("# FireNet Cascade")
        gr.Markdown("Local FireNetCNN screens every image. Uncertain cases escalate to Glimmer via Fireworks, then to Spark via the Meta API. A human confirms any alert — nothing is automated.")
        gr.HTML(_status_html())
        with gr.Row():
            with gr.Column(scale=3):
                img = gr.Image(type="filepath", label="Image — upload or webcam", sources=["upload", "webcam"])
                mode = gr.Radio(["live", "cached"], value="cached", label="Mode (cached is labelled; live makes real model calls)")
                threshold = gr.Slider(0, 1, value=0.6, step=0.05, label="Escalation threshold")
                sam = gr.Checkbox(label="SAM evidence on fire/smoke (local Ultralytics model, downloaded once)", value=False)
                run = gr.Button("Run cascade", variant="primary", elem_classes=["primary"])
            with gr.Column(scale=2):
                verdict = gr.HTML(_verdict_html(None))
                detail = gr.JSON(label="Tier trail — tier, label, confidence, latency, cost", elem_classes=["mono"])
                overlay = gr.Image(label="SAM overlay (local Ultralytics segmentation)", interactive=False)
                confirm_out = gr.Textbox(label="Human review", interactive=False)
                confirm = gr.Button("Human confirms alert — no automated action")
        def _run(p, m, t, s):
            r = decide_full(p, use_cache=(m == "cached"), threshold=t)
            ov = None
            if s and r["label"] in {"fire", "smoke"}:
                from src.sam_client import SamClient
                seg = SamClient().segment(p)
                r["sam"] = seg
                ov = seg["overlay"]
            return _verdict_html(r), r, ov
        run.click(_run, [img, mode, threshold, sam], [verdict, detail, overlay])
        confirm.click(lambda: "Confirmed by human — no automated action taken.", outputs=confirm_out)
    demo.launch()


if __name__ == "__main__":
    launch()
