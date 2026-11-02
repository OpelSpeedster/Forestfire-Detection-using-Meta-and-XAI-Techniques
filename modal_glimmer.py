"""LEGACY Modal deployment: Muse Glimmer 30B as an OpenAI-compatible endpoint.

Superseded: Glimmer now runs via Fireworks AI (FIREWORKS_API_KEY in .env).
This file is kept only for the Modal fallback path in get_glimmer_client().

Serves HF checkpoint ``meta-models/Muse-Glimmer-30B`` (Apache 2.0) via
vLLM's own OpenAI-compatible server behind Modal ``web_server``.

Runs on Modal's free-tier GPU by default (T4, no payment method needed).
`modal` ships with the `modal` package in requirements.txt; inside this
project's venv the commands are:
    .venv\Scripts\modal setup            # paste MODAL token once
    .venv\Scripts\modal deploy modal_glimmer.py   # prints the https endpoint
    .venv\Scripts\modal run modal_glimmer.py::check

GPU choice via MODAL_GPU env (default T4, free tier). The full 30B
checkpoint needs ~60GB+ and does NOT fit on a T4 (16GB): with the
default, set HF_MODEL_ID to a smaller Glimmer checkpoint (or a
quantized AWQ build) so it fits. Only set MODAL_GPU=A100-80GB if
your workspace has payment verified and you want the full 30B.

Then in .env (Modal key stays primary, Meta/Fireworks stay fallback):
    MODAL_API_KEY=<your key>             # or MODAL_TOKEN_ID / MODAL_TOKEN_SECRET
    GLIMMER_BASE_URL=https://<your-app>--<workspace>.modal.run/v1
    GLIMMER_MODEL=muse-glimmer-30b

Verify without spending on eval:
    python -c "from src.vlm_client import get_vlm_client; c=get_vlm_client(); print(c.model, c.base_url)"
    curl $GLIMMER_BASE_URL/models -H "Authorization: Bearer $MODAL_API_KEY"
"""

import os

import modal

APP_NAME = "firenet-glimmer"
HF_MODEL_ID = os.environ.get("HF_MODEL_ID", "meta-models/Muse-Glimmer-30B")
SERVED_NAME = os.environ.get("GLIMMER_MODEL", "muse-glimmer-30b")
# Free-tier default: T4 needs no payment method. Override with
# MODAL_GPU=A100-80GB (payment-verified workspaces) for the full 30B.
GPU = os.environ.get("MODAL_GPU", "T4")
VLLM_PORT = 8000
SMALL_GPU = GPU.strip().upper() in {"T4", "L4"}

image = (
    modal.Image.from_registry("vllm/vllm-openai:v0.11.0", add_python="3.11")
    .env({"HF_HUB_OFFLINE": "0"})
)

app = modal.App(APP_NAME)


# If the checkpoint ever gates, create a Modal secret named "huggingface-secret"
# holding HF_TOKEN and add: secrets=[modal.Secret.from_name("huggingface-secret")],
@app.function(
    image=image,
    gpu=GPU,
    timeout=3600,
    scaledown_window=300,
)
@modal.web_server(port=VLLM_PORT, startup_timeout=30 * 60)
def serve():
    """Boot vLLM's OpenAI server: POST /v1/chat/completions, GET /v1/models."""
    import subprocess
    import sys

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", HF_MODEL_ID,
        "--served-model-name", SERVED_NAME,
        "--host", "0.0.0.0",
        "--port", str(VLLM_PORT),
        "--trust-remote-code",
        "--gpu-memory-utilization", "0.9",
    ]
    if SMALL_GPU:
        # Keep a small/free GPU from OOMing on long contexts; the full
        # 30B checkpoint still needs a big GPU — pair T4 with a small
        # or quantized HF_MODEL_ID rather than the full 30B weights.
        cmd += ["--max-model-len", "2048", "--enforce-eager"]
    # Vision input for Glimmer: vLLM enables it from the model's config;
    # if image_url is rejected, check the container logs for the supported modality flag.
    subprocess.Popen(cmd).wait()


@app.local_entrypoint()
def check():
    """Smoke-test helper (runs locally, hits the deployed URL from env)."""
    import sys
    import pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from src.env import load_env
    load_env()
    import httpx

    base = os.environ.get("GLIMMER_BASE_URL", "").rstrip("/")
    key = os.environ.get("MODAL_API_KEY", "")
    if not base or not key or "your-modal" in base:
        print("Set GLIMMER_BASE_URL + MODAL_API_KEY in .env first.")
        raise SystemExit(2)
    r = httpx.get(f"{base}/models", headers={"Authorization": f"Bearer {key}"}, timeout=30.0)
    r.raise_for_status()
    print("models:", [m.get("id") for m in r.json().get("data", [])])
