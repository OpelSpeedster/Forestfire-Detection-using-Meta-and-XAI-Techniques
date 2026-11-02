import json
import mimetypes
import os
import pathlib
import threading
import time

from src.cache import cache_key, image_hash
from src.env import is_placeholder, load_env

load_env()

PROMPT = 'Classify as {"label":"fire|smoke|no_fire","confidence":0-1,"rationale":"one sentence"} low temp.'

# Docs-first defaults (verify in dashboards before spending):
# Meta Model API base https://api.meta.ai/v1, recommended model muse-spark-1.3.
# Glimmer (open-weight) is served via Fireworks AI (OpenAI-compatible
# https://api.fireworks.ai/inference/v1); set GLIMMER_MODEL to the exact
# Fireworks model ID from your Fireworks dashboard/account.
META_API_BASE_DEFAULT = "https://api.meta.ai/v1"
FIREWORKS_BASE_DEFAULT = "https://api.fireworks.ai/inference/v1"


def spark_model() -> str:
    load_env()
    return os.environ.get("SPARK_MODEL", "muse-spark-1.3")


def glimmer_model() -> str:
    load_env()
    return os.environ.get("GLIMMER_MODEL", "muse-glimmer-30b")


def sam_model() -> str:
    load_env()
    return os.environ.get("SAM_MODEL", os.environ.get("SAM_WEIGHTS", "sam_b.pt"))


# Back-compat constants (prefer the functions above; these are read at import).
SPARK_MODEL_DEFAULT = spark_model()
GLIMMER_MODEL_DEFAULT = glimmer_model()
SAM_MODEL_DEFAULT = sam_model()

class BudgetExhausted(RuntimeError):
    pass

def parse_response(text: str) -> dict:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("No JSON object found in response")
    obj = json.loads(text[start : end + 1])
    label = obj.get("label")
    if label not in {"fire", "smoke", "no_fire"}:
        raise ValueError(f"Invalid label {label!r}; expected fire/smoke/no_fire")
    conf = obj.get("confidence")
    if not isinstance(conf, (int, float)) or not (0 <= float(conf) <= 1):
        raise ValueError(f"Invalid confidence {conf!r}; expected 0-1")
    if not obj.get("rationale"):
        obj["rationale"] = ""
    return obj

def image_mime(path: str) -> str:
    mime, _ = mimetypes.guess_type(path)
    if mime and mime.startswith("image/"):
        return mime
    ext = pathlib.Path(path).suffix.lower()
    return {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
            ".webp": "image/webp", ".bmp": "image/bmp", ".gif": "image/gif"}.get(ext, "image/jpeg")

def modal_api_key() -> str:
    # Modal console/API key: explicit key wins, else token id:secret pair.
    k = os.environ.get("MODAL_API_KEY", "").strip()
    if k:
        return k
    tid = os.environ.get("MODAL_TOKEN_ID", "").strip()
    tsec = os.environ.get("MODAL_TOKEN_SECRET", "").strip()
    if tid and tsec:
        return f"{tid}:{tsec}"
    return ""

class VLMClient:
    def __init__(self, model: str | None = None, base_url: str = "", api_key: str = "", cache_dir: str = "results/cache", budget_usd: float = 20, concurrency: int = 4):
        load_env()
        self.model = model or spark_model()
        self.base_url = base_url or os.environ.get("META_API_BASE", META_API_BASE_DEFAULT)
        raw_key = api_key or os.environ.get("META_API_KEY", "") or modal_api_key() or os.environ.get("FIREWORKS_API_KEY", "")
        self.api_key = "" if is_placeholder(raw_key) else raw_key
        self.cache_dir = pathlib.Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.budget_usd = float(budget_usd)
        self._spent_usd = 0.0
        if concurrency < 1:
            raise ValueError("concurrency must be >= 1")
        self.concurrency = concurrency
        self._semaphore = threading.Semaphore(concurrency)

    def _estimate_cost_usd(self, total_tokens: int) -> float:
        # USD per 1M tokens (override via VLM_USD_PER_1M_TOKENS); Spark docs cite ~1.25 input.
        rate = float(os.environ.get("VLM_USD_PER_1M_TOKENS", "1.25"))
        return (max(0, int(total_tokens)) / 1_000_000.0) * rate

    def _check_budget(self, est: float):
        if self._spent_usd + est > self.budget_usd:
            raise BudgetExhausted("budget guard halted spend")

    def _charge(self, cost_usd: float) -> None:
        self._spent_usd += float(cost_usd)

    def classify(self, image_path: str, use_cache: bool = True) -> dict:
        import base64
        import time as _time
        import httpx
        p = pathlib.Path(image_path)
        if not p.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        ih = image_hash(image_path)
        key = cache_key(self.model, ih, PROMPT)
        cached = self.cache_dir / f"{key}.json"
        if use_cache and cached.exists():
            return json.loads(cached.read_text())
        if not self.base_url or not self.api_key:
            raise RuntimeError(
                f"VLM '{self.model}' not configured (base_url/api_key missing). "
                "Set META_API_BASE + META_API_KEY (or FIREWORKS_API_KEY) in .env. "
                "Refusing to return a placeholder label so the demo cannot hallucinate."
            )
        self._check_budget(0.001)
        mime = image_mime(str(p))
        b64 = base64.b64encode(p.read_bytes()).decode()
        payload = {
            "model": self.model,
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
            ]}],
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        t0 = _time.time()
        last_err = None
        with self._semaphore:
            for attempt in range(3):
                try:
                    r = httpx.post(f"{self.base_url.rstrip('/')}/chat/completions", json=payload, headers=headers, timeout=60.0)
                    r.raise_for_status()
                    data = r.json()
                    text = data["choices"][0]["message"]["content"]
                    obj = parse_response(text)
                    latency_ms = int((_time.time() - t0) * 1000)
                    usage = data.get("usage", {})
                    tokens = int(usage.get("total_tokens", 0))
                    cost_usd = self._estimate_cost_usd(tokens)
                    self._charge(cost_usd)
                    result = {"label": obj["label"], "confidence": float(obj["confidence"]), "rationale": obj.get("rationale", ""),
                              "latency_ms": latency_ms, "tokens": tokens, "cost_usd": cost_usd}
                    cached.write_text(json.dumps(result))
                    return result
                except BudgetExhausted:
                    raise
                except Exception as e:
                    last_err = e
                    _time.sleep(2 ** attempt)
        raise RuntimeError(f"VLM call failed after 3 attempts: {last_err}")

def _budget() -> float:
    load_env()
    return float(os.environ.get("BUDGET_USD", "20"))


def get_spark_client() -> VLMClient:
    """Meta Spark via META_API_KEY on the Meta Model API. Fail-closed if unset."""
    load_env()
    key = os.environ.get("META_API_KEY", "")
    if is_placeholder(key):
        raise RuntimeError("Spark not configured: set a real META_API_KEY in .env")
    return VLMClient(model=spark_model(),
                     base_url=os.environ.get("META_API_BASE", META_API_BASE_DEFAULT),
                     api_key=key, budget_usd=_budget())


def get_glimmer_client() -> VLMClient:
    """Glimmer via Fireworks AI (primary); legacy Modal deployment still works as fallback."""
    load_env()
    fw_key = os.environ.get("FIREWORKS_API_KEY", "")
    url = os.environ.get("GLIMMER_BASE_URL", "").strip() or FIREWORKS_BASE_DEFAULT
    if not is_placeholder(fw_key) and "your-" not in url:
        return VLMClient(model=glimmer_model(), base_url=url, api_key=fw_key, budget_usd=_budget())
    # Legacy Modal-hosted Glimmer (modal_glimmer.py).
    mkey = modal_api_key()
    if url and "your-" not in url and not is_placeholder(mkey):
        return VLMClient(model=glimmer_model(), base_url=url, api_key=mkey, budget_usd=_budget())
    raise RuntimeError(
        "Glimmer not configured: set FIREWORKS_API_KEY + GLIMMER_MODEL "
        "(exact Fireworks model ID) in .env; GLIMMER_BASE_URL defaults to "
        "https://api.fireworks.ai/inference/v1.")


def get_client(tier: str) -> VLMClient:
    if tier == "spark":
        return get_spark_client()
    if tier == "glimmer":
        return get_glimmer_client()
    raise ValueError(f"Unknown tier {tier!r}; expected 'glimmer' or 'spark'")


def list_models() -> list:
    """Every model the demo/eval can reach, with live configured status.

    Replaces the old single-client factory that trapped callers between
    Glimmer and Spark: each tier is listed and constructed independently,
    so one unconfigured tier never blocks the others (SAM included).
    """
    load_env()
    out = []
    for tier, model, kind in (("glimmer", glimmer_model(), "chat"),
                              ("spark", spark_model(), "chat"),
                              ("sam", sam_model(), "segmentation")):
        try:
            if tier == "sam":
                from src.sam_client import SamClient
                ok = SamClient().configured()
            else:
                get_client(tier)
                ok = True
        except Exception as e:  # noqa: BLE001 - status listing must not raise
            out.append({"tier": tier, "model": model, "kind": kind,
                        "configured": False, "reason": str(e)})
        else:
            out.append({"tier": tier, "model": model, "kind": kind,
                        "configured": ok, "reason": "" if ok else "not configured"})
    return out


def get_vlm_client() -> VLMClient:
    """Back-compat escalation client: Glimmer on Modal first, Spark fallback."""
    try:
        return get_glimmer_client()
    except RuntimeError:
        return get_spark_client()
