import json
import os
import pathlib
import time
import hashlib

from src.cache import cache_key, image_hash

PROMPT = 'Classify as {"label":"fire|smoke|no_fire","confidence":0-1,"rationale":"one sentence"} low temp.'

# Docs-first defaults (verify in dev.meta.ai dashboard before spending):
# Meta Model API base https://api.meta.ai/v1, recommended model muse-spark-1.3
# (also serves 1.2/1.1; Responses API POST /v1/responses is Meta's default, Chat Completions also served).
# Glimmer 30B (Apache 2.0 open-weight) is NOT on Meta API; use Modal/Fireworks/Baseten deployment.
# SAM 3.1 model id sam-3.1 via media-segmentation (segmentImage). Pricing/limits unconfirmed — check dashboard.
META_API_BASE_DEFAULT = "https://api.meta.ai/v1"
SPARK_MODEL_DEFAULT = os.environ.get("SPARK_MODEL", "muse-spark-1.3")
GLIMMER_MODEL_DEFAULT = os.environ.get("GLIMMER_MODEL", "muse-glimmer-30b")
SAM_MODEL_DEFAULT = os.environ.get("SAM_MODEL", "sam-3.1")

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

class VLMClient:
    def __init__(self, model: str | None = None, base_url: str = "", api_key: str = "", cache_dir: str = "results/cache", budget_usd: float = 20, concurrency: int = 4):
        self.model = model or SPARK_MODEL_DEFAULT
        self.base_url = base_url or os.environ.get("META_API_BASE", META_API_BASE_DEFAULT)
        self.api_key = api_key or os.environ.get("META_API_KEY", "") or os.environ.get("FIREWORKS_API_KEY", "")
        self.cache_dir = pathlib.Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.budget_usd = float(budget_usd)
        self._spent_usd = 0.0
        self.concurrency = concurrency

    def _check_budget(self, est: float):
        if self._spent_usd + est > self.budget_usd:
            raise BudgetExhausted("budget guard halted spend")

    def classify(self, image_path: str) -> dict:
        import base64
        import time as _time
        import httpx
        p = pathlib.Path(image_path)
        if not p.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        ih = image_hash(image_path)
        key = cache_key(self.model, ih, PROMPT)
        cached = self.cache_dir / f"{key}.json"
        if cached.exists():
            return json.loads(cached.read_text())
        if not self.base_url or not self.api_key:
            raise RuntimeError(
                f"VLM '{self.model}' not configured (base_url/api_key missing). "
                "Set META_API_BASE + META_API_KEY (or FIREWORKS_API_KEY) in .env. "
                "Refusing to return a placeholder label so the demo cannot hallucinate."
            )
        self._check_budget(0.001)
        b64 = base64.b64encode(p.read_bytes()).decode()
        payload = {
            "model": self.model,
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ]}],
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        t0 = _time.time()
        last_err = None
        for attempt in range(3):
            try:
                r = httpx.post(f"{self.base_url.rstrip('/')}/chat/completions", json=payload, headers=headers, timeout=60.0)
                r.raise_for_status()
                data = r.json()
                text = data["choices"][0]["message"]["content"]
                obj = parse_response(text)
                latency_ms = int((_time.time() - t0) * 1000)
                usage = data.get("usage", {})
                result = {"label": obj["label"], "confidence": float(obj["confidence"]), "rationale": obj.get("rationale", ""),
                          "latency_ms": latency_ms, "tokens": int(usage.get("total_tokens", 0)), "cost_usd": 0.0}
                cached.write_text(json.dumps(result))
                return result
            except Exception as e:
                last_err = e
                _time.sleep(2 ** attempt)
        raise RuntimeError(f"VLM call failed after 3 attempts: {last_err}")
