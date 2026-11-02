import json
import pytest
from src.vlm_client import VLMClient, image_mime, get_vlm_client

def _fake_response():
    class R:
        def raise_for_status(self):
            pass
        def json(self):
            return {"choices": [{"message": {"content": '{"label":"fire","confidence":0.9,"rationale":"flames"}'}}],
                    "usage": {"total_tokens": 1000000}}
    return R()

def test_budget_charges_successful_call(tmp_path, monkeypatch):
    img = tmp_path / "a.png"
    img.write_bytes(b"\x89PNG fake")
    c = VLMClient(model="muse-spark-1.3", base_url="https://api.meta.ai/v1", api_key="k", cache_dir=str(tmp_path / "c"), budget_usd=20)
    monkeypatch.setattr("httpx.post", lambda *a, **k: _fake_response())
    out = c.classify(str(img))
    assert out["label"] == "fire"
    # 1M tokens at 1.25/1M = 1.25 must be charged, not 0.0
    assert c._spent_usd == pytest.approx(1.25)
    assert out["cost_usd"] == pytest.approx(1.25)

def test_image_mime_by_format(tmp_path):
    assert image_mime("x.png") == "image/png"
    assert image_mime("x.jpg") == "image/jpeg"
    assert image_mime("x.webp") == "image/webp"

def test_concurrency_validation(tmp_path):
    with pytest.raises(ValueError, match="concurrency"):
        VLMClient(cache_dir=str(tmp_path), concurrency=0)

def test_modal_priority_over_meta(tmp_path, monkeypatch):
    monkeypatch.setenv("MODAL_API_KEY", "modal-key-123")
    monkeypatch.setenv("GLIMMER_BASE_URL", "https://example.modal.run/v1")
    monkeypatch.setenv("GLIMMER_MODEL", "muse-glimmer-30b")
    # Keep Meta/Fireworks placeholders present to prove Modal wins but they survive as fallback
    monkeypatch.setenv("META_API_KEY", "meta-placeholder")
    c = get_vlm_client()
    assert c.model == "muse-glimmer-30b"
    assert c.base_url == "https://example.modal.run/v1"
    assert c.api_key == "modal-key-123"
