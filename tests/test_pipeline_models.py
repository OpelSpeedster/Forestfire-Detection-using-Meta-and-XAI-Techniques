import pytest


class FakeVLM:
    def __init__(self, label, conf=0.9):
        self._label, self._conf = label, conf

    def classify(self, image_path, use_cache=True):
        return {"label": self._label, "confidence": self._conf,
                "latency_ms": 10, "cost_usd": 0.001}


def _pairs():
    return [("a.jpg", "smoke", "test"), ("b.jpg", "fire", "test")]


def test_all_six_strategies_present():
    import scripts.evaluate as ev
    local = lambda p: ("smoke", 0.9, {}) if p == "a.jpg" else ("fire", 0.9, {})
    out = ev.evaluate_strategies(_pairs(), glimmer=FakeVLM("smoke"),
                                 spark=FakeVLM("fire"), local_predict=local)
    assert set(out) == set(ev.STRATEGIES)
    assert all(v["status"] == "done" for v in out.values())
    assert out["spark-only"]["accuracy"] == 0.5


def test_unconfigured_tiers_pending_not_faked():
    import scripts.evaluate as ev
    out = ev.evaluate_strategies(_pairs(), glimmer=None, spark=None,
                                 local_predict=lambda p: ("smoke", 0.9, {}))
    assert out["local-only"]["status"] == "done"
    assert out["glimmer-only"]["status"] == "pending"
    assert "accuracy" not in out["spark-only"]


def test_model_list_covers_glimmer_spark_sam(monkeypatch):
    monkeypatch.setenv("META_API_KEY", "real-looking-key")
    monkeypatch.setenv("FIREWORKS_API_KEY", "fw-real-looking-key")
    monkeypatch.setenv("GLIMMER_BASE_URL", "https://api.fireworks.ai/inference/v1")
    monkeypatch.setenv("GLIMMER_MODEL", "accounts/fireworks/models/glimmer-test")
    from src.vlm_client import list_models
    models = {m["tier"]: m for m in list_models()}
    assert set(models) == {"glimmer", "spark", "sam"}
    assert models["spark"]["configured"] is True
    assert models["glimmer"]["configured"] is True  # Fireworks primary, no Modal needed


def test_glimmer_missing_key_stays_unconfigured(monkeypatch):
    monkeypatch.delenv("FIREWORKS_API_KEY", raising=False)
    monkeypatch.delenv("MODAL_API_KEY", raising=False)
    monkeypatch.delenv("MODAL_TOKEN_ID", raising=False)
    monkeypatch.delenv("MODAL_TOKEN_SECRET", raising=False)
    from src.vlm_client import get_glimmer_client
    with pytest.raises(RuntimeError, match="not configured"):
        get_glimmer_client()


def test_sam_missing_file_raises_before_weights(tmp_path):
    from src.sam_client import SamClient, sam_weights
    assert sam_weights().endswith(".pt")
    with pytest.raises(FileNotFoundError):
        SamClient().segment(str(tmp_path / "nope.jpg"))


def test_env_placeholders_rejected():
    from src.env import is_placeholder
    assert is_placeholder("your-meta-api-key")
    assert is_placeholder("")
    assert not is_placeholder("ak-real-key")


def test_demo_has_no_comparison_tab():
    import pathlib
    src = pathlib.Path("app/demo.py").read_text()
    assert "Comparison" not in src and "gr.Tab" not in src
