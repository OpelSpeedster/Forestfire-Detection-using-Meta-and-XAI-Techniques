import pytest
import app.demo as demo

class FakeVLM:
    def __init__(self, label="smoke"):
        self._label = label
    def classify(self, image_path):
        return {"label": self._label, "confidence": 0.77, "rationale": "ok"}

def test_demo_high_confidence_stays_local(monkeypatch):
    monkeypatch.setattr("src.classifier.predict", lambda p: ("no_fire", 0.95, {"fire": 0.02, "no_fire": 0.95, "smoke": 0.03}))
    # VLM must not be called; make factory explode if used
    def boom():
        raise AssertionError("VLM should not be called")
    monkeypatch.setattr("src.vlm_client.get_vlm_client", boom)
    r = demo.decide("any.jpg", use_cache=True, threshold=0.6)
    assert r["tier"] == "local" and r["label"] == "no_fire"

def test_demo_low_confidence_uses_vlm_label(monkeypatch):
    monkeypatch.setattr("src.classifier.predict", lambda p: ("fire", 0.2, {"fire": 0.2, "no_fire": 0.3, "smoke": 0.5}))
    monkeypatch.setattr("src.vlm_client.get_vlm_client", lambda: FakeVLM(label="smoke"))
    r = demo.decide("any.jpg", use_cache=True, threshold=0.6)
    # Must return VLM label, not hardcoded fire
    assert r["tier"] == "glimmer" and r["label"] == "smoke" and r["confidence"] == 0.77

def test_demo_classifier_failure_surfaces(monkeypatch):
    def fail(p):
        raise FileNotFoundError("Weights not found")
    monkeypatch.setattr("src.classifier.predict", fail)
    with pytest.raises(FileNotFoundError):
        demo.decide("x.jpg", use_cache=True, threshold=0.6)

def test_demo_vlm_failure_surfaces(monkeypatch):
    monkeypatch.setattr("src.classifier.predict", lambda p: ("smoke", 0.2, {"smoke": 0.2, "fire": 0.4, "no_fire": 0.4}))
    class BadVLM:
        def classify(self, image_path):
            raise RuntimeError("VLM not configured")
    monkeypatch.setattr("src.vlm_client.get_vlm_client", lambda: BadVLM())
    with pytest.raises(RuntimeError, match="not configured"):
        demo.decide("x.jpg", use_cache=True, threshold=0.6)
