import pytest
import scripts.evaluate as ev

def test_empty_pairs_rejected():
    with pytest.raises(ValueError, match="No images"):
        ev.run_local_baseline([])

def test_prediction_failures_raise_not_counted(monkeypatch):
    # Force every predict to fail; must raise instead of counting as no_fire
    def fail(p):
        raise RuntimeError("boom")
    monkeypatch.setattr("src.classifier.predict", fail)
    pairs = [("a.jpg", "fire", "test"), ("b.jpg", "smoke", "test")]
    with pytest.raises(RuntimeError, match="prediction"):
        ev.run_local_baseline(pairs)
