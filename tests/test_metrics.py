from src.metrics import prf1

def test_per_class_recall():
    y_true = ["smoke", "smoke", "fire"]
    y_pred = ["smoke", "fire", "fire"]
    m = prf1(y_true, y_pred)
    assert "smoke" in m and "recall" in m["smoke"]
    assert m["smoke"]["recall"] == 0.5

def test_accuracy():
    from src.metrics import accuracy
    assert accuracy(["a", "b"], ["a", "b"]) == 1.0
    assert accuracy(["a", "b"], ["a", "a"]) == 0.5

def test_metrics_reject_mismatched_lengths():
    import pytest
    from src.metrics import accuracy
    with pytest.raises(ValueError, match="length mismatch"):
        prf1(["smoke"], ["smoke", "fire"])
    with pytest.raises(ValueError, match="length mismatch"):
        accuracy(["smoke"], [])
