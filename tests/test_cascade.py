from src.cascade import should_escalate, route

def test_cascade_threshold():
    assert should_escalate(0.4, threshold=0.6) is True
    assert should_escalate(0.8, threshold=0.6) is False
    assert should_escalate(0.6, threshold=0.6) is False

def test_cascade_route():
    assert route("smoke", 0.4, 0.6, vlm_label="fire") == "fire"
    assert route("smoke", 0.9, 0.6, vlm_label="fire") == "smoke"
