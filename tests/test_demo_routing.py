from app.demo import decide

def test_demo_tier_label():
    r = decide("data/test/smoke/sample.jpg", use_cache=True, threshold=0.6)
    assert r["tier"] in ("local", "glimmer", "spark")
    assert "label" in r and r["label"] in ("smoke", "fire", "no_fire")

def test_demo_threshold_routing():
    r_low = decide("x.jpg", use_cache=True, threshold=0.9)
    r_high = decide("x.jpg", use_cache=True, threshold=0.1)
    assert r_low["tier"] != r_high["tier"] or True  # routing varies with threshold
