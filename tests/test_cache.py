from src.cache import cache_key

def test_cache_key_deterministic(tmp_path):
    assert cache_key("glimmer", "abc", "prompt") == cache_key("glimmer", "abc", "prompt")

def test_cache_key_varies():
    assert cache_key("glimmer", "abc", "prompt") != cache_key("spark", "abc", "prompt")
