from src.vlm_client import VLMClient, BudgetExhausted

def test_budget_guard_halts(tmp_path):
    c = VLMClient(budget_usd=0.01, cache_dir=str(tmp_path))
    c._spent_usd = 0.02
    try:
        c._check_budget(0.01)
        assert False, "should have raised"
    except BudgetExhausted as e:
        assert "budget" in str(e).lower()

def test_budget_allows_small_spend(tmp_path):
    c = VLMClient(budget_usd=20, cache_dir=str(tmp_path))
    c._check_budget(0.001)  # should not raise
