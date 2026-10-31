import os
import pathlib

def test_setup_downloads_weights(tmp_path, monkeypatch):
    monkeypatch.setenv("FIRENET_WEIGHTS", str(tmp_path / "FirenetCNN1.h5"))
    # reimport to pick up env
    import importlib
    import scripts.setup as s
    importlib.reload(s)
    p = s.fetch_weights(dry_run=True)
    assert pathlib.Path(p).exists()
    # reset to default for other tests
    monkeypatch.delenv("FIRENET_WEIGHTS", raising=False)
    importlib.reload(s)
