"""Load .env once, without overriding real environment variables. Never logs values."""
import os
import pathlib

_loaded = False


def load_env() -> None:
    global _loaded
    if _loaded:
        return
    _loaded = True
    root = pathlib.Path(__file__).resolve().parents[1]
    env_file = root / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def is_placeholder(value: str) -> bool:
    v = (value or "").strip().lower()
    return not v or v.startswith("your-") or v in {"[redacted]", "changeme", "xxx"} or "your-" in v
