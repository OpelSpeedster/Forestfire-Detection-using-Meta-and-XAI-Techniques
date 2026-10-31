import os
import pathlib
import urllib.request

WEIGHTS_URL = os.environ.get("WEIGHTS_URL", "https://example.com/FirenetCNN1.h5")
DEST = pathlib.Path(os.environ.get("FIRENET_WEIGHTS", "external/FirenetCNN1.h5"))

def fetch_weights(dry_run=False, url: str | None = None):
    target_url = url or WEIGHTS_URL
    DEST.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        DEST.write_bytes(b"dry-run")
        return DEST
    # If weights already present (manual copy), skip download
    if DEST.exists() and DEST.stat().st_size > 1024:
        print(f"Weights already present at {DEST} ({DEST.stat().st_size} bytes) — skipping download.")
        return DEST
    # Placeholder URL check — don't hit example.com
    if "example.com" in target_url:
        print(f"[setup] WEIGHTS_URL is placeholder ({target_url}).")
        print(f"  Fix: set WEIGHTS_URL to a real release URL, or")
        print(f"  manually place FirenetCNN1.h5 from github.com/OpelSpeedster/Forest-Fire-Detection-Using-FirenetCNN-and-XAI-Techniques into {DEST}")
        print(f"  Then re-run: .venv\\Scripts\\python scripts/setup.py")
        if DEST.exists():
            return DEST
        raise SystemExit(2)
    try:
        print(f"Downloading weights from {target_url} -> {DEST} ...")
        urllib.request.urlretrieve(target_url, str(DEST))
    except Exception as e:
        print(f"[setup] Download failed: {e}")
        print(f"  Manually download FirenetCNN1.h5 and place it at {DEST}")
        print(f"  You can set WEIGHTS_URL env var to a direct release URL and retry.")
        raise SystemExit(2)
    return DEST

if __name__ == "__main__":
    p = fetch_weights()
    print(f"Weights at {p}. Place test images in data/forestfire-classifier-dataset/test/{{smoke,fire,no_fire}} (or data/test/{{smoke,fire,no_fire}})")
