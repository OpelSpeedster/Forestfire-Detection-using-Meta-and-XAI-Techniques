"""SAM evidence via a local Ultralytics model (auto-downloaded weights).

Masks are qualitative evidence, not ground truth. The Ultralytics SAM
predictor does not take text prompts, so there are no "smoke" / "flames"
concepts here: one call segments everything visible, and the caller treats
the overlay + mask-area fraction as supporting evidence for a human reviewer.
"""
import os
import pathlib

from src.env import load_env

WEIGHTS_DIR = pathlib.Path(os.environ.get("SAM_WEIGHTS_DIR", "external/sam"))


def sam_weights() -> str:
    load_env()
    return os.environ.get("SAM_WEIGHTS", "sam_b.pt")


class SamClient:
    def __init__(self, weights: str = ""):
        load_env()
        self.weights = weights or sam_weights()
        self.weights_dir = pathlib.Path(os.environ.get("SAM_WEIGHTS_DIR", str(WEIGHTS_DIR)))

    def available(self) -> bool:
        try:
            import ultralytics  # noqa: F401
            return True
        except Exception:
            return False

    def configured(self) -> bool:
        # Local model: "configured" means the library is installed. Weights
        # download on first use into SAM_WEIGHTS_DIR.
        return self.available()

    def _load_model(self):
        try:
            from ultralytics import SAM
        except Exception as e:
            raise RuntimeError(
                "SAM not available: install it with `pip install -r requirements.txt` "
                f"(needs the `ultralytics` package). ({e})"
            )
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        target = self.weights_dir / pathlib.Path(self.weights).name
        if target.exists():
            return SAM(str(target))
        # Ultralytics resolves known weight names (e.g. sam_b.pt) and
        # downloads them automatically; SAM_WEIGHTS_DIR keeps them local.
        try:
            return SAM(self.weights)
        except Exception as e:
            raise RuntimeError(f"SAM weights download/load failed for {self.weights!r}: {e}")

    def segment(self, image_path: str, overlay_path: str = "") -> dict:
        """Segment everything visible. Returns {weights, mask_area_fraction, n_masks, overlay}."""
        p = pathlib.Path(image_path)
        if not p.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        model = self._load_model()
        try:
            results = model.predict(str(p), verbose=False)
        except Exception as e:
            raise RuntimeError(f"SAM inference failed: {e}")
        r0 = results[0] if results else None
        masks = getattr(r0, "masks", None)
        n = 0
        frac = 0.0
        overlay = overlay_path or str(p.parent / f"{p.stem}_sam_overlay{p.suffix or '.jpg'}")
        try:
            import numpy as np
            from PIL import Image
            img = Image.open(p).convert("RGB")
            w, h = img.size
            canvas = np.asarray(img).astype(float)
            if masks is not None and getattr(masks, "data", None) is not None:
                import torch.nn.functional as F
                import torch
                data = masks.data
                if not isinstance(data, torch.Tensor):
                    data = torch.as_tensor(np.asarray(data))
                n = int(data.shape[0])
                acc = torch.zeros((h, w), dtype=torch.float32)
                for m in data:
                    m = m.float()
                    if m.shape != (h, w):
                        m = F.interpolate(m[None, None], size=(h, w), mode="nearest")[0, 0]
                    acc = torch.maximum(acc, m)
                frac = float((acc > 0.5).float().mean().item())
                tint = (acc > 0.5).numpy()
                canvas[tint] = canvas[tint] * 0.5 + np.array([255, 107, 53]) * 0.5
            Image.fromarray(canvas.astype("uint8")).save(overlay)
        except Exception as e:
            raise RuntimeError(f"SAM overlay rendering failed: {e}")
        return {"weights": self.weights, "mask_area_fraction": frac,
                "n_masks": n, "overlay": overlay}


def mask_trend(fractions: list) -> str:
    if len(fractions) < 2:
        return "insufficient frames"
    delta = fractions[-1] - fractions[0]
    if delta > 0.02:
        return "growing"
    if delta < -0.02:
        return "shrinking"
    return "stable"
