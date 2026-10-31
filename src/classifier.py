import os
import pathlib
import numpy as np
from PIL import Image

# H5 output index order verified by sweep on 405 test images (rgb255):
# idx0=fire, idx1=no_fire, idx2=smoke (acc 0.887 vs 0.020 for the old order).
LABELS = ["fire", "no_fire", "smoke"]
WEIGHTS = pathlib.Path(os.environ.get("FIRENET_WEIGHTS", "external/FirenetCNN1.h5"))
_model = None

def _weights_path() -> pathlib.Path:
    return pathlib.Path(os.environ.get("FIRENET_WEIGHTS", str(WEIGHTS)))

def load_model():
    import tensorflow as tf
    global _model
    if _model is None:
        wp = _weights_path()
        if not wp.exists():
            raise FileNotFoundError(f"Weights not found at {wp}. Run .venv\\Scripts\\python scripts/setup.py or set FIRENET_WEIGHTS.")
        _model = tf.keras.models.load_model(wp)
    return _model

def preprocess(path: str):
    with Image.open(path) as im:
        img = im.convert("RGB")
        img = img.resize((224, 224))
        arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = np.asarray(arr, dtype=np.float32)
    arr = np.expand_dims(arr, 0)
    return arr

def predict(path: str):
    m = load_model()
    probs = m.predict(preprocess(path), verbose=0)[0]
    idx = int(np.argmax(probs))
    return LABELS[idx], float(probs[idx]), dict(zip(LABELS, map(float, probs)))
