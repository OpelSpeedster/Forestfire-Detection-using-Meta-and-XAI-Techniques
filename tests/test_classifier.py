import pathlib
import numpy as np

def test_classifier_preprocess_shape(tmp_path):
    from PIL import Image
    from src.classifier import preprocess, LABELS
    # create dummy RGB image
    p = tmp_path / "sample.jpg"
    Image.new("RGB", (300, 200), color=(10, 20, 30)).save(p)
    arr = preprocess(str(p))
    assert arr.shape == (1, 224, 224, 3)
    assert LABELS == ["fire", "no_fire", "smoke"]
    # values normalized 0-1
    assert float(arr.max()) <= 1.0 and float(arr.min()) >= 0.0

def test_classifier_predict_with_mock(monkeypatch, tmp_path):
    from PIL import Image
    import src.classifier as clf

    class FakeModel:
        def predict(self, arr, verbose=0):
            # idx0 highest -> fire under verified H5 order
            return np.array([[0.7, 0.2, 0.1]])

    monkeypatch.setattr(clf, "load_model", lambda: FakeModel())
    p = tmp_path / "sample.jpg"
    Image.new("RGB", (224, 224), color=(0, 0, 0)).save(p)
    label, conf, probs = clf.predict(str(p))
    assert label == "fire"
    assert 0 <= conf <= 1
    assert set(probs.keys()) == {"smoke", "fire", "no_fire"}
