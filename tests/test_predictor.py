from pathlib import Path
import os
import pytest
import numpy as np
import cv2
from ml_object_detector.models.predictor import YoloPredictor


# ----------------------------------------------------------------------#
# 1.  Light-weight stand-in for `ultralytics.YOLO`
# ----------------------------------------------------------------------#
class DummyYOLO:
    """
    Mimics just enough of ultralytics.YOLO for a unit test.
    """

    def __init__(self, *args, **kwargs):
        """ "Collects calls for later inspection"""
        self.predict_calls = []

    def info(self, *args, **kwargs):
        """No-op; predictor calls this on init."""
        return None

    # Records args, returns a fake “Results”
    def predict(
        self, source, save, save_txt, save_conf, project, name, conf, verbose, exist_ok
    ):

        # Record the call so test can assert on it
        self.predict_calls.append(
            {
                "source": Path(source),
                "project": Path(project),
                "name": name,
                "conf": conf,
            }
        )

        # Fabricate a minimal Results-like object
        Box = type("Box", (), {})  # empty stub
        dummy_res = type(
            "Res",
            (),
            {
                "boxes": [Box(), Box()],
                "save_dir": Path(project) / name,
                "speed": {"preprocess": 1.0, "inference": 2.0, "postprocess": 1.0},
                "orig_shape": (640, 480, 3),
            },
        )
        return [dummy_res]


# ----------------------------------------------------------------------#
# 2.  Fixture: YoloPredictor wired to DummyYOLO
# ----------------------------------------------------------------------#
@pytest.fixture()
def predictor_with_dummy(monkeypatch, tmp_path):
    # Patch the external dependency
    dummy = DummyYOLO()

    # Patch the YOLO class in the predictor module
    monkeypatch.setattr(
        "ml_object_detector.models.predictor.YOLO", lambda *args, **kwargs: dummy
    )

    # Also patch SOURCE_DIR to point to an empy temp dir
    monkeypatch.setattr("ml_object_detector.models.predictor.SOURCE_DIR", tmp_path)

    # touch a fake image so the glob finds 1 file
    (tmp_path / "img1.jpg").touch()

    return YoloPredictor(), dummy, tmp_path

# Helper to the integration test
def _write_tiny_jpeg(path: Path) -> None:
    """Create a 1×1 black JPEG so OpenCV can decode it."""
    black = np.zeros((1, 1, 3), dtype=np.uint8)
    cv2.imwrite(str(path), black)

# ----------------------------------------------------------------------#
# 3.  Unit test — no heavy model download, runs in < 100 ms
# ----------------------------------------------------------------------#
@pytest.mark.unit
def test_predict_images_infolder_uses_dummy(predictor_with_dummy):
    predictor, dummy, src_dir = predictor_with_dummy

    out_dir = src_dir / "out"
    results = predictor.predict_images_in_folder(
        folder=src_dir, out_dir=out_dir, conf=0.5
    )

    # predict should be called exactly once
    assert len(dummy.predict_calls) == 1

    call = dummy.predict_calls[0]
    assert call["source"] == src_dir
    assert call["conf"] == 0.5

    # predict_images_in_folder should return our fabricated object
    assert len(results) == 1
    assert len(results[0].boxes) == 2


# ----------------------------------------------------------------------#
# 4.  Optional “smoke” integration test (slow — real model)
# ----------------------------------------------------------------------#
RUN_INT = os.getenv("RUN_INTEGRATION") == "1"

@pytest.mark.integration
@pytest.mark.skipif(not RUN_INT, reason="set RUN_INTEGRATION=1 to run")
def test_predictor_smoke(tmp_path):
    """
    End-to-end check with the *real* model.
    Remove @skip if you have weights cached locally.
    """
    predictor = YoloPredictor()

    out_dir = tmp_path / "out"
    src_dir = tmp_path / "src"
    src_dir.mkdir()

    # write a valid mock image
    _write_tiny_jpeg(src_dir / "img1.jpg")

    results = predictor.predict_images_in_folder(folder=src_dir, out_dir=out_dir)
    assert len(results) == 1
    assert (out_dir / "labels").exists()
