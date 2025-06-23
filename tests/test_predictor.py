from pathlib import Path
import shutil
import os

import pytest
from models.predictor import YoloPredictor, SOURCE_DIR, OUTPUT_DIR


@pytest.fixture()
def clean_output(tmp_path):
    """
    Yield a scratch directory inside OUTPUT_DIR and wipe it afterwards.
    """
    destination = OUTPUT_DIR / "pytest_run"
    if destination.exists():
        shutil.rmtree(destination)

    destination.mkdir(parents=True, exist_ok=True)
    yield destination
    shutil.rmtree(destination)


def test_predictor(clean_output):
    """
    Smoke-test that prediciton runs end-to-end.
    """

    predictor = YoloPredictor()

    results = predictor.predict_images_in_folder(
        folder=SOURCE_DIR, out_dir=clean_output
    )

    # Results list length
    img_extensions = {".jpg", ".jpeg", ".png"}
    img_count = sum(1 for photo in SOURCE_DIR.iterdir() if photo.suffix.lower() in img_extensions)
    assert len(results) == img_count, "Mismatch between images and results."

    # annotated images were written
    written = list(clean_output.glob("*"))
    assert written, "No output files saved"
    assert any (file.suffix.lower() in img_extensions for file in written)

    for r in results:
        assert hasattr(r, "boxes"), "Result missing .boxes attribute)"
