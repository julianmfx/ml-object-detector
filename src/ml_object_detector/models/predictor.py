from pathlib import Path
import logging

from ultralytics import YOLO
from config.load_config import load_config

cfg = load_config()

# ------------------------------------------------------------------
# 1.  Establish the project root
# ------------------------------------------------------------------
ROOT = Path(cfg["base_dir"])

# ------------------------------------------------------------------
# 2.  Build all other paths relative to that root
# ------------------------------------------------------------------
SOURCE_DIR   = ROOT / cfg["input_dir"]                  # data/raw
OUTPUT_DIR   = ROOT / cfg["output_dir"]                 # data/processed
MODEL_DIR    = ROOT / cfg["model_dir"]                  # models
MODEL_PATH   = ROOT / cfg["model_path"]                 # models/weight/yolov8n.pt
LOGS_DIR     = ROOT / cfg["logs_dir"]                   # logs

CONF_THRESH  = float(cfg["confidence_threshold"])       # 0.8

logging.info("Loading YOLO weights from %s", MODEL_PATH)


class YoloPredictor:
    """
    Convenience wrapper around `ultralytics.YOLO` that
    1) loads the model once and
    2) offers a single `predict_folder` method.
    """

    def __init__(self, model_path: str | Path = MODEL_PATH) -> None:
        self.model = YOLO(model_path)
        self.model.info()
        logging.info("YOLO model loaded and ready.")

    def predict_images_in_folder(
        self,
        folder: str | Path | None = None,
        out_dir: str | Path | None = None,
        conf: float | None = None,
    ):
        """
        Run inference on **all** images in `folder` and
        save annotated copies to `out_dir`.

        Parameters
        ----------
        folder   : source directory with .jpg/.png files
        out_dir  : where the annotated images should go
        conf     : confidence threshold (0–1)
        """
        # fall back to YAML defaults only when arguments aren’t provided
        folder  = Path(folder  or SOURCE_DIR)
        out_dir = Path(out_dir or OUTPUT_DIR)
        conf    = float(conf if conf is not None else CONF_THRESH)
        out_dir.mkdir(parents=True, exist_ok=True)
        results = self.model.predict(
            source=folder,
            save=True,
            save_txt=True,
            save_conf=True,
            project=out_dir.parent,
            name=out_dir.name,
            exist_ok=True,
            conf=conf,
        )
        return results
