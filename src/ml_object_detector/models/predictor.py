from dataclasses import dataclass
from pathlib import Path
from typing import List
from ultralytics import YOLO
from ultralytics.engine.results import Results
from ml_object_detector.config.load_config import load_config
from ml_object_detector.utils.logging import setup_logs

cfg = load_config()
log = setup_logs()

# ------------------------------------------------------------------
# 1.  Establish the project root
# ------------------------------------------------------------------
ROOT = Path(cfg["ROOT"])

# ------------------------------------------------------------------
# 2.  Build all other paths relative to that root
# ------------------------------------------------------------------
SOURCE_DIR = ROOT / cfg["input_dir"]  # data/raw
OUTPUT_DIR = ROOT / cfg["output_dir"]  # data/processed
MODEL_DIR = ROOT / cfg["model_dir"]  # ml_object_detector/models/weights
MODEL_NAME = cfg["model_name"]  # yolov8n.pt
LOGS_DIR = ROOT / cfg["logs_dir"]  # logs
CONF_THRESH = float(cfg["confidence_threshold"])  # 0.8
MODEL_PATH = MODEL_DIR / MODEL_NAME  # ml_object_detector/models/weights/yolov8n.pt

log.info("Loading YOLO weights from %s", MODEL_NAME)


@dataclass
class OneResult:
    boxed_path: Path
    labels: int
    speed_ms: float


class YoloPredictor:
    """
    Convenience wrapper around `ultralytics.YOLO` that
    1) loads the model once and
    2) offers a single `predict_images_in_folder` method.
    """

    def __init__(self, model_path: str | Path = MODEL_PATH) -> None:
        self.model = YOLO(model_path)
        self.model.info()
        log.info("YOLO model loaded and ready.")

    def predict_one(
        self, img_path: Path, out_dir: Path | None = None, conf: float | None = None
    ) -> OneResult:
        out_dir = Path(out_dir or OUTPUT_DIR)
        conf = float(conf if conf is not None else CONF_THRESH)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Run YOLO ------------------------------
        res = self.model.predict(
            source=str(img_path),
            save=True,
            save_txt=True,
            save_conf=True,
            project=str(out_dir.parent),
            name=out_dir.name,
            conf=conf,
            verbose=False,
            exist_ok=True,
        )[0]
        boxed = Path(res.save_dir) / f"{img_path.stem}.jpg"
        return OneResult(boxed, len(res.boxes), sum(res.speed.values()))

    def predict_images_in_folder(
        self,
        folder: str | Path | None = None,
        out_dir: str | Path | None = None,
        conf: float | None = None,
    ) -> List[Results]:
        """
        Run inference on **all** images in `folder` and
        save annotated copies plus labels to `out_dir`.

        Parameters
        ----------
        folder   : source directory with .jpg/.png files
        out_dir  : where the annotated images should go
        conf     : confidence threshold (0–1)

        Returns
        -------
        list[ultralytics.engine.results.Results]
            One `Results` object per image.

        """
        # fall back to YAML defaults only when arguments aren’t provided
        folder = Path(folder or SOURCE_DIR)
        out_dir = Path(out_dir or OUTPUT_DIR)
        conf = float(conf if conf is not None else CONF_THRESH)
        out_dir.mkdir(parents=True, exist_ok=True)
        results: List[Results] = self.model.predict(
            source=folder,
            save=True,
            save_txt=True,
            save_conf=True,
            project=out_dir.parent,
            name=out_dir.name,
            exist_ok=True,
            conf=conf,
            verbose=False,  # silence internal prints, avoid log line duplications
        )
        if results:
            sp = results[0].speed
            shape = getattr(results[0], "orig_shape", "unknown")
            log.info(
                "Speed: %.1fms preprocess, %.1fms inference, %.1fms postprocess "
                "per image at shape %s",
                sp.get("preprocess", 0.0),
                sp.get("inference", 0.0),
                sp.get("postprocess", 0.0),
                shape,
            )
        else:
            log.warning("YOLO.predict() returned no results - folder may be empty")
        log.info("Results saved to %s", out_dir)
        n_labels = sum(len(r.boxes) for r in results)
        log.info(
            "%d label%s saved to %s/labels",
            n_labels,
            "" if n_labels == 1 else "s",
            out_dir,
        )
        return results
