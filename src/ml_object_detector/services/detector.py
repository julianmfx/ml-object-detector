import asyncio
import uuid
import shutil
from pathlib import Path
from datetime import datetime
from fastapi import BackgroundTasks, UploadFile
from ml_object_detector.models.predictor import YoloPredictor
from ml_object_detector.postprocess.analysis import build_summaries
from ml_object_detector.postprocess.html_report import write_html_report
from ml_object_detector.utils.email_alarm import send_alarm_email
from ml_object_detector.utils.fs import ensure_directory_exists
from ml_object_detector.utils.clean_query_names import slugify
from ml_object_detector.config.load_config import load_config

cfg = load_config()
ROOT = Path(cfg["ROOT"])
PROCESSED = ROOT / cfg["output_dir"]
REPORTS = ROOT / cfg["reports_dir"]

model = YoloPredictor()
locks: dict[str, asyncio.Lock] = {}


def acquire_lock(ip: str) -> asyncio.Lock:
    lock = locks.setdefault(ip, asyncio.Lock())
    if lock.locked():
        raise RuntimeError("Previous job still running")
    return lock


def run_yolo_and_report(src_dir: Path, conf: float, run_id: str) -> Path:
    processed_dir = PROCESSED / run_id
    ensure_directory_exists(processed_dir)
    results = model.predict_images_in_folder(src_dir, processed_dir, conf)
    summaries = build_summaries(results, conf, run_id)
    report = write_html_report(summaries, REPORTS, run_id)
    if not summaries and len(results) > 0:
        send_alarm_email(run_id, len(results))
    return report


def save_uploads(files: list[UploadFile], dest_dir: Path) -> list[Path]:
    ensure_directory_exists(dest_dir)
    paths = []

    for file in files:
        ext = Path(file.filename).suffix
        tmp = dest_dir / f"{uuid.uuid4()}{ext}"
        with tmp.open("wb") as out:
            shutil.copyfileobj(file.file, out)
        paths.append(tmp)
    return paths
