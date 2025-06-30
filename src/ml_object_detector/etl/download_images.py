"""Download images from Pexels API"""

import os
import requests
import logging
import logging.config
import hashlib
import uuid
from datetime import datetime
from pathlib import Path
from ml_object_detector.config.load_config import load_config
from dotenv import load_dotenv

load_dotenv()
cfg = load_config()
PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")
if not PEXELS_API_KEY:
    raise RuntimeError("PEXELS_API_KEY missing. Put it in .env.")
HEADERS = {"Authorization": PEXELS_API_KEY}
BASE_DIR = Path(cfg["ROOT"])
DESTINATION_DIR = Path(BASE_DIR / cfg["input_dir"])
DESTINATION_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR = Path(BASE_DIR / cfg["logs_dir"])


def make_immutable_name(raw_bytes: bytes, ext: str = "") -> str:
    """
    Return the first 12 hex chars of SHA-1 + extension.
    For identical bytes you always get the same name.
    """
    hash = hashlib.sha1(raw_bytes).hexdigest()[:12]
    return f"{hash}{ext}"


def setup_logs(log_path=LOGS_DIR / "download_images.log") -> logging.Logger:
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cfg = {
        "version": 1,
        "disable_existing_loggers": False,  # leave 3-party logs intact
        "formatters": {
            "std": {"format": "%(asctime)s [%(levelname)s] %(message)s"},
        },
        "handlers": {
            # Console — INFO+
            "console": {
                "class": "logging.StreamHandler",
                "level": "INFO",
                "formatter": "std",
            },
            # Rotating file — DEBUG & up
            "debug_file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "DEBUG",
                "formatter": "std",
                "filename": Path(log_path),
                "maxBytes": 5_000_000,  # 5 MB
                "backupCount": 3,  # keep 3 old
            },
        },
        "loggers": {
            # Your app logger
            "download_logger": {
                "handlers": ["console", "debug_file"],
                "level": "DEBUG",
                "propagate": False,
            }
        },
    }

    logging.config.dictConfig(cfg)
    return logging.getLogger("download_logger")


def download_image(
    query: str,
    n: int = 5,
    log: logging.Logger | None = None,
    dest_dir: Path | None = None,
) -> list[Path]:
    """Download *n* images for *query* into DESTINATION_DIR."""
    assert isinstance(n, int) and n > 0, "n must be a positive integer"
    log = log or logging.getLogger("download_logger")

    DEST = Path(dest_dir or DESTINATION_DIR)           # NEW
    DEST.mkdir(parents=True, exist_ok=True)            # ensure

    url = "https://api.pexels.com/v1/search"
    params = {"query": query, "per_page": n}
    response = requests.get(url, headers=HEADERS, params=params, timeout=30)
    response.raise_for_status()

    downloaded = 0
    log.info("Files will saved in %s", DEST)
    log.info("Downloading %d images for query '%s'", n, query)

    saved: list[Path] = []
    for photo in response.json()["photos"]:
        image_url = photo["src"]["original"]

        # 1. Download image bytes
        image_bytes = requests.get(image_url, timeout=30).content

        # 2. Derive a deterministic filename
        ext = Path(image_url).suffix
        hashed_name = make_immutable_name(image_bytes, ext)
        filename = DEST / hashed_name

        if filename.exists():
            log.info(f"File {filename} already exists, skipping download.")
            continue

        # Ensure parent directory exists
        filename.parent.mkdir(parents=True, exist_ok=True)
        filename.write_bytes(image_bytes)

        downloaded += 1
        saved.append(filename)
        log.debug("Downloaded: %s", filename)

    log.info("Requested %d photos, saved %d photos", n, downloaded)
    log.info("Details saved in %s", LOGS_DIR / "download_images.log")
    return saved
