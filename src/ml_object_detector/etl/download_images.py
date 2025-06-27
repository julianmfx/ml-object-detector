"""Download images from Pexels API"""
import os
import requests
import logging
import logging.config
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

def setup_logs(log_path=LOGS_DIR / "download_images.log") -> logging.Logger:
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cfg = {
        "version": 1,
        "disable_existing_loggers": False,    # leave 3-party logs intact
        "formatters": {
            "std":  {"format": "%(asctime)s [%(levelname)s] %(message)s"},
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
                "maxBytes": 5_000_000,          # 5 MB
                "backupCount": 3,               # keep 3 old
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


def download_image(query: str, n: int = 5, log: logging.Logger | None = None) -> None:
    """Download *n* images for *query* into DESTINATION_DIR."""
    assert isinstance(n, int) and n > 0, "n must be a positive integer"
    log = log or logging.getLogger("download_logger")

    url = "https://api.pexels.com/v1/search"
    params = {"query": query, "per_page": n}
    response = requests.get(url, headers=HEADERS, params=params, timeout=30)
    response.raise_for_status()

    downloaded = 0
    log.info("Files will saved in %s", DESTINATION_DIR)
    log.info("Downloading %d images for query '%s'", n, query)

    for photo in response.json()["photos"]:
        image_url = photo["src"]["original"]
        filename = DESTINATION_DIR / image_url.split("/")[-1].split("?")[0]

        if filename.exists():
            log.info(f"File {filename} already exists, skipping download.")
            continue

        image_bytes = requests.get(image_url, timeout=30).content
        filename.write_bytes(image_bytes)
        downloaded += 1
        log.debug("Downloaded: %s", filename)

    log.info("Requested %d photos, saved %d photos", n, downloaded)
    log.info("Details saved in %s", LOGS_DIR / "download_images.log")
