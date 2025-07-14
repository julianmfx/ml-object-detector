import logging
import logging.config
import logging.handlers
from pathlib import Path

from ml_object_detector.config.load_config import load_config

cfg = load_config()
BASE_DIR = Path(cfg["ROOT"])
LOGS_DIR = Path(BASE_DIR / cfg["logs_dir"])

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
