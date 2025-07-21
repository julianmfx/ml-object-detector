import logging
import uuid
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from ml_object_detector.api import register_routers
from ml_object_detector.config.load_config import load_config
from ml_object_detector.utils.logging import setup_logs
from ml_object_detector.utils.fs import ensure_directory_exists

# Initialization

cfg = load_config()
setup_logs()    # Sets global logging config
log = logging.getLogger(__name__)

## Define paths
ROOT = Path(cfg["ROOT"])
REPORTS_DIR = ROOT / cfg["reports_dir"]
PROCESSED_IMAGES_DIR = ROOT / cfg["output_dir"]
STATIC_DIR = ROOT / cfg["static_dir"]

for path in (REPORTS_DIR, PROCESSED_IMAGES_DIR, STATIC_DIR):
    ensure_directory_exists(path)

# Mount API
app = FastAPI(title="ml-object-detector API")

# Middleware: inject a per-request ID into logs and response headers
@app.middleware("http")
async def add_request_id(request, call_next):
    request_id = str(uuid.uuid4())

    # attach context to all log records produced during this request
    request.state.log = logging.LoggerAdapter(
        logging.getLogger(__name__), {"req_id": request_id}
    )

    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response

# Static mounts
app.mount("/reports", StaticFiles(directory=REPORTS_DIR, html=True), name="reports")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/processed", StaticFiles(directory=PROCESSED_IMAGES_DIR), name="processed")


register_routers(app)
