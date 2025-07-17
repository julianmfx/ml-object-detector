from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException,
    BackgroundTasks,
    Form,
    Request,
)
from starlette.status import HTTP_429_TOO_MANY_REQUESTS
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import asyncio

from pathlib import Path
import shutil
import uuid
from datetime import datetime

from ml_object_detector.config.load_config import load_config
from ml_object_detector.etl.download_images import download_image
from ml_object_detector.utils.logging import setup_logs
from ml_object_detector.utils.fs import ensure_directory_exists
from ml_object_detector.utils.email_alarm import send_alarm_email
from ml_object_detector.utils.clean_query_names import slugify
from ml_object_detector.models.predictor import YoloPredictor
from ml_object_detector.postprocess.analysis import build_summaries
from ml_object_detector.postprocess.html_report import write_html_report


# Initialization

cfg = load_config()
log = setup_logs()

ROOT = Path(cfg["ROOT"])
REPORTS_DIR = ROOT / cfg["reports_dir"]
PROCESSED_IMAGES_DIR = ROOT / cfg["output_dir"]
STATIC_DIR = ROOT / cfg["static_dir"]

ensure_directory_exists(REPORTS_DIR)
ensure_directory_exists(PROCESSED_IMAGES_DIR)
ensure_directory_exists(STATIC_DIR)

model = YoloPredictor()

# maps client IP
client_locks: dict[str, asyncio.Lock] = {}

# Mount API
app = FastAPI(title="ml-object-detector API")

# Routing API to serve HTML reports
app.mount(
    "/reports",  # URL prefix
    StaticFiles(directory=REPORTS_DIR, html=True),
    name="reports",
)

# Routing API to load favicon.ico
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Routing API to serve model artifacts (images)
app.mount("/processed", StaticFiles(directory=PROCESSED_IMAGES_DIR), name="processed")


# Helpers

def _save_uploads(files: list[UploadFile], dest_dir: Path) -> list[Path]:
    ensure_directory_exists(dest_dir)
    paths = []

    for file in files:
        ext = Path(file.filename).suffix
        tmp = dest_dir / f"{uuid.uuid4()}{ext}"
        with tmp.open("wb") as out:
            shutil.copyfileobj(file.file, out)
        paths.append(tmp)
    return paths


def _run_yolo_and_report(source_dir: Path, conf: float, run_id: str) -> Path:
    # point to the folder that now contains the images
    processed_run_dir = PROCESSED_IMAGES_DIR / run_id
    ensure_directory_exists(processed_run_dir)

    results = model.predict_images_in_folder(
        folder=source_dir, out_dir=processed_run_dir, conf=conf
    )
    summaries = build_summaries(results, conf_threshold=conf, run_id=run_id)
    report_dir = ROOT / cfg["reports_dir"]
    ensure_directory_exists(report_dir)
    report = write_html_report(
        summaries=summaries, reports_dir=report_dir, run_id=run_id
    )

    # If there are not rows, it means there were not objects detected
    # If there are more than 0 results, it means that at least one image was uploaded
    if not summaries and len(results) > 0:
        send_alarm_email(run_id, len(results))

    return report


# Routes/Endpoints

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse(STATIC_DIR / "favicon.ico")

@app.get("/", response_class=HTMLResponse)
async def docs_page():
    """
    Quick manual-testing page (upload *or* query).
    """

    return """
<!DOCTYPE html>
<html>
    <head>
        <link rel="icon" href="/static/favicon.ico" type="image/x-icon">
        <title>YOLO object detector</title>
    </head>
    <body>
    <h1>YOLO object detector</h1>

    <h2>1. Try with your own images</h2>
    <form action="/detect_upload" method="post" enctype="multipart/form-data">
      <input type="file"   name="files" multiple accept="image/*">
      <input type="number" step="0.01" min="0" max="1" name="conf" value="0.8">
      <button class="detect-btn" type="submit">Detect</button>
    </form>

    <h2>2. Or search &amp; auto-download from Pexels</h2>
    <form action="/detect_query" method="post" class="mb-4">
      <input type="text"   name="query" placeholder="picnic, surfing" required>
      <input type="number" name="n"     min="1"  max="15" value="5">
      <input type="number" name="conf"  step="0.01" min="0" max="1" value="0.8">
      <button class="detect-btn" type="submit">Detect</button>
    </form>

    <p><small>Prefer raw JSON? Open <code>/docs</code> for Swagger UI.</small></p>

    <script>
      document.querySelectorAll("form").forEach(f => {
        f.addEventListener("submit", () => {
          const btn = f.querySelector(".detect-btn");
          if (btn) {
            btn.disabled = true;
            btn.textContent = "Processing…";
          }
        });
      });
    </script>
    """


@app.post("/detect_upload")
async def detect_upload(
    request: Request,
    background: BackgroundTasks,
    files: list[UploadFile] = File(...),
    conf: float = Form(0.8),
):
    # One-job-per-IP locking
    client_ip = request.client.host
    lock = client_locks.setdefault(client_ip, asyncio.Lock())

    if lock.locked():
        return JSONResponse(
            {"detail": "Previsou detection still processing."},
            status_code=HTTP_429_TOO_MANY_REQUESTS,
        )

    await lock.acquire()

    def release_lock_then(fn, *args, **kw):
        """
        Run fn(*args, **kw) and always release the lock afterwards.
        """
        try:
            fn(*args, **kw)
        finally:
            lock.release()

    try:
        # Validation -----------------------------------
        if not files:
            raise HTTPException(400, "At least one image is required")
        if not (0.0 <= conf <= 1.0):
            raise HTTPException(400, "Confidence threshold must be between 0 and 1!")

        # Build run_id ----------------
        timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        slug_base = Path(files[0].filename).stem if len(files) == 1 else "bulk_upload"
        run_id = f"{slugify(slug_base)}_{timestamp}"

        # Save uploads ----------------
        run_raw_dir = ROOT / cfg["input_dir"] / run_id  # data/raw/<run_id>/
        ensure_directory_exists(run_raw_dir)
        paths = _save_uploads(files, dest_dir=run_raw_dir)

        # Single-image path (single input, fast response) ----------------
        if len(paths) == 1:
            processed_dir = PROCESSED_IMAGES_DIR / run_id
            ensure_directory_exists(processed_dir)

            one = model.predict_one(
                img_path=paths[0], out_dir=processed_dir, conf=conf
            )
            boxed_path = one.boxed_path

            # Write entry in the log file
            log.info(
                "run_id=%s file=%s detections=%d inference_ms=%.1f saved_to=%s",
                run_id,
                boxed_path.name,
                one.labels,
                one.speed_ms,
                boxed_path,
            )

            # Heavy report / e-mail re runs in the background
            # Will release lock when done
            public_url = f"/processed/{run_id}/{boxed_path.name}"
            background.add_task(
                release_lock_then, _run_yolo_and_report, run_raw_dir, conf, run_id
            )

            return RedirectResponse(
                url=public_url,
                status_code=303
            )

        # Multi-image path (bulk, background) -------------------------------
        background.add_task(
            release_lock_then, _run_yolo_and_report, run_raw_dir, conf, run_id
        )

        report_name = f"report_{run_id}.html"
        accepts_html = "text/html" in request.headers.get("accept", "").lower()

        if accepts_html:
            # Brower to the processing page like detect_query endpoint
            return RedirectResponse(
                url=f"/processing/{run_id}/{report_name}", status_code=303
            )

        # Fallback to JSON if not HTML client
        return JSONResponse(
            {
                "run_id": run_id,
                "status": "processing",
                "images": [p.name for p in paths],
                "poll_url": f"/processing/{run_id}/{report_name}",
                "report_hint": f"Check {cfg['uploads_dir']} soon.",
                "log_file": "/logs/upload_images.log",
            }
        )

    except Exception:
        # Make sure we don't leave the lock hanging on unexpected errors
        lock.release()
        raise

@app.post("/detect_query")
async def detect_query(
    request: Request,
    background: BackgroundTasks,
    query: str = Form(...),
    n: int = Form(5),
    conf: float = Form(0.8),
):
    """
    1. Downloads <n> images from Pexels for every comma-separated term
       in *query*.
    2. Kicks off YOLO + HTML-report generation **in the background**.
    3. Immediately redirects the browser to a lightweight “processing…”
       page that polls until the report is ready.
    """
    client_ip = request.client.host
    lock = client_locks.setdefault(client_ip, asyncio.Lock())

    if lock.locked():
        return JSONResponse(
            {"detail": "Previous detection stil precessing."},
            status_code=HTTP_429_TOO_MANY_REQUESTS,
        )

    await lock.acquire()

    # Validation --------------------
    if not query:
        raise HTTPException(
            400, "A query is required in order to process de object detector."
        )
    if not (0 <= n <= 15):
        raise HTTPException(400, "The number of images must be between 0 and 15.")

    if not (0.0 <= conf <= 1.0):
        raise HTTPException(400, "Confidence threshold must be between 0 and 1!")

    # Folder set-up ───────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    query_slug = slugify(query.replace(",", " "))

    run_id = f"{query_slug}_{timestamp}"
    run_raw_dir = ROOT / cfg["input_dir"] / run_id
    ensure_directory_exists(run_raw_dir)

    for term in [q.strip() for q in query.split(",") if q.strip()]:
        download_image(term, n=n, log=log, dest_dir=run_raw_dir)

    # Kick off heavy task ───────────────────────────────────────────────
    def task_wrapper():
        try:
            _run_yolo_and_report(run_raw_dir, conf, run_id)
        finally:
            lock.release()

    background.add_task(task_wrapper)

    report_name = f"report_{run_id}.html"
    accepts_html = "text/html" in request.headers.get("accept", "").lower()

    if accepts_html:
        # Browser to the report
        return RedirectResponse(
            url=f"/processing/{run_id}/{report_name}", status_code=303
        )

    return {
        "run_id": run_id,
        "status": "processing",
        "poll_url": f"/processing/{run_id}/{report_name}",
        "report_hint": f"/reports/{report_name} (once ready)",
        "log_file": "/logs/download_images.log",
    }


@app.get("/processing/{run_id}/{report_name}", response_class=HTMLResponse)
async def processing(report_name: str, run_id: str):
    report_path = REPORTS_DIR / report_name

    if report_path.exists():
        # Work finished → jump to the real report
        return RedirectResponse(url=f"/reports/{report_name}", status_code=303)

    # Still crunching – show spinner + auto refresh
    return f"""
    <html>
      <head>
        <title>Processing…</title>
        <meta http-equiv="refresh" content="2">
        <style>
          @keyframes spin {{ 0% {{transform:rotate(0deg)}} 100% {{transform:rotate(360deg)}} }}
          .loader {{
              border: 8px solid #f3f3f3; border-top: 8px solid #3498db;
              border-radius: 50%; width: 60px; height: 60px;
              animation: spin 1s linear infinite; margin:40px auto;
          }}
          body {{font-family: sans-serif; text-align:center; padding-top:40px}}
        </style>
      </head>
      <body>
        <h2>Running object detection…</h2>
        <p>This page will refresh automatically.</p>
        <div class="loader"></div>
      </body>
    </html>
    """
