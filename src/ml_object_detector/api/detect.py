from fastapi import (
    APIRouter,
    Request,
    UploadFile,
    File,
    Form,
    BackgroundTasks,
    HTTPException,
    status,
)
from starlette.status import HTTP_429_TOO_MANY_REQUESTS
from fastapi.responses import JSONResponse, RedirectResponse
from pathlib import Path
from datetime import datetime
import asyncio
from ml_object_detector.services.detector import (
    acquire_lock,
    run_yolo_and_report,
    save_uploads,
    model,
    log,
    ROOT,
    PROCESSED,
    cfg,
)
from ml_object_detector.utils.fs import ensure_directory_exists
from ml_object_detector.utils.clean_query_names import slugify
from ml_object_detector.etl.download_images import download_image

router = APIRouter(tags=["Detection"])


def release_lock_then(lock, fn, *args, **kw):
    """
    Run fn(*args, **kw) and always release the lock afterwards.
    """
    try:
        fn(*args, **kw)
    finally:
        lock.release()


@router.post("/detect_upload")
async def detect_upload(
    request: Request,
    background: BackgroundTasks,
    files: list[UploadFile] = File(...),
    conf: float = Form(0.8),
):
    client_ip = request.client.host
    lock = acquire_lock(client_ip)

    if lock.locked():
        return JSONResponse(
            {"detail": "Previsou detection still processing."},
            status_code=HTTP_429_TOO_MANY_REQUESTS,
        )

    await lock.acquire()

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
        paths = save_uploads(files, dest_dir=run_raw_dir)

        # Single-image path (single input, fast response) ----------------
        if len(paths) == 1:
            processed_dir = PROCESSED / run_id
            ensure_directory_exists(processed_dir)

            one = model.predict_one(img_path=paths[0], out_dir=processed_dir, conf=conf)
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
                release_lock_then, lock, run_yolo_and_report, run_raw_dir, conf, run_id
            )

            return RedirectResponse(url=public_url, status_code=303)

        # Multi-image path (bulk, background) -------------------------------
        background.add_task(
            release_lock_then, lock, run_yolo_and_report, run_raw_dir, conf, run_id
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


@router.post("/detect_query")
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
    lock = acquire_lock(client_ip)

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

    background.add_task(release_lock_then, lock, run_yolo_and_report, run_raw_dir, conf, run_id)


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
