from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import shutil
import uuid

from ml_object_detector.config.load_config import load_config
from ml_object_detector.etl.download_images import download_image, setup_logs
from ml_object_detector.models.predictor import YoloPredictor
from ml_object_detector.postprocess.analysis import build_summaries
from ml_object_detector.postprocess.html_report import write_html_report


# Initialization

cfg = load_config()
ROOT = Path(cfg["ROOT"])
log = setup_logs()
app = FastAPI(title="ml-object-detector API")
app.mount(
    "/reports",                               # URL prefix
    StaticFiles(directory=ROOT / cfg["reports_dir"]),
    name="reports",
)
model = YoloPredictor()

# Helpers


def _save_uploads(files: list[UploadFile]) -> list[Path]:
    dest_dir = ROOT / cfg["uploads_dir"]
    dest_dir.mkdir(exist_ok=True, parents=True)
    paths = []

    for file in files:
        ext = Path(file.filename).suffix
        tmp = dest_dir / f"{uuid.uuid4()}{ext}"
        with tmp.open("wb") as out:
            shutil.copyfileobj(file.file, out)
        paths.append(tmp)
    return paths


def _run_yolo_and_report(image_paths: list[Path], conf: float) -> Path:
    # point to the folder that now contains the images
    images_dir = ROOT / cfg["input_dir"]

    results = model.predict_images_in_folder(folder=images_dir, conf=conf)
    summaries = build_summaries(results, conf_threshold=conf)
    report_dir = ROOT / cfg["reports_dir"]
    return write_html_report(summaries, report_dir)


# Routes
@app.get("/", response_class=HTMLResponse)
async def docs_page():
    """
    Quick manual-testing page (upload *or* query).
    """

    return """
    <h1>YOLO object detector</h1>
    <h2>1. Try with your own images</h2>
    <form action="/detect_upload" method="post" enctype="multipart/form-data">
      <input type="file" name="files" multiple accept="image/*">
      <input type="number" step="0.01" min="0" max="1" name="conf" value="0.8">
      <button type="submit">Detect</button>
    </form>

    <h2>2. Or search &amp; auto-download from Pexels</h2>
    <form action="/detect_query" method="post" class="mb-4">
      <input type="text"   name="query" placeholder="picnic, surfing" required>
      <input type="number" name="n"     min="1"  max="15" value="5">
      <input type="number" name="conf"  step="0.01" min="0" max="1" value="0.8">
      <button type="submit">Detect</button>
    </form>

    <p><small>Prefer raw JSON? Open <code>/docs</code> for Swagger UI.</small></p>
    """


@app.post("/detect_upload")
async def detect_upload(
    background: BackgroundTasks, files: list[UploadFile] = File(...), conf: float = 0.8
):
    if not (0.0 <= conf <= 1.0):
        raise HTTPException(400, "Confidence threshold must be between 0 and 1!")

    paths = _save_uploads(files)
    # run heavy work in background so the request returns fast
    background.add_task(_run_yolo_and_report, paths, conf)

    return JSONResponse(
        {
            "status": "processing",
            "images": [p.name for p in paths],
            "report_hint": f"Check {cfg['results_dir']} soon.",
        }
    )


@app.post("/detect_query")
async def detect_query(
    request: Request,
    query: str = Form(...), n: int = Form(5), conf: float = Form(0.8)
):
    """
    Replicates the CLI's ETL path: downlaod images for a query string.
    Run model and reutnr HTML report path.
    """
    if not query:
        raise HTTPException(
            400, "A query is required in order to process de object detector."
        )
    if not (0 <= n <= 15):
        raise HTTPException(400, "The number of images must be between 0 and 15.")

    if not (0.0 <= conf <= 1.0):
        raise HTTPException(400, "Confidence threshold must be between 0 and 1!")

    image_paths: list[Path] = []
    for term in [q.strip() for q in query.split(",") if q.strip()]:
        image_paths.extend(download_image(term, n=n, log=log))

    report_path = _run_yolo_and_report(image_paths, conf)
    report_url = f"reports/{report_path.name}"

    if "text/html" in request.headers.get("accept",""):
        # Browser to the report
        return RedirectResponse(url=report_url, status_code=303)

    return {
        "images": [p.name for p in image_paths],
        "html_report": str(report_path.relative_to(ROOT)),
        "log_file": str(
            (ROOT / cfg["logs_dir"] / "download_images.log").relative_to(ROOT)
        ),
    }
