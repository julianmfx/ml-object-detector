from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, FileResponse, RedirectResponse
from pathlib import Path
from ml_object_detector.config.load_config import load_config

cfg = load_config()
ROOT = Path(cfg["ROOT"])
STATIC_DIR = ROOT / cfg["static_dir"]
REPORTS_DIR = ROOT / cfg["reports_dir"]

router = APIRouter(tags=["Home"])

# Routes/Endpoints

@router.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse(STATIC_DIR / "favicon.ico")

@router.get("/", response_class=HTMLResponse)
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

@router.get("/processing/{run_id}/{report_name}", response_class=HTMLResponse)
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
