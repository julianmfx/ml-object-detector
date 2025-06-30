from __future__ import annotations
from pathlib import Path
from datetime import datetime

from jinja2 import Environment, FileSystemLoader
from ml_object_detector.config.load_config import load_config


def _get_env() -> Environment:
    cfg = load_config()
    TEMPLATE_DIR = Path(cfg["ROOT"]) / cfg["template_dir"]
    env = Environment(loader=FileSystemLoader(TEMPLATE_DIR), autoescape=True)
    return env


def write_html_report(
    summaries: list[dict], reports_dir: Path, run_id: str | None = None
) -> Path:
    """
    Render results.html.j2 into <reports_dir>/object_detector_report_<run_id>.html

    Parameters
    ----------
    summaries : list[dict]
        Pre-computed summaries (one per detection).
        Each *row["image"]* is already relative, e.g.  "20250630T190215/aa3f09c1d2e4.jpeg".
    reports_dir : Path
        Folder where reports live (served at `/reports`).
    run_id : str | None
        If None we generate a timestamp `YYYYMMDDThhmmss`. Pass your own to keep it stable
        across multi-step pipelines.

    Returns
    -------
    Path
        Absolute path of the freshly written HTML report.
    """
    run_id = run_id or datetime.now().strftime("%Y%m%dT%H%M%S")

    env = _get_env()
    template = env.get_template("results.html.j2")

    html = template.render(run_date=datetime.now(), rows=summaries)

    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / f"object_detector_report.html_{run_id}.html"
    report_path.write_text(html, encoding="utf-8")
    return report_path
