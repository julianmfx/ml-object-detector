from pathlib import Path
import os
from jinja2 import Environment, FileSystemLoader
from datetime import datetime
from ml_object_detector.config.load_config import load_config


def _get_env() -> Environment:
    cfg = load_config()
    TEMPLATE_DIR = Path(cfg["ROOT"]) / cfg["template_dir"]
    return Environment(loader=FileSystemLoader(TEMPLATE_DIR), autoescape=True)


def write_html_report(summaries: list[dict], out_dir: Path) -> Path:
    """_summary_

    Args:
        summaries (list[dict]): _description_
        out_dir (Path): _description_

    Returns:
        Path: _description_
    """
    env = _get_env()
    template = env.get_template("results.html.j2")
    html = template.render(run_date=datetime.now(), rows=summaries)

    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "object_detector_report.html"
    report_path.write_text(html, encoding="utf-8")
    return report_path
