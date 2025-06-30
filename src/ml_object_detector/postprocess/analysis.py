from pathlib import Path
from typing import List, Dict
from ultralytics.engine.results import Results


def build_summaries(
    results: List[Results],
    conf_threshold: float,
    run_id: str = "",
) -> List[Dict[str, str | float]]:
    """
    Build a list of dictionaries, one per detected object, ready for the
    Jinja2 template / CSV / JSON export.

    Keys per row
    ------------
    image   : str   –  When *run_id* is given, this becomes
                       "<run_id>/<filename>" so that the HTML template
                       can generate           "/processed/<run_id>/<filename>".
                       If *run_id* == "", it falls back to just "filename"
                       (preserves old CLI behaviour).
    object  : str   –  Human-readable class label.
    conf    : float –  Confidence score, 0‒1.

    Parameters
    ----------
    results        : list[ultralytics.engine.results.Results]
    conf_threshold : float
        Minimum confidence required to keep a detection.
    run_id         : str, optional
        Timestamp or UUID that namespaces this inference run; used only to
        prefix the *image* field. Default "" for backward compatibility.

    >>>   Example:
        >>> "Image beach_01.jpg has been identified with surfboard "
        "with the 92.4% level of confidence."
    """

    rows: list[Dict[str, str | float]] = []
    prefix = f"{run_id}/" if run_id else ""      # e.g. "20250630T190215/"

    for result in results:
        image_name = Path(result.path).name
        for cls_id, score in zip(result.boxes.cls, result.boxes.conf):
            if score < conf_threshold:
                continue
            object_name = result.names[int(cls_id)]
            rows.append(
                {
                    "image": prefix + image_name,
                    "object": object_name,
                    "conf": float(score),
                }
            )
    return rows


def summarise_predictions(
    results: List[Results],
    conf_threshold: float,
) -> List[str]:
    """
    Produce friendly log lines, reusing ``build_summaries``
    """

    return [
        (
            f"Image {row['image']} has been identified with "
            f"{row['object']} with the {row['conf']:.1%} level of confidence."
        )
        for row in build_summaries(results, conf_threshold)
    ]
