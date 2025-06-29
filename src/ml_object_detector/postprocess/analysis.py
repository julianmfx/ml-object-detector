from pathlib import Path
from typing import List, Dict
from ultralytics.engine.results import Results


def build_summaries(
    results: List[Results],
    conf_threshold: float,
) -> List[Dict[str, str | float]]:
    """
    Create a list[dict] ready for HTML/JSON/CSV export.

    Each dict has:
        - image  : file name (str)
        - object : detected class label (str)
        - conf   : confidence as float 0-1

    >>>   Example:
        >>> "Image beach_01.jpg has been identified with surfboard "
        "with the 92.4% level of confidence."
    """

    rows: list[str] = []
    for result in results:
        image_name = Path(result.path).name
        for cls_id, score in zip(result.boxes.cls, result.boxes.conf):
            if score < conf_threshold:
                continue
            object_name = result.names[int(cls_id)]
            rows.append(
                {
                    "image": image_name,
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
