#!/usr/bin/env python

"""
run_pipeline.py
-----------------

Interactive ETL -> YOLO prediction pipeline.

"""

from __future__ import annotations
from pathlib import Path
from typing import List

import importlib

from ml_object_detector.config.load_config import load_config
from ml_object_detector.etl.download_images import download_image, setup_logs
from ml_object_detector.models.predictor import YoloPredictor


# helper
def run_custom_etl(queries: list[str], n: int, log) -> None:
    """
    Run a custom ETL process that downloads images based on the provided queries.

    Parameters
    ----------
    queries : list of str
        List of search queries to download images for.
    n : int
        Number of images to download for each query.
    log : logging.Logger
        Logger instance for logging messages.
    """
    if n == 0:
        log.info("Image predictor has been canceled, no images will be downloaded.")
        return False

    for query in queries:
        log.info("Downloading %d images for query: %s", n, query)
        download_image(query, n=n, log=log)

    log.info("ETL finished for: %s", ", ".join(queries))
    return True


# Main
def main() -> None:
    log = setup_logs()

    # Ask the user
    answer = (
        input(
            "Do you want to run the image recognition with the default settings (5 images for 'picnic' and 'surfing' each?) [yes/no]:\n"
        )
        .strip()
        .lower()
    )

    try:
        if answer in ("", "yes", "y"):
            # run existing ETL
            etl = importlib.import_module("ml_object_detector.cli.run_etl")
            etl.main()

        else:
            query_raw = input(
                "Enter up to 4 search terms (comma-separated):\n"
                "\tFor example: 'picnic, surfing, beach'\n"
            ).strip()
            queries = [q.strip() for q in query_raw.split(",") if q.strip()]

            if not queries:
                log.error("No valid queries provided - aborting.")
                return

            if len(queries) > 4:
                log.warning(
                    "You provided more than 4 queries, only the first 4 will be used."
                )
                queries = queries[:4]

            log.info("Queries to be used: %s", ", ".join(queries))

            n_raw = input(
                "How many images to download for each query? (Maximum is 15, default is 5)\n"
            ).strip()

            if n_raw == "":
                n = 5  # default value

            elif n_raw.isdigit():
                n = int(n_raw)

                if not 0 <= n <= 15:
                    log.error("Number of images must be between 0 and 15 - aborting.")
                    return

            else:
                raise ValueError("Invalid input for number of images.")


            downloaded = run_custom_etl(queries, n, log)
            if not downloaded:
                log.error("ETL process was canceled or failed - aborting.")
                return

    except ModuleNotFoundError as e:
        log.error("Could not locate ETL module: %s", e)
        return
    except RuntimeError as e:  # e.g. network failure in download_image
        log.error("Download failed: %s", e)
        return

    # Prepare source folder for prediction
    cfg = load_config()
    images_dir = Path(cfg["ROOT"]) / cfg["input_dir"]

    log.info("Starting the prediction pipeline...")
    log.info("Loading configuration from config.yaml")
    log.info("Using images from: %s", cfg["input_dir"])

    conf_raw = input(
        "Enter the confidence threshold for predictions (0.0 to 1.0, default is 0.8): "
    ).strip()

    if conf_raw == "":
        conf = float(cfg["confidence_threshold"])
    else:
        try:
            conf = float(conf_raw)
            if not (0.0 <= conf <= 1.0):
                raise ValueError("Confidence threshold must be between 0.0 and 1.0.")
        except ValueError:
            log.error("Invalid input for confidence threshold - aborting.")
            return

    # Check if images are found
    has_images = any(images_dir.glob("*.jpg")) \
                or any(images_dir.glob("*.jpeg")) \
                or any(images_dir.glob("*.png"))

    if not has_images:
        log.warning("No images found in %s - skipping prediction step.", images_dir)
        return

    YoloPredictor().predict_images_in_folder(folder=images_dir, conf=conf)

    log.info("Pipeline finished successfully.")


if __name__ == "__main__":
    main()
