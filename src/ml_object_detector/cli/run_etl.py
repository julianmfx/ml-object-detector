#!/usr/bin/env python
from ml_object_detector.etl.download_images import download_image
import logging

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    download_image("picnic", n=10)
    download_image("surfing", n=10)
    logging.info("ETL finished.")
