#!/usr/bin/env python
from ml_object_detector.etl.download_images import download_image
import logging

def main() -> None:
    """Main function to run the ETL process."""
    logging.basicConfig(level=logging.INFO)
    download_image("picnic", n=10)
    download_image("surfing", n=10)
    logging.info("ETL finished.")

if __name__ == "__main__":
    main()
# This script is the entry point for running the ETL process.
