#!/usr/bin/env python
from ml_object_detector.etl.download_images import download_image
from ml_object_detector.etl.download_images import setup_logs

def main() -> None:
    """Main function to run the ETL process."""
    log = setup_logs()
    log.info("Starting ETL process...")
    download_image("picnic", n=5, log=log)
    download_image("surfing", n=5, log=log)
    log.info("ETL finished.")

if __name__ == "__main__":
    main()
# This script is the entry point for running the ETL process.
