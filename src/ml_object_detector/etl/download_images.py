"""Download images from Pexels API"""
import os
import requests
import logging
from pathlib import Path
from ml_object_detector.config.load_config import load_config
from dotenv import load_dotenv

load_dotenv()
cfg = load_config()
PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")
if not PEXELS_API_KEY:
    raise RuntimeError("PEXELS_API_KEY missing. Put it in .env.")
HEADERS = {"Authorization": PEXELS_API_KEY}
BASE_DIR = Path(cfg["base_dir"]).resolve()
DESTINATION_DIR = Path(BASE_DIR / cfg["input_dir"])
DESTINATION_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def download_image(query: str, n: int = 10) -> None:
    """Download *n* images for *query* into DESTINATION_DIR."""
    url = "https://api.pexels.com/v1/search"
    params = {"query": query, "per_page": n}
    response = requests.get(url, headers=HEADERS, params=params, timeout=30)
    response.raise_for_status()

    for photo in response.json()["photos"]:
        image_url = photo["src"]["original"]
        filename = DESTINATION_DIR / image_url.split("/")[-1].split("?")[0]

        if filename.exists():
            logging.info(f"File {filename} already exists, skipping download.")
            continue

        logging.info("Downloading %s", filename)
        image_bytes = requests.get(image_url, timeout=30).content
        filename.write_bytes(image_bytes)
