# Briefing

This project main goal is to develop and strengthen my skills in logging, project architecture, Machine Learning (ML) pipelines, FastAPI development and Docker containerization.

---

# Current Features

The project includes a fully functional ML pipeline has a consistent pipeline, which can be launched by running the `ml-api` script in the terminal (inside a virtual enviroment). Once the local FastAPI server is up and running, it offers two key features:

## 1. Image Detection
Upload one or multiple images for object detection:

* **Single image**: returns the image with detected objects.
* **Bulk upload**: returns an HTML report summarizing all detected objects in the images.

## 2. Image Query and Download

Download images from an external API using a list of keywords (comma-separated):

* Returns an HTML report summarazing all detected objects in the images.

## Configuration options

For both use cases, users can:

* **Set a confidence threshold** for the object detection model.
* **Activate an email alert** if no object is detected in the input images.


# Development

The image detector is deployed through a API using FastAPI. Then, it will conteneirized with Docker so that any person can use it.

# Next steps

1. Restrict upload to images
2. Ship through docker to test in another computers.

## Further details

For both **bulk uploads** and **image queries**, the pipeline saves the raw and processed images into a local folder, uniquely identified by a `run_id`.

* Each `run_id` is constructed using a descriptive slug, which is bulk or query-based-slug combined with a timestamp  in the format `YYYY-MM-DD-T-HH-MM-SS`.
* This ensures that each executing has a distinct folder for traceability and debugging.
* Reports and images are stored within these run-specific folder for easier navigation and retrieval.
