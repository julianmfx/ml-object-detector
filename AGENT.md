# ML Object Detector - Agent Guide

## Commands
- **Test**: `pytest` (run all), `pytest tests/test_predictor.py` (single test), `pytest -m unit` (unit only), `pytest -m integration` (integration only)
- **Pre-commit**: `pre-commit run --all-files` (lint/format), `pre-commit install` (setup hooks)
- **API**: `ml-api` (start FastAPI server), `ml-api --reload` (dev mode)
- **CLI**: `ml-etl` (ETL pipeline), `ml-pipeline` (ML pipeline)

## Architecture
- **Structure**: Clean architecture with `src/ml_object_detector/` containing `api/`, `cli/`, `config/`, `domain/`, `etl/`, `models/`, `postprocess/`, `services/`, `utils/`
- **API**: FastAPI app with endpoints for image detection and bulk processing
- **ML**: YOLO-based object detection with confidence thresholds
- **Storage**: Local filesystem with run-specific folders using `run_id` (slug + timestamp)
- **Config**: YAML-based configuration in `config/config.yaml`

## Code Style
- **Imports**: Absolute imports from `ml_object_detector.*`, standard library first, third-party second, local last
- **Types**: Use dataclasses and type hints (`from typing import List`)
- **Logging**: Use `setup_logs()` from `utils.logging`, log at info level for key operations
- **Paths**: Use `pathlib.Path`, load from config with `ROOT = Path(cfg["ROOT"])`
- **Tests**: pytest with custom markers (`@pytest.mark.unit`, `@pytest.mark.integration`), mock external dependencies
- **Naming**: Snake_case for variables/functions, PascalCase for classes (e.g., `YoloPredictor`)
