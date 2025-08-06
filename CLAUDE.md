# ML Object Detector - Claude Context

## Project Overview
A comprehensive ML object detection system built for learning MLOps skills. Uses YOLOv8 for object detection with both web API and CLI interfaces. Supports uploading images or querying Pexels API for detection with configurable confidence thresholds, email alerts, and HTML report generation.

## Architecture
- **API Layer**: FastAPI web application with file upload and detection endpoints
- **ML Core**: YOLOv8 wrapper for single/batch image prediction
- **Services**: File validation, detection orchestration
- **ETL Pipeline**: Pexels API integration for image downloads
- **Post-processing**: HTML report generation and email notifications
- **CLI Tools**: Interactive pipeline and ETL execution

## Key Files & Entry Points
- `src/ml_object_detector/api/fastapi_app.py` - Main FastAPI application
- `src/ml_object_detector/models/predictor.py` - YOLOv8 wrapper class
- `src/ml_object_detector/services/file_inspection.py` - File validation service
- `src/ml_object_detector/config/config.yaml` - Central configuration
- `pyproject.toml` - Dependencies and CLI command definitions

## CLI Commands
- `ml-api` - Start FastAPI web server
- `ml-pipeline` - Interactive detection pipeline
- `ml-etl` - Download images from Pexels API

## Development Commands
- **Test**: `pytest` (unit tests) or `pytest -m integration` (integration tests)
- **Lint**: `ruff check`
- **Format**: `ruff format`
- **Install**: `pip install -e .`
- **Install dev tools**: `pip install -e ".[dev]"`
- **Install API dependencies**: `pip install -e ".[api]"`

## Development Rules
- **Always use virtual environment**: All commands should run inside the venv
- **Explain code creation**: When creating new code, always explain the reasoning and purpose
- **Test dependencies**: Install test dependencies as needed (httpx for FastAPI testing)
- **Commit messages**: Never include "Generated with Claude Code" or similar phrases in commits

## Configuration Files

### pyproject.toml
- **Build system**: Uses setuptools with wheel
- **Dependencies**: Core deps include `aiofiles`, `python-magic`
- **Optional dependencies**:
  - `dev`: pre-commit, nbconvert, nbdime
  - `api`: FastAPI 0.115.14, uvicorn, python-multipart
- **CLI scripts**: Defines `ml-etl`, `ml-pipeline`, `ml-api` commands
- **Package structure**: Source code in `src/` directory

### pytest.ini
- **Test markers**:
  - `unit`: Fast, fully-mocked tests (default tier)
  - `integration`: Slow, needs external assets or heavy models

## Environment Variables
- `PEXELS_API_KEY` - Required for image downloads from Pexels API

## Current Branch
`refactor/api-structure` - Working on API structure improvements and file upload restrictions

## Project Goals
Learning and demonstrating MLOps skills including:
- Data Science workflows
- API development with FastAPI
- Logging and monitoring
- Testing strategies (unit/integration)
- Project architecture and clean code practices
