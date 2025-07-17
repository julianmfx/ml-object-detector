def main() -> None:
    """
    Run the FastAPI server that lives in fastapi_app.py
    Usage:
        $ ml-api # Featuls to 0.0.0.0:8000
        # ml-api --port 9000
    """
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="YOLO object-detector API")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true",
                        help="Auto-reload on code changes (dev only)")
    args = parser.parse_args()
    uvicorn.run(
        "ml_object_detector.app.fastapi_app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )
