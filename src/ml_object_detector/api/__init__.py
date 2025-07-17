from fastapi import FastAPI
from .home    import router as home_router
from .upload  import router as upload_router
from .detect  import router as detect_router

def register_routers(app: FastAPI) -> None:
    for r in (home_router, upload_router, detect_router):
        app.include_router(r)
