from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from ml_object_detector.services.file_inspection import inspect_image
from ml_object_detector.domain.errors import InvalidImageError
from fastapi.responses import JSONResponse

router = APIRouter(prefix="/images", tags=["Uploads"])

async def guard_image(file: UploadFile = File(...)) -> bytes:
    data = await file.read()
    try:
        inspect_image(data)
    except InvalidImageError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return data  # Verified image bytes

@router.post("/upload_image")
async def upload_image(img: bytes = Depends(guard_image)):
    # This endpoint receives a verified image file as bytes.
    # You can plug into model.predict_one() here if needed.
    return JSONResponse(content={"status": "received", "size_bytes": len(img)})
