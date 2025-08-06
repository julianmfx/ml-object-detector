from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from ml_object_detector.services.file_inspection import inspect_uploaded_file
from ml_object_detector.domain.errors import InvalidImageError
from fastapi.responses import JSONResponse

router = APIRouter(prefix="/images", tags=["Upload"])

async def guard_image(file: UploadFile = File(...)) -> bytes:
    try:
        await inspect_uploaded_file(file)
        # Reset file pointer after inspection
        await file.seek(0)
        data = await file.read()
    except InvalidImageError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return data  # Verified image bytes

@router.post("/upload_image")
async def upload_image(img: bytes = Depends(guard_image)):
    # This endpoint receives a verified image file as bytes.
    # You can plug into model.predict_one() here if needed.
    return JSONResponse(content={"status": "received", "size_bytes": len(img)})
