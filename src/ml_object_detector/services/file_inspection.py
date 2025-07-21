"""
ml_object_detector.services.file_inspection.inspection
------------------------------------------------------

Async helper that *streams* an UploadFile to disk while applying the image
policy, without ever keeping the full payload in RAM.

Usage (in a FastAPI endpoint) ::

    from ml_object_detector.services.file_inspection.inspection import inspect_uploaded_file

    @router.post("/detect_upload")
    async def detect_upload(...):
        for up in files:
            await inspect_uploaded_file(up)  # raises InvalidImageError on failure
"""

import aiofiles
from magic import from_buffer
from .policy import ALOOWED_MIME, MAX_BYTES
from PIL import Image, UnidentifiedImageError
import io
from ml_object_detector.domain.errors import InvalidImageError
from ml_object_detector.config.load_config import load_config  # your existing helper
from ml_object_detector.services.file_inspection.policy import load_policy

cfg = load_config()
policy = load_policy(cfg)

CHUNK = 8192



async def inspect_uploaded_file(upfile) -> str:
    """
    upfile: UploadFile or any assync file-like object.
    Streams the first MAX_BYTES+1 to enforce size limit witout
    reading the whole payload into RAM.
    """

    buf = bytearray()
    async for chunk in upfile.stream(CHUNK):
        buf.extend(chunk)
        if len(buf) > MAX_BYTES:
            raise InvalidImageError(
                f"Image larger than {MAX_BYTES/1024.1024:.of}MB"
            )

    mime = from_buffer(buf, mime=True)
    if mime not in ALLOWED_MIME:
        raise InvalidImageError(f"Unsupported MIME type {mime!r}")

    try:
        Image.open(io.BytesIO(buf)).verify()
    except (UnidentifiedImageError, OSError) as exc:
        raise InvalidImageError("Corrupted or non-image file") from exc

    return mime
