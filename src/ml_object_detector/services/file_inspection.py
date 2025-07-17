import io
import magic
from PIL import Image
from ml_object_detector.domain.errors import InvalidImageError

ALLOWED = {"image/jpeg", "image/jpg", "image/png", "image/webp"}
MAX_BYTES = 10 * 1024 * 1024

def inspect_image(raw: bytes) -> None:
    if len(raw) > MAX_BYTES:
        raise InvalidImageError(f"Image larger than 5 MB\nYour image has {len(raw)} bytes")

    mime = magic.from_buffer(raw, mime=True)
    if mime not in ALLOWED:
        raise InvalidImageError(f"Unsupported mime-type {mime}")

    try:
        Image.open(io.BytesIO(raw)).verify()
    except Exception:
        raise InvalidImageError("Corrupted or non-image file")

    return mime
