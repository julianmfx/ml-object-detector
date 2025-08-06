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
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Tuple

import aiofiles
from magic import from_buffer
# Policy loaded dynamically below
from PIL import Image, UnidentifiedImageError
from starlette.datastructures import UploadFile

from ml_object_detector.domain.errors import InvalidImageError
from ml_object_detector.config.load_config import load_config  # your existing helper
from .policy import load_policy

# ---------------------------------------------------------------

cfg = load_config()
policy = load_policy(cfg)
CHUNK = 8192

# ---------------------------------------------------------------

async def inspect_uploaded_file(upfile: UploadFile) -> Tuple[str, Path]:
    """
    Stream *upfile* to a temporary file, validate it, and return:

        (mime_type, tmp_path)

    The caller can then move/rename `tmp_path` or unlink it.

    Raises
    ------
    InvalidImageError
        If the file is too large, not an allowed MIME type,
        or Pillow fails to verify it.
    """

    # 1) Create a named temporary file path (close df immediately)
    file_descriptor, temporary_name = tempfile.mkstemp(prefix="upload_", suffix=".img")
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)

    total = 0
    sniff = bytearray() # first 32 KB for MIME sniffing

    try:
        # 2) Stream from client to disk, chunk-by-chunk
        async with aiofiles.open(temporary_path, "wb") as destination:
            while True:
                chunk = await upfile.read(CHUNK)
                if not chunk:
                    break

                total += len(chunk)
                if total > policy.max_bytes:
                    raise InvalidImageError(
                        f"Image larger than {policy.max_bytes/1024/1024:.2f} MB"
                    )

                if len(sniff) < 32_768: # capture initial bytes
                    need = 32_768 - len(sniff)
                    sniff.extend(chunk[:need])

                await destination.write(chunk)

        # 3) Validate MIME type
        mime = from_buffer(bytes(sniff), mime=True).lower()
        if mime not in policy.allowed_mime:
            raise InvalidImageError(f"Unsupported MIME type {mime!r}")

        # 4) Let Pillow verify integrity (blocking but tiny files).
        with Image.open(temporary_path) as img:
            img.verify()

        return mime, temporary_path

    except (InvalidImageError, UnidentifiedImageError, OSError) as exc:
        # Ensure temp file is removed on any validation failure.
        temporary_path.unlink(missing_ok=True)
        raise InvalidImageError(str(exc)) from exc
