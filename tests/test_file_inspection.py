#!/usr/bin/env python3
"""Quick test for file_inspection functionality"""

import io
from pathlib import Path

import pytest
from starlette.datastructures import UploadFile

from ml_object_detector.services.file_inspection import (inspect_uploaded_file, policy)
from ml_object_detector.domain.errors import InvalidImageError

# Helpers ---------------


class MemUploadFile(UploadFile):
    """UploadFile mock backed by bytes in memory"""

    def __init__(self, filename: str, content: bytes):
        super().__init__(
            file=io.BytesIO(content), filename=filename, size=len(content)
        )
        self._content = content

    async def stream(self, chunk_size: int):
        for i in range(0, len(self._content), chunk_size):
            yield self._content[i : i + chunk_size]


def load_sample_images() -> list[Path]:
    samples_dir = Path(__file__).parent.parent / "input_images"
    return (
        list(samples_dir.glob("*.jpg"))
        + list(samples_dir.glob("*.jpeg"))
        + list(samples_dir.glob("*.png"))
    )


# Tests ------------


@pytest.mark.asyncio
@pytest.mark.parametrize("image_path", load_sample_images(), ids=lambda p: p.name)
@pytest.mark.unit
async def test_good_images_pass(image_path: Path):
    """All valid sample images should pass the inspector."""
    content = image_path.read_bytes()
    upload = MemUploadFile(image_path.name, content)

    mime, temporary_path = await inspect_uploaded_file(upload)

    assert mime.startswith("image/")
    assert temporary_path.exists()
    temporary_path.unlink(missing_ok=True)  # Clean up


@pytest.mark.asyncio
@pytest.mark.unit
async def test_oversize_image_fails():
    """An image exceeding hard_limit_mb should raise InvalidImageError."""
    # Re-use a samll sample image and inflate it 20x to exceed 8 MB
    sample = load_sample_images()[0].read_bytes()
    inflated = sample * 100
    assert len(inflated) > policy.max_bytes

    upload = MemUploadFile("big.jpg", inflated)

    with pytest.raises(InvalidImageError, match="larger than"):
        await inspect_uploaded_file(upload)

    print(f"Oversize payload: {len(inflated)/1_048_576:.1f} MB")

@pytest.mark.asyncio
@pytest.mark.unit
async def test_wrong_mime_fails():
    """A non-image file should be rejected by MIME check."""
    bogus = b"%PDF-1.4 pretend I am a PDF file"
    upload = MemUploadFile("doc.pdf", bogus)

    with pytest.raises(InvalidImageError, match="MIME type"):
        await inspect_uploaded_file(upload)
