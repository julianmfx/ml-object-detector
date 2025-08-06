#!/usr/bin/env python3
"""Tests for upload.py API endpoints"""

import io
import pytest
from fastapi.testclient import TestClient
from fastapi import FastAPI
from PIL import Image

from ml_object_detector.api.upload import router


@pytest.fixture
def app():
    """FastAPI app with upload router for testing"""
    app = FastAPI()
    app.include_router(router)
    return app


@pytest.fixture
def client(app):
    """Test client for API calls"""
    return TestClient(app)


@pytest.fixture
def valid_image_bytes():
    """Generate a valid PNG image as bytes"""
    img = Image.new("RGB", (100, 100), color="red")
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer.getvalue()


@pytest.fixture
def invalid_file_bytes():
    """Generate invalid file content"""
    return b"This is not an image file"


@pytest.mark.unit
def test_upload_valid_image(client, valid_image_bytes):
    """Test successful upload of a valid image"""
    response = client.post(
        "/images/upload_image",
        files={"file": ("test.png", valid_image_bytes, "image/png")}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "received"
    assert data["size_bytes"] == len(valid_image_bytes)


@pytest.mark.unit
def test_upload_invalid_file(client, invalid_file_bytes):
    """Test rejection of invalid file"""
    response = client.post(
        "/images/upload_image",
        files={"file": ("test.txt", invalid_file_bytes, "text/plain")}
    )

    assert response.status_code == 422
    assert "detail" in response.json()


@pytest.mark.unit
def test_upload_no_file(client):
    """Test endpoint with no file provided"""
    response = client.post("/images/upload_image")

    assert response.status_code == 422


@pytest.mark.unit
def test_upload_large_file(client):
    """Test rejection of oversized file"""
    # Create a large dummy file (1MB of random data)
    large_content = b"x" * (1024 * 1024)

    response = client.post(
        "/images/upload_image",
        files={"file": ("large.png", large_content, "image/png")}
    )

    # Should reject large files
    assert response.status_code == 422


@pytest.mark.unit
def test_upload_empty_file(client):
    """Test rejection of empty file"""
    response = client.post(
        "/images/upload_image",
        files={"file": ("empty.png", b"", "image/png")}
    )

    # Should reject empty files
    assert response.status_code == 422
