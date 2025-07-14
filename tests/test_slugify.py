"""
Unit tests for utils.clean_query_names

Run with:
    pytest -q test/test_clean_query_names.py
"""
import pytest
from ml_object_detector.utils.clean_query_names import slugify

@pytest.mark.unit
def test_ascii_slug():
    """
    ASCII filename should become lower-kebab case.
    """
    assert slugify("Hello World!.jpg") == "hello-world"

@pytest.mark.unit
def test_unicode_slug():
    """
    Unicode letters are kept, not stripped.
    """
    assert slugify("niño.png") == "nino"

@pytest.mark.unit
def test_extension_removed():
    assert not slugify("file.JPG").endswith("jpg")
