"""
Tests for multimodal utility functions.

This module tests the utility functions for handling multimodal inputs:
- MIME type detection
- Base64 encoding/decoding
- URL validation
- File size calculation
- PDF page counting
- Path resolution and validation
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from mad_spark_alt.utils.multimodal_utils import (
    detect_mime_type,
    read_file_as_base64,
    validate_url,
    get_file_size,
    get_pdf_page_count,
    resolve_file_path,
    validate_file_path,
)


class TestMIMETypeDetection:
    """Test MIME type detection for various file formats."""

    def test_detect_image_mime_types(self):
        """Test MIME type detection for image files."""
        assert detect_mime_type(Path("image.png")) == "image/png"
        assert detect_mime_type(Path("photo.jpg")) == "image/jpeg"
        assert detect_mime_type(Path("photo.jpeg")) == "image/jpeg"
        assert detect_mime_type(Path("graphic.webp")) == "image/webp"

    def test_detect_document_mime_types(self):
        """Test MIME type detection for document files."""
        assert detect_mime_type(Path("document.pdf")) == "application/pdf"
        assert detect_mime_type(Path("readme.txt")) == "text/plain"
        assert detect_mime_type(Path("README.md")) == "text/markdown"
        assert detect_mime_type(Path("index.html")) == "text/html"

    def test_detect_mime_type_case_insensitive(self):
        """Test MIME type detection is case-insensitive."""
        assert detect_mime_type(Path("IMAGE.PNG")) == "image/png"
        assert detect_mime_type(Path("DOCUMENT.PDF")) == "application/pdf"

    def test_detect_mime_type_unknown_extension(self):
        """Test MIME type detection returns default for unknown extensions."""
        mime_type = detect_mime_type(Path("file.unknown"))
        assert mime_type in ["application/octet-stream", None]  # Default behavior


class TestBase64Encoding:
    """Test base64 encoding/decoding for files."""

    def test_read_file_as_base64(self, tmp_path):
        """Test reading file and encoding as base64."""
        # Create test file
        test_file = tmp_path / "test.txt"
        test_content = b"Hello, World!"
        test_file.write_bytes(test_content)

        # Read and encode
        base64_data, mime_type = read_file_as_base64(test_file)

        # Verify
        assert isinstance(base64_data, str)
        assert len(base64_data) > 0
        assert mime_type == "text/plain"

        # Verify we can decode it back
        import base64
        decoded = base64.b64decode(base64_data)
        assert decoded == test_content

    def test_read_binary_file_as_base64(self, tmp_path):
        """Test reading binary file and encoding as base64."""
        # Create test binary file
        test_file = tmp_path / "test.png"
        test_content = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"  # PNG header
        test_file.write_bytes(test_content)

        # Read and encode
        base64_data, mime_type = read_file_as_base64(test_file)

        # Verify
        assert isinstance(base64_data, str)
        assert mime_type == "image/png"

        # Verify roundtrip
        import base64
        decoded = base64.b64decode(base64_data)
        assert decoded == test_content


class TestURLValidation:
    """Test URL validation."""

    def test_validate_url_with_http(self):
        """Test validation accepts HTTP URLs."""
        # Should not raise
        validate_url("http://example.com")
        validate_url("http://example.com/path")

    def test_validate_url_with_https(self):
        """Test validation accepts HTTPS URLs."""
        # Should not raise
        validate_url("https://example.com")
        validate_url("https://example.com/path?query=value")

    def test_validate_url_rejects_invalid_scheme(self):
        """Test validation rejects invalid URL schemes."""
        with pytest.raises(ValueError, match="Invalid URL scheme"):
            validate_url("ftp://example.com")
        with pytest.raises(ValueError, match="Invalid URL scheme"):
            validate_url("file:///path/to/file")

    def test_validate_url_rejects_malformed(self):
        """Test validation rejects malformed URLs."""
        with pytest.raises(ValueError, match="Invalid URL scheme"):
            validate_url("not a url")
        with pytest.raises(ValueError, match="non-empty string"):
            validate_url("")
        with pytest.raises(ValueError, match="domain name is missing"):
            validate_url("http://")


class TestFileSizeCalculation:
    """Test file size calculation."""

    def test_get_file_size(self, tmp_path):
        """Test getting file size in bytes."""
        # Create test file with known size
        test_file = tmp_path / "test.txt"
        test_content = b"A" * 1024  # 1KB
        test_file.write_bytes(test_content)

        # Get size
        size = get_file_size(test_file)

        assert size == 1024

    def test_get_file_size_empty_file(self, tmp_path):
        """Test getting size of empty file."""
        test_file = tmp_path / "empty.txt"
        test_file.write_bytes(b"")

        size = get_file_size(test_file)

        assert size == 0

    def test_get_file_size_large_file(self, tmp_path):
        """Test getting size of large file."""
        test_file = tmp_path / "large.bin"
        test_content = b"X" * (5 * 1024 * 1024)  # 5MB
        test_file.write_bytes(test_content)

        size = get_file_size(test_file)

        assert size == 5 * 1024 * 1024


class TestPDFPageCount:
    """Test PDF page counting."""

    def test_get_pdf_page_count_without_pypdf2(self):
        """Test PDF page count returns None when PyPDF2 not available."""
        with patch.dict('sys.modules', {'PyPDF2': None}):
            # Create fake path (doesn't need to exist for this test)
            result = get_pdf_page_count(Path("/fake/path.pdf"))

            # Should return None gracefully
            assert result is None

    @pytest.mark.skipif(True, reason="PyPDF2 is optional dependency")
    def test_get_pdf_page_count_with_pypdf2(self):
        """Test PDF page count with PyPDF2 available (skipped if not installed)."""
        # This test would require PyPDF2 to be installed
        # Kept as placeholder for when optional dependencies are installed
        pass


class TestPathResolution:
    """Test file path resolution and validation."""

    def test_resolve_file_path_relative(self, tmp_path):
        """Test resolving relative file path."""
        # Create test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("test")

        # Change to tmp directory
        original_cwd = Path.cwd()
        try:
            os.chdir(tmp_path)

            # Resolve relative path
            resolved = resolve_file_path("test.txt")

            assert resolved.is_absolute()
            assert resolved.exists()
            assert resolved.name == "test.txt"
        finally:
            os.chdir(original_cwd)

    def test_resolve_file_path_absolute(self, tmp_path):
        """Test resolving absolute file path."""
        # Create test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("test")

        # Resolve absolute path
        resolved = resolve_file_path(str(test_file))

        assert resolved.is_absolute()
        assert resolved == test_file

    def test_resolve_file_path_with_home_directory(self, tmp_path):
        """Test resolving path with ~ (home directory)."""
        # Create a test file in a known location
        import shutil

        # Create test file in tmp directory
        test_file = tmp_path / ".test_multimodal_file.txt"
        test_file.write_text("test")

        # Copy to home directory temporarily
        home_dir = Path.home()
        home_test_file = home_dir / ".test_multimodal_file.txt"

        try:
            shutil.copy(test_file, home_test_file)

            # Test resolving with ~
            resolved = resolve_file_path("~/.test_multimodal_file.txt")
            assert resolved.is_absolute()
            assert resolved.exists()
            assert resolved == home_test_file
        finally:
            # Clean up
            if home_test_file.exists():
                home_test_file.unlink()

    def test_resolve_file_path_nonexistent_raises_error(self):
        """Test resolving nonexistent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="File not found"):
            resolve_file_path("/nonexistent/path/to/file.txt")

    def test_resolve_file_path_directory_raises_error(self, tmp_path):
        """Test resolving directory (not file) raises ValueError."""
        with pytest.raises(ValueError, match="Path is not a file"):
            resolve_file_path(str(tmp_path))


class TestFilePathValidation:
    """Test file path validation for safety."""

    def test_validate_file_path_normal_file(self, tmp_path):
        """Test validation passes for normal file."""
        # Create small file
        test_file = tmp_path / "small.txt"
        test_file.write_text("small content")

        # Should not raise
        validate_file_path(test_file)

    def test_validate_file_path_large_file_raises(self, tmp_path):
        """Test validation rejects files exceeding size limit."""
        # Create file larger than default limit (50MB)
        test_file = tmp_path / "large.bin"
        # Write 51MB
        test_file.write_bytes(b"X" * (51 * 1024 * 1024))

        with pytest.raises(ValueError, match="File too large"):
            validate_file_path(test_file, max_size_mb=50)

    def test_validate_file_path_custom_size_limit(self, tmp_path):
        """Test validation with custom size limit."""
        # Create 2MB file
        test_file = tmp_path / "medium.bin"
        test_file.write_bytes(b"X" * (2 * 1024 * 1024))

        # Should raise with 1MB limit
        with pytest.raises(ValueError, match="File too large"):
            validate_file_path(test_file, max_size_mb=1)

        # Should pass with 3MB limit
        validate_file_path(test_file, max_size_mb=3)

    def test_validate_file_path_readable_check(self, tmp_path):
        """Test validation checks if file is readable."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        # Mock os.access to simulate unreadable file
        with patch('os.access', return_value=False):
            with pytest.raises(PermissionError, match="Cannot read file"):
                validate_file_path(test_file)


class TestPathExpansion:
    """Test path expansion with environment variables."""

    def test_resolve_file_path_with_env_var(self, tmp_path):
        """Test resolving path with environment variable."""
        # Create test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("test")

        # Set environment variable
        os.environ["TEST_DIR"] = str(tmp_path)

        try:
            # Resolve path with env variable
            resolved = resolve_file_path("$TEST_DIR/test.txt")

            assert resolved.exists()
            assert resolved.name == "test.txt"
        finally:
            # Clean up
            del os.environ["TEST_DIR"]
