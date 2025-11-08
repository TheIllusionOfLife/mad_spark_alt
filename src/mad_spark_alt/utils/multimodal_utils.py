"""
Multimodal Utility Functions for Mad Spark Alt.

This module provides utility functions for handling multimodal inputs including:
- MIME type detection
- Base64 encoding for file data
- URL validation
- File size calculation
- PDF page counting (optional with PyPDF2)
- File path resolution and validation

These utilities support the multimodal feature set for processing images,
documents, and URLs in the QADI system.
"""

import base64
import logging
import mimetypes
import os
from pathlib import Path
from typing import Optional, Tuple
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


def detect_mime_type(file_path: Path) -> str:
    """
    Detect MIME type from file extension.

    Uses Python's mimetypes module with custom mappings for common formats.

    Args:
        file_path: Path to the file

    Returns:
        MIME type string (e.g., "image/png", "application/pdf")
        Returns "application/octet-stream" if type cannot be determined

    Example:
        >>> mime_type = detect_mime_type(Path("diagram.png"))
        >>> print(mime_type)  # "image/png"
    """
    # Get MIME type from extension
    mime_type, _ = mimetypes.guess_type(str(file_path))

    # If mimetypes doesn't recognize it, try custom mappings
    if mime_type is None:
        extension = file_path.suffix.lower()
        custom_mappings = {
            ".md": "text/markdown",
            ".markdown": "text/markdown",
            ".heic": "image/heic",
            ".heif": "image/heif",
        }
        mime_type = custom_mappings.get(extension, "application/octet-stream")

    return mime_type


def read_file_as_base64(file_path: Path) -> Tuple[str, str]:
    """
    Read file and encode as base64 string.

    Args:
        file_path: Path to the file to encode

    Returns:
        Tuple of (base64_string, mime_type)

    Raises:
        FileNotFoundError: If file doesn't exist
        PermissionError: If file is not readable

    Example:
        >>> base64_data, mime = read_file_as_base64(Path("image.png"))
        >>> print(f"Encoded {len(base64_data)} characters as {mime}")
    """
    # Validate file exists and is readable
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if not os.access(file_path, os.R_OK):
        raise PermissionError(f"Cannot read file: {file_path}")

    # Read file in binary mode
    with open(file_path, "rb") as f:
        file_data = f.read()

    # Encode to base64
    base64_data = base64.b64encode(file_data).decode("utf-8")

    # Detect MIME type
    mime_type = detect_mime_type(file_path)

    return base64_data, mime_type


def validate_url(url: str) -> bool:
    """
    Validate URL format and scheme.

    Checks that URL is properly formatted and uses HTTP/HTTPS scheme.

    Args:
        url: URL string to validate

    Returns:
        True if URL is valid, False otherwise

    Example:
        >>> validate_url("https://example.com/article")
        True
        >>> validate_url("not a url")
        False
    """
    if not url:
        return False

    try:
        parsed = urlparse(url)

        # Check scheme is http or https
        if parsed.scheme not in ["http", "https"]:
            return False

        # Check netloc (domain) is present
        if not parsed.netloc:
            return False

        return True

    except Exception as e:
        logger.debug(f"URL validation failed for '{url}': {e}")
        return False


def get_file_size(file_path: Path) -> int:
    """
    Get file size in bytes.

    Args:
        file_path: Path to the file

    Returns:
        File size in bytes

    Raises:
        FileNotFoundError: If file doesn't exist

    Example:
        >>> size = get_file_size(Path("document.pdf"))
        >>> print(f"File is {size / (1024*1024):.2f} MB")
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    return file_path.stat().st_size


def get_pdf_page_count(file_path: Path) -> Optional[int]:
    """
    Get page count from PDF file.

    Requires PyPDF2 package (optional dependency).
    Returns None if PyPDF2 is not installed or PDF cannot be read.

    Args:
        file_path: Path to PDF file

    Returns:
        Number of pages, or None if cannot be determined

    Example:
        >>> page_count = get_pdf_page_count(Path("report.pdf"))
        >>> if page_count:
        ...     print(f"PDF has {page_count} pages")
        ... else:
        ...     print("Could not determine page count")
    """
    try:
        import PyPDF2

        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            return len(reader.pages)

    except ImportError:
        logger.debug(
            "PyPDF2 not installed. Cannot count PDF pages. "
            "Install with: uv pip install 'mad-spark-alt[multimodal]'"
        )
        return None

    except Exception as e:
        logger.warning(f"Error counting PDF pages for {file_path}: {e}")
        return None


def resolve_file_path(file_path: str) -> Path:
    """
    Resolve and validate file path with proper expansion.

    Supports:
    - Relative paths (./file.png, ../file.png, file.png)
    - Absolute paths (/full/path/to/file.png)
    - Home directory (~/ or ~user/)
    - Environment variables ($HOME/file.png)

    Args:
        file_path: Path string from user input

    Returns:
        Resolved absolute Path object

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If path is a directory, not a file

    Example:
        >>> path = resolve_file_path("~/Documents/report.pdf")
        >>> print(path)  # /Users/name/Documents/report.pdf
    """
    # Expand ~ and environment variables
    expanded = os.path.expanduser(os.path.expandvars(file_path))

    # Convert to Path and resolve to absolute
    path = Path(expanded).resolve()

    # Validate existence
    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {file_path}\n"
            f"Resolved to: {path}\n"
            f"Current directory: {Path.cwd()}"
        )

    # Validate it's a file (not directory)
    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")

    return path


def validate_file_path(path: Path, max_size_mb: int = 50) -> None:
    """
    Validate file is safe to process.

    Checks:
    - File size is within limit
    - File is readable

    Args:
        path: Path object to validate
        max_size_mb: Maximum file size in MB (default: 50MB)

    Raises:
        ValueError: If file is too large
        PermissionError: If file is not readable

    Example:
        >>> path = Path("large_file.bin")
        >>> try:
        ...     validate_file_path(path, max_size_mb=20)
        ... except ValueError as e:
        ...     print(f"Validation failed: {e}")
    """
    # Size check (prevent huge files)
    size_bytes = path.stat().st_size
    size_mb = size_bytes / (1024 * 1024)

    if size_mb > max_size_mb:
        raise ValueError(
            f"File too large: {size_mb:.1f}MB (max {max_size_mb}MB)\n"
            f"For files >20MB, consider using Gemini File API\n"
            f"File: {path}"
        )

    # Readable check
    if not os.access(path, os.R_OK):
        raise PermissionError(f"Cannot read file: {path}")
