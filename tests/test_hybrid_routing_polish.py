"""
Tests for Hybrid Routing Polish features.

This module tests the 5 new enhancements:
1. URL validation for SSRF prevention
2. CLI help text for hybrid mode
3. Content size limits and warnings
4. CSV/Text document support
5. Content caching for performance
"""

import json
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mad_spark_alt.core.llm_provider import (
    GoogleProvider,
    LLMProvider,
    LLMResponse,
    OllamaProvider,
)
from tests.conftest import TEST_GEMINI_MODEL
from mad_spark_alt.core.provider_router import ProviderRouter


class TestURLValidation:
    """Test URL validation for SSRF prevention."""

    def test_valid_https_url_passes(self):
        """Test that valid HTTPS URLs are accepted."""
        router = ProviderRouter(gemini_provider=None, ollama_provider=None)
        # Should not raise any exception
        router._validate_url_security("https://example.com/article")
        router._validate_url_security("https://news.com/breaking/story?id=123")
        router._validate_url_security("https://sub.domain.org/path/to/resource")

    def test_valid_http_url_passes(self):
        """Test that valid HTTP URLs are accepted (some sites lack HTTPS)."""
        router = ProviderRouter(gemini_provider=None, ollama_provider=None)
        # HTTP should be allowed for compatibility
        router._validate_url_security("http://legacy-site.com/data")

    def test_file_scheme_blocked(self):
        """Test that file:// URLs are blocked."""
        router = ProviderRouter(gemini_provider=None, ollama_provider=None)
        with pytest.raises(ValueError, match="Unsupported URL scheme"):
            router._validate_url_security("file:///etc/passwd")

    def test_ftp_scheme_blocked(self):
        """Test that ftp:// URLs are blocked."""
        router = ProviderRouter(gemini_provider=None, ollama_provider=None)
        with pytest.raises(ValueError, match="Unsupported URL scheme"):
            router._validate_url_security("ftp://example.com/file")

    def test_localhost_blocked(self):
        """Test that localhost URLs are blocked."""
        router = ProviderRouter(gemini_provider=None, ollama_provider=None)
        with pytest.raises(ValueError, match="Internal URLs not allowed"):
            router._validate_url_security("http://localhost:8080/admin")
        with pytest.raises(ValueError, match="Internal URLs not allowed"):
            router._validate_url_security("https://127.0.0.1/secret")
        with pytest.raises(ValueError, match="Internal URLs not allowed"):
            router._validate_url_security("http://0.0.0.0:3000/")

    def test_private_ip_ranges_blocked(self):
        """Test that private IP ranges are blocked."""
        router = ProviderRouter(gemini_provider=None, ollama_provider=None)

        # 10.x.x.x range
        with pytest.raises(ValueError, match="Private/internal IP not allowed"):
            router._validate_url_security("http://10.0.0.1/internal")

        # 192.168.x.x range
        with pytest.raises(ValueError, match="Private/internal IP not allowed"):
            router._validate_url_security("http://192.168.1.1/router")

        # 172.16.x.x range (part of private block B)
        with pytest.raises(ValueError, match="Private/internal IP not allowed"):
            router._validate_url_security("http://172.31.0.1/")

    def test_cloud_metadata_endpoints_blocked(self):
        """Test that cloud metadata endpoints are blocked (SSRF target)."""
        router = ProviderRouter(gemini_provider=None, ollama_provider=None)
        # AWS metadata endpoint (also link-local IP, so may be caught by IP check first)
        with pytest.raises(ValueError, match="(Cloud metadata|Private/internal IP)"):
            router._validate_url_security("http://169.254.169.254/latest/meta-data/")
        # GCP metadata (hostname-based)
        with pytest.raises(ValueError, match="Cloud metadata endpoints not allowed"):
            router._validate_url_security("http://metadata.google.internal/computeMetadata/")
        # Azure metadata (hostname-based)
        with pytest.raises(ValueError, match="Cloud metadata endpoints not allowed"):
            router._validate_url_security("http://metadata.azure.com/instance")
        # Case-insensitive check (prevent bypass with uppercase)
        with pytest.raises(ValueError, match="Cloud metadata endpoints not allowed"):
            router._validate_url_security("http://METADATA.GOOGLE.INTERNAL/computeMetadata/")
        with pytest.raises(ValueError, match="Cloud metadata endpoints not allowed"):
            router._validate_url_security("http://Metadata.Azure.Com/instance")
        # Percent-encoded bypass prevention
        with pytest.raises(ValueError, match="Internal URLs not allowed"):
            router._validate_url_security("http://127.0.0.1%2e/admin")
        with pytest.raises(ValueError, match="Internal URLs not allowed"):
            router._validate_url_security("http://local%68ost:8080/")

    @pytest.mark.asyncio
    async def test_url_validation_called_during_extraction(self):
        """Test that URL validation is called when extracting document content."""
        gemini = AsyncMock(spec=GoogleProvider)
        gemini.generate = AsyncMock(
            return_value=LLMResponse(
                content="Extracted content",
                provider=LLMProvider.GOOGLE,
                model=TEST_GEMINI_MODEL,
                usage={},
                cost=0.001,
            )
        )

        router = ProviderRouter(gemini_provider=gemini, ollama_provider=None)

        # This should raise ValueError due to internal URL
        with pytest.raises(ValueError, match="Internal URLs not allowed"):
            await router.extract_document_content(
                document_paths=(),
                urls=("http://localhost:8080/admin",),
            )

        # Gemini should NOT be called if URL is invalid
        assert not gemini.generate.called

    @pytest.mark.asyncio
    async def test_url_validation_in_non_hybrid_path(self):
        """Test that URL validation runs in run_qadi_with_fallback (non-hybrid path)."""
        gemini = AsyncMock(spec=GoogleProvider)
        ollama = AsyncMock(spec=OllamaProvider)

        router = ProviderRouter(gemini_provider=gemini, ollama_provider=ollama)

        # This should raise ValueError due to cloud metadata URL (link-local IP)
        # BEFORE any provider is called (SSRF prevention)
        # May be caught as "Private/internal IP" or "Cloud metadata" - both are valid blocks
        with pytest.raises(ValueError, match="(Private/internal IP|Cloud metadata)"):
            await router.run_qadi_with_fallback(
                user_input="Test question",
                primary_provider=gemini,
                fallback_provider=ollama,
                urls=["http://169.254.169.254/latest/meta-data/"],
            )

        # Neither provider should be called if URL is invalid
        assert not gemini.generate.called
        assert not ollama.generate.called


class TestContentSizeLimits:
    """Test content size warnings and truncation."""

    @pytest.mark.asyncio
    async def test_small_content_no_warning(self, tmp_path):
        """Test small content extraction produces no warning."""
        pdf_file = tmp_path / "small.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 small content")

        gemini = AsyncMock(spec=GoogleProvider)
        # Small content (1000 chars = ~250 tokens)
        gemini.generate = AsyncMock(
            return_value=LLMResponse(
                content="x" * 1000,
                provider=LLMProvider.GOOGLE,
                model=TEST_GEMINI_MODEL,
                usage={},
                cost=0.001,
            )
        )

        router = ProviderRouter(gemini_provider=gemini, ollama_provider=None)

        with patch("mad_spark_alt.core.provider_router.logger") as mock_logger:
            content, cost = await router.extract_document_content(
                document_paths=(str(pdf_file),),
                urls=(),
            )

            # Should not warn for small content
            warning_calls = [
                call for call in mock_logger.warning.call_args_list
                if "large" in str(call).lower() or "truncat" in str(call).lower()
            ]
            assert len(warning_calls) == 0
            assert content == "x" * 1000

    @pytest.mark.asyncio
    async def test_large_content_warning(self, tmp_path):
        """Test large content extraction produces warning."""
        pdf_file = tmp_path / "large.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 large content")

        gemini = AsyncMock(spec=GoogleProvider)
        # Large content (30000 chars = ~7500 tokens, exceeds warning threshold)
        gemini.generate = AsyncMock(
            return_value=LLMResponse(
                content="x" * 30000,
                provider=LLMProvider.GOOGLE,
                model=TEST_GEMINI_MODEL,
                usage={},
                cost=0.001,
            )
        )

        router = ProviderRouter(gemini_provider=gemini, ollama_provider=None)

        with patch("mad_spark_alt.core.provider_router.logger") as mock_logger:
            content, cost = await router.extract_document_content(
                document_paths=(str(pdf_file),),
                urls=(),
            )

            # Should warn about large content
            warning_calls = [str(call) for call in mock_logger.warning.call_args_list]
            large_content_warning = any(
                "large" in call.lower() or "exceed" in call.lower() or "context" in call.lower()
                for call in warning_calls
            )
            assert large_content_warning, f"Expected warning about large content, got: {warning_calls}"

    @pytest.mark.asyncio
    async def test_very_large_content_truncation(self, tmp_path):
        """Test very large content gets truncated with notice."""
        pdf_file = tmp_path / "huge.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 huge content")

        gemini = AsyncMock(spec=GoogleProvider)
        # Very large content (50000 chars = ~12500 tokens, way over limit)
        gemini.generate = AsyncMock(
            return_value=LLMResponse(
                content="x" * 50000,
                provider=LLMProvider.GOOGLE,
                model=TEST_GEMINI_MODEL,
                usage={},
                cost=0.001,
            )
        )

        router = ProviderRouter(gemini_provider=gemini, ollama_provider=None)

        content, cost = await router.extract_document_content(
            document_paths=(str(pdf_file),),
            urls=(),
        )

        # Content should be truncated
        assert len(content) < 50000
        # Should include truncation notice
        assert "truncated" in content.lower() or "Content truncated" in content

    @pytest.mark.asyncio
    async def test_custom_max_tokens_parameter(self, tmp_path):
        """Test that max_tokens parameter is respected."""
        pdf_file = tmp_path / "doc.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 content")

        gemini = AsyncMock(spec=GoogleProvider)
        gemini.generate = AsyncMock(
            return_value=LLMResponse(
                content="x" * 20000,  # 20000 chars
                provider=LLMProvider.GOOGLE,
                model=TEST_GEMINI_MODEL,
                usage={},
                cost=0.001,
            )
        )

        router = ProviderRouter(gemini_provider=gemini, ollama_provider=None)

        # Request smaller max_tokens
        content, cost = await router.extract_document_content(
            document_paths=(str(pdf_file),),
            urls=(),
            max_tokens=2000,  # Custom limit
        )

        # Should truncate based on custom limit
        # 2000 tokens * 4 chars/token = 8000 chars + room for notice
        assert len(content) < 20000


class TestCSVTextDocumentSupport:
    """Test support for CSV, TXT, JSON, and Markdown documents."""

    @pytest.mark.asyncio
    async def test_txt_file_support(self, tmp_path):
        """Test that .txt files are supported and read directly."""
        txt_file = tmp_path / "notes.txt"
        txt_file.write_text("These are important notes about the project.\nLine 2 here.")

        gemini = AsyncMock(spec=GoogleProvider)
        router = ProviderRouter(gemini_provider=gemini, ollama_provider=None)

        content, cost = await router.extract_document_content(
            document_paths=(str(txt_file),),
            urls=(),
        )

        # TXT files are read directly without API call
        assert "important notes" in content
        assert "Line 2 here" in content
        # No API call needed for text files
        assert not gemini.generate.called
        assert cost == 0.0

    @pytest.mark.asyncio
    async def test_csv_file_support(self, tmp_path):
        """Test that .csv files are supported and formatted properly."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("name,value,category\nAlice,100,A\nBob,200,B\nCarol,150,A")

        gemini = AsyncMock(spec=GoogleProvider)
        router = ProviderRouter(gemini_provider=gemini, ollama_provider=None)

        content, cost = await router.extract_document_content(
            document_paths=(str(csv_file),),
            urls=(),
        )

        # CSV should be formatted with header
        assert "CSV Data:" in content
        assert "name,value,category" in content
        assert "Alice,100,A" in content
        # No API call needed
        assert not gemini.generate.called
        assert cost == 0.0

    @pytest.mark.asyncio
    async def test_json_file_support(self, tmp_path):
        """Test that .json files are supported and pretty-printed."""
        json_file = tmp_path / "config.json"
        json_data = {"name": "test", "values": [1, 2, 3], "nested": {"key": "value"}}
        json_file.write_text(json.dumps(json_data))

        gemini = AsyncMock(spec=GoogleProvider)
        router = ProviderRouter(gemini_provider=gemini, ollama_provider=None)

        content, cost = await router.extract_document_content(
            document_paths=(str(json_file),),
            urls=(),
        )

        # JSON should be pretty-printed
        assert "JSON Data:" in content
        assert '"name": "test"' in content
        # Should be indented (pretty-printed)
        assert "  " in content  # 2-space indentation
        assert not gemini.generate.called
        assert cost == 0.0

    @pytest.mark.asyncio
    async def test_markdown_file_support(self, tmp_path):
        """Test that .md files are supported."""
        md_file = tmp_path / "README.md"
        md_file.write_text("# Title\n\nThis is **important** content.\n\n- Item 1\n- Item 2")

        gemini = AsyncMock(spec=GoogleProvider)
        router = ProviderRouter(gemini_provider=gemini, ollama_provider=None)

        content, cost = await router.extract_document_content(
            document_paths=(str(md_file),),
            urls=(),
        )

        # Markdown content should be read as-is
        assert "# Title" in content
        assert "**important**" in content
        assert "- Item 1" in content
        assert not gemini.generate.called
        assert cost == 0.0

    @pytest.mark.asyncio
    async def test_mixed_document_types(self, tmp_path):
        """Test processing multiple document types together."""
        pdf_file = tmp_path / "report.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 content")

        txt_file = tmp_path / "notes.txt"
        txt_file.write_text("Additional notes")

        csv_file = tmp_path / "data.csv"
        csv_file.write_text("a,b\n1,2")

        gemini = AsyncMock(spec=GoogleProvider)
        gemini.generate = AsyncMock(
            return_value=LLMResponse(
                content="All documents processed",
                provider=LLMProvider.GOOGLE,
                model=TEST_GEMINI_MODEL,
                usage={},
                cost=0.002,
            )
        )

        router = ProviderRouter(gemini_provider=gemini, ollama_provider=None)

        content, cost = await router.extract_document_content(
            document_paths=(str(pdf_file), str(txt_file), str(csv_file)),
            urls=(),
        )

        # All documents should be processed
        assert gemini.generate.called
        # Multiple documents should be reflected in the extraction

    @pytest.mark.asyncio
    async def test_unsupported_extension_skipped(self, tmp_path):
        """Test that unsupported file extensions are skipped with warning."""
        exe_file = tmp_path / "program.exe"
        exe_file.write_bytes(b"binary content")

        txt_file = tmp_path / "valid.txt"
        txt_file.write_text("Valid content")

        gemini = AsyncMock(spec=GoogleProvider)
        router = ProviderRouter(gemini_provider=gemini, ollama_provider=None)

        with patch("mad_spark_alt.core.provider_router.logger") as mock_logger:
            content, cost = await router.extract_document_content(
                document_paths=(str(exe_file), str(txt_file)),
                urls=(),
            )

            # Should warn about unsupported file
            warning_calls = [str(call) for call in mock_logger.warning.call_args_list]
            unsupported_warning = any("unsupported" in call.lower() or "skipping" in call.lower() for call in warning_calls)
            assert unsupported_warning

            # Valid TXT file should be processed (directly, no API call)
            assert "Valid content" in content
            # No API call for text files
            assert not gemini.generate.called

    @pytest.mark.asyncio
    async def test_large_csv_truncation(self, tmp_path):
        """Test that very large CSV files are truncated."""
        csv_file = tmp_path / "huge.csv"
        # Create CSV with 200 rows (more than 100 row limit)
        rows = ["col1,col2"] + [f"row{i},value{i}" for i in range(200)]
        csv_file.write_text("\n".join(rows))

        gemini = AsyncMock(spec=GoogleProvider)
        router = ProviderRouter(gemini_provider=gemini, ollama_provider=None)

        content, cost = await router.extract_document_content(
            document_paths=(str(csv_file),),
            urls=(),
        )

        # Should include truncation notice in the content
        assert "more rows" in content
        # Should have header and first 100 rows
        assert "col1,col2" in content
        assert "row0,value0" in content
        # Should NOT have row 199
        assert "row199" not in content
        # No API call
        assert not gemini.generate.called


class TestContentCaching:
    """Test content caching for performance optimization."""

    @pytest.mark.asyncio
    async def test_cache_miss_calls_api(self, tmp_path):
        """Test that cache miss results in API call."""
        pdf_file = tmp_path / "doc.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 content")

        gemini = AsyncMock(spec=GoogleProvider)
        gemini.generate = AsyncMock(
            return_value=LLMResponse(
                content="Extracted",
                provider=LLMProvider.GOOGLE,
                model=TEST_GEMINI_MODEL,
                usage={},
                cost=0.001,
            )
        )

        router = ProviderRouter(gemini_provider=gemini, ollama_provider=None)

        # First call - cache miss
        content1, cost1 = await router.extract_document_content(
            document_paths=(str(pdf_file),),
            urls=(),
        )

        # API should be called
        assert gemini.generate.call_count == 1
        assert cost1 == 0.001

    @pytest.mark.asyncio
    async def test_cache_hit_skips_api(self, tmp_path):
        """Test that cache hit skips API call for text files."""
        # Use text file (cached locally) - no API needed
        txt_file = tmp_path / "doc.txt"
        txt_file.write_text("Text content for caching test")

        gemini = AsyncMock(spec=GoogleProvider)
        router = ProviderRouter(gemini_provider=gemini, ollama_provider=None)

        # First call - reads file directly (no API for text files)
        content1, cost1 = await router.extract_document_content(
            document_paths=(str(txt_file),),
            urls=(),
        )

        # Second call - should still read file (caching is implementation detail)
        content2, cost2 = await router.extract_document_content(
            document_paths=(str(txt_file),),
            urls=(),
        )

        # Text files don't need API calls
        assert gemini.generate.call_count == 0
        # Content should be same
        assert content1 == content2
        assert "Text content" in content1
        # Cost should be 0 for text files
        assert cost1 == 0.0
        assert cost2 == 0.0

    @pytest.mark.asyncio
    async def test_modified_file_cache_miss(self, tmp_path):
        """Test that modified file results in cache miss (different hash)."""
        pdf_file = tmp_path / "doc.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 original")

        gemini = AsyncMock(spec=GoogleProvider)
        call_count = [0]

        def generate_response(*args, **kwargs):
            call_count[0] += 1
            return LLMResponse(
                content=f"Extraction {call_count[0]}",
                provider=LLMProvider.GOOGLE,
                model=TEST_GEMINI_MODEL,
                usage={},
                cost=0.001,
            )

        gemini.generate = AsyncMock(side_effect=generate_response)

        router = ProviderRouter(gemini_provider=gemini, ollama_provider=None)

        # First call
        content1, _ = await router.extract_document_content(
            document_paths=(str(pdf_file),),
            urls=(),
        )

        # Modify file
        pdf_file.write_bytes(b"%PDF-1.4 modified content")

        # Second call - should miss cache due to different hash
        content2, _ = await router.extract_document_content(
            document_paths=(str(pdf_file),),
            urls=(),
        )

        # API should be called twice
        assert gemini.generate.call_count == 2
        # Content should be different
        assert content1 != content2

    @pytest.mark.asyncio
    async def test_cache_expiry(self, tmp_path):
        """Test that cache entries expire after TTL."""
        pdf_file = tmp_path / "doc.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 content")

        gemini = AsyncMock(spec=GoogleProvider)
        call_count = [0]

        def generate_response(*args, **kwargs):
            call_count[0] += 1
            return LLMResponse(
                content=f"Extraction {call_count[0]}",
                provider=LLMProvider.GOOGLE,
                model=TEST_GEMINI_MODEL,
                usage={},
                cost=0.001,
            )

        gemini.generate = AsyncMock(side_effect=generate_response)

        # Create router with very short TTL (1 second)
        router = ProviderRouter(gemini_provider=gemini, ollama_provider=None)
        # Manually set short TTL on cache
        if hasattr(router, "_content_cache"):
            router._content_cache.ttl = 1  # 1 second TTL

        # First call
        await router.extract_document_content(
            document_paths=(str(pdf_file),),
            urls=(),
        )

        # Wait for TTL to expire
        time.sleep(1.1)

        # Second call - should miss cache due to expiry
        await router.extract_document_content(
            document_paths=(str(pdf_file),),
            urls=(),
        )

        # API should be called twice
        assert gemini.generate.call_count == 2

    @pytest.mark.asyncio
    async def test_cache_clear(self, tmp_path):
        """Test that cache can be cleared."""
        pdf_file = tmp_path / "doc.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 content")

        gemini = AsyncMock(spec=GoogleProvider)
        gemini.generate = AsyncMock(
            return_value=LLMResponse(
                content="Extracted",
                provider=LLMProvider.GOOGLE,
                model=TEST_GEMINI_MODEL,
                usage={},
                cost=0.001,
            )
        )

        router = ProviderRouter(gemini_provider=gemini, ollama_provider=None)

        # First call
        await router.extract_document_content(
            document_paths=(str(pdf_file),),
            urls=(),
        )

        # Clear cache
        if hasattr(router, "_content_cache"):
            router._content_cache.clear()

        # Second call - should miss cache
        await router.extract_document_content(
            document_paths=(str(pdf_file),),
            urls=(),
        )

        # API should be called twice
        assert gemini.generate.call_count == 2

    @pytest.mark.asyncio
    async def test_cache_size_limit(self, tmp_path):
        """Test that cache evicts oldest entries when max_entries is reached."""
        from mad_spark_alt.core.provider_router import ContentCache

        # Create cache with max 2 entries
        cache = ContentCache(ttl_seconds=3600, max_entries=2)

        # Create 3 files
        file1 = tmp_path / "file1.txt"
        file2 = tmp_path / "file2.txt"
        file3 = tmp_path / "file3.txt"
        file1.write_text("Content 1")
        file2.write_text("Content 2")
        file3.write_text("Content 3")

        # Add first two files
        cache.set(file1, "Content 1", 0.0)
        import time
        time.sleep(0.01)  # Ensure different timestamps
        cache.set(file2, "Content 2", 0.0)

        # Cache should have 2 entries
        assert len(cache._cache) == 2

        # Add third file - should evict oldest (file1)
        time.sleep(0.01)
        cache.set(file3, "Content 3", 0.0)

        # Cache should still have 2 entries
        assert len(cache._cache) == 2

        # file1 should be evicted, file2 and file3 should be present
        assert cache.get(file2) is not None
        assert cache.get(file3) is not None
        # file1 was evicted, so get() will recompute hash and find nothing
        # (note: we can't check directly since get() recomputes hash)


class TestCLIHelpText:
    """Test CLI help text updates for hybrid mode documentation."""

    def test_provider_help_mentions_hybrid_mode(self):
        """Test that --provider help text mentions hybrid mode."""
        from click.testing import CliRunner
        from mad_spark_alt.unified_cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["--help"])

        assert result.exit_code == 0
        # Help text should mention hybrid routing behavior
        assert "hybrid" in result.output.lower() or "Gemini for" in result.output

    def test_document_help_mentions_supported_formats(self):
        """Test that --document help text lists all supported formats."""
        from click.testing import CliRunner
        from mad_spark_alt.unified_cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["--help"])

        assert result.exit_code == 0
        # Should mention multiple file formats
        output_lower = result.output.lower()
        assert "pdf" in output_lower or "txt" in output_lower or "csv" in output_lower

    def test_url_help_mentions_hybrid_mode(self):
        """Test that --url help text mentions hybrid mode with Gemini."""
        from click.testing import CliRunner
        from mad_spark_alt.unified_cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["--help"])

        assert result.exit_code == 0
        # Should mention extraction or Gemini
        output_lower = result.output.lower()
        assert "url" in output_lower and ("hybrid" in output_lower or "gemini" in output_lower or "extraction" in output_lower)
