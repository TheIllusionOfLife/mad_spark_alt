"""Tests for Ollama image path normalization.

This module tests the fix for Japanese UAT Issue #1:
Ollama requires absolute image paths (or "./" prefix) to recognize images correctly.

The fix normalizes relative paths to absolute in OllamaProvider._build_messages().
"""

import socket
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio

from mad_spark_alt.core.llm_provider import LLMProvider, LLMRequest, OllamaProvider
from mad_spark_alt.core.multimodal import (
    MultimodalInput,
    MultimodalInputType,
    MultimodalSourceType,
)


class TestOllamaImagePathNormalization:
    """Unit tests for path normalization using mocks."""

    @pytest_asyncio.fixture
    async def mocked_provider_and_paths(self):
        """Fixture for OllamaProvider with mocked dependencies.

        Provides:
        - provider: OllamaProvider instance
        - captured_paths: list to capture paths passed to read_file_as_base64

        Automatically cleans up provider after test.
        """
        provider = OllamaProvider()
        captured_paths = []

        def capture_path(path):
            captured_paths.append(str(path))
            return ("iVBORw0KGgoAAAANS==", "image/png")

        mock_response_data = {
            "message": {"content": "Test description"},
            "done": True,
            "prompt_eval_count": 10,
            "eval_count": 5,
        }

        with patch(
            'mad_spark_alt.core.llm_provider.safe_aiohttp_request',
            new=AsyncMock(return_value=mock_response_data)
        ), patch(
            'mad_spark_alt.core.llm_provider.read_file_as_base64',
            side_effect=capture_path
        ):
            yield provider, captured_paths

        await provider.close()

    @pytest.mark.asyncio
    async def test_relative_path_gets_normalized_to_absolute(
        self, mocked_provider_and_paths
    ):
        """Test that relative paths are converted to absolute paths."""
        provider, captured_paths = mocked_provider_and_paths

        multimodal_input = MultimodalInput(
            input_type=MultimodalInputType.IMAGE,
            source_type=MultimodalSourceType.FILE_PATH,
            data="english.png",  # Relative path
            mime_type="image/png"
        )

        request = LLMRequest(
            user_prompt="What does this image say?",
            multimodal_inputs=[multimodal_input]
        )

        await provider.generate(request)

        # Verify path was normalized to absolute
        assert len(captured_paths) == 1
        captured = Path(captured_paths[0])
        assert captured.is_absolute()
        assert captured.name == "english.png"

    @pytest.mark.asyncio
    async def test_absolute_path_unchanged(self, mocked_provider_and_paths):
        """Test that absolute paths remain unchanged."""
        provider, captured_paths = mocked_provider_and_paths

        # Use a valid absolute path within project
        abs_path = str(Path.cwd() / "test_image.png")

        multimodal_input = MultimodalInput(
            input_type=MultimodalInputType.IMAGE,
            source_type=MultimodalSourceType.FILE_PATH,
            data=abs_path,
            mime_type="image/png"
        )

        request = LLMRequest(
            user_prompt="What does this image say?",
            multimodal_inputs=[multimodal_input]
        )

        await provider.generate(request)

        # Verify path is still absolute
        assert len(captured_paths) == 1
        captured = Path(captured_paths[0])
        assert captured.is_absolute()
        assert captured.name == "test_image.png"

    @pytest.mark.asyncio
    async def test_path_with_spaces_handled(self, mocked_provider_and_paths):
        """Test that paths with spaces are handled correctly."""
        provider, captured_paths = mocked_provider_and_paths

        multimodal_input = MultimodalInput(
            input_type=MultimodalInputType.IMAGE,
            source_type=MultimodalSourceType.FILE_PATH,
            data="file with spaces.png",
            mime_type="image/png"
        )

        request = LLMRequest(
            user_prompt="What does this image say?",
            multimodal_inputs=[multimodal_input]
        )

        await provider.generate(request)

        # Verify path with spaces was preserved
        assert len(captured_paths) == 1
        captured = Path(captured_paths[0])
        assert captured.is_absolute()
        assert captured.name == "file with spaces.png"

    @pytest.mark.asyncio
    async def test_unicode_filename_handled(self, mocked_provider_and_paths):
        """Test that Unicode filenames (Japanese, etc.) are handled correctly."""
        provider, captured_paths = mocked_provider_and_paths

        multimodal_input = MultimodalInput(
            input_type=MultimodalInputType.IMAGE,
            source_type=MultimodalSourceType.FILE_PATH,
            data="日本語.png",  # Japanese filename
            mime_type="image/png"
        )

        request = LLMRequest(
            user_prompt="この画像には何と書いてありますか？",
            multimodal_inputs=[multimodal_input]
        )

        await provider.generate(request)

        # Verify Unicode filename was preserved
        assert len(captured_paths) == 1
        captured = Path(captured_paths[0])
        assert captured.is_absolute()
        assert captured.name == "日本語.png"

    @pytest.mark.asyncio
    async def test_path_traversal_rejected(self):
        """Test that relative path traversal attempts are rejected.

        Security test: Ensures malicious relative paths like ../../../etc/passwd
        are rejected before file reading.
        """
        provider = OllamaProvider()
        try:
            with pytest.raises(ValueError, match="Relative path.*resolves outside project directory"):
                multimodal_input = MultimodalInput(
                    input_type=MultimodalInputType.IMAGE,
                    source_type=MultimodalSourceType.FILE_PATH,
                    data="../../../etc/passwd",
                    mime_type="image/png"
                )
                request = LLMRequest(
                    user_prompt="What does this say?",
                    multimodal_inputs=[multimodal_input]
                )
                await provider.generate(request)
        finally:
            await provider.close()

    @pytest.mark.asyncio
    async def test_absolute_path_outside_cwd_allowed(self):
        """Test that absolute paths outside CWD are allowed.

        Absolute paths like /tmp/image.png or /data/uploads/photo.jpg
        should be allowed - users explicitly specified them.
        Only relative paths that traverse outside CWD should be blocked.
        """
        provider = OllamaProvider()

        def mock_read_base64(path):
            # Mock successful read of absolute path outside CWD
            return ("iVBORw0KGgoAAAANS==", "image/png")

        with patch(
            'mad_spark_alt.core.llm_provider.safe_aiohttp_request',
            new=AsyncMock(return_value={"message": {"content": "test"}, "done": True})
        ), patch(
            'mad_spark_alt.core.llm_provider.read_file_as_base64',
            side_effect=mock_read_base64
        ):
            # Use absolute path outside CWD (e.g., /tmp)
            multimodal_input = MultimodalInput(
                input_type=MultimodalInputType.IMAGE,
                source_type=MultimodalSourceType.FILE_PATH,
                data="/tmp/test_image.png",
                mime_type="image/png"
            )
            request = LLMRequest(
                user_prompt="What does this say?",
                multimodal_inputs=[multimodal_input]
            )

            # Should NOT raise ValueError - absolute paths are allowed
            response = await provider.generate(request)
            assert response is not None

        await provider.close()


@pytest.mark.ollama
@pytest.mark.integration
class TestOllamaImagePathsIntegration:
    """Integration tests with real Ollama server and real image files."""

    @pytest.fixture
    def check_ollama_available(self):
        """Check if Ollama server is available."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('localhost', 11434))
        sock.close()
        if result != 0:
            pytest.skip("Ollama server not running on localhost:11434")

    @pytest.fixture
    def test_image_exists(self):
        """Check if test image exists."""
        image_path = Path("tests/fixtures/images/english.png")
        if not image_path.exists():
            pytest.skip(f"Test image not found: {image_path}")
        return image_path

    @pytest.mark.asyncio
    async def test_real_ollama_with_relative_path(
        self,
        check_ollama_available,
        test_image_exists
    ):
        """Test real Ollama with relative image path.

        This test verifies that Ollama can correctly read and understand
        image content when given a relative path like "tests/fixtures/images/english.png".
        """
        provider = OllamaProvider()

        # Use relative path (as users would naturally type it)
        relative_path = "tests/fixtures/images/english.png"

        multimodal_input = MultimodalInput(
            input_type=MultimodalInputType.IMAGE,
            source_type=MultimodalSourceType.FILE_PATH,
            data=relative_path,
            mime_type="image/png",
            file_size=Path(relative_path).stat().st_size
        )

        request = LLMRequest(
            user_prompt="What text does this image contain? Be specific.",
            multimodal_inputs=[multimodal_input],
            max_tokens=200
        )

        response = await provider.generate(request)

        # Verify response
        assert response.provider == LLMProvider.OLLAMA
        assert len(response.content) > 0
        assert response.cost == 0.0

        # The image should be understood correctly
        # english.png contains "I want to become an astronaut"
        content_lower = response.content.lower()
        assert "astronaut" in content_lower or "space" in content_lower, \
            f"Expected content about astronaut, got: {response.content}"

        print("\n[Relative Path Test]")
        print(f"  Path: {relative_path}")
        print(f"  Response Time: {response.response_time:.2f}s")
        print(f"  Content: {response.content[:100]}...")

        await provider.close()

    @pytest.mark.asyncio
    async def test_real_ollama_with_japanese_filename(
        self,
        check_ollama_available
    ):
        """Test real Ollama with Japanese filename.

        This test verifies that Ollama handles Unicode filenames correctly.
        """
        japanese_image = Path("tests/fixtures/images/日本語.png")
        if not japanese_image.exists():
            pytest.skip(f"Japanese test image not found: {japanese_image}")

        provider = OllamaProvider()

        multimodal_input = MultimodalInput(
            input_type=MultimodalInputType.IMAGE,
            source_type=MultimodalSourceType.FILE_PATH,
            data=str(japanese_image),
            mime_type="image/png",
            file_size=japanese_image.stat().st_size
        )

        request = LLMRequest(
            user_prompt="この画像には何と書いてありますか？",  # FULLWIDTH ? is correct for Japanese
            multimodal_inputs=[multimodal_input],
            max_tokens=200
        )

        response = await provider.generate(request)

        # Verify response
        assert response.provider == LLMProvider.OLLAMA
        assert len(response.content) > 0
        assert response.cost == 0.0

        # The image should be processed (actual content varies by model)
        assert response.response_time > 0

        print("\n[Japanese Filename Test]")
        print(f"  Path: {japanese_image}")
        print(f"  Response Time: {response.response_time:.2f}s")
        print(f"  Content: {response.content[:100]}...")

        await provider.close()

    @pytest.mark.asyncio
    async def test_real_ollama_with_absolute_path(
        self,
        check_ollama_available,
        test_image_exists
    ):
        """Test real Ollama with absolute path.

        This test verifies that absolute paths continue to work correctly.
        """
        # Get absolute path
        abs_path = test_image_exists.resolve()

        provider = OllamaProvider()

        multimodal_input = MultimodalInput(
            input_type=MultimodalInputType.IMAGE,
            source_type=MultimodalSourceType.FILE_PATH,
            data=str(abs_path),
            mime_type="image/png",
            file_size=abs_path.stat().st_size
        )

        request = LLMRequest(
            user_prompt="What text does this image contain?",
            multimodal_inputs=[multimodal_input],
            max_tokens=200
        )

        response = await provider.generate(request)

        # Verify response
        assert response.provider == LLMProvider.OLLAMA
        assert len(response.content) > 0
        assert response.cost == 0.0

        # Verify image content was understood
        content_lower = response.content.lower()
        assert "astronaut" in content_lower or "space" in content_lower, \
            f"Expected content about astronaut, got: {response.content}"

        print("\n[Absolute Path Test]")
        print(f"  Path: {abs_path}")
        print(f"  Response Time: {response.response_time:.2f}s")
        print(f"  Content: {response.content[:100]}...")

        await provider.close()
