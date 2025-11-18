"""
Integration tests for Ollama image path handling.

This module tests that Ollama correctly handles various image path formats:
- Relative paths (e.g., "image.png")
- Absolute paths (e.g., "/full/path/to/image.png")
- Paths with spaces (e.g., "file with spaces.png")
- Paths with Unicode characters (e.g., "日本語.png")

These tests require:
1. Ollama server running (localhost:11434)
2. gemma3:12b-it-qat model installed
3. Test image files in the project root
"""

import os
from pathlib import Path
from unittest.mock import patch, AsyncMock

import pytest

from mad_spark_alt.core.llm_provider import (
    LLMProvider,
    LLMRequest,
    OllamaProvider,
)
from mad_spark_alt.core.multimodal import MultimodalInput, MultimodalInputType, MultimodalSourceType


class TestOllamaImagePathNormalization:
    """Test that OllamaProvider normalizes image paths correctly."""

    @pytest.mark.asyncio
    async def test_relative_path_gets_normalized_to_absolute(self):
        """Test that relative paths are converted to absolute paths.

        This is a unit test with mocked HTTP request to verify path normalization logic.
        """
        provider = OllamaProvider()

        # Mock the HTTP request
        mock_response_data = {
            "message": {"content": "Test image description"},
            "done": True,
            "prompt_eval_count": 10,
            "eval_count": 5,
        }

        # Track the actual path that gets passed to read_file_as_base64
        captured_paths = []

        original_read = None
        from mad_spark_alt.core.llm_provider import read_file_as_base64
        original_read = read_file_as_base64

        def capture_path(path):
            captured_paths.append(str(path))
            # Return mock base64 data
            return ("iVBORw0KGgoAAAANS==", "image/png")

        with patch(
            'mad_spark_alt.core.llm_provider.safe_aiohttp_request',
            new=AsyncMock(return_value=mock_response_data)
        ), patch(
            'mad_spark_alt.core.llm_provider.read_file_as_base64',
            side_effect=capture_path
        ):
            # Create request with relative path
            relative_path = "english.png"
            multimodal_input = MultimodalInput(
                input_type=MultimodalInputType.IMAGE,
                source_type=MultimodalSourceType.FILE_PATH,
                data=relative_path,
                mime_type="image/png"
            )

            request = LLMRequest(
                user_prompt="What does this image say?",
                multimodal_inputs=[multimodal_input]
            )

            await provider.generate(request)

            # Verify the path was normalized to absolute
            assert len(captured_paths) == 1
            captured = Path(captured_paths[0])

            # Path should be absolute after normalization
            assert captured.is_absolute(), f"Expected absolute path, got: {captured}"

            # Path should end with the original filename
            assert captured.name == "english.png"

        await provider.close()

    @pytest.mark.asyncio
    async def test_absolute_path_unchanged(self):
        """Test that absolute paths are passed through unchanged."""
        provider = OllamaProvider()

        mock_response_data = {
            "message": {"content": "Test description"},
            "done": True,
            "prompt_eval_count": 10,
            "eval_count": 5,
        }

        captured_paths = []

        def capture_path(path):
            captured_paths.append(str(path))
            return ("iVBORw0KGgoAAAANS==", "image/png")

        with patch(
            'mad_spark_alt.core.llm_provider.safe_aiohttp_request',
            new=AsyncMock(return_value=mock_response_data)
        ), patch(
            'mad_spark_alt.core.llm_provider.read_file_as_base64',
            side_effect=capture_path
        ):
            # Use absolute path
            absolute_path = str(Path("/tmp/test_image.png").absolute())
            multimodal_input = MultimodalInput(
                input_type=MultimodalInputType.IMAGE,
                source_type=MultimodalSourceType.FILE_PATH,
                data=absolute_path,
                mime_type="image/png"
            )

            request = LLMRequest(
                user_prompt="Describe this",
                multimodal_inputs=[multimodal_input]
            )

            await provider.generate(request)

            # Verify absolute path is still absolute
            assert len(captured_paths) == 1
            captured = Path(captured_paths[0])
            assert captured.is_absolute()
            assert str(captured) == absolute_path

        await provider.close()

    @pytest.mark.asyncio
    async def test_path_with_spaces_handled(self):
        """Test that paths with spaces are handled correctly.

        Python's Path handles spaces automatically when passed to file operations,
        but we need to ensure the path is absolute for Ollama compatibility.
        """
        provider = OllamaProvider()

        mock_response_data = {
            "message": {"content": "Image with spaces"},
            "done": True,
            "prompt_eval_count": 10,
            "eval_count": 5,
        }

        captured_paths = []

        def capture_path(path):
            captured_paths.append(str(path))
            return ("iVBORw0KGgoAAAANS==", "image/png")

        with patch(
            'mad_spark_alt.core.llm_provider.safe_aiohttp_request',
            new=AsyncMock(return_value=mock_response_data)
        ), patch(
            'mad_spark_alt.core.llm_provider.read_file_as_base64',
            side_effect=capture_path
        ):
            # Relative path with spaces
            relative_path_with_spaces = "file with spaces.png"
            multimodal_input = MultimodalInput(
                input_type=MultimodalInputType.IMAGE,
                source_type=MultimodalSourceType.FILE_PATH,
                data=relative_path_with_spaces,
                mime_type="image/png"
            )

            request = LLMRequest(
                user_prompt="What is this?",
                multimodal_inputs=[multimodal_input]
            )

            await provider.generate(request)

            # Verify path is absolute and contains spaces
            assert len(captured_paths) == 1
            captured = Path(captured_paths[0])
            assert captured.is_absolute()
            assert "file with spaces.png" in str(captured)

        await provider.close()

    @pytest.mark.asyncio
    async def test_unicode_filename_handled(self):
        """Test that Unicode filenames (Japanese) are handled correctly."""
        provider = OllamaProvider()

        mock_response_data = {
            "message": {"content": "Japanese filename image"},
            "done": True,
            "prompt_eval_count": 10,
            "eval_count": 5,
        }

        captured_paths = []

        def capture_path(path):
            captured_paths.append(str(path))
            return ("iVBORw0KGgoAAAANS==", "image/png")

        with patch(
            'mad_spark_alt.core.llm_provider.safe_aiohttp_request',
            new=AsyncMock(return_value=mock_response_data)
        ), patch(
            'mad_spark_alt.core.llm_provider.read_file_as_base64',
            side_effect=capture_path
        ):
            # Japanese filename
            japanese_filename = "日本語.png"
            multimodal_input = MultimodalInput(
                input_type=MultimodalInputType.IMAGE,
                source_type=MultimodalSourceType.FILE_PATH,
                data=japanese_filename,
                mime_type="image/png"
            )

            request = LLMRequest(
                user_prompt="何と書いてありますか？",
                multimodal_inputs=[multimodal_input]
            )

            await provider.generate(request)

            # Verify path is absolute and preserves Unicode
            assert len(captured_paths) == 1
            captured = Path(captured_paths[0])
            assert captured.is_absolute()
            assert "日本語.png" in str(captured)

        await provider.close()


@pytest.mark.ollama
@pytest.mark.integration
class TestOllamaImagePathsIntegration:
    """Integration tests with real Ollama server and real image files."""

    @pytest.fixture
    def check_ollama_available(self):
        """Check if Ollama server is available."""
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('localhost', 11434))
        sock.close()
        if result != 0:
            pytest.skip("Ollama server not running on localhost:11434")

    @pytest.fixture
    def test_image_exists(self):
        """Check if test image exists."""
        image_path = Path("english.png")
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
        image content when given a relative path like "english.png".
        """
        provider = OllamaProvider()

        # Use relative path (as users would naturally type it)
        relative_path = "english.png"

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

        print(f"\n[Relative Path Test]")
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
        japanese_image = Path("日本語.png")
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
            user_prompt="この画像には何と書いてありますか？",
            multimodal_inputs=[multimodal_input],
            max_tokens=200
        )

        response = await provider.generate(request)

        assert response.provider == LLMProvider.OLLAMA
        assert len(response.content) > 0

        # Should contain relevant Japanese content
        # 日本語.png contains "わたしの夢は宇宙飛行士です。"
        content = response.content
        assert "宇宙" in content or "astronaut" in content.lower(), \
            f"Expected content about space/astronaut, got: {content}"

        print(f"\n[Japanese Filename Test]")
        print(f"  Path: {japanese_image}")
        print(f"  Response Time: {response.response_time:.2f}s")
        print(f"  Content: {content[:100]}...")

        await provider.close()

    @pytest.mark.asyncio
    async def test_real_ollama_with_absolute_path(
        self,
        check_ollama_available,
        test_image_exists
    ):
        """Test real Ollama with absolute image path."""
        provider = OllamaProvider()

        # Use absolute path
        absolute_path = str(Path("english.png").absolute())

        multimodal_input = MultimodalInput(
            input_type=MultimodalInputType.IMAGE,
            source_type=MultimodalSourceType.FILE_PATH,
            data=absolute_path,
            mime_type="image/png",
            file_size=Path(absolute_path).stat().st_size
        )

        request = LLMRequest(
            user_prompt="What does this image say?",
            multimodal_inputs=[multimodal_input],
            max_tokens=200
        )

        response = await provider.generate(request)

        assert response.provider == LLMProvider.OLLAMA
        assert len(response.content) > 0

        content_lower = response.content.lower()
        assert "astronaut" in content_lower or "space" in content_lower, \
            f"Expected content about astronaut, got: {response.content}"

        print(f"\n[Absolute Path Test]")
        print(f"  Path: {absolute_path}")
        print(f"  Response Time: {response.response_time:.2f}s")
        print(f"  Content: {response.content[:100]}...")

        await provider.close()
