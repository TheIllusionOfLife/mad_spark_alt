"""
Tests for ProviderRouter - multi-provider routing and fallback logic.

This module tests intelligent provider selection and graceful degradation.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from mad_spark_alt.core.provider_router import ProviderRouter, ProviderSelection
from mad_spark_alt.core.llm_provider import (
    GoogleProvider,
    LLMProvider,
    LLMRequest,
    LLMResponse,
    OllamaProvider,
)
from mad_spark_alt.core.retry import LLMError
from mad_spark_alt.core.simple_qadi_orchestrator import SimpleQADIResult
from tests.conftest import TEST_GEMINI_MODEL


class TestProviderRouterSelection:
    """Test provider selection logic."""

    def test_router_initialization(self):
        """Test ProviderRouter initializes correctly."""
        gemini = MagicMock(spec=GoogleProvider)
        ollama = MagicMock(spec=OllamaProvider)

        router = ProviderRouter(gemini, ollama)

        assert router.gemini_provider == gemini
        assert router.ollama_provider == ollama
        assert router.default_strategy == ProviderSelection.AUTO

    def test_force_gemini_selection(self):
        """Test explicit Gemini provider selection."""
        gemini = MagicMock(spec=GoogleProvider)
        ollama = MagicMock(spec=OllamaProvider)
        router = ProviderRouter(gemini, ollama)

        provider, is_hybrid = router.select_provider(
            force_provider=ProviderSelection.GEMINI
        )

        assert provider == gemini
        assert is_hybrid is False

    def test_force_ollama_selection_text_only(self):
        """Test explicit Ollama selection for text input."""
        gemini = MagicMock(spec=GoogleProvider)
        ollama = MagicMock(spec=OllamaProvider)
        router = ProviderRouter(gemini, ollama)

        provider, is_hybrid = router.select_provider(
            has_documents=False,
            has_urls=False,
            force_provider=ProviderSelection.OLLAMA,
        )

        assert provider == ollama
        assert is_hybrid is False

    def test_force_ollama_with_documents_raises_error(self):
        """Test Ollama + documents raises validation error."""
        gemini = MagicMock(spec=GoogleProvider)
        ollama = MagicMock(spec=OllamaProvider)
        router = ProviderRouter(gemini, ollama)

        with pytest.raises(ValueError, match="doesn't support --document"):
            router.select_provider(
                has_documents=True, force_provider=ProviderSelection.OLLAMA
            )

    def test_force_ollama_with_urls_raises_error(self):
        """Test Ollama + URLs raises validation error."""
        gemini = MagicMock(spec=GoogleProvider)
        ollama = MagicMock(spec=OllamaProvider)
        router = ProviderRouter(gemini, ollama)

        with pytest.raises(ValueError, match="doesn't support --document"):
            router.select_provider(
                has_urls=True, force_provider=ProviderSelection.OLLAMA
            )

    def test_auto_selection_documents_to_gemini_hybrid(self):
        """Test auto-selection routes documents to Gemini (hybrid mode)."""
        gemini = MagicMock(spec=GoogleProvider)
        ollama = MagicMock(spec=OllamaProvider)
        router = ProviderRouter(gemini, ollama)

        provider, is_hybrid = router.select_provider(has_documents=True)

        assert provider == gemini
        assert is_hybrid is True  # Hybrid mode: Gemini preprocesses

    def test_auto_selection_urls_to_gemini_hybrid(self):
        """Test auto-selection routes URLs to Gemini (hybrid mode)."""
        gemini = MagicMock(spec=GoogleProvider)
        ollama = MagicMock(spec=OllamaProvider)
        router = ProviderRouter(gemini, ollama)

        provider, is_hybrid = router.select_provider(has_urls=True)

        assert provider == gemini
        assert is_hybrid is True

    def test_auto_selection_text_to_ollama(self):
        """Test auto-selection routes text-only to Ollama."""
        gemini = MagicMock(spec=GoogleProvider)
        ollama = MagicMock(spec=OllamaProvider)
        router = ProviderRouter(gemini, ollama)

        provider, is_hybrid = router.select_provider(
            has_documents=False, has_urls=False
        )

        assert provider == ollama
        assert is_hybrid is False

    def test_auto_selection_fallback_to_gemini_when_no_ollama(self):
        """Test auto-selection falls back to Gemini when Ollama unavailable."""
        gemini = MagicMock(spec=GoogleProvider)
        router = ProviderRouter(gemini_provider=gemini, ollama_provider=None)

        provider, is_hybrid = router.select_provider()

        assert provider == gemini
        assert is_hybrid is False

    def test_no_providers_raises_error(self):
        """Test error when no providers available."""
        router = ProviderRouter(gemini_provider=None, ollama_provider=None)

        with pytest.raises(ValueError, match="No LLM providers available"):
            router.select_provider()

    def test_documents_without_gemini_raises_error(self):
        """Test error when documents requested but Gemini unavailable."""
        ollama = MagicMock(spec=OllamaProvider)
        router = ProviderRouter(gemini_provider=None, ollama_provider=ollama)

        with pytest.raises(ValueError, match="require Gemini API"):
            router.select_provider(has_documents=True)


class TestProviderRouterFallback:
    """Test fallback mechanism."""

    @pytest.mark.asyncio
    async def test_primary_provider_success_no_fallback(self):
        """Test successful primary provider doesn't trigger fallback."""
        gemini = MagicMock(spec=GoogleProvider)
        ollama = MagicMock(spec=OllamaProvider)

        # Mock successful Ollama response
        mock_response = LLMResponse(
            content="Test response",
            provider=LLMProvider.OLLAMA,
            model="gemma3:12b",
            usage={"total_tokens": 100},
            cost=0.0,
            response_time=1.0,
        )
        ollama.generate = AsyncMock(return_value=mock_response)

        router = ProviderRouter(gemini, ollama)

        request = LLMRequest(user_prompt="Test")
        response = await router.generate_with_fallback(request, ollama)

        assert response.content == "Test response"
        assert response.provider == LLMProvider.OLLAMA
        ollama.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_ollama_failure_falls_back_to_gemini(self):
        """Test Ollama failure triggers Gemini fallback."""
        gemini = MagicMock(spec=GoogleProvider)
        ollama = MagicMock(spec=OllamaProvider)

        # Mock Ollama failure
        ollama.generate = AsyncMock(side_effect=Exception("Ollama server down"))

        # Mock Gemini success
        gemini_response = LLMResponse(
            content="Fallback response",
            provider=LLMProvider.GOOGLE,
            model=TEST_GEMINI_MODEL,
            usage={"total_tokens": 100},
            cost=0.001,
            response_time=2.0,
        )
        gemini.generate = AsyncMock(return_value=gemini_response)

        router = ProviderRouter(gemini, ollama)

        request = LLMRequest(user_prompt="Test")
        response = await router.generate_with_fallback(request, ollama)

        assert response.content == "Fallback response"
        assert response.provider == LLMProvider.GOOGLE
        ollama.generate.assert_called_once()
        gemini.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_both_providers_fail_raises_error(self):
        """Test both providers failing raises combined error."""
        gemini = MagicMock(spec=GoogleProvider)
        ollama = MagicMock(spec=OllamaProvider)

        # Mock both failing
        ollama.generate = AsyncMock(side_effect=Exception("Ollama down"))
        gemini.generate = AsyncMock(side_effect=Exception("Gemini API error"))

        router = ProviderRouter(gemini, ollama)

        request = LLMRequest(user_prompt="Test")

        with pytest.raises(LLMError, match="Both providers failed"):
            await router.generate_with_fallback(request, ollama)

    @pytest.mark.asyncio
    async def test_fallback_disabled_raises_primary_error(self):
        """Test fallback disabled raises original error."""
        gemini = MagicMock(spec=GoogleProvider)
        ollama = MagicMock(spec=OllamaProvider)

        # Mock Ollama failure
        ollama.generate = AsyncMock(side_effect=Exception("Ollama error"))

        router = ProviderRouter(gemini, ollama)

        request = LLMRequest(user_prompt="Test")

        with pytest.raises(Exception, match="Ollama error"):
            await router.generate_with_fallback(
                request, ollama, enable_fallback=False
            )

        # Gemini should not be called
        assert not gemini.generate.called


class TestProviderRouterStatus:
    """Test provider status reporting."""

    def test_status_with_both_providers(self):
        """Test status reporting with both providers available."""
        gemini = MagicMock(spec=GoogleProvider)
        gemini.get_available_models.return_value = [
            MagicMock(model_name=TEST_GEMINI_MODEL)
        ]

        ollama = MagicMock(spec=OllamaProvider)
        ollama.model = "gemma3:12b"
        ollama.base_url = "http://localhost:11434"

        router = ProviderRouter(gemini, ollama)
        status = router.get_provider_status()

        assert status["gemini"]["available"] is True
        assert status["gemini"]["model"] == TEST_GEMINI_MODEL
        assert status["ollama"]["available"] is True
        assert status["ollama"]["model"] == "gemma3:12b"

    def test_status_with_only_gemini(self):
        """Test status when only Gemini available."""
        gemini = MagicMock(spec=GoogleProvider)
        gemini.get_available_models.return_value = [
            MagicMock(model_name=TEST_GEMINI_MODEL)
        ]

        router = ProviderRouter(gemini_provider=gemini, ollama_provider=None)
        status = router.get_provider_status()

        assert status["gemini"]["available"] is True
        assert status["ollama"]["available"] is False

    def test_status_with_no_providers(self):
        """Test status when no providers available."""
        router = ProviderRouter(gemini_provider=None, ollama_provider=None)
        status = router.get_provider_status()

        assert status["gemini"]["available"] is False
        assert status["ollama"]["available"] is False


class TestQADICycleFallback:
    """Test QADI cycle fallback for SDK usage."""

    @pytest.mark.asyncio
    async def test_primary_provider_succeeds_no_fallback(self):
        """Test successful primary provider returns without fallback."""
        gemini = MagicMock(spec=GoogleProvider)
        ollama = MagicMock(spec=OllamaProvider)

        # Create mock QADI result
        mock_result = MagicMock(spec=SimpleQADIResult)
        mock_result.core_question = "Test question"
        mock_result.hypotheses = ["Hypothesis 1"]
        mock_result.evaluations = []
        mock_result.action_plan = []

        router = ProviderRouter(gemini, ollama)

        with patch(
            "mad_spark_alt.core.simple_qadi_orchestrator.SimpleQADIOrchestrator"
        ) as mock_orchestrator_class:
            mock_orchestrator = MagicMock()
            mock_orchestrator.run_qadi_cycle = AsyncMock(return_value=mock_result)
            mock_orchestrator_class.return_value = mock_orchestrator

            result, active_provider, used_fallback = await router.run_qadi_with_fallback(
                user_input="How can AI improve education?",
                primary_provider=ollama,
                fallback_provider=gemini,
            )

            assert result == mock_result
            assert active_provider == ollama
            assert used_fallback is False
            mock_orchestrator.run_qadi_cycle.assert_called_once()

    @pytest.mark.asyncio
    async def test_ollama_failure_falls_back_to_gemini(self):
        """Test Ollama failure triggers Gemini fallback for QADI cycle."""
        gemini = MagicMock(spec=GoogleProvider)
        ollama = MagicMock(spec=OllamaProvider)

        # Create mock QADI result for fallback
        mock_fallback_result = MagicMock(spec=SimpleQADIResult)
        mock_fallback_result.core_question = "Fallback question"
        mock_fallback_result.hypotheses = ["Fallback hypothesis"]

        router = ProviderRouter(gemini, ollama)

        with patch(
            "mad_spark_alt.core.simple_qadi_orchestrator.SimpleQADIOrchestrator"
        ) as mock_orchestrator_class:
            # First call (Ollama) fails, second call (Gemini) succeeds
            call_count = [0]

            def create_orchestrator(*args, **kwargs):
                call_count[0] += 1
                mock = MagicMock()
                if call_count[0] == 1:
                    # First orchestrator (Ollama) fails
                    mock.run_qadi_cycle = AsyncMock(
                        side_effect=ConnectionError("Ollama server unavailable")
                    )
                else:
                    # Second orchestrator (Gemini) succeeds
                    mock.run_qadi_cycle = AsyncMock(return_value=mock_fallback_result)
                return mock

            mock_orchestrator_class.side_effect = create_orchestrator

            result, active_provider, used_fallback = await router.run_qadi_with_fallback(
                user_input="How can AI improve education?",
                primary_provider=ollama,
                fallback_provider=gemini,
            )

            assert result == mock_fallback_result
            assert active_provider == gemini
            assert used_fallback is True
            assert call_count[0] == 2  # Two orchestrators created

    @pytest.mark.asyncio
    async def test_ollama_failure_with_keyword_detection(self):
        """Test Ollama failure detection with specific keywords."""
        gemini = MagicMock(spec=GoogleProvider)
        ollama = MagicMock(spec=OllamaProvider)

        mock_result = MagicMock(spec=SimpleQADIResult)

        router = ProviderRouter(gemini, ollama)

        # Test various Ollama failure keywords
        failure_messages = [
            "Failed to generate response",
            "Failed to parse JSON",
            "Failed to extract hypotheses",
            "Failed to score ideas",
            "Max retries exceeded for Ollama",
        ]

        for failure_msg in failure_messages:
            with patch(
                "mad_spark_alt.core.simple_qadi_orchestrator.SimpleQADIOrchestrator"
            ) as mock_orch_class:
                call_count = [0]

                def create_orch(*args, **kwargs):
                    call_count[0] += 1
                    mock = MagicMock()
                    if call_count[0] == 1:
                        mock.run_qadi_cycle = AsyncMock(
                            side_effect=RuntimeError(failure_msg)
                        )
                    else:
                        mock.run_qadi_cycle = AsyncMock(return_value=mock_result)
                    return mock

                mock_orch_class.side_effect = create_orch

                _, active_provider, used_fallback = await router.run_qadi_with_fallback(
                    user_input="Test",
                    primary_provider=ollama,
                    fallback_provider=gemini,
                )

                assert active_provider == gemini, f"Failed for: {failure_msg}"
                assert used_fallback is True, f"Failed for: {failure_msg}"

    @pytest.mark.asyncio
    async def test_non_ollama_failure_not_caught(self):
        """Test non-Ollama failures are not caught (prevents masking bugs)."""
        gemini = MagicMock(spec=GoogleProvider)
        ollama = MagicMock(spec=OllamaProvider)

        router = ProviderRouter(gemini, ollama)

        with patch(
            "mad_spark_alt.core.simple_qadi_orchestrator.SimpleQADIOrchestrator"
        ) as mock_orch_class:
            mock_orchestrator = MagicMock()
            # This is a programming bug, should not be caught
            mock_orchestrator.run_qadi_cycle = AsyncMock(
                side_effect=TypeError("Invalid argument type")
            )
            mock_orch_class.return_value = mock_orchestrator

            with pytest.raises(TypeError, match="Invalid argument type"):
                await router.run_qadi_with_fallback(
                    user_input="Test",
                    primary_provider=ollama,
                    fallback_provider=gemini,
                )

    @pytest.mark.asyncio
    async def test_no_fallback_provider_reraises_error(self):
        """Test error is reraised when no fallback provider available."""
        gemini = MagicMock(spec=GoogleProvider)
        ollama = MagicMock(spec=OllamaProvider)

        router = ProviderRouter(gemini, ollama)

        with patch(
            "mad_spark_alt.core.simple_qadi_orchestrator.SimpleQADIOrchestrator"
        ) as mock_orch_class:
            mock_orchestrator = MagicMock()
            mock_orchestrator.run_qadi_cycle = AsyncMock(
                side_effect=ConnectionError("Ollama connection failed")
            )
            mock_orch_class.return_value = mock_orchestrator

            # No fallback provider
            with pytest.raises(ConnectionError, match="Ollama connection failed"):
                await router.run_qadi_with_fallback(
                    user_input="Test",
                    primary_provider=ollama,
                    fallback_provider=None,  # No fallback
                )

    @pytest.mark.asyncio
    async def test_both_providers_fail_raises_original_error(self):
        """Test when both providers fail, the fallback error is raised."""
        gemini = MagicMock(spec=GoogleProvider)
        ollama = MagicMock(spec=OllamaProvider)

        router = ProviderRouter(gemini, ollama)

        with patch(
            "mad_spark_alt.core.simple_qadi_orchestrator.SimpleQADIOrchestrator"
        ) as mock_orch_class:
            call_count = [0]

            def create_orch(*args, **kwargs):
                call_count[0] += 1
                mock = MagicMock()
                if call_count[0] == 1:
                    mock.run_qadi_cycle = AsyncMock(
                        side_effect=ConnectionError("Ollama down")
                    )
                else:
                    mock.run_qadi_cycle = AsyncMock(
                        side_effect=Exception("Gemini API error")
                    )
                return mock

            mock_orch_class.side_effect = create_orch

            # Should raise the Gemini error (second failure)
            with pytest.raises(Exception, match="Gemini API error"):
                await router.run_qadi_with_fallback(
                    user_input="Test",
                    primary_provider=ollama,
                    fallback_provider=gemini,
                )

    @pytest.mark.asyncio
    async def test_sdk_usage_pattern(self):
        """Test SDK users can use fallback directly without CLI."""
        # This demonstrates the SDK usage pattern
        gemini = MagicMock(spec=GoogleProvider)
        ollama = MagicMock(spec=OllamaProvider)

        mock_result = MagicMock(spec=SimpleQADIResult)
        mock_result.core_question = "SDK Test"
        mock_result.hypotheses = ["SDK Hypothesis"]

        router = ProviderRouter(gemini, ollama)

        with patch(
            "mad_spark_alt.core.simple_qadi_orchestrator.SimpleQADIOrchestrator"
        ) as mock_orch_class:
            mock_orchestrator = MagicMock()
            mock_orchestrator.run_qadi_cycle = AsyncMock(return_value=mock_result)
            mock_orch_class.return_value = mock_orchestrator

            # SDK usage - configure temperature and hypotheses
            result, provider, fallback_used = await router.run_qadi_with_fallback(
                user_input="How can we improve user experience?",
                primary_provider=ollama,
                fallback_provider=gemini,
                temperature_override=0.9,
                num_hypotheses=5,
            )

            # Verify orchestrator was configured correctly
            mock_orch_class.assert_called_once_with(
                temperature_override=0.9,
                num_hypotheses=5,
                llm_provider=ollama,
            )

            assert result.core_question == "SDK Test"
            assert provider == ollama
            assert fallback_used is False

    @pytest.mark.asyncio
    async def test_parameters_passed_to_orchestrator(self):
        """Test all parameters are correctly passed to orchestrator."""
        gemini = MagicMock(spec=GoogleProvider)
        ollama = MagicMock(spec=OllamaProvider)

        mock_result = MagicMock(spec=SimpleQADIResult)
        router = ProviderRouter(gemini, ollama)

        with patch(
            "mad_spark_alt.core.simple_qadi_orchestrator.SimpleQADIOrchestrator"
        ) as mock_orch_class:
            mock_orchestrator = MagicMock()
            mock_orchestrator.run_qadi_cycle = AsyncMock(return_value=mock_result)
            mock_orch_class.return_value = mock_orchestrator

            multimodal_inputs = [MagicMock()]
            urls = ["https://example.com"]

            await router.run_qadi_with_fallback(
                user_input="Test input",
                primary_provider=ollama,
                fallback_provider=gemini,
                temperature_override=0.7,
                num_hypotheses=4,
                multimodal_inputs=multimodal_inputs,
                urls=urls,
            )

            # Verify run_qadi_cycle was called with correct parameters
            mock_orchestrator.run_qadi_cycle.assert_called_once_with(
                "Test input",
                multimodal_inputs=multimodal_inputs,
                urls=urls,
            )


class TestHybridRouting:
    """Test hybrid routing: Gemini extracts documents/URLs, Ollama runs QADI."""

    @pytest.mark.asyncio
    async def test_extract_document_content_with_urls(self):
        """Test Gemini extracts content from URLs."""
        gemini = AsyncMock(spec=GoogleProvider)
        gemini.generate = AsyncMock(
            return_value=LLMResponse(
                content="Extracted content from the webpage about AI safety...",
                provider=LLMProvider.GOOGLE,
                model=TEST_GEMINI_MODEL,
                usage={"prompt_tokens": 100, "completion_tokens": 200},
                cost=0.0012,
            )
        )

        router = ProviderRouter(gemini_provider=gemini, ollama_provider=None)

        extracted_text, cost = await router.extract_document_content(
            document_paths=(),
            urls=("https://example.com/article",),
        )

        assert "Extracted content" in extracted_text
        assert cost == 0.0012
        # Verify Gemini was called with URLs
        call_args = gemini.generate.call_args[0][0]
        assert call_args.urls == ["https://example.com/article"]

    @pytest.mark.asyncio
    async def test_extract_document_content_with_pdf(self, tmp_path):
        """Test Gemini extracts content from PDF documents."""
        # Create a mock PDF file
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 fake pdf content")

        gemini = AsyncMock(spec=GoogleProvider)
        gemini.generate = AsyncMock(
            return_value=LLMResponse(
                content="Extracted: Financial report shows Q3 revenue...",
                provider=LLMProvider.GOOGLE,
                model=TEST_GEMINI_MODEL,
                usage={"prompt_tokens": 500, "completion_tokens": 1000},
                cost=0.0045,
            )
        )

        router = ProviderRouter(gemini_provider=gemini, ollama_provider=None)

        extracted_text, cost = await router.extract_document_content(
            document_paths=(str(pdf_file),),
            urls=(),
        )

        assert "Financial report" in extracted_text
        assert cost == 0.0045
        # Verify Gemini was called with multimodal inputs (PDF)
        call_args = gemini.generate.call_args[0][0]
        assert call_args.multimodal_inputs is not None
        assert len(call_args.multimodal_inputs) == 1

    @pytest.mark.asyncio
    async def test_extract_document_content_mixed(self, tmp_path):
        """Test extraction with both documents and URLs."""
        pdf_file = tmp_path / "report.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 content")

        gemini = AsyncMock(spec=GoogleProvider)
        gemini.generate = AsyncMock(
            return_value=LLMResponse(
                content="Combined: PDF data plus web content...",
                provider=LLMProvider.GOOGLE,
                model=TEST_GEMINI_MODEL,
                usage={"prompt_tokens": 800, "completion_tokens": 1500},
                cost=0.0078,
            )
        )

        router = ProviderRouter(gemini_provider=gemini, ollama_provider=None)

        extracted_text, cost = await router.extract_document_content(
            document_paths=(str(pdf_file),),
            urls=("https://example.com/data",),
        )

        assert "Combined" in extracted_text
        assert cost == 0.0078

    @pytest.mark.asyncio
    async def test_extract_without_gemini_raises_error(self):
        """Test extraction fails without Gemini provider."""
        router = ProviderRouter(gemini_provider=None, ollama_provider=None)

        with pytest.raises(ValueError, match="Gemini provider required"):
            await router.extract_document_content(
                document_paths=(), urls=("https://example.com",)
            )

    @pytest.mark.asyncio
    async def test_extract_with_all_invalid_documents_raises_error(self, tmp_path):
        """Test extraction fails fast when all documents are invalid (not found or unsupported)."""
        # Create a non-supported file (binary executable)
        exe_file = tmp_path / "program.exe"
        exe_file.write_bytes(b"binary content")

        gemini = AsyncMock(spec=GoogleProvider)
        router = ProviderRouter(gemini_provider=gemini, ollama_provider=None)

        # Only unsupported file and no URLs - should raise ValueError
        with pytest.raises(ValueError, match="No valid documents or URLs"):
            await router.extract_document_content(
                document_paths=(str(exe_file),),
                urls=(),
            )

        # Gemini should NOT be called since no valid inputs
        assert not gemini.generate.called

    @pytest.mark.asyncio
    async def test_extract_with_missing_documents_raises_error(self):
        """Test extraction fails when all documents are missing."""
        gemini = AsyncMock(spec=GoogleProvider)
        router = ProviderRouter(gemini_provider=gemini, ollama_provider=None)

        # Non-existent files and no URLs - should raise ValueError
        with pytest.raises(ValueError, match="No valid documents or URLs"):
            await router.extract_document_content(
                document_paths=("/nonexistent/file.pdf", "/also/missing.pdf"),
                urls=(),
            )

        # Gemini should NOT be called
        assert not gemini.generate.called

    @pytest.mark.asyncio
    async def test_run_hybrid_qadi_gemini_extract_ollama_qadi(self, tmp_path):
        """Test full hybrid flow: Gemini extracts, Ollama runs QADI."""
        pdf_file = tmp_path / "data.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 content")

        # Mock Gemini for extraction
        gemini = AsyncMock(spec=GoogleProvider)
        gemini.generate = AsyncMock(
            return_value=LLMResponse(
                content="Market analysis shows 20% growth in Q3...",
                provider=LLMProvider.GOOGLE,
                model=TEST_GEMINI_MODEL,
                usage={"prompt_tokens": 300, "completion_tokens": 500},
                cost=0.0025,
            )
        )

        # Mock Ollama for QADI
        ollama = AsyncMock(spec=OllamaProvider)

        # Mock SimpleQADIOrchestrator and its result
        mock_result = SimpleQADIResult(
            core_question="What drives market growth?",
            hypotheses=["H1", "H2"],
            hypothesis_scores=[],
            final_answer="Analysis complete",
            action_plan=["Step 1"],
            verification_examples=["Example 1"],
            verification_conclusion="Valid",
            total_llm_cost=0.0,  # Ollama is free
        )

        router = ProviderRouter(gemini_provider=gemini, ollama_provider=ollama)

        with patch(
            "mad_spark_alt.core.simple_qadi_orchestrator.SimpleQADIOrchestrator"
        ) as mock_orch_class:
            mock_orchestrator = AsyncMock()
            mock_orchestrator.run_qadi_cycle = AsyncMock(return_value=mock_result)
            mock_orch_class.return_value = mock_orchestrator

            result, provider, used_fallback, metadata = await router.run_hybrid_qadi(
                user_input="Analyze market trends",
                document_paths=(str(pdf_file),),
                urls=(),
                image_paths=(),
                temperature_override=0.8,
                num_hypotheses=3,
            )

            # Verify Gemini was used for extraction
            assert gemini.generate.called

            # Verify Ollama was selected for QADI
            assert provider == ollama
            assert used_fallback is False

            # Verify metadata
            assert metadata["preprocessing_cost"] == 0.0025
            assert metadata["hybrid_mode"] is True
            assert metadata["documents_processed"] == 1

            # Verify QADI cycle received enhanced input with context
            call_args = mock_orchestrator.run_qadi_cycle.call_args[1]
            user_input_arg = call_args.get("user_input") or mock_orchestrator.run_qadi_cycle.call_args[0][0]
            assert "Market analysis shows 20% growth" in user_input_arg
            assert "Analyze market trends" in user_input_arg

    @pytest.mark.asyncio
    async def test_run_hybrid_qadi_images_passed_directly_to_ollama(self, tmp_path):
        """Test images go directly to Ollama (no preprocessing)."""
        # Create mock image file
        img_file = tmp_path / "chart.png"
        img_file.write_bytes(b"fake png data")

        gemini = AsyncMock(spec=GoogleProvider)
        gemini.generate = AsyncMock(
            return_value=LLMResponse(
                content="URL content here",
                provider=LLMProvider.GOOGLE,
                model=TEST_GEMINI_MODEL,
                usage={},
                cost=0.001,
            )
        )

        ollama = AsyncMock(spec=OllamaProvider)
        mock_result = SimpleQADIResult(
            core_question="Q",
            hypotheses=["H1"],
            hypothesis_scores=[],
            final_answer="Answer",
            action_plan=["Step"],
            verification_examples=["Ex"],
            verification_conclusion="Valid",
            total_llm_cost=0.0,
        )

        router = ProviderRouter(gemini_provider=gemini, ollama_provider=ollama)

        with patch(
            "mad_spark_alt.core.simple_qadi_orchestrator.SimpleQADIOrchestrator"
        ) as mock_orch_class:
            mock_orchestrator = AsyncMock()
            mock_orchestrator.run_qadi_cycle = AsyncMock(return_value=mock_result)
            mock_orch_class.return_value = mock_orchestrator

            _result, _provider, _used_fallback, _metadata = await router.run_hybrid_qadi(
                user_input="Analyze the chart",
                document_paths=(),
                urls=("https://example.com",),
                image_paths=(str(img_file),),
            )

            # Verify images were passed to Ollama's QADI cycle
            call_args = mock_orchestrator.run_qadi_cycle.call_args[1]
            multimodal = call_args.get("multimodal_inputs")
            assert multimodal is not None
            assert len(multimodal) == 1
            assert str(img_file) in multimodal[0].data

    @pytest.mark.asyncio
    async def test_run_hybrid_qadi_fallback_when_ollama_unavailable(self, tmp_path):
        """Test fallback to Gemini-only when Ollama unavailable."""
        pdf_file = tmp_path / "doc.pdf"
        pdf_file.write_bytes(b"%PDF-1.4")

        gemini = AsyncMock(spec=GoogleProvider)
        gemini.generate = AsyncMock(
            return_value=LLMResponse(
                content="Extracted content",
                provider=LLMProvider.GOOGLE,
                model=TEST_GEMINI_MODEL,
                usage={},
                cost=0.002,
            )
        )

        # No Ollama provider
        router = ProviderRouter(gemini_provider=gemini, ollama_provider=None)

        mock_result = SimpleQADIResult(
            core_question="Q",
            hypotheses=["H1"],
            hypothesis_scores=[],
            final_answer="Answer",
            action_plan=["Step"],
            verification_examples=["Ex"],
            verification_conclusion="Valid",
            total_llm_cost=0.01,
        )

        with patch(
            "mad_spark_alt.core.simple_qadi_orchestrator.SimpleQADIOrchestrator"
        ) as mock_orch_class:
            mock_orchestrator = AsyncMock()
            mock_orchestrator.run_qadi_cycle = AsyncMock(return_value=mock_result)
            mock_orch_class.return_value = mock_orchestrator

            _result, provider, used_fallback, metadata = await router.run_hybrid_qadi(
                user_input="Question",
                document_paths=(str(pdf_file),),
                urls=(),
                image_paths=(),
            )

            # Should fallback to Gemini
            assert provider == gemini
            assert used_fallback is True
            assert metadata["hybrid_mode"] is False  # Not truly hybrid

    @pytest.mark.asyncio
    async def test_run_hybrid_qadi_fallback_on_ollama_error(self, tmp_path):
        """Test fallback when Ollama fails during QADI."""
        pdf_file = tmp_path / "doc.pdf"
        pdf_file.write_bytes(b"%PDF-1.4")

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

        ollama = AsyncMock(spec=OllamaProvider)

        router = ProviderRouter(gemini_provider=gemini, ollama_provider=ollama)

        mock_result = SimpleQADIResult(
            core_question="Q",
            hypotheses=["H1"],
            hypothesis_scores=[],
            final_answer="Answer",
            action_plan=["Step"],
            verification_examples=["Ex"],
            verification_conclusion="Valid",
            total_llm_cost=0.01,
        )

        call_count = 0

        async def fail_on_first_call(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First call (Ollama) fails with Ollama-specific error
                raise ConnectionError("Ollama connection refused")
            return mock_result

        with patch(
            "mad_spark_alt.core.simple_qadi_orchestrator.SimpleQADIOrchestrator"
        ) as mock_orch_class:
            mock_orchestrator = AsyncMock()
            mock_orchestrator.run_qadi_cycle = AsyncMock(side_effect=fail_on_first_call)
            mock_orch_class.return_value = mock_orchestrator

            _result, provider, used_fallback, _metadata = await router.run_hybrid_qadi(
                user_input="Question",
                document_paths=(str(pdf_file),),
                urls=(),
                image_paths=(),
            )

            # Should fallback to Gemini after Ollama failure
            assert provider == gemini
            assert used_fallback is True

    @pytest.mark.asyncio
    async def test_run_hybrid_qadi_without_gemini_raises_error(self):
        """Test hybrid routing fails without Gemini provider."""
        ollama = AsyncMock(spec=OllamaProvider)
        router = ProviderRouter(gemini_provider=None, ollama_provider=ollama)

        with pytest.raises(ValueError, match="Gemini provider required"):
            await router.run_hybrid_qadi(
                user_input="Question",
                document_paths=(),
                urls=("https://example.com",),
                image_paths=(),
            )

    @pytest.mark.asyncio
    async def test_run_hybrid_qadi_metadata_tracking(self, tmp_path):
        """Test metadata accurately tracks hybrid operation details."""
        pdf1 = tmp_path / "doc1.pdf"
        pdf2 = tmp_path / "doc2.pdf"
        pdf1.write_bytes(b"%PDF-1.4")
        pdf2.write_bytes(b"%PDF-1.4")

        img = tmp_path / "image.jpg"
        img.write_bytes(b"fake jpg")

        gemini = AsyncMock(spec=GoogleProvider)
        gemini.generate = AsyncMock(
            return_value=LLMResponse(
                content="x" * 5000,  # 5000 characters extracted
                provider=LLMProvider.GOOGLE,
                model=TEST_GEMINI_MODEL,
                usage={},
                cost=0.0089,
            )
        )

        ollama = AsyncMock(spec=OllamaProvider)
        mock_result = SimpleQADIResult(
            core_question="Q",
            hypotheses=["H1"],
            hypothesis_scores=[],
            final_answer="Answer",
            action_plan=["Step"],
            verification_examples=["Ex"],
            verification_conclusion="Valid",
            total_llm_cost=0.0,
        )

        router = ProviderRouter(gemini_provider=gemini, ollama_provider=ollama)

        with patch(
            "mad_spark_alt.core.simple_qadi_orchestrator.SimpleQADIOrchestrator"
        ) as mock_orch_class:
            mock_orchestrator = AsyncMock()
            mock_orchestrator.run_qadi_cycle = AsyncMock(return_value=mock_result)
            mock_orch_class.return_value = mock_orchestrator

            _result, _provider, _used_fallback, metadata = await router.run_hybrid_qadi(
                user_input="Question",
                document_paths=(str(pdf1), str(pdf2)),
                urls=("https://url1.com", "https://url2.com", "https://url3.com"),
                image_paths=(str(img),),
            )

            # Verify all metadata fields
            assert metadata["preprocessing_cost"] == 0.0089
            assert metadata["extracted_content_length"] == 5000
            assert metadata["documents_processed"] == 2
            assert metadata["urls_processed"] == 3
            assert metadata["images_passed_to_ollama"] == 1
            assert metadata["hybrid_mode"] is True

    @pytest.mark.asyncio
    async def test_non_ollama_failure_not_caught_in_hybrid(self, tmp_path):
        """Test non-Ollama failures in hybrid routing are not caught (prevents masking bugs)."""
        pdf_file = tmp_path / "doc.pdf"
        pdf_file.write_bytes(b"%PDF-1.4")

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

        ollama = AsyncMock(spec=OllamaProvider)
        router = ProviderRouter(gemini_provider=gemini, ollama_provider=ollama)

        with patch(
            "mad_spark_alt.core.simple_qadi_orchestrator.SimpleQADIOrchestrator"
        ) as mock_orch_class:
            mock_orchestrator = AsyncMock()
            # This is a programming bug (TypeError), not Ollama failure - should NOT be caught
            mock_orchestrator.run_qadi_cycle = AsyncMock(
                side_effect=TypeError("Invalid argument type - programming bug")
            )
            mock_orch_class.return_value = mock_orchestrator

            # Programming bugs should propagate, not trigger fallback
            with pytest.raises(TypeError, match="Invalid argument type"):
                await router.run_hybrid_qadi(
                    user_input="Question",
                    document_paths=(str(pdf_file),),
                    urls=(),
                    image_paths=(),
                )

    @pytest.mark.asyncio
    async def test_empty_extraction_result_warning(self, tmp_path):
        """Test behavior when Gemini returns empty string from extraction."""
        pdf_file = tmp_path / "doc.pdf"
        pdf_file.write_bytes(b"%PDF-1.4")

        gemini = AsyncMock(spec=GoogleProvider)
        # Gemini returns empty string (document might be empty/unreadable)
        gemini.generate = AsyncMock(
            return_value=LLMResponse(
                content="",  # Empty extraction result
                provider=LLMProvider.GOOGLE,
                model=TEST_GEMINI_MODEL,
                usage={},
                cost=0.001,
            )
        )

        ollama = AsyncMock(spec=OllamaProvider)
        mock_result = SimpleQADIResult(
            core_question="Q",
            hypotheses=["H1"],
            hypothesis_scores=[],
            final_answer="Answer",
            action_plan=["Step"],
            verification_examples=["Ex"],
            verification_conclusion="Valid",
            total_llm_cost=0.0,
        )

        router = ProviderRouter(gemini_provider=gemini, ollama_provider=ollama)

        with patch(
            "mad_spark_alt.core.simple_qadi_orchestrator.SimpleQADIOrchestrator"
        ) as mock_orch_class:
            mock_orchestrator = AsyncMock()
            mock_orchestrator.run_qadi_cycle = AsyncMock(return_value=mock_result)
            mock_orch_class.return_value = mock_orchestrator

            _result, _provider, _used_fallback, metadata = await router.run_hybrid_qadi(
                user_input="Question",
                document_paths=(str(pdf_file),),
                urls=(),
                image_paths=(),
            )

            # Metadata should reflect empty extraction
            assert metadata["extracted_content_length"] == 0
            # QADI should still receive the (empty) context
            call_args = mock_orchestrator.run_qadi_cycle.call_args[1]
            user_input_arg = call_args.get("user_input") or mock_orchestrator.run_qadi_cycle.call_args[0][0]
            # Context block should be present but empty
            assert "Context from documents/URLs:" in user_input_arg
            assert "Question:" in user_input_arg
