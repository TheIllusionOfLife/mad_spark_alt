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
            model="gemma3:12b-it-qat",
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
            model="gemini-2.5-flash",
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
            MagicMock(model_name="gemini-2.5-flash")
        ]

        ollama = MagicMock(spec=OllamaProvider)
        ollama.model = "gemma3:12b-it-qat"
        ollama.base_url = "http://localhost:11434"

        router = ProviderRouter(gemini, ollama)
        status = router.get_provider_status()

        assert status["gemini"]["available"] is True
        assert status["gemini"]["model"] == "gemini-2.5-flash"
        assert status["ollama"]["available"] is True
        assert status["ollama"]["model"] == "gemma3:12b-it-qat"

    def test_status_with_only_gemini(self):
        """Test status when only Gemini available."""
        gemini = MagicMock(spec=GoogleProvider)
        gemini.get_available_models.return_value = [
            MagicMock(model_name="gemini-2.5-flash")
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
