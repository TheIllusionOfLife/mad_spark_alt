"""
End-to-end integration test for Ollama provider.

Tests the complete flow: CLI → ProviderRouter → Orchestrator → Ollama
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from mad_spark_alt.core.llm_provider import OllamaProvider, LLMProvider, LLMResponse
from mad_spark_alt.core.simple_qadi_orchestrator import SimpleQADIOrchestrator


@pytest.mark.ollama
@pytest.mark.integration
class TestOllamaEndToEndIntegration:
    """End-to-end tests validating full Ollama integration path."""

    @pytest.fixture
    def check_ollama_available(self):
        """Check if Ollama server is available."""
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('localhost', 11434))
        sock.close()
        if result != 0:
            pytest.skip("Ollama server not running on localhost:11434")

    @pytest.mark.asyncio
    async def test_orchestrator_uses_ollama_provider(
        self,
        check_ollama_available
    ):
        """Verify SimpleQADIOrchestrator actually uses provided Ollama provider."""
        ollama_provider = OllamaProvider()

        # Create orchestrator with Ollama provider
        orchestrator = SimpleQADIOrchestrator(
            temperature_override=0.7,
            num_hypotheses=3,
            llm_provider=ollama_provider
        )

        # Verify provider is stored
        assert orchestrator.llm_provider is not None
        assert isinstance(orchestrator.llm_provider, OllamaProvider)

        # Run QADI cycle - this will actually call Ollama
        result = await orchestrator.run_qadi_cycle(
            user_input="What is quantum computing in one sentence?",
            max_retries=2
        )

        # Verify results
        assert result.core_question is not None
        assert len(result.hypotheses) >= 3
        assert result.final_answer is not None

        # Verify Ollama was used (cost should be $0.00)
        assert result.total_llm_cost == 0.0

    @pytest.mark.asyncio
    async def test_provider_routing_with_force_ollama(self):
        """Test ProviderRouter correctly selects Ollama when forced."""
        from mad_spark_alt.core.provider_router import ProviderRouter, ProviderSelection
        from mad_spark_alt.core.llm_provider import GoogleProvider

        # Create both providers
        gemini_provider = GoogleProvider(api_key="test-key")
        ollama_provider = OllamaProvider()

        # Create router
        router = ProviderRouter(
            gemini_provider=gemini_provider,
            ollama_provider=ollama_provider
        )

        # Force Ollama selection - use enum for type safety
        selected, is_hybrid = router.select_provider(
            has_documents=False,
            has_urls=False,
            force_provider=ProviderSelection.OLLAMA
        )

        # Verify Ollama was selected
        assert isinstance(selected, OllamaProvider)
        assert is_hybrid is False

    def test_cli_to_ollama_integration_mock(self):
        """Test CLI → Router → Orchestrator path with mocked Ollama.

        This test verifies that provider routing correctly passes OllamaProvider
        to the orchestrator when --provider ollama is specified.
        """
        from click.testing import CliRunner
        from mad_spark_alt.unified_cli import main
        from mad_spark_alt.core.llm_provider import GoogleProvider
        from mad_spark_alt.core.simple_qadi_orchestrator import SimpleQADIResult

        runner = CliRunner()

        # Create a proper SimpleQADIResult mock (not MagicMock) to avoid attribute issues
        mock_result = SimpleQADIResult(
            core_question="Test question",
            hypotheses=["H1"],
            hypothesis_scores=[],
            final_answer="Test answer",
            action_plan=["Step 1", "Step 2", "Step 3"],
            verification_examples=[],
            verification_conclusion="Test",
            total_llm_cost=0.0,
            synthesized_ideas=[],
        )

        # Capture the primary_provider passed to run_qadi_with_fallback
        captured_provider = {}

        async def mock_run_qadi(**kwargs):
            captured_provider['primary'] = kwargs.get('primary_provider')
            return (mock_result, kwargs.get('primary_provider'), False)

        # Setup complete mock environment
        with patch('mad_spark_alt.unified_cli.load_env_file'):
            with patch('mad_spark_alt.unified_cli.os.getenv', return_value='test-key'):
                with patch('mad_spark_alt.unified_cli.setup_llm_providers', new_callable=AsyncMock):
                    mock_gemini = MagicMock(spec=GoogleProvider)
                    with patch('mad_spark_alt.unified_cli.get_google_provider', return_value=mock_gemini):
                        # Mock Ollama connectivity check to succeed
                        with patch('aiohttp.ClientSession.get') as mock_get:
                            mock_resp = AsyncMock()
                            mock_resp.status = 200
                            mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
                            mock_resp.__aexit__ = AsyncMock(return_value=False)
                            mock_get.return_value = mock_resp

                            # Mock the provider router's run_qadi_with_fallback method
                            with patch(
                                'mad_spark_alt.core.provider_router.ProviderRouter.run_qadi_with_fallback',
                                new=AsyncMock(side_effect=mock_run_qadi)
                            ):
                                # This simulates: msa "test" --provider ollama
                                result = runner.invoke(main, ['--provider', 'ollama', 'What is quantum computing?'])

        # Verify command completed successfully
        assert result.exit_code == 0
        # Verify Ollama provider was passed as primary
        assert 'primary' in captured_provider
        assert isinstance(captured_provider['primary'], OllamaProvider)

    @pytest.mark.asyncio
    async def test_ollama_fallback_to_gemini(self):
        """Test fallback from failed Ollama to Gemini."""
        from mad_spark_alt.core.provider_router import ProviderRouter
        from mad_spark_alt.core.llm_provider import GoogleProvider, LLMRequest
        from mad_spark_alt.core.retry import LLMError, ErrorType

        # Create providers
        gemini_provider = GoogleProvider(api_key="test-key")
        ollama_provider = OllamaProvider(base_url="http://localhost:99999")  # Wrong port

        router = ProviderRouter(
            gemini_provider=gemini_provider,
            ollama_provider=ollama_provider
        )

        # Mock Gemini response
        mock_gemini_response = LLMResponse(
            content="Fallback response",
            provider=LLMProvider.GOOGLE,
            model="gemini-2.5-flash",
            cost=0.001,
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            response_time=0.5
        )

        request = LLMRequest(user_prompt="Test")

        with patch.object(gemini_provider, 'generate', new=AsyncMock(return_value=mock_gemini_response)):
            # Should fail on Ollama, fall back to Gemini
            response = await router.generate_with_fallback(
                request,
                ollama_provider,
                enable_fallback=True
            )

            # Verify Gemini was used as fallback
            assert response.provider == LLMProvider.GOOGLE
            assert response.content == "Fallback response"
