"""
Test CLI/SDK parity - ensures CLI uses core SimpleQADIOrchestrator.

This test verifies that the CLI doesn't have custom orchestrator logic
that diverges from the SDK behavior.
"""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from mad_spark_alt.core.simple_qadi_orchestrator import SimpleQADIOrchestrator
from mad_spark_alt.unified_cli import main
from click.testing import CliRunner


def test_cli_imports_simple_qadi_orchestrator():
    """Verify CLI imports SimpleQADIOrchestrator from core."""
    from mad_spark_alt import unified_cli

    # Check that SimpleQADIOrchestrator is accessible
    assert hasattr(unified_cli, 'SimpleQADIOrchestrator')

    # Verify it's the core orchestrator, not a custom one
    from mad_spark_alt.core.simple_qadi_orchestrator import SimpleQADIOrchestrator as CoreOrchestrator
    assert unified_cli.SimpleQADIOrchestrator is CoreOrchestrator


def test_cli_does_not_have_simpler_qadi_orchestrator():
    """Verify SimplerQADIOrchestrator class doesn't exist in CLI."""
    from mad_spark_alt import unified_cli

    # SimplerQADIOrchestrator should NOT exist
    assert not hasattr(unified_cli, 'SimplerQADIOrchestrator')


def test_cli_does_not_have_simpler_qadi_prompts():
    """Verify SimplerQADIPrompts class doesn't exist in CLI."""
    from mad_spark_alt import unified_cli

    # SimplerQADIPrompts should NOT exist
    assert not hasattr(unified_cli, 'SimplerQADIPrompts')


def test_cli_instantiates_core_orchestrator():
    """Verify CLI instantiates the core SimpleQADIOrchestrator, not a custom one."""
    from mad_spark_alt.core.llm_provider import GoogleProvider, OllamaProvider as RealOllamaProvider

    runner = CliRunner()

    # Mock the orchestrator to track instantiation
    # Note: Since centralized fallback, orchestrator is now instantiated in ProviderRouter.run_qadi_with_fallback()
    # We need to patch at the source module (simple_qadi_orchestrator) where it's imported
    with patch('mad_spark_alt.core.simple_qadi_orchestrator.SimpleQADIOrchestrator') as MockOrchestrator:
        # Setup mock
        mock_instance = MagicMock()
        mock_result = MagicMock()
        mock_result.core_question = "Test question"
        mock_result.hypotheses = ["H1", "H2", "H3"]
        mock_result.hypothesis_scores = []
        mock_result.final_answer = "Test answer"
        mock_result.action_plan = []
        mock_result.verification_examples = []
        mock_result.verification_conclusion = "Test"
        mock_result.total_llm_cost = 0.0
        mock_result.synthesized_ideas = []

        mock_instance.run_qadi_cycle = AsyncMock(return_value=mock_result)
        MockOrchestrator.return_value = mock_instance

        # Mock all async dependencies
        with patch('mad_spark_alt.unified_cli.load_env_file'):
            with patch('mad_spark_alt.unified_cli.os.getenv', return_value='test-api-key'):
                with patch('mad_spark_alt.unified_cli.setup_llm_providers', new_callable=AsyncMock):
                    # Mock get_google_provider to return a mock provider (must be instance of GoogleProvider)
                    mock_gemini = MagicMock(spec=GoogleProvider)
                    with patch('mad_spark_alt.unified_cli.get_google_provider', return_value=mock_gemini):
                        # Mock OllamaProvider constructor to raise OSError (simulates CI environment without Ollama)
                        # Keep OllamaProvider as real class so isinstance() checks work
                        with patch.object(RealOllamaProvider, '__init__', side_effect=OSError("Ollama not available")):
                            # Run CLI (will use Gemini provider when Ollama unavailable)
                            result = runner.invoke(main, ['Test question'])

        # Verify SimpleQADIOrchestrator was instantiated with correct parameters
        MockOrchestrator.assert_called_once()
        args, kwargs = MockOrchestrator.call_args
        assert kwargs['temperature_override'] is None
        assert kwargs['num_hypotheses'] == 3
        assert 'llm_provider' in kwargs  # Provider is now passed from unified_cli


def test_cli_uses_core_qadi_prompts():
    """Verify CLI uses QADIPrompts from core, not custom prompts."""
    from mad_spark_alt.core.simple_qadi_orchestrator import SimpleQADIOrchestrator
    from mad_spark_alt.core.qadi_prompts import QADIPrompts

    # Create orchestrator instance
    orchestrator = SimpleQADIOrchestrator()

    # Verify it uses QADIPrompts
    assert hasattr(orchestrator, 'prompts')
    assert isinstance(orchestrator.prompts, QADIPrompts)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_cli_and_sdk_produce_same_phase1_format():
    """
    Integration test: Verify CLI and SDK produce the same Phase 1 output format.

    This test requires a real API key and is marked as integration.
    """
    import os
    from mad_spark_alt.core.simple_qadi_orchestrator import SimpleQADIOrchestrator

    # Skip if no API key
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        pytest.skip("GOOGLE_API_KEY not set")

    # Initialize orchestrator (same as CLI would use)
    orchestrator = SimpleQADIOrchestrator()

    # Test with a simple question
    test_question = "How can we reduce plastic waste?"

    # Run QADI cycle
    result = await orchestrator.run_qadi_cycle(test_question)

    # Verify core question format - should reliably start with "Q: "
    # as specified by get_questioning_prompt format requirement
    assert result.core_question.startswith("Q: "), "Core question should start with 'Q: '"
    assert len(result.core_question) > 10, "Core question should be substantial"

    # The question should be more than just a literal restatement
    # (this is the key difference between SimplerQADI and SimpleQADI)
    # SDK version should provide analytical refinement
    assert result.core_question != f"Q: {test_question}"
