"""
Tests for CLI export functionality.

Following TDD approach - tests written before implementation.
"""

import json
import pytest
from pathlib import Path
from click.testing import CliRunner
from unittest.mock import AsyncMock, patch, MagicMock

from mad_spark_alt.unified_cli import main
from mad_spark_alt.core.simple_qadi_orchestrator import SimpleQADIResult
from mad_spark_alt.core.phase_logic import HypothesisScore
from mad_spark_alt.evolution.interfaces import EvolutionResult


@pytest.fixture
def runner():
    """Click CLI test runner."""
    return CliRunner()


@pytest.fixture
def mock_provider_setup():
    """Mock provider setup for CLI tests in CI environment."""
    from mad_spark_alt.core.llm_provider import GoogleProvider, OllamaProvider

    with patch('mad_spark_alt.unified_cli.setup_llm_providers', new_callable=AsyncMock):
        # Mock get_google_provider to return a mock provider (must be instance of GoogleProvider)
        mock_gemini = MagicMock(spec=GoogleProvider)
        with patch('mad_spark_alt.unified_cli.get_google_provider', return_value=mock_gemini):
            # Mock OllamaProvider constructor to raise OSError (simulates CI environment without Ollama)
            # Keep OllamaProvider as real class so isinstance() checks work
            with patch.object(OllamaProvider, '__init__', side_effect=OSError("Ollama not available")):
                with patch.dict('os.environ', {'GOOGLE_API_KEY': 'test-key'}):
                    yield mock_gemini


@pytest.fixture
def mock_qadi_result():
    """Mock QADI result for testing."""
    from mad_spark_alt.core.interfaces import GeneratedIdea, ThinkingMethod

    return SimpleQADIResult(
        core_question="How to improve productivity?",
        hypotheses=["Time management", "Automation", "Focus techniques"],
        hypothesis_scores=[
            HypothesisScore(0.9, 0.8, 0.9, 0.7, 0.8, 0.82),
            HypothesisScore(0.8, 0.7, 0.6, 0.8, 0.7, 0.72),
            HypothesisScore(0.85, 0.75, 0.8, 0.75, 0.75, 0.78),
        ],
        final_answer="Combine time management with automation and focus techniques.",
        action_plan=["Set up calendar", "Automate repetitive tasks", "Practice deep work"],
        verification_examples=["Case study: Developer productivity increased 40%"],
        verification_conclusion="Proven effective across multiple domains.",
        total_llm_cost=0.0082,
        synthesized_ideas=[
            GeneratedIdea(
                content="Time management",
                thinking_method=ThinkingMethod.ABDUCTION,
                agent_name="Agent",
                generation_prompt="Prompt",
            ),
            GeneratedIdea(
                content="Automation",
                thinking_method=ThinkingMethod.ABDUCTION,
                agent_name="Agent",
                generation_prompt="Prompt",
            ),
        ],
    )


@pytest.fixture
def mock_evolution_result():
    """Mock evolution result for testing."""
    from mad_spark_alt.core.interfaces import GeneratedIdea, ThinkingMethod
    from mad_spark_alt.evolution.interfaces import IndividualFitness

    idea = GeneratedIdea(
        content="Evolved idea",
        thinking_method=ThinkingMethod.ABDUCTION,
        agent_name="Agent",
        generation_prompt="Prompt",
    )

    # Create a mock individual with fitness
    individual = IndividualFitness(
        idea=idea,
        impact=0.8,
        feasibility=0.8,
        accessibility=0.8,
        sustainability=0.8,
        scalability=0.8,
        overall_fitness=0.8,
    )

    return EvolutionResult(
        final_population=[individual],  # Non-empty for success=True
        best_ideas=[idea],
        generation_snapshots=[],
        total_generations=3,
        execution_time=90.0,
        evolution_metrics={"diversity_avg": 0.75},
    )


class TestCLIExportJSON:
    """Test JSON export via CLI."""

    def test_export_json_with_output_flag(self, runner, tmp_path, mock_qadi_result, mock_provider_setup):
        """Test that --output flag exports results to JSON file."""
        output_file = tmp_path / "test_output.json"

        # Mock the orchestrator
        with patch('mad_spark_alt.unified_cli.SimpleQADIOrchestrator') as mock_orch_class:
            mock_orch = AsyncMock()
            mock_orch.run_qadi_cycle.return_value = mock_qadi_result
            mock_orch_class.return_value = mock_orch

            # Provider setup already mocked by fixture
            result = runner.invoke(
                main,
                ['--output', str(output_file), '--format', 'json', 'Test question'],
                catch_exceptions=False
            )

        # Verify command completed
        assert result.exit_code == 0

        # Verify file was created
        assert output_file.exists()

        # Verify JSON content
        with open(output_file) as f:
            data = json.load(f)

        assert data["core_question"] == "How to improve productivity?"
        assert len(data["hypotheses"]) == 3
        assert data["final_answer"] == "Combine time management with automation and focus techniques."


    def test_export_json_default_format(self, runner, tmp_path, mock_qadi_result, mock_provider_setup):
        """Test that JSON is default format when --output is specified."""
        output_file = tmp_path / "output.json"

        with patch('mad_spark_alt.unified_cli.SimpleQADIOrchestrator') as mock_orch_class:
            mock_orch = AsyncMock()
            mock_orch.run_qadi_cycle.return_value = mock_qadi_result
            mock_orch_class.return_value = mock_orch

            # Provider setup already mocked by fixture
            result = runner.invoke(
                main,
                ['--output', str(output_file), 'Test question'],
                catch_exceptions=False
            )

        assert result.exit_code == 0
        assert output_file.exists()

    
    def test_export_json_with_evolution(self, runner, tmp_path, mock_qadi_result, mock_evolution_result):
        """Test exporting QADI + evolution results to JSON."""
        output_file = tmp_path / "evolution_output.json"

        with patch('mad_spark_alt.unified_cli.SimpleQADIOrchestrator') as mock_orch_class:
            mock_orch = AsyncMock()
            mock_orch.run_qadi_cycle.return_value = mock_qadi_result
            mock_orch_class.return_value = mock_orch

            with patch('mad_spark_alt.unified_cli.GeneticAlgorithm') as mock_ga_class:
                mock_ga = MagicMock()
                mock_ga.evolve = AsyncMock(return_value=mock_evolution_result)
                mock_ga_class.return_value = mock_ga

                with patch('mad_spark_alt.unified_cli.get_google_provider', return_value=MagicMock()):
                    with patch('mad_spark_alt.unified_cli.setup_llm_providers', new_callable=AsyncMock):
                        with patch.dict('os.environ', {'GOOGLE_API_KEY': 'test-key'}):
                            result = runner.invoke(
                                main,
                                ['--evolve', '--output', str(output_file), '--format', 'json', 'Test'],
                                catch_exceptions=False
                            )

        assert result.exit_code == 0
        assert output_file.exists()

        with open(output_file) as f:
            data = json.load(f)

        # Should have both QADI and evolution sections
        assert "qadi_analysis" in data
        assert "evolution_results" in data
        assert data["evolution_results"]["total_generations"] == 3


class TestCLIExportMarkdown:
    """Test Markdown export via CLI."""


    def test_export_markdown_with_format_flag(self, runner, tmp_path, mock_qadi_result, mock_provider_setup):
        """Test that --format md exports results to Markdown file."""
        output_file = tmp_path / "test_output.md"

        with patch('mad_spark_alt.unified_cli.SimpleQADIOrchestrator') as mock_orch_class:
            mock_orch = AsyncMock()
            mock_orch.run_qadi_cycle.return_value = mock_qadi_result
            mock_orch_class.return_value = mock_orch

            # Provider setup already mocked by fixture
            result = runner.invoke(
                main,
                ['--output', str(output_file), '--format', 'md', 'Test question'],
                catch_exceptions=False
            )

        assert result.exit_code == 0
        assert output_file.exists()

        content = output_file.read_text()

        assert "# QADI Analysis Results" in content
        assert "## Core Question" in content
        assert "How to improve productivity?" in content

    
    def test_export_markdown_with_evolution(self, runner, tmp_path, mock_qadi_result, mock_evolution_result):
        """Test exporting QADI + evolution to Markdown."""
        output_file = tmp_path / "evolution_output.md"

        with patch('mad_spark_alt.unified_cli.SimpleQADIOrchestrator') as mock_orch_class:
            mock_orch = AsyncMock()
            mock_orch.run_qadi_cycle.return_value = mock_qadi_result
            mock_orch_class.return_value = mock_orch

            with patch('mad_spark_alt.unified_cli.GeneticAlgorithm') as mock_ga_class:
                mock_ga = MagicMock()
                mock_ga.evolve = AsyncMock(return_value=mock_evolution_result)
                mock_ga_class.return_value = mock_ga

                with patch('mad_spark_alt.unified_cli.get_google_provider', return_value=MagicMock()):
                    with patch('mad_spark_alt.unified_cli.setup_llm_providers', new_callable=AsyncMock):
                        with patch.dict('os.environ', {'GOOGLE_API_KEY': 'test-key'}):
                            result = runner.invoke(
                                main,
                                ['--evolve', '--output', str(output_file), '--format', 'md', 'Test'],
                                catch_exceptions=False
                            )

        assert result.exit_code == 0
        assert output_file.exists()

        content = output_file.read_text()

        assert "# QADI Analysis Results" in content
        assert "## Evolution Results" in content


class TestCLIExportValidation:
    """Test export validation and error handling."""

    
    def test_invalid_format_rejected(self, runner):
        """Test that invalid format option is rejected."""
        with patch.dict('os.environ', {'GOOGLE_API_KEY': 'test-key'}):
            result = runner.invoke(
                main,
                ['--output', 'output.txt', '--format', 'txt', 'Test'],
            )

        # Should fail with invalid choice error
        assert result.exit_code != 0
        assert "txt" in result.output or "Invalid value" in result.output


    def test_output_creates_parent_directories(self, runner, tmp_path, mock_qadi_result, mock_provider_setup):
        """Test that export creates parent directories if needed."""
        output_file = tmp_path / "subdir1" / "subdir2" / "output.json"

        with patch('mad_spark_alt.unified_cli.SimpleQADIOrchestrator') as mock_orch_class:
            mock_orch = AsyncMock()
            mock_orch.run_qadi_cycle.return_value = mock_qadi_result
            mock_orch_class.return_value = mock_orch

            # Provider setup already mocked by fixture
            result = runner.invoke(
                main,
                ['--output', str(output_file), 'Test'],
                catch_exceptions=False
            )

        assert result.exit_code == 0
        assert output_file.exists()
        assert output_file.parent.exists()
