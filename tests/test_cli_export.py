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
def mock_qadi_result():
    """Mock QADI result for testing."""
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
    )


@pytest.fixture
def mock_evolution_result():
    """Mock evolution result for testing."""
    return EvolutionResult(
        final_population=[],
        best_ideas=[],
        generation_snapshots=[],
        total_generations=3,
        execution_time=90.0,
        evolution_metrics={"diversity_avg": 0.75},
    )


class TestCLIExportJSON:
    """Test JSON export via CLI."""

    @pytest.mark.asyncio
    async def test_export_json_with_output_flag(self, runner, tmp_path, mock_qadi_result):
        """Test that --output flag exports results to JSON file."""
        output_file = tmp_path / "test_output.json"

        # Mock the orchestrator
        with patch('mad_spark_alt.unified_cli.SimpleQADIOrchestrator') as mock_orch_class:
            mock_orch = AsyncMock()
            mock_orch.run_qadi_cycle.return_value = mock_qadi_result
            mock_orch_class.return_value = mock_orch

            # Mock LLM setup
            with patch('mad_spark_alt.unified_cli.setup_llm_providers', new_callable=AsyncMock):
                with patch.dict('os.environ', {'GOOGLE_API_KEY': 'test-key'}):
                    result = runner.invoke(
                        main,
                        ['Test question', '--output', str(output_file), '--format', 'json'],
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

    @pytest.mark.asyncio
    async def test_export_json_default_format(self, runner, tmp_path, mock_qadi_result):
        """Test that JSON is default format when --output is specified."""
        output_file = tmp_path / "output.json"

        with patch('mad_spark_alt.unified_cli.SimpleQADIOrchestrator') as mock_orch_class:
            mock_orch = AsyncMock()
            mock_orch.run_qadi_cycle.return_value = mock_qadi_result
            mock_orch_class.return_value = mock_orch

            with patch('mad_spark_alt.unified_cli.setup_llm_providers', new_callable=AsyncMock):
                with patch.dict('os.environ', {'GOOGLE_API_KEY': 'test-key'}):
                    result = runner.invoke(
                        main,
                        ['Test question', '--output', str(output_file)],
                        catch_exceptions=False
                    )

        assert result.exit_code == 0
        assert output_file.exists()

    @pytest.mark.asyncio
    async def test_export_json_with_evolution(self, runner, tmp_path, mock_qadi_result, mock_evolution_result):
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

                with patch('mad_spark_alt.unified_cli.setup_llm_providers', new_callable=AsyncMock):
                    with patch.dict('os.environ', {'GOOGLE_API_KEY': 'test-key'}):
                        result = runner.invoke(
                            main,
                            ['Test', '--evolve', '--output', str(output_file), '--format', 'json'],
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

    @pytest.mark.asyncio
    async def test_export_markdown_with_format_flag(self, runner, tmp_path, mock_qadi_result):
        """Test that --format md exports results to Markdown file."""
        output_file = tmp_path / "test_output.md"

        with patch('mad_spark_alt.unified_cli.SimpleQADIOrchestrator') as mock_orch_class:
            mock_orch = AsyncMock()
            mock_orch.run_qadi_cycle.return_value = mock_qadi_result
            mock_orch_class.return_value = mock_orch

            with patch('mad_spark_alt.unified_cli.setup_llm_providers', new_callable=AsyncMock):
                with patch.dict('os.environ', {'GOOGLE_API_KEY': 'test-key'}):
                    result = runner.invoke(
                        main,
                        ['Test question', '--output', str(output_file), '--format', 'md'],
                        catch_exceptions=False
                    )

        assert result.exit_code == 0
        assert output_file.exists()

        content = output_file.read_text()

        assert "# QADI Analysis Results" in content
        assert "## Core Question" in content
        assert "How to improve productivity?" in content

    @pytest.mark.asyncio
    async def test_export_markdown_with_evolution(self, runner, tmp_path, mock_qadi_result, mock_evolution_result):
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

                with patch('mad_spark_alt.unified_cli.setup_llm_providers', new_callable=AsyncMock):
                    with patch.dict('os.environ', {'GOOGLE_API_KEY': 'test-key'}):
                        result = runner.invoke(
                            main,
                            ['Test', '--evolve', '--output', str(output_file), '--format', 'md'],
                            catch_exceptions=False
                        )

        assert result.exit_code == 0
        assert output_file.exists()

        content = output_file.read_text()

        assert "# QADI Analysis Results" in content
        assert "## Evolution Results" in content


class TestCLIExportValidation:
    """Test export validation and error handling."""

    @pytest.mark.asyncio
    async def test_invalid_format_rejected(self, runner):
        """Test that invalid format option is rejected."""
        with patch.dict('os.environ', {'GOOGLE_API_KEY': 'test-key'}):
            result = runner.invoke(
                main,
                ['Test', '--output', 'output.txt', '--format', 'txt'],
            )

        # Should fail with invalid choice error
        assert result.exit_code != 0
        assert "txt" in result.output or "Invalid value" in result.output

    @pytest.mark.asyncio
    async def test_output_creates_parent_directories(self, runner, tmp_path, mock_qadi_result):
        """Test that export creates parent directories if needed."""
        output_file = tmp_path / "subdir1" / "subdir2" / "output.json"

        with patch('mad_spark_alt.unified_cli.SimpleQADIOrchestrator') as mock_orch_class:
            mock_orch = AsyncMock()
            mock_orch.run_qadi_cycle.return_value = mock_qadi_result
            mock_orch_class.return_value = mock_orch

            with patch('mad_spark_alt.unified_cli.setup_llm_providers', new_callable=AsyncMock):
                with patch.dict('os.environ', {'GOOGLE_API_KEY': 'test-key'}):
                    result = runner.invoke(
                        main,
                        ['Test', '--output', str(output_file)],
                        catch_exceptions=False
                    )

        assert result.exit_code == 0
        assert output_file.exists()
        assert output_file.parent.exists()
