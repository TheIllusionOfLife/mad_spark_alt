"""
Tests for --evaluate flag mode (creativity evaluation).

Tests the new --evaluate/--eval flag that replaces the broken 'evaluate' subcommand.
"""

import pytest
from click.testing import CliRunner
from unittest.mock import AsyncMock, MagicMock, patch
import tempfile
from pathlib import Path


class TestEvaluateFlagMode:
    """Test --evaluate flag functionality."""

    def test_evaluate_flag_with_text(self):
        """--evaluate flag should work with text argument."""
        from mad_spark_alt.unified_cli import main

        runner = CliRunner()
        result = runner.invoke(main, ['--evaluate', 'test text for evaluation'])

        # Should not fail with "No such command"
        assert result.exit_code != 2
        assert "No such command" not in result.output

    def test_eval_short_flag(self):
        """--eval should work as alias for --evaluate."""
        from mad_spark_alt.unified_cli import main

        runner = CliRunner()
        result = runner.invoke(main, ['--eval', 'test text'])

        assert result.exit_code != 2
        assert "No such command" not in result.output

    def test_evaluate_both_option_orders(self):
        """Options after positional should work (PR #157 compatibility)."""
        from mad_spark_alt.unified_cli import main

        runner = CliRunner()

        # Options before positional
        result1 = runner.invoke(main, ['--evaluate', 'test text'])

        # Options after positional (PR #157 feature)
        result2 = runner.invoke(main, ['test text', '--evaluate'])

        # Both should work
        assert result1.exit_code != 2
        assert result2.exit_code != 2

    def test_evaluate_with_evaluate_with_option(self):
        """--evaluate_with should filter evaluators."""
        from mad_spark_alt.unified_cli import main

        runner = CliRunner()
        result = runner.invoke(main, [
            '--evaluate', 'test text',
            '--evaluate_with', 'diversity_evaluator'
        ])

        assert result.exit_code != 2
        assert "No such option" not in result.output

    def test_evaluate_with_multiple_evaluators(self):
        """--evaluate_with should accept comma-separated evaluators."""
        from mad_spark_alt.unified_cli import main

        runner = CliRunner()
        result = runner.invoke(main, [
            '--evaluate', 'test text',
            '--evaluate_with', 'diversity_evaluator,quality_evaluator'
        ])

        assert result.exit_code != 2

    def test_evaluate_with_file_option(self):
        """--evaluate with --file should read text from file."""
        from mad_spark_alt.unified_cli import main

        runner = CliRunner()

        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write('test content for evaluation')
            temp_path = f.name

        try:
            result = runner.invoke(main, ['--evaluate', '--file', temp_path])
            assert result.exit_code != 2
            assert "No such option: --file" not in result.output
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_evaluate_with_output_options(self):
        """--evaluate should work with --output and --format options."""
        from mad_spark_alt.unified_cli import main

        runner = CliRunner()

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            output_path = f.name

        try:
            result = runner.invoke(main, [
                '--evaluate', 'test text',
                '--output', output_path,
                '--format', 'json'
            ])
            assert result.exit_code != 2
        finally:
            Path(output_path).unlink(missing_ok=True)

    def test_evaluate_with_evolve_error(self):
        """--evaluate and --evolve should be mutually exclusive."""
        from mad_spark_alt.unified_cli import main

        runner = CliRunner()
        result = runner.invoke(main, ['--evaluate', '--evolve', 'test text'])

        # Should exit with error
        assert result.exit_code == 1
        assert 'Cannot use --evolve with --evaluate' in result.output

    def test_evaluate_with_temperature_warning(self):
        """--temperature with --evaluate should show warning."""
        from mad_spark_alt.unified_cli import main

        runner = CliRunner()
        result = runner.invoke(main, [
            '--evaluate', 'test text',
            '--temperature', '1.5'
        ])

        # Should work but show warning
        assert 'Warning' in result.output
        assert 'temperature' in result.output.lower()

    def test_evaluate_with_image_warning(self):
        """--image with --evaluate should show warning."""
        from mad_spark_alt.unified_cli import main

        runner = CliRunner()

        # Create temporary image file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            temp_path = f.name

        try:
            result = runner.invoke(main, [
                '--evaluate', 'test text',
                '--image', temp_path
            ])

            # Should work but show warning
            assert 'Warning' in result.output
            assert '--image' in result.output or 'Multimodal' in result.output
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_evaluate_with_document_warning(self):
        """--document with --evaluate should show warning."""
        from mad_spark_alt.unified_cli import main

        runner = CliRunner()

        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as f:
            temp_path = f.name

        try:
            result = runner.invoke(main, [
                '--evaluate', 'test text',
                '--document', temp_path
            ])

            assert 'Warning' in result.output
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_evaluate_no_input_shows_help(self):
        """--evaluate without input should show help or error."""
        from mad_spark_alt.unified_cli import main

        runner = CliRunner()
        result = runner.invoke(main, ['--evaluate'])

        # Should either show help or error message
        assert result.exit_code != 0 or '--help' in result.output.lower()


class TestBackwardCompatibility:
    """Test that QADI mode and other commands still work."""

    def test_qadi_mode_still_works(self):
        """Default QADI mode should be unchanged."""
        from mad_spark_alt.unified_cli import main

        runner = CliRunner()

        with patch('mad_spark_alt.unified_cli._run_qadi_sync') as mock_qadi:
            with patch('os.getenv', return_value='fake-key'):
                result = runner.invoke(main, ['test question'])

        assert result.exit_code == 0
        assert mock_qadi.called

    def test_qadi_with_evolve_still_works(self):
        """--evolve flag should still work in QADI mode."""
        from mad_spark_alt.unified_cli import main

        runner = CliRunner()

        with patch('mad_spark_alt.unified_cli._run_qadi_sync') as mock_qadi:
            with patch('os.getenv', return_value='fake-key'):
                result = runner.invoke(main, ['test question', '--evolve'])

        assert result.exit_code == 0
        assert mock_qadi.called

    def test_list_evaluators_subcommand_still_works(self):
        """list-evaluators subcommand should be unchanged."""
        from mad_spark_alt.unified_cli import main

        runner = CliRunner()
        result = runner.invoke(main, ['list-evaluators'])

        assert result.exit_code == 0
        assert 'evaluator' in result.output.lower()
