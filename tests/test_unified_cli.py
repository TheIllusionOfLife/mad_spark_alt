"""
Comprehensive tests for unified CLI with default QADI command.

Tests the new CLI structure where:
- Default command (no subcommand) runs QADI analysis
- Multimodal support in default command
- Subcommands for evolve, evaluate, batch-evaluate, compare, list-evaluators
"""

import pytest
from click.testing import CliRunner
from unittest.mock import AsyncMock, MagicMock, patch
import tempfile
from pathlib import Path


class TestUnifiedCLIDefaultCommand:
    """Test default QADI command (no subcommand needed)."""

    def test_help_shows_default_usage(self):
        """Help text should show QADI as default command."""
        from mad_spark_alt.unified_cli import main

        runner = CliRunner()
        result = runner.invoke(main, ['--help'])

        assert result.exit_code == 0
        assert 'Mad Spark Alt' in result.output
        assert 'QADI Analysis' in result.output or 'Run QADI analysis directly' in result.output

    def test_no_args_shows_help(self):
        """Running with no arguments should show help."""
        from mad_spark_alt.unified_cli import main

        runner = CliRunner()
        result = runner.invoke(main, [])

        assert result.exit_code == 0
        assert '--help' in result.output or 'Usage:' in result.output

    def test_default_qadi_runs_without_subcommand(self):
        """QADI should run when invoked without subcommand."""
        from mad_spark_alt.unified_cli import main

        with patch('mad_spark_alt.unified_cli._run_qadi_sync') as mock_qadi:
            with patch('os.getenv', return_value='fake-key'):
                runner = CliRunner()
                result = runner.invoke(main, ['Test question'])

                # Should not fail
                assert result.exit_code == 0
                # Should call QADI analysis
                mock_qadi.assert_called_once()

    def test_default_command_with_temperature(self):
        """Default command should accept --temperature option."""
        from mad_spark_alt.unified_cli import main

        with patch('mad_spark_alt.unified_cli._run_qadi_sync') as mock_qadi:
            with patch('os.getenv', return_value='fake-key'):
                runner = CliRunner()
                # Options must come BEFORE positional arguments in Click
                result = runner.invoke(main, ['--temperature', '1.2', 'Test question'])

                assert result.exit_code == 0
                # Verify temperature was passed
                call_kwargs = mock_qadi.call_args[1]
                assert call_kwargs['temperature'] == 1.2

    def test_default_command_with_verbose(self):
        """Default command should accept --verbose flag."""
        from mad_spark_alt.unified_cli import main

        with patch('mad_spark_alt.unified_cli._run_qadi_sync') as mock_qadi:
            with patch('os.getenv', return_value='fake-key'):
                runner = CliRunner()
                # Options must come BEFORE positional arguments in Click
                result = runner.invoke(main, ['--verbose', 'Test question'])

                assert result.exit_code == 0
                # Verify verbose was passed
                call_kwargs = mock_qadi.call_args[1]
                assert call_kwargs['verbose'] is True


class TestUnifiedCLIMultimodal:
    """Test multimodal support in default command."""

    def test_default_command_with_image(self):
        """Default command should accept --image option."""
        from mad_spark_alt.unified_cli import main

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            tmp_path = tmp.name
            tmp.write(b'fake image data')

        try:
            with patch('mad_spark_alt.unified_cli._run_qadi_sync') as mock_qadi:
                with patch('os.getenv', return_value='fake-key'):
                    runner = CliRunner()
                    result = runner.invoke(main, ['--image', tmp_path, 'Test question'])

                    assert result.exit_code == 0
                    # Verify image_paths was passed
                    call_kwargs = mock_qadi.call_args[1]
                    assert len(call_kwargs['image_paths']) == 1
                    assert tmp_path in call_kwargs['image_paths']
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_default_command_with_document(self):
        """Default command should accept --document option."""
        from mad_spark_alt.unified_cli import main

        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
            tmp_path = tmp.name
            tmp.write(b'fake pdf data')

        try:
            with patch('mad_spark_alt.unified_cli._run_qadi_sync') as mock_qadi:
                with patch('os.getenv', return_value='fake-key'):
                    runner = CliRunner()
                    result = runner.invoke(main, ['--document', tmp_path, 'Test question'])

                    assert result.exit_code == 0
                    # Verify document_paths was passed
                    call_kwargs = mock_qadi.call_args[1]
                    assert len(call_kwargs['document_paths']) == 1
                    assert tmp_path in call_kwargs['document_paths']
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_default_command_with_url(self):
        """Default command should accept --url option."""
        from mad_spark_alt.unified_cli import main

        with patch('mad_spark_alt.unified_cli._run_qadi_sync') as mock_qadi:
            with patch('os.getenv', return_value='fake-key'):
                runner = CliRunner()
                result = runner.invoke(main, ['--url', 'https://example.com', 'Test question'])

                assert result.exit_code == 0
                # Verify urls was passed
                call_kwargs = mock_qadi.call_args[1]
                assert len(call_kwargs['urls']) == 1
                assert 'https://example.com' in call_kwargs['urls']


class TestUnifiedCLIEvolution:
    """Test evolution support in default command."""

    def test_default_command_with_evolve_flag(self):
        """Default command should support --evolve flag."""
        from mad_spark_alt.unified_cli import main

        with patch('mad_spark_alt.unified_cli._run_qadi_sync') as mock_qadi:
            with patch('os.getenv', return_value='fake-key'):
                runner = CliRunner()
                result = runner.invoke(main, ['--evolve', 'Test question'])

                assert result.exit_code == 0
                # Verify evolve was passed
                call_kwargs = mock_qadi.call_args[1]
                assert call_kwargs['evolve'] is True

    def test_default_command_evolution_with_generations(self):
        """Default command should accept --generations with --evolve."""
        from mad_spark_alt.unified_cli import main

        with patch('mad_spark_alt.unified_cli._run_qadi_sync') as mock_qadi:
            with patch('os.getenv', return_value='fake-key'):
                runner = CliRunner()
                result = runner.invoke(main, ['--evolve', '--generations', '3', 'Test'])

                assert result.exit_code == 0
                # Verify generations was passed
                call_kwargs = mock_qadi.call_args[1]
                assert call_kwargs['generations'] == 3


class TestUnifiedCLISubcommands:
    """Test explicit subcommands."""

    def test_evolve_subcommand_exists(self):
        """Evolve subcommand should be available."""
        from mad_spark_alt.unified_cli import main

        runner = CliRunner()
        result = runner.invoke(main, ['evolve', '--help'])

        assert result.exit_code == 0
        assert 'evolve' in result.output.lower() or 'Evolution' in result.output

    def test_evaluate_subcommand_exists(self):
        """Evaluate subcommand should be available."""
        from mad_spark_alt.unified_cli import main

        runner = CliRunner()
        result = runner.invoke(main, ['evaluate', '--help'])

        assert result.exit_code == 0
        assert 'evaluate' in result.output.lower() or 'Evaluate' in result.output

    def test_list_evaluators_subcommand_exists(self):
        """List-evaluators subcommand should be available."""
        from mad_spark_alt.unified_cli import main

        runner = CliRunner()
        result = runner.invoke(main, ['list-evaluators', '--help'])

        assert result.exit_code == 0


class TestUnifiedCLIErrorHandling:
    """Test error handling and validation."""

    def test_invalid_temperature_shows_error(self):
        """Invalid temperature should show error."""
        from mad_spark_alt.unified_cli import main

        runner = CliRunner()
        result = runner.invoke(main, ['Test', '--temperature', '3.0'])

        # Should fail or show error
        assert result.exit_code != 0 or 'error' in result.output.lower()

    def test_missing_google_api_key_shows_error(self):
        """Missing GOOGLE_API_KEY should show helpful error."""
        from mad_spark_alt.unified_cli import main

        with patch('os.getenv', return_value=None):  # No API key
            runner = CliRunner()
            result = runner.invoke(main, ['Test question'])

            assert result.exit_code != 0
            assert 'GOOGLE_API_KEY' in result.output or 'API key' in result.output
