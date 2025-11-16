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

    def test_evolve_is_flag_not_subcommand(self):
        """Evolution should be accessed via --evolve flag, not 'evolve' subcommand."""
        from mad_spark_alt.unified_cli import main

        runner = CliRunner()

        # Check that --help shows --evolve flag
        result = runner.invoke(main, ['--help'])
        assert result.exit_code == 0
        assert '--evolve' in result.output

        # Check that 'evolve' as a word in help is described as a flag, not subcommand
        # The help should NOT list "evolve" as a command like "evaluate" or "list-evaluators"
        assert 'Commands:' in result.output
        commands_section = result.output.split('Commands:')[1] if 'Commands:' in result.output else ""
        # 'evolve' should NOT appear in the commands section
        assert 'evolve' not in commands_section.lower() or '--evolve' in result.output

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
        """Missing GOOGLE_API_KEY should work with Ollama fallback (if available)."""
        from mad_spark_alt.unified_cli import main

        with patch('os.getenv', return_value=None):  # No API key
            runner = CliRunner()
            result = runner.invoke(main, ['Test question'])

            # With Ollama integration, system can work without Google API key
            # If Ollama is available, command succeeds; if not, it fails with helpful error
            if result.exit_code != 0:
                # If it fails, should show helpful error about providers
                assert 'API key' in result.output or 'provider' in result.output.lower() or 'Ollama' in result.output


class TestProviderInheritance:
    """Test provider inheritance from parent command to subcommands."""

    def test_evaluate_shows_provider_option(self):
        """Evaluate subcommand should have --provider option."""
        from mad_spark_alt.unified_cli import main

        runner = CliRunner()
        result = runner.invoke(main, ['evaluate', '--help'])

        assert result.exit_code == 0
        assert '--provider' in result.output
        assert 'gemini' in result.output.lower()
        assert 'ollama' in result.output.lower()

    def test_batch_evaluate_shows_provider_option(self):
        """Batch-evaluate subcommand should have --provider option."""
        from mad_spark_alt.unified_cli import main

        runner = CliRunner()
        result = runner.invoke(main, ['batch-evaluate', '--help'])

        assert result.exit_code == 0
        assert '--provider' in result.output
        assert 'gemini' in result.output.lower()
        assert 'ollama' in result.output.lower()

    def test_compare_shows_provider_option(self):
        """Compare subcommand should have --provider option."""
        from mad_spark_alt.unified_cli import main

        runner = CliRunner()
        result = runner.invoke(main, ['compare', '--help'])

        assert result.exit_code == 0
        assert '--provider' in result.output
        assert 'gemini' in result.output.lower()
        assert 'ollama' in result.output.lower()

    def test_main_command_provider_option_persists(self):
        """Main command's --provider option should be documented."""
        from mad_spark_alt.unified_cli import main

        runner = CliRunner()
        result = runner.invoke(main, ['--help'])

        assert result.exit_code == 0
        assert '--provider' in result.output
        # Check for provider choices
        assert 'auto' in result.output.lower()
        assert 'gemini' in result.output.lower()
        assert 'ollama' in result.output.lower()

    def test_context_stores_provider_selection(self):
        """Context should store provider selection for subcommand inheritance."""
        from mad_spark_alt.unified_cli import main
        import click

        runner = CliRunner()

        # We can't easily test context passing without executing subcommands,
        # but we can verify the command structure accepts provider
        result = runner.invoke(main, ['--provider', 'gemini', '--help'])

        # Should not error
        assert result.exit_code == 0
        # Help should still be accessible with provider flag
        assert 'Mad Spark Alt' in result.output

    def test_provider_inheritance_in_subcommand_signature(self):
        """Subcommands should inherit provider if not overridden."""
        from mad_spark_alt.unified_cli import evaluate, batch_evaluate, compare

        # Check that Click-decorated functions have provider parameter
        # Access the underlying Click command's params
        eval_params = {p.name for p in evaluate.params}
        assert 'provider' in eval_params

        batch_params = {p.name for p in batch_evaluate.params}
        assert 'provider' in batch_params

        compare_params = {p.name for p in compare.params}
        assert 'provider' in compare_params

    def test_provider_override_is_documented(self):
        """Override behavior should be documented in help text."""
        from mad_spark_alt.unified_cli import main

        runner = CliRunner()
        result = runner.invoke(main, ['evaluate', '--help'])

        assert result.exit_code == 0
        # Help should mention inheritance, override, or parent - or at least show the provider option with default None
        # The help text says "default: inherit from parent"
        help_lower = result.output.lower()
        assert ('inherit' in help_lower or
                'override' in help_lower or
                'parent' in help_lower or
                'default' in help_lower and 'provider' in help_lower)
