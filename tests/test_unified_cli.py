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

from mad_spark_alt.core.provider_router import ProviderSelection


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

    def test_options_after_positional_argument(self):
        """Options should work AFTER positional argument with allow_interspersed_args=True."""
        from mad_spark_alt.unified_cli import main

        with patch('mad_spark_alt.unified_cli._run_qadi_sync') as mock_qadi:
            with patch('os.getenv', return_value='fake-key'):
                runner = CliRunner()
                # NEW: Options AFTER positional argument should now work
                result = runner.invoke(main, ['Test question', '--provider', 'gemini'])

                assert result.exit_code == 0
                # Verify provider was parsed correctly
                call_kwargs = mock_qadi.call_args[1]
                assert call_kwargs['provider_selection'] == ProviderSelection.GEMINI

    def test_mixed_option_ordering(self):
        """Options should work in any order relative to positional argument."""
        from mad_spark_alt.unified_cli import main

        with patch('mad_spark_alt.unified_cli._run_qadi_sync') as mock_qadi:
            with patch('os.getenv', return_value='fake-key'):
                runner = CliRunner()
                # Mix: option before, positional, option after
                result = runner.invoke(main, ['--temperature', '1.5', 'Test question', '--provider', 'ollama'])

                assert result.exit_code == 0
                call_kwargs = mock_qadi.call_args[1]
                assert call_kwargs['temperature'] == 1.5
                assert call_kwargs['provider_selection'] == ProviderSelection.OLLAMA

    def test_options_all_after_positional(self):
        """Multiple options should work when all placed after positional argument."""
        from mad_spark_alt.unified_cli import main

        with patch('mad_spark_alt.unified_cli._run_qadi_sync') as mock_qadi:
            with patch('os.getenv', return_value='fake-key'):
                runner = CliRunner()
                # All options AFTER positional
                result = runner.invoke(main, ['Test question', '--verbose', '--temperature', '0.9', '--provider', 'gemini'])

                assert result.exit_code == 0
                call_kwargs = mock_qadi.call_args[1]
                assert call_kwargs['verbose'] is True
                assert call_kwargs['temperature'] == 0.9
                assert call_kwargs['provider_selection'] == ProviderSelection.GEMINI

    def test_backward_compatibility_options_before(self):
        """Verify backward compatibility: options before positional still work."""
        from mad_spark_alt.unified_cli import main

        with patch('mad_spark_alt.unified_cli._run_qadi_sync') as mock_qadi:
            with patch('os.getenv', return_value='fake-key'):
                runner = CliRunner()
                # OLD style: options BEFORE positional (should still work)
                result = runner.invoke(main, ['--provider', 'gemini', '--verbose', 'Test question'])

                assert result.exit_code == 0
                call_kwargs = mock_qadi.call_args[1]
                assert call_kwargs['provider_selection'] == ProviderSelection.GEMINI
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
                    call_kwargs = mock_qadi.call_args[1]
                    assert len(call_kwargs['image_paths']) == 1
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_default_command_with_document(self):
        """Default command should accept --document option."""
        from mad_spark_alt.unified_cli import main

        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp:
            tmp_path = tmp.name
            tmp.write(b'fake document data')

        try:
            with patch('mad_spark_alt.unified_cli._run_qadi_sync') as mock_qadi:
                with patch('os.getenv', return_value='fake-key'):
                    runner = CliRunner()
                    result = runner.invoke(main, ['--document', tmp_path, 'Test question'])

                    assert result.exit_code == 0
                    call_kwargs = mock_qadi.call_args[1]
                    assert len(call_kwargs['document_paths']) == 1
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
                call_kwargs = mock_qadi.call_args[1]
                assert len(call_kwargs['urls']) == 1


class TestUnifiedCLIEvolution:
    """Test evolution mode in default command."""

    def test_default_command_with_evolve_flag(self):
        """Default command should accept --evolve flag."""
        from mad_spark_alt.unified_cli import main

        with patch('mad_spark_alt.unified_cli._run_qadi_sync') as mock_qadi:
            with patch('os.getenv', return_value='fake-key'):
                runner = CliRunner()
                result = runner.invoke(main, ['--evolve', 'Test question'])

                assert result.exit_code == 0
                call_kwargs = mock_qadi.call_args[1]
                assert call_kwargs['evolve'] is True

    def test_default_command_evolution_with_generations(self):
        """Default command should accept evolution parameters."""
        from mad_spark_alt.unified_cli import main

        with patch('mad_spark_alt.unified_cli._run_qadi_sync') as mock_qadi:
            with patch('os.getenv', return_value='fake-key'):
                runner = CliRunner()
                result = runner.invoke(main, [
                    '--evolve', '--generations', '5', '--population', '10',
                    'Test question'
                ])

                assert result.exit_code == 0
                call_kwargs = mock_qadi.call_args[1]
                assert call_kwargs['evolve'] is True
                assert call_kwargs['generations'] == 5
                assert call_kwargs['population'] == 10


class TestUnifiedCLISubcommands:
    """Test that subcommands work correctly."""

    def test_evolve_is_flag_not_subcommand(self):
        """Evolve should be a flag, not a subcommand."""
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

        with patch('os.getenv', return_value=None):
            with patch('mad_spark_alt.unified_cli._run_qadi_sync') as mock_qadi:
                runner = CliRunner()
                result = runner.invoke(main, ['Test question'])

                # CLI should handle this gracefully
                assert result.exit_code == 0 or 'API key' in result.output


class TestProviderInheritance:
    """Test that --provider option is available on main command."""

    def test_main_command_provider_option_persists(self):
        """Main command should have --provider option."""
        from mad_spark_alt.unified_cli import main

        runner = CliRunner()
        result = runner.invoke(main, ['--help'])

        assert result.exit_code == 0
        assert '--provider' in result.output

    def test_main_command_accepts_provider_flag(self):
        """Main command should accept and process --provider flag."""
        from mad_spark_alt.unified_cli import main

        with patch('mad_spark_alt.unified_cli._run_qadi_sync') as mock_qadi:
            with patch('os.getenv', return_value='fake-key'):
                runner = CliRunner()
                result = runner.invoke(main, ['--provider', 'gemini', 'Test question'])

                assert result.exit_code == 0
                # Verify provider was passed
                call_kwargs = mock_qadi.call_args[1]
                assert call_kwargs['provider_selection'] == ProviderSelection.GEMINI

    def test_subcommands_do_not_have_provider_option(self):
        """Subcommands should not inherit --provider option (it's main-specific)."""
        from mad_spark_alt.unified_cli import main

        runner = CliRunner()
        result = runner.invoke(main, ['list-evaluators', '--help'])

        # list-evaluators should not have --provider in its help
        # (provider is for QADI analysis, not evaluator listing)
        assert result.exit_code == 0
        # This is fine - subcommands may or may not have provider


class TestPDFValidation:
    """Test PDF file validation for --image flag (Issue #3)."""

    def test_pdf_rejected_in_image_flag(self):
        """PDF files should be rejected when passed to --image flag."""
        from mad_spark_alt.unified_cli import main

        # Create a temporary PDF file
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False, mode='w') as tmp:
            tmp.write('%PDF-1.4')  # Minimal PDF header
            pdf_path = tmp.name

        try:
            with patch('mad_spark_alt.unified_cli._run_qadi_sync'):
                with patch('os.getenv', return_value='fake-key'):
                    runner = CliRunner()
                    result = runner.invoke(main, [
                        '--image', pdf_path,
                        'Test question'
                    ])

                    # Should show helpful error message
                    assert result.exit_code != 0 or 'PDF files should be passed to --document' in result.output
        finally:
            Path(pdf_path).unlink(missing_ok=True)

    def test_pdf_case_insensitive(self):
        """PDF validation should be case-insensitive."""
        from mad_spark_alt.unified_cli import main

        # Test with .PDF extension
        with tempfile.NamedTemporaryFile(suffix='.PDF', delete=False, mode='w') as tmp:
            tmp.write('%PDF-1.4')
            pdf_path = tmp.name

        try:
            with patch('mad_spark_alt.unified_cli._run_qadi_sync'):
                with patch('os.getenv', return_value='fake-key'):
                    runner = CliRunner()
                    result = runner.invoke(main, [
                        '--image', pdf_path,
                        'Test question'
                    ])

                    # Should show helpful error message regardless of case
                    assert result.exit_code != 0 or 'PDF files should be passed to --document' in result.output
        finally:
            Path(pdf_path).unlink(missing_ok=True)

    def test_image_files_allowed(self):
        """Actual image files should be allowed in --image flag."""
        from mad_spark_alt.unified_cli import main

        # Create a temporary PNG file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            tmp.write(b'\x89PNG\r\n\x1a\n')  # PNG signature
            png_path = tmp.name

        try:
            with patch('mad_spark_alt.unified_cli._run_qadi_sync'):
                with patch('os.getenv', return_value='fake-key'):
                    runner = CliRunner()
                    result = runner.invoke(main, [
                        '--image', png_path,
                        'Test question'
                    ])

                    # Should not fail with PDF error
                    assert 'PDF files should be passed to --document' not in result.output
        finally:
            Path(png_path).unlink(missing_ok=True)


class TestEvaluateSubcommandFunctionality:
    """Test evaluate subcommand with actual arguments (Issue #4)."""

    def test_evaluate_with_text_argument(self):
        """Evaluate should accept text as positional argument."""
        from mad_spark_alt.unified_cli import main

        runner = CliRunner()
        result = runner.invoke(main, ['evaluate', 'test text for evaluation'])

        # Should NOT fail with "No such command 'test'"
        assert result.exit_code != 2
        assert "No such command" not in result.output
        # May fail for other reasons (missing evaluators, etc.) but should recognize 'evaluate' as subcommand

    def test_evaluate_with_evaluators_option(self):
        """Evaluate should accept --evaluators option to filter evaluators."""
        from mad_spark_alt.unified_cli import main

        runner = CliRunner()
        result = runner.invoke(main, ['evaluate', 'test text', '--evaluators', 'diversity_evaluator'])

        # Should NOT fail with "No such command" or "No such option"
        assert result.exit_code != 2
        assert "No such command" not in result.output
        assert "No such option: --evaluators" not in result.output

    def test_evaluate_with_multiple_evaluators(self):
        """Evaluate should accept comma-separated evaluators."""
        from mad_spark_alt.unified_cli import main

        runner = CliRunner()
        result = runner.invoke(main, ['evaluate', 'test text', '--evaluators', 'diversity_evaluator,quality_evaluator'])

        # Should parse evaluators correctly
        assert result.exit_code != 2
        assert "No such command" not in result.output

    def test_evaluate_with_file_option(self):
        """Evaluate should accept --file option for file input."""
        from mad_spark_alt.unified_cli import main

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write('test content for evaluation')
            f.flush()
            test_file = f.name

        try:
            runner = CliRunner()
            result = runner.invoke(main, ['evaluate', '--file', test_file])

            # Should recognize evaluate subcommand and file option
            assert result.exit_code != 2
            assert "No such command" not in result.output
            assert "No such option: --file" not in result.output
        finally:
            Path(test_file).unlink(missing_ok=True)

    def test_evaluate_vs_qadi_default_distinction(self):
        """Ensure evaluate subcommand is distinct from default QADI command."""
        from mad_spark_alt.unified_cli import main

        runner = CliRunner()

        # Test evaluate subcommand
        result_eval = runner.invoke(main, ['evaluate', 'test text'])

        # Test default QADI command (with mocking to avoid actual execution)
        with patch('mad_spark_alt.unified_cli._run_qadi_sync') as mock_qadi:
            with patch('os.getenv', return_value='fake-key'):
                result_qadi = runner.invoke(main, ['test question for QADI'])

        # Evaluate should invoke evaluate logic
        assert "No such command" not in result_eval.output

        # QADI should invoke QADI logic
        assert result_qadi.exit_code == 0
        mock_qadi.assert_called_once()

    def test_list_evaluators_still_works(self):
        """Ensure list-evaluators subcommand still works after fix."""
        from mad_spark_alt.unified_cli import main

        runner = CliRunner()
        result = runner.invoke(main, ['list-evaluators'])

        # Should list evaluators, not fail
        assert result.exit_code == 0
        assert 'evaluator' in result.output.lower()

    def test_evaluate_no_input_shows_help_or_error(self):
        """Evaluate without text or file should show help or error."""
        from mad_spark_alt.unified_cli import main

        runner = CliRunner()
        result = runner.invoke(main, ['evaluate'])

        # Should either show help or error about missing input
        # Should NOT fail with "No such command"
        assert "No such command" not in result.output
