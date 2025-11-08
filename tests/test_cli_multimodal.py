"""
Integration tests for CLI multimodal options (--image, --document, --url).
"""

import pytest
import tempfile
from pathlib import Path
from click.testing import CliRunner
from unittest.mock import AsyncMock, MagicMock, patch

from mad_spark_alt.cli import main
from mad_spark_alt.core.multimodal import MultimodalInput, MultimodalInputType, MultimodalSourceType


class TestCLIMultimodalOptions:
    """Test CLI options for multimodal inputs."""

    def test_image_option_single_file(self):
        """Test --image option with single file."""
        runner = CliRunner()

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name
            tmp.write(b"fake image data")

        try:
            with patch('mad_spark_alt.cli.asyncio.run') as mock_run:
                mock_run.return_value = None
                result = runner.invoke(main, [
                    'evolve',
                    'Analyze this design',
                    '--image', tmp_path
                ])

                # Should accept the option
                assert result.exit_code == 0 or "--image" not in result.output
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_image_option_multiple_files(self):
        """Test --image option with multiple files."""
        runner = CliRunner()

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp1, \
             tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp2:
            tmp_path1 = tmp1.name
            tmp_path2 = tmp2.name
            tmp1.write(b"fake image 1")
            tmp2.write(b"fake image 2")

        try:
            with patch('mad_spark_alt.cli.asyncio.run') as mock_run:
                mock_run.return_value = None
                result = runner.invoke(main, [
                    'evolve',
                    'Compare these designs',
                    '--image', tmp_path1,
                    '--image', tmp_path2
                ])

                # Should accept multiple images
                assert result.exit_code == 0 or "--image" not in result.output
        finally:
            Path(tmp_path1).unlink(missing_ok=True)
            Path(tmp_path2).unlink(missing_ok=True)

    def test_image_option_short_form(self):
        """Test -i short form for --image."""
        runner = CliRunner()

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name
            tmp.write(b"fake image data")

        try:
            with patch('mad_spark_alt.cli.asyncio.run') as mock_run:
                mock_run.return_value = None
                result = runner.invoke(main, [
                    'evolve',
                    'Analyze this',
                    '-i', tmp_path
                ])

                # Should accept short form
                assert result.exit_code == 0 or "-i" not in result.output
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_document_option_single_file(self):
        """Test --document option with single file."""
        runner = CliRunner()

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = tmp.name
            tmp.write(b"%PDF-1.4 fake pdf")

        try:
            with patch('mad_spark_alt.cli.asyncio.run') as mock_run:
                mock_run.return_value = None
                result = runner.invoke(main, [
                    'evolve',
                    'Summarize this report',
                    '--document', tmp_path
                ])

                # Should accept the option
                assert result.exit_code == 0 or "--document" not in result.output
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_document_option_multiple_files(self):
        """Test --document option with multiple files."""
        runner = CliRunner()

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp1, \
             tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp2:
            tmp_path1 = tmp1.name
            tmp_path2 = tmp2.name
            tmp1.write(b"%PDF-1.4 fake pdf 1")
            tmp2.write(b"%PDF-1.4 fake pdf 2")

        try:
            with patch('mad_spark_alt.cli.asyncio.run') as mock_run:
                mock_run.return_value = None
                result = runner.invoke(main, [
                    'evolve',
                    'Compare these reports',
                    '--document', tmp_path1,
                    '--document', tmp_path2
                ])

                # Should accept multiple documents
                assert result.exit_code == 0 or "--document" not in result.output
        finally:
            Path(tmp_path1).unlink(missing_ok=True)
            Path(tmp_path2).unlink(missing_ok=True)

    def test_document_option_short_form(self):
        """Test -d short form for --document."""
        runner = CliRunner()

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = tmp.name
            tmp.write(b"%PDF-1.4 fake pdf")

        try:
            with patch('mad_spark_alt.cli.asyncio.run') as mock_run:
                mock_run.return_value = None
                result = runner.invoke(main, [
                    'evolve',
                    'Analyze this',
                    '-d', tmp_path
                ])

                # Should accept short form
                assert result.exit_code == 0 or "-d" not in result.output
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_url_option_single_url(self):
        """Test --url option with single URL."""
        runner = CliRunner()

        with patch('mad_spark_alt.cli.asyncio.run') as mock_run:
            mock_run.return_value = None
            result = runner.invoke(main, [
                'evolve',
                'Analyze this article',
                '--url', 'https://example.com/article'
            ])

            # Should accept the option
            assert result.exit_code == 0 or "--url" not in result.output

    def test_url_option_multiple_urls(self):
        """Test --url option with multiple URLs."""
        runner = CliRunner()

        with patch('mad_spark_alt.cli.asyncio.run') as mock_run:
            mock_run.return_value = None
            result = runner.invoke(main, [
                'evolve',
                'Compare these articles',
                '--url', 'https://example.com/article1',
                '--url', 'https://example.com/article2'
            ])

            # Should accept multiple URLs
            assert result.exit_code == 0 or "--url" not in result.output

    def test_url_option_short_form(self):
        """Test -u short form for --url."""
        runner = CliRunner()

        with patch('mad_spark_alt.cli.asyncio.run') as mock_run:
            mock_run.return_value = None
            result = runner.invoke(main, [
                'evolve',
                'Analyze this',
                '-u', 'https://example.com/test'
            ])

            # Should accept short form
            assert result.exit_code == 0 or "-u" not in result.output

    def test_combined_multimodal_options(self):
        """Test combining multiple multimodal option types."""
        runner = CliRunner()

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_img, \
             tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_doc:
            img_path = tmp_img.name
            doc_path = tmp_doc.name
            tmp_img.write(b"fake image")
            tmp_doc.write(b"%PDF-1.4 fake pdf")

        try:
            with patch('mad_spark_alt.cli.asyncio.run') as mock_run:
                mock_run.return_value = None
                result = runner.invoke(main, [
                    'evolve',
                    'Synthesize insights',
                    '--image', img_path,
                    '--document', doc_path,
                    '--url', 'https://example.com/article'
                ])

                # Should accept all options together
                assert result.exit_code == 0 or "No such option" not in result.output
        finally:
            Path(img_path).unlink(missing_ok=True)
            Path(doc_path).unlink(missing_ok=True)

    def test_image_file_not_found(self):
        """Test error handling for non-existent image file."""
        runner = CliRunner()

        result = runner.invoke(main, [
            'evolve',
            'Analyze this',
            '--image', '/nonexistent/image.png'
        ])

        # Should fail with file not found error
        assert result.exit_code != 0
        assert "does not exist" in result.output.lower() or "error" in result.output.lower()

    def test_document_file_not_found(self):
        """Test error handling for non-existent document file."""
        runner = CliRunner()

        result = runner.invoke(main, [
            'evolve',
            'Analyze this',
            '--document', '/nonexistent/document.pdf'
        ])

        # Should fail with file not found error
        assert result.exit_code != 0
        assert "does not exist" in result.output.lower() or "error" in result.output.lower()


class TestCLIMultimodalProcessing:
    """Test that CLI correctly processes multimodal inputs."""

    @pytest.mark.asyncio
    async def test_image_processing_creates_multimodal_input(self):
        """Test that image files are converted to MultimodalInput objects."""
        from mad_spark_alt.cli import _run_evolution_pipeline

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name
            tmp.write(b"fake image data")

        try:
            with patch('mad_spark_alt.cli.SimpleQADIOrchestrator') as mock_orch_class:
                mock_orch = AsyncMock()
                mock_orch_class.return_value = mock_orch
                mock_orch.run_qadi_cycle.return_value = MagicMock(
                    synthesized_ideas=[MagicMock(content="test", metadata={})],
                    total_llm_cost=0.01,
                    total_images_processed=1,
                    total_pages_processed=0,
                    total_urls_processed=0,
                )

                await _run_evolution_pipeline(
                    problem="Test",
                    context=None,
                    generations=2,
                    population=2,
                    temperature=None,
                    output_file=None,
                    traditional=True,
                    diversity_method="jaccard",
                    image_paths=(tmp_path,),
                    document_paths=(),
                    urls=(),
                )

                # Verify run_qadi_cycle was called
                assert mock_orch.run_qadi_cycle.called
                call_kwargs = mock_orch.run_qadi_cycle.call_args.kwargs

                # Verify multimodal_inputs was passed
                assert 'multimodal_inputs' in call_kwargs
                multimodal_inputs = call_kwargs['multimodal_inputs']

                # Verify it's a list with one MultimodalInput
                assert isinstance(multimodal_inputs, list)
                assert len(multimodal_inputs) == 1

                # Verify the MultimodalInput properties
                mm_input = multimodal_inputs[0]
                assert mm_input.input_type == MultimodalInputType.IMAGE
                assert mm_input.source_type == MultimodalSourceType.FILE_PATH
                assert str(Path(tmp_path).absolute()) in mm_input.data
                assert mm_input.mime_type.startswith("image/")
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_document_processing_creates_multimodal_input(self):
        """Test that document files are converted to MultimodalInput objects."""
        from mad_spark_alt.cli import _run_evolution_pipeline

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = tmp.name
            tmp.write(b"%PDF-1.4 fake pdf")

        try:
            with patch('mad_spark_alt.cli.SimpleQADIOrchestrator') as mock_orch_class:
                mock_orch = AsyncMock()
                mock_orch_class.return_value = mock_orch
                mock_orch.run_qadi_cycle.return_value = MagicMock(
                    synthesized_ideas=[MagicMock(content="test", metadata={})],
                    total_llm_cost=0.01,
                    total_images_processed=0,
                    total_pages_processed=1,
                    total_urls_processed=0,
                )

                await _run_evolution_pipeline(
                    problem="Test",
                    context=None,
                    generations=2,
                    population=2,
                    temperature=None,
                    output_file=None,
                    traditional=True,
                    diversity_method="jaccard",
                    image_paths=(),
                    document_paths=(tmp_path,),
                    urls=(),
                )

                # Verify multimodal_inputs was passed
                call_kwargs = mock_orch.run_qadi_cycle.call_args.kwargs
                multimodal_inputs = call_kwargs['multimodal_inputs']

                # Verify it's a list with one MultimodalInput
                assert isinstance(multimodal_inputs, list)
                assert len(multimodal_inputs) == 1

                # Verify the MultimodalInput properties
                mm_input = multimodal_inputs[0]
                assert mm_input.input_type == MultimodalInputType.DOCUMENT
                assert mm_input.source_type == MultimodalSourceType.FILE_PATH
                assert str(Path(tmp_path).absolute()) in mm_input.data
                assert mm_input.mime_type == "application/pdf"
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_url_processing(self):
        """Test that URLs are passed correctly."""
        from mad_spark_alt.cli import _run_evolution_pipeline

        with patch('mad_spark_alt.cli.SimpleQADIOrchestrator') as mock_orch_class:
            mock_orch = AsyncMock()
            mock_orch_class.return_value = mock_orch
            mock_orch.run_qadi_cycle.return_value = MagicMock(
                synthesized_ideas=[MagicMock(content="test", metadata={})],
                total_llm_cost=0.01,
                total_images_processed=0,
                total_pages_processed=0,
                total_urls_processed=2,
            )

            test_urls = ("https://example.com/article1", "https://example.com/article2")

            await _run_evolution_pipeline(
                problem="Test",
                context=None,
                generations=2,
                population=2,
                temperature=None,
                output_file=None,
                traditional=True,
                diversity_method="jaccard",
                image_paths=(),
                document_paths=(),
                urls=test_urls,
            )

            # Verify urls were passed
            call_kwargs = mock_orch.run_qadi_cycle.call_args.kwargs
            passed_urls = call_kwargs['urls']

            # Verify URLs match
            assert passed_urls == list(test_urls)

    @pytest.mark.asyncio
    async def test_multimodal_stats_display(self):
        """Test that multimodal processing stats are displayed."""
        from mad_spark_alt.cli import _run_evolution_pipeline

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name
            tmp.write(b"fake image")

        try:
            with patch('mad_spark_alt.cli.SimpleQADIOrchestrator') as mock_orch_class, \
                 patch('mad_spark_alt.cli.console') as mock_console:
                mock_orch = AsyncMock()
                mock_orch_class.return_value = mock_orch
                mock_orch.run_qadi_cycle.return_value = MagicMock(
                    synthesized_ideas=[MagicMock(content="test", metadata={})],
                    total_llm_cost=0.01,
                    total_images_processed=1,
                    total_pages_processed=0,
                    total_urls_processed=1,
                )

                await _run_evolution_pipeline(
                    problem="Test",
                    context=None,
                    generations=2,
                    population=2,
                    temperature=None,
                    output_file=None,
                    traditional=True,
                    diversity_method="jaccard",
                    image_paths=(tmp_path,),
                    document_paths=(),
                    urls=("https://example.com",),
                )

                # Verify console.print was called with multimodal stats
                print_calls = [str(call) for call in mock_console.print.call_args_list]
                stats_printed = any("Processed:" in str(call) or "images" in str(call) or "URLs" in str(call)
                                   for call in print_calls)
                assert stats_printed, "Multimodal stats should be displayed"
        finally:
            Path(tmp_path).unlink(missing_ok=True)


class TestCLIMultimodalHelp:
    """Test help text for multimodal options."""

    def test_evolve_help_shows_image_option(self):
        """Test that --help shows the --image option."""
        runner = CliRunner()
        result = runner.invoke(main, ['evolve', '--help'])

        assert result.exit_code == 0
        assert '--image' in result.output
        assert '-i' in result.output
        assert 'image file' in result.output.lower()

    def test_evolve_help_shows_document_option(self):
        """Test that --help shows the --document option."""
        runner = CliRunner()
        result = runner.invoke(main, ['evolve', '--help'])

        assert result.exit_code == 0
        assert '--document' in result.output
        assert '-d' in result.output
        assert 'document file' in result.output.lower() or 'PDF' in result.output

    def test_evolve_help_shows_url_option(self):
        """Test that --help shows the --url option."""
        runner = CliRunner()
        result = runner.invoke(main, ['evolve', '--help'])

        assert result.exit_code == 0
        assert '--url' in result.output
        assert '-u' in result.output
