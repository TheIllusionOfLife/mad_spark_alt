"""
Tests for removal of quick mode functionality and checkpoint frequency updates.
"""

import pytest
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, call
import tempfile
import shutil
import json

from mad_spark_alt.core.interfaces import GeneratedIdea, ThinkingMethod
from mad_spark_alt.evolution import GeneticAlgorithm, EvolutionRequest, EvolutionConfig
from mad_spark_alt.evolution.interfaces import IndividualFitness
from mad_spark_alt.cli import evolve, _run_evolution_pipeline


class TestQuickModeRemoval:
    """Test suite for verifying quick mode has been removed properly."""
    
    def test_evolve_command_no_quick_option(self):
        """Verify that --quick option has been removed from CLI."""
        from click.testing import CliRunner
        from mad_spark_alt.cli import main
        
        runner = CliRunner()
        # Test that --quick option is not recognized
        result = runner.invoke(main, ['evolve', 'test problem', '--quick'])
        assert result.exit_code != 0
        assert "--quick" in result.output or "no such option" in result.output
        
    def test_evolve_command_help_no_quick_mention(self):
        """Verify that help text doesn't mention quick mode."""
        from click.testing import CliRunner
        from mad_spark_alt.cli import main
        
        runner = CliRunner()
        result = runner.invoke(main, ['evolve', '--help'])
        assert result.exit_code == 0
        assert "--quick" not in result.output
        assert "-q" not in result.output
        assert "Quick mode" not in result.output
        

class TestCheckpointFrequency:
    """Test suite for checkpoint frequency changes."""
    
    def test_checkpoint_interval_default(self):
        """Verify default checkpoint interval is 1."""
        ga = GeneticAlgorithm(
            checkpoint_dir=".test_checkpoints",
            checkpoint_interval=1
        )
        assert ga.checkpoint_interval == 1
        
    def test_checkpointer_enabled_with_dir(self):
        """Verify checkpointer is created when checkpoint_dir is provided."""
        ga = GeneticAlgorithm(
            checkpoint_dir=".test_checkpoints",
            checkpoint_interval=1
        )
        assert ga.checkpointer is not None
        
    def test_checkpointer_disabled_without_dir(self):
        """Verify checkpointer is None when checkpoint_dir is None."""
        ga = GeneticAlgorithm(
            checkpoint_dir=None,
            checkpoint_interval=1
        )
        assert ga.checkpointer is None
                    
    @pytest.mark.asyncio
    async def test_cli_checkpoint_config(self):
        """Verify CLI passes correct checkpoint configuration to GeneticAlgorithm."""
        from unittest.mock import patch, AsyncMock, MagicMock
        from mad_spark_alt.cli import _run_evolution_pipeline
        
        with patch('mad_spark_alt.cli.GeneticAlgorithm') as mock_ga_class:
            # Mock the genetic algorithm instance
            mock_ga = AsyncMock()
            mock_ga_class.return_value = mock_ga
            mock_ga.evolve.return_value = MagicMock(
                final_population=[],
                best_ideas=[],
                evolution_metrics={},
                error_message=None,
                success=True,
                execution_time=1.0,
                total_generations=2
            )
            
            with patch('mad_spark_alt.cli.SimpleQADIOrchestrator') as mock_orch_class:
                mock_orch = AsyncMock()
                mock_orch_class.return_value = mock_orch
                mock_orch.run_qadi_cycle.return_value = MagicMock(
                    synthesized_ideas=[MagicMock(content="test", metadata={})],
                    total_llm_cost=0.01
                )
                
                # Run the pipeline
                await _run_evolution_pipeline(
                    problem="test",
                    context="test context",
                    generations=2,
                    population=2,
                    temperature=None,
                    output_file=None,
                    traditional=False
                )
                
                # Verify GeneticAlgorithm was created with correct checkpoint config
                mock_ga_class.assert_called_once()
                call_kwargs = mock_ga_class.call_args.kwargs
                assert call_kwargs['checkpoint_dir'] == ".evolution_checkpoints"
                assert call_kwargs['checkpoint_interval'] == 1
                

class TestParameterValidation:
    """Test that parameter validation still works without quick mode."""
    
    @pytest.mark.asyncio
    async def test_generations_validation(self):
        """Verify generations must be between 2 and 5."""
        from click.testing import CliRunner
        from mad_spark_alt.cli import main
        
        runner = CliRunner()
        
        # Test below minimum
        result = runner.invoke(main, ['evolve', 'test', '--generations', '1'])
        assert result.exit_code != 0
        assert "Generations must be between 2 and 5" in result.output
        
        # Test above maximum
        result = runner.invoke(main, ['evolve', 'test', '--generations', '6'])
        assert result.exit_code != 0
        assert "Generations must be between 2 and 5" in result.output
        
    @pytest.mark.asyncio
    async def test_population_validation(self):
        """Verify population must be between 2 and 10."""
        from click.testing import CliRunner
        from mad_spark_alt.cli import main
        
        runner = CliRunner()
        
        # Test below minimum
        result = runner.invoke(main, ['evolve', 'test', '--population', '1'])
        assert result.exit_code != 0
        assert "Population size must be between 2 and 10" in result.output
        
        # Test above maximum
        result = runner.invoke(main, ['evolve', 'test', '--population', '11'])
        assert result.exit_code != 0
        assert "Population size must be between 2 and 10" in result.output


class TestDefaultValues:
    """Test that default values are correct after quick mode removal."""
    
    def test_default_generations_is_2(self):
        """Verify default generations is 2."""
        from mad_spark_alt.cli import evolve
        import click
        
        # Get the evolve command's parameters
        for param in evolve.params:
            if param.name == 'generations':
                assert param.default == 2
                break
        else:
            pytest.fail("generations parameter not found")
            
    def test_default_population_is_5(self):
        """Verify default population is 5."""
        from mad_spark_alt.cli import evolve
        import click
        
        # Get the evolve command's parameters
        for param in evolve.params:
            if param.name == 'population':
                assert param.default == 5
                break
        else:
            pytest.fail("population parameter not found")


class TestIntegrationScenarios:
    """Integration tests for various user scenarios."""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_evolution_with_real_api(self):
        """Test evolution with real Google API key."""
        import os
        if not os.getenv("GOOGLE_API_KEY"):
            pytest.skip("GOOGLE_API_KEY not available")
            
        from mad_spark_alt.cli import _run_evolution_pipeline
        
        # Test with minimal settings
        await _run_evolution_pipeline(
            problem="How to reduce plastic waste?",
            context="Focus on practical solutions",
            generations=2,  # Minimum
            population=2,   # Minimum
            temperature=0.7,
            output_file=None,
            traditional=False
        )
        
    @pytest.mark.asyncio 
    async def test_evolution_checkpoint_creation(self):
        """Test that checkpoints are created during evolution."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir) / "checkpoints"
            
            # Create initial ideas
            initial_ideas = [
                GeneratedIdea(
                    content=f"Solution {i}",
                    thinking_method=ThinkingMethod.ABDUCTION,
                    agent_name="test",
                    generation_prompt="test"
                )
                for i in range(2)
            ]
            
            # Mock the fitness evaluator
            from unittest.mock import patch, AsyncMock
            with patch('mad_spark_alt.evolution.fitness.FitnessEvaluator') as mock_eval_class:
                mock_evaluator = AsyncMock()
                mock_eval_class.return_value = mock_evaluator
                
                # Mock evaluate_batch to return proper results
                mock_evaluator.evaluate_batch.return_value = [
                    {"overall_score": 0.8, "diversity_score": 0.7}
                    for _ in range(10)  # Enough for multiple calls
                ]
                
                # Create GA with checkpoint interval = 1
                ga = GeneticAlgorithm(
                    checkpoint_dir=str(checkpoint_dir),
                    checkpoint_interval=1,
                    use_cache=False
                )
                
                # Run 2 generations
                config = EvolutionConfig(
                    population_size=2,
                    generations=2,
                    max_parallel_evaluations=2,  # Must not exceed population_size
                )
                
                request = EvolutionRequest(
                    initial_population=initial_ideas,
                    config=config,
                    context="Test evolution context"  # Add required context
                )
                
                # Run evolution
                result = await ga.evolve(request)
                
                # Verify evolution succeeded
                assert result.success
                
                # Verify checkpoints were created (at least 1)
                checkpoint_files = list(checkpoint_dir.glob("*.json"))
                assert len(checkpoint_files) >= 1, f"Expected at least 1 checkpoint, found {len(checkpoint_files)}"
                
                # Verify checkpoint content
                for checkpoint_file in checkpoint_files:
                    with open(checkpoint_file) as f:
                        data = json.load(f)
                        assert "generation" in data
                        assert "population" in data
                        assert "config" in data
                        assert data["generation"] >= 1  # Should be 1 or higher
            
            
class TestDocumentationUpdates:
    """Verify documentation has been updated correctly."""
    
    def test_no_quick_mode_in_commands_md(self):
        """Verify COMMANDS.md doesn't mention quick mode."""
        commands_path = Path("COMMANDS.md")
        if commands_path.exists():
            content = commands_path.read_text()
            assert "--quick" not in content
            assert "-q" not in content.split()  # Check as whole word
            
    def test_no_quick_mode_in_cli_usage(self):
        """Verify cli_usage.md doesn't mention quick mode."""
        cli_usage_path = Path("docs/cli_usage.md")
        if cli_usage_path.exists():
            content = cli_usage_path.read_text()
            assert "--quick" not in content
            
    def test_no_quick_mode_in_session_handover(self):
        """Verify SESSION_HANDOVER.md doesn't mention quick mode."""
        handover_path = Path("SESSION_HANDOVER.md")
        if handover_path.exists():
            content = handover_path.read_text()
            # Allow historical mentions but not as current examples
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if "--quick" in line and i > 0:
                    # Check if it's in a code block or example
                    prev_line = lines[i-1] if i > 0 else ""
                    if "```" in prev_line or "example" in prev_line.lower():
                        pytest.fail(f"Quick mode found in example on line {i+1}")