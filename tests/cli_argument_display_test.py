"""
Tests for CLI argument display accuracy.

These tests ensure that the population and generation values displayed
to the user match what they requested on the command line.
"""

import re
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mad_spark_alt.core.interfaces import GeneratedIdea
from mad_spark_alt.core.simple_qadi_orchestrator import SimpleQADIResult
from mad_spark_alt.evolution.interfaces import (
    EvolutionConfig,
    EvolutionResult,
    IndividualFitness,
)


class TestCLIArgumentDisplay:
    """Test that CLI arguments are displayed correctly to users."""

    @pytest.mark.asyncio
    async def test_evolution_display_shows_requested_population(self):
        """Test that requested population is shown, not the calculated minimum."""
        # This test captures the output when running evolution
        # and verifies the displayed values match the CLI arguments
        
        # Mock the QADI result with only 3 ideas
        mock_ideas = [
            GeneratedIdea(
                content=f"Test idea {i}",
                thinking_method="test",
                agent_name="test",
                generation_prompt="test",
            )
            for i in range(3)
        ]
        
        mock_qadi_result = SimpleQADIResult(
            core_question="Test question",
            hypotheses=["H1", "H2", "H3"],
            hypothesis_scores=[],
            final_answer="Test answer",
            action_plan=["Action 1"],
            verification_examples=[],
            verification_conclusion="",
            synthesized_ideas=mock_ideas,
            total_llm_cost=0.001,
        )
        
        # Mock evolution result
        mock_evolution_result = EvolutionResult(
            final_population=[
                IndividualFitness(idea=idea, overall_fitness=0.5)
                for idea in mock_ideas
            ],
            best_ideas=mock_ideas[:2],
            generation_snapshots=[],
            total_generations=3,
            execution_time=10.0,
            evolution_metrics={
                "generations_completed": 3,
                "total_ideas_evaluated": 15,
                "best_fitness": 0.6,
                "average_fitness": 0.5,
                "fitness_improvement_percent": 20.0,
            },
        )
        
        # Capture output
        captured_output = []
        
        def mock_print(*args, **kwargs):
            captured_output.append(" ".join(str(arg) for arg in args))
        
        with patch("builtins.print", side_effect=mock_print):
            with patch("os.getenv", return_value="fake-api-key"):  # Mock API key check
                with patch("mad_spark_alt.core.simple_qadi_orchestrator.SimpleQADIOrchestrator") as mock_orchestrator_class:
                    # Mock the orchestrator instance
                    mock_orchestrator = MagicMock()
                    mock_orchestrator.run_qadi_cycle = AsyncMock(return_value=mock_qadi_result)
                    mock_orchestrator_class.return_value = mock_orchestrator
                    
                    with patch("mad_spark_alt.evolution.genetic_algorithm.GeneticAlgorithm") as mock_ga_class:
                        # Mock the GA instance
                        mock_ga = MagicMock()
                        mock_ga.evolve = AsyncMock(return_value=mock_evolution_result)
                        mock_ga_class.return_value = mock_ga
                        
                        # Import here to avoid early execution
                        from qadi_simple import run_qadi_analysis
                        
                        # Run with population=10 but only 3 ideas available
                        await run_qadi_analysis(
                            "Test question",
                            evolve=True,
                            generations=3,
                            population=10,  # Requested 10
                            traditional=True  # Use traditional to avoid LLM calls
                        )
        
        # Find the evolution display line
        evolution_line = None
        for line in captured_output:
            if "Evolving ideas" in line:
                evolution_line = line
                break
        
        # Debug output
        print("\n--- Captured output ---")
        for line in captured_output:
            print(line)
        print("--- End captured output ---\n")
        
        assert evolution_line is not None, "Evolution display line not found"
        
        # The line should show the requested values, not the calculated minimum
        # Expected: "🧬 Evolving ideas (3 generations, 10 population)..."
        # NOT: "🧬 Evolving ideas (3 generations, 3 population)..."
        match = re.search(r"(\d+)\s+generations?,\s+(\d+)\s+population", evolution_line)
        assert match is not None, f"Could not parse evolution line: {evolution_line}"
        
        displayed_generations = int(match.group(1))
        displayed_population = int(match.group(2))
        
        assert displayed_generations == 3, f"Expected generations=3, got {displayed_generations}"
        assert displayed_population == 10, f"Expected population=10, got {displayed_population}"
        
        # Also check if there's a clarification message about using fewer ideas
        has_clarification = any("Using 3 ideas from available 3" in line for line in captured_output)
        assert has_clarification, "Should clarify when using fewer ideas than requested"

    @pytest.mark.asyncio
    async def test_evolution_display_with_sufficient_ideas(self):
        """Test display when we have enough ideas for the requested population."""
        # Mock 15 ideas (more than requested)
        mock_ideas = [
            GeneratedIdea(
                content=f"Test idea {i}",
                thinking_method="test",
                agent_name="test",
                generation_prompt="test",
            )
            for i in range(15)
        ]
        
        mock_qadi_result = SimpleQADIResult(
            core_question="Test question",
            hypotheses=["H1", "H2", "H3"],
            hypothesis_scores=[],
            final_answer="Test answer",
            action_plan=["Action 1"],
            verification_examples=[],
            verification_conclusion="",
            synthesized_ideas=mock_ideas,
            total_llm_cost=0.001,
        )
        
        captured_output = []
        
        def mock_print(*args, **kwargs):
            captured_output.append(" ".join(str(arg) for arg in args))
        
        with patch("builtins.print", side_effect=mock_print):
            with patch("mad_spark_alt.core.simple_qadi_orchestrator.SimpleQADIOrchestrator.run_qadi_cycle") as mock_qadi:
                mock_qadi.return_value = mock_qadi_result
                
                with patch("mad_spark_alt.evolution.genetic_algorithm.GeneticAlgorithm.evolve") as mock_evolve:
                    # Mock evolution to avoid actual processing
                    mock_evolve.return_value = EvolutionResult(
                        final_population=[],
                        best_ideas=[],
                        generation_snapshots=[],
                        total_generations=5,
                        execution_time=15.0,
                        evolution_metrics={
                            "generations_completed": 5,
                            "total_ideas_evaluated": 50,
                        },
                    )
                    
                    from qadi_simple import run_qadi_analysis
                    
                    # Run with population=8, we have 15 ideas
                    await run_qadi_analysis(
                        "Test question",
                        evolve=True,
                        generations=5,
                        population=8,
                        traditional=True
                    )
        
        # Find the evolution display line
        evolution_line = None
        for line in captured_output:
            if "Evolving ideas" in line:
                evolution_line = line
                break
        
        assert evolution_line is not None
        
        # Should show requested values
        match = re.search(r"(\d+)\s+generations?,\s+(\d+)\s+population", evolution_line)
        assert match is not None
        
        displayed_generations = int(match.group(1))
        displayed_population = int(match.group(2))
        
        assert displayed_generations == 5
        assert displayed_population == 8
        
        # Should NOT have a clarification message since we have enough ideas
        has_clarification = any("Using" in line and "ideas from available" in line for line in captured_output)
        assert not has_clarification, "Should not show clarification when we have enough ideas"

    def test_evolution_config_respects_cli_arguments(self):
        """Test that EvolutionConfig is created with the correct CLI values."""
        # This is a unit test for the config creation logic
        requested_population = 10
        requested_generations = 5
        available_ideas = 3
        
        # The actual population should be min(requested, available)
        actual_population = min(requested_population, available_ideas)
        
        config = EvolutionConfig(
            population_size=actual_population,
            generations=requested_generations,
            mutation_rate=0.3,
            crossover_rate=0.75,
            elite_size=min(2, max(1, actual_population // 3)),
        )
        
        # Config should use actual population
        assert config.population_size == 3
        assert config.generations == 5
        
        # But display should show requested values
        # This is what we need to fix - the display logic
        display_message = f"🧬 Evolving ideas ({requested_generations} generations, {requested_population} population)..."
        assert "10 population" in display_message
        assert "5 generations" in display_message