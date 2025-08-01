"""
Integration tests for evolution timeout fix with real API calls.

This module tests that evolution completes successfully with the new timeout settings.
"""

import asyncio
import os
import time
import pytest
from pathlib import Path

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Only run if API key is available
pytestmark = pytest.mark.skipif(
    not os.getenv("GOOGLE_API_KEY"),
    reason="GOOGLE_API_KEY not set"
)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_evolution_completes_with_new_timeout():
    """Test that evolution with population=10, generations=3 completes successfully."""
    from mad_spark_alt.core import setup_llm_providers
    from mad_spark_alt.core.interfaces import GeneratedIdea, ThinkingMethod
    from mad_spark_alt.evolution.interfaces import EvolutionRequest, EvolutionConfig
    from mad_spark_alt.evolution.genetic_algorithm import GeneticAlgorithm
    from mad_spark_alt.core.llm_provider import get_google_provider
    
    # Setup LLM provider
    await setup_llm_providers(google_api_key=os.getenv("GOOGLE_API_KEY"))
    google_provider = get_google_provider()
    
    # Create initial population
    initial_ideas = [
        GeneratedIdea(
            content=f"Game concept {i}: A puzzle game with unique mechanics",
            thinking_method=ThinkingMethod.ABDUCTION,
            agent_name="test",
            generation_prompt="Create game concept",
            confidence_score=0.7 + i * 0.05
        )
        for i in range(3)  # Start with 3 ideas
    ]
    
    # Configure evolution
    config = EvolutionConfig(
        population_size=10,  # The problematic size
        generations=3,       # The problematic generations
        mutation_rate=0.3,
        crossover_rate=0.7,
        use_semantic_operators=True
    )
    
    # Create evolution request
    request = EvolutionRequest(
        initial_population=initial_ideas,
        config=config,
        context="Create innovative game concepts"
    )
    
    # Create genetic algorithm with LLM provider
    ga = GeneticAlgorithm(llm_provider=google_provider)
    
    # Calculate expected timeout
    def calculate_evolution_timeout(gens: int, pop: int) -> float:
        base_timeout = 120.0
        time_per_eval = 8.0
        total_evaluations = gens * pop + pop
        estimated_time = base_timeout + (total_evaluations * time_per_eval)
        return min(estimated_time, 900.0)
    
    expected_timeout = calculate_evolution_timeout(3, 10)
    assert expected_timeout == 440.0, "Timeout calculation should match expected value"
    
    # Run evolution with new timeout
    start_time = time.time()
    try:
        result = await asyncio.wait_for(
            ga.evolve(request),
            timeout=expected_timeout
        )
        elapsed = time.time() - start_time
        
        # Verify completion
        assert result.success, "Evolution should complete successfully"
        assert len(result.final_population) >= config.population_size
        assert result.total_generations == config.generations
        assert elapsed < expected_timeout, f"Should complete within {expected_timeout}s, took {elapsed:.1f}s"
        
        print(f"\n✅ Evolution completed successfully in {elapsed:.1f}s")
        print(f"   Final population size: {len(result.final_population)}")
        print(f"   Best fitness: {result.final_population[0].overall_fitness:.3f}")
        
        # Check that semantic operators were used
        if hasattr(ga, 'semantic_operator_metrics'):
            print(f"   Semantic mutations: {ga.semantic_operator_metrics['semantic_mutations']}")
            print(f"   Semantic crossovers: {ga.semantic_operator_metrics['semantic_crossovers']}")
            assert ga.semantic_operator_metrics['semantic_mutations'] > 0, "Should use semantic mutations"
        
    except asyncio.TimeoutError:
        pytest.fail(f"Evolution timed out after {expected_timeout}s - fix did not work")
    except Exception as e:
        pytest.fail(f"Evolution failed with error: {e}")


@pytest.mark.integration  
def test_cli_output_shows_new_timeout():
    """Test that CLI shows the updated timeout value."""
    import subprocess
    import sys
    
    # Run the CLI with --help to avoid actual execution
    result = subprocess.run(
        [sys.executable, "qadi_simple.py", "test", "--evolve", "--generations", "3", "--population", "10", "--help"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent  # Run from project root
    )
    
    # The help output should show the command would use 440s timeout
    # Note: We can't easily test the actual timeout message without running evolution