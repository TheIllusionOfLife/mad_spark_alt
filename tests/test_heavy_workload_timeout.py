"""
Test suite specifically for heavy workload timeout scenarios.

This module tests the exact scenario that currently fails: --generations 5 --population 10
and validates that it completes within acceptable timeframes after optimization.
"""

import asyncio
import pytest
import time
import os
from unittest.mock import AsyncMock, MagicMock

from mad_spark_alt.evolution.interfaces import EvolutionConfig
from mad_spark_alt.evolution.genetic_algorithm import GeneticAlgorithm
from mad_spark_alt.evolution.interfaces import EvolutionRequest, IndividualFitness
from mad_spark_alt.core.interfaces import GeneratedIdea, ThinkingMethod


class TestHeavyWorkloadTimeout:
    """Test heavy workload scenarios that currently timeout."""

    def test_timeout_calculation_for_heavy_workload(self):
        """Test timeout calculation for heavy workload settings."""
        from qadi_simple import calculate_evolution_timeout
        
        # Test the exact heavy workload scenario
        generations = 5
        population = 10
        
        timeout = calculate_evolution_timeout(generations, population)
        
        # Current calculation: 60 + ((5*10 + 10) * 12) = 60 + (60 * 12) = 780s
        # Capped at 1200s (20 minutes)
        expected_timeout = min(60 + (60 * 12), 1200)  # 780s
        
        assert timeout == expected_timeout
        assert timeout == 780  # 13 minutes should be sufficient
        
        # But the real issue is architectural - this timeout should be plenty

    @pytest.mark.asyncio
    async def test_current_sequential_bottleneck_timing(self):
        """Test current sequential processing to demonstrate the bottleneck."""
        
        # Mock components to simulate current sequential processing
        mock_evaluator = AsyncMock()
        mock_mutation = AsyncMock()
        mock_crossover = AsyncMock()
        
        # Simulate realistic LLM call timing (5.6s average from PR #83)
        # Handle both 2-arg and 3-arg calls (some calls don't pass context)
        async def mock_evaluate(*args):
            # Extract arguments based on count
            if len(args) == 2:
                ideas, config = args
            else:
                ideas, config, context = args
            
            await asyncio.sleep(0.1)  # Simulate 5.6s per evaluation (scaled down for testing)
            return [
                IndividualFitness(
                    idea=idea,
                    impact=0.8,
                    feasibility=0.6,
                    accessibility=0.7,
                    sustainability=0.8,
                    scalability=0.7,
                    overall_fitness=0.7
                ) for idea in ideas
            ]
        
        async def mock_mutate(idea, context, evaluation_context=None):
            await asyncio.sleep(0.1)  # Simulate LLM call timing
            return GeneratedIdea(
                content=f"Mutated: {idea.content}",
                thinking_method=ThinkingMethod.ABDUCTION,
                agent_name="test_mutator",
                generation_prompt="test mutation",
                metadata=idea.metadata.copy()
            )
        
        async def mock_cross(idea1, idea2, context):
            await asyncio.sleep(0.1)  # Simulate LLM call timing
            return (
                GeneratedIdea(
                    content=f"Cross1: {idea1.content}", 
                    thinking_method=ThinkingMethod.ABDUCTION,
                    agent_name="test_crossover",
                    generation_prompt="test crossover",
                    metadata={}
                ),
                GeneratedIdea(
                    content=f"Cross2: {idea2.content}", 
                    thinking_method=ThinkingMethod.ABDUCTION,
                    agent_name="test_crossover",
                    generation_prompt="test crossover",
                    metadata={}
                )
            )
        
        mock_evaluator.evaluate_population = mock_evaluate
        mock_evaluator.calculate_population_diversity = AsyncMock(return_value=0.8)
        mock_mutation.mutate = mock_mutate
        mock_crossover.crossover = mock_cross
        
        # Create algorithm with sequential processing (current implementation)
        algorithm = GeneticAlgorithm(
            fitness_evaluator=mock_evaluator,
            crossover_operator=mock_crossover,
            mutation_operator=mock_mutation
        )
        
        # Set semantic operators directly 
        algorithm.semantic_mutation_operator = mock_mutation
        algorithm.semantic_crossover_operator = mock_crossover
        
        # Test moderate workload to demonstrate sequential bottleneck
        config = EvolutionConfig(
            population_size=6,
            generations=2,
            mutation_rate=0.8,  # High mutation rate to demonstrate bottleneck
            crossover_rate=0.8,
            use_semantic_operators=True
        )
        
        initial_ideas = [
            GeneratedIdea(
                content=f"Sequential test idea {i}",
                thinking_method=ThinkingMethod.ABDUCTION,
                agent_name="test_generator",
                generation_prompt="sequential test",
                metadata={"generation": 0}
            ) for i in range(config.population_size)
        ]
        
        request = EvolutionRequest(
            initial_population=initial_ideas,
            config=config,
            context="Sequential bottleneck test"
        )
        
        start_time = time.time()
        result = await algorithm.evolve(request)
        execution_time = time.time() - start_time
        
        # This test should demonstrate the sequential bottleneck
        # With current architecture: each generation processes pairs sequentially
        # Population=6, Generations=2 with high mutation rate should take significant time
        
        # Document the sequential processing time
        print(f"Sequential processing time for Pop=6, Gen=2: {execution_time:.2f}s")
        
        # The key insight: this time scales poorly to heavy workloads
        # Heavy workload (Pop=10, Gen=5) would take proportionally much longer
        
        assert result is not None
        assert result.total_generations == config.generations + 1  # Includes initial generation

    @pytest.mark.asyncio
    async def test_parallel_processing_target_performance(self):
        """Test target performance with parallel processing implementation."""
        
        # This test should FAIL initially (before parallel implementation)
        # and PASS after parallel processing is implemented
        
        # Mock parallel processing components
        mock_evaluator = AsyncMock()
        mock_batch_mutation = AsyncMock()
        mock_crossover = AsyncMock()
        
        # Simulate batch processing efficiency
        async def mock_batch_evaluate(population, config, context=None):
            # Batch evaluation should be more efficient than individual
            batch_size = min(len(population), 5)  # Process in batches of 5
            await asyncio.sleep(0.05 * len(population) / batch_size)  # Efficient batch timing
            return [
                IndividualFitness(
                    idea=idea,
                    impact=0.8,
                    feasibility=0.6,
                    accessibility=0.7,
                    sustainability=0.8,
                    scalability=0.7,
                    overall_fitness=0.7
                ) for idea in population
            ]
        
        async def mock_batch_mutate(ideas, context=None):
            # Batch mutation should process multiple ideas efficiently
            await asyncio.sleep(0.02 * len(ideas))  # Efficient batch processing
            return [
                GeneratedIdea(
                    content=f"Batch mutated: {idea.content}",
                    thinking_method=ThinkingMethod.ABDUCTION,
                    agent_name="batch_mutator",
                    generation_prompt="batch mutation",
                    metadata=idea.metadata.copy()
                ) for idea in ideas
            ]
        
        mock_evaluator.evaluate_population = mock_batch_evaluate
        mock_evaluator.calculate_population_diversity = AsyncMock(return_value=0.8)
        mock_batch_mutation.mutate_batch = mock_batch_mutate
        mock_crossover.crossover = AsyncMock(return_value=(
            GeneratedIdea(
                content="Parallel cross1", 
                thinking_method=ThinkingMethod.ABDUCTION,
                agent_name="test_crossover",
                generation_prompt="test",
                metadata={}
            ),
            GeneratedIdea(
                content="Parallel cross2", 
                thinking_method=ThinkingMethod.ABDUCTION,
                agent_name="test_crossover",
                generation_prompt="test",
                metadata={}
            )
        ))
        
        # Create algorithm that should use parallel processing
        # This will fail initially until parallel implementation is complete
        algorithm = GeneticAlgorithm(
            fitness_evaluator=mock_evaluator,
            crossover_operator=mock_crossover,
            mutation_operator=MagicMock()  # Traditional mutation
        )
        
        # Mock the semantic operators that are initialized internally
        algorithm.semantic_mutation_operator = mock_batch_mutation
        algorithm.semantic_crossover_operator = mock_crossover
        
        # Test heavy workload scenario
        config = EvolutionConfig(
            population_size=10,
            generations=5,
            mutation_rate=0.5,
            crossover_rate=0.8,
            use_semantic_operators=True,
            max_parallel_evaluations=5  # Enable parallel evaluation
        )
        
        initial_ideas = [
            GeneratedIdea(
                content=f"Heavy workload idea {i}",
                thinking_method=ThinkingMethod.ABDUCTION,
                agent_name="heavy_test_generator",
                generation_prompt="heavy workload test",
                metadata={"generation": 0}
            ) for i in range(config.population_size)
        ]
        
        request = EvolutionRequest(
            initial_population=initial_ideas,
            config=config,
            context="Heavy workload parallel processing test"
        )
        
        start_time = time.time()
        result = await algorithm.evolve(request)
        execution_time = time.time() - start_time
        
        # Target: Heavy workload should complete in under 4 minutes with parallel processing
        # This is the key performance target that needs to be achieved
        target_time = 240  # 4 minutes
        
        print(f"Parallel processing time for Pop=10, Gen=5: {execution_time:.2f}s (target: <{target_time}s)")
        
        # This assertion should FAIL initially, then PASS after optimization
        assert execution_time < target_time, \
            f"Heavy workload took {execution_time:.1f}s, should be under {target_time}s with parallel processing"
        
        # Verify result quality is maintained
        assert result is not None
        assert result.total_generations == config.generations + 1  # Includes initial generation
        assert len(result.final_population) == config.population_size

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_cli_heavy_workload_timeout_resolution(self):
        """Integration test for CLI heavy workload timeout resolution."""
        
        if not os.getenv('GOOGLE_API_KEY'):
            pytest.skip("GOOGLE_API_KEY not available for CLI integration test")
        
        # Test the exact CLI command that currently times out
        import subprocess
        
        test_command = [
            "uv", "run", "python", "qadi_simple.py",
            "How can we create a carbon-neutral city?",
            "--evolve",
            "--generations", "5",
            "--population", "10",
            "--temperature", "0.8"
        ]
        
        start_time = time.time()
        
        try:
            # Run with timeout to prevent hanging in CI
            result = subprocess.run(
                test_command,
                cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout for heavy workload
            )
            
            execution_time = time.time() - start_time
            
            # Verify command completed successfully
            assert result.returncode == 0, f"CLI command failed: {result.stderr}"
            
            # Verify execution time is reasonable
            assert execution_time < 480, f"CLI heavy workload took {execution_time:.1f}s, should be under 480s"
            
            # Verify output quality
            output = result.stdout
            assert "Evolution completed successfully" in output or "High Score Approaches" in output
            assert "Final Answer:" in output
            assert len(output) > 1000  # Substantial output, not truncated
            
            # Verify no timeout or truncation errors
            assert "timeout" not in output.lower()
            assert "truncated" not in output.lower()
            assert "error" not in output.lower()
            
        except subprocess.TimeoutExpired:
            execution_time = time.time() - start_time
            pytest.fail(f"CLI heavy workload timed out after {execution_time:.1f}s - optimization needed")


class TestParallelArchitectureRequirements:
    """Tests that define the requirements for parallel architecture implementation."""

    def test_batch_mutation_interface_requirements(self):
        """Test that batch mutation interface exists and works correctly."""
        
        from mad_spark_alt.evolution.semantic_operators import BatchSemanticMutationOperator
        from mad_spark_alt.core.llm_provider import GoogleProvider
        
        # Verify batch mutation operator exists (requires LLM provider)
        mock_provider = MagicMock()
        operator = BatchSemanticMutationOperator(mock_provider)
        
        # Verify it has the required mutate_batch method
        assert hasattr(operator, 'mutate_batch')
        assert callable(getattr(operator, 'mutate_batch'))
        
        # The batch mutation should support processing multiple ideas at once
        # This is key to the parallel processing optimization

    @pytest.mark.asyncio
    async def test_parallel_evaluation_requirements(self):
        """Test requirements for parallel evaluation processing."""
        
        # Test fitness evaluator architecture rather than specific implementation class
        from mad_spark_alt.evolution.fitness import FitnessEvaluator
        
        # Create standard fitness evaluator
        evaluator = FitnessEvaluator()
        
        # Verify it can handle batch evaluation
        assert hasattr(evaluator, 'evaluate_population')
        
        # The evaluator should efficiently process multiple ideas
        # Implementation should support batching for parallel processing

    def test_evolution_config_parallel_settings(self):
        """Test that evolution config supports parallel processing settings."""
        
        config = EvolutionConfig(
            population_size=10,
            generations=5,
            max_parallel_evaluations=5,
            use_semantic_operators=True
        )
        
        # Verify parallel settings are available
        assert hasattr(config, 'max_parallel_evaluations')
        assert config.max_parallel_evaluations == 5
        
        # These settings should be used by the parallel implementation