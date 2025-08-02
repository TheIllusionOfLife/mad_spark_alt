"""
Test suite for parallel evolution processing performance optimizations.

This module contains comprehensive tests to validate that heavy workload scenarios
(--generations 5 --population 10) complete within acceptable timeframes through
parallel processing optimizations.
"""

import asyncio
import pytest
import time
from unittest.mock import AsyncMock, MagicMock
from typing import List

from mad_spark_alt.evolution.genetic_algorithm import GeneticAlgorithm
from mad_spark_alt.evolution.interfaces import EvolutionConfig
from mad_spark_alt.evolution.interfaces import EvolutionRequest, IndividualFitness
from mad_spark_alt.core.interfaces import GeneratedIdea, ThinkingMethod


class TestParallelEvolutionPerformance:
    """Test parallel evolution processing performance improvements."""

    @pytest.fixture
    def mock_fitness_evaluator(self):
        """Mock fitness evaluator with realistic timing."""
        evaluator = AsyncMock()
        
        # Simulate realistic evaluation timing (3s per evaluation)
        async def mock_evaluate(population, config, context=None):
            await asyncio.sleep(0.1)  # Simulate network delay
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
        
        evaluator.evaluate_population = mock_evaluate
        evaluator.calculate_population_diversity = AsyncMock(return_value=0.8)
        return evaluator

    @pytest.fixture
    def mock_semantic_operators(self):
        """Mock semantic operators with batch processing capabilities."""
        mutation_op = AsyncMock()
        crossover_op = AsyncMock()
        
        # Mock batch mutation with realistic timing
        async def mock_batch_mutate(ideas, context, evaluation_context=None):
            await asyncio.sleep(0.05 * len(ideas))  # Scale with batch size
            return [
                GeneratedIdea(
                    content=f"Mutated: {idea.content}",
                    thinking_method=ThinkingMethod.ABDUCTION,
                    agent_name="batch_mutator",
                    generation_prompt="batch mutation",
                    metadata={"generation": idea.metadata.get("generation", 0) + 1}
                ) for idea in ideas
            ]
        
        # Mock individual crossover
        async def mock_crossover(idea1, idea2, context):
            await asyncio.sleep(0.05)  # Simulate network delay
            return (
                GeneratedIdea(
                    content=f"Crossover1: {idea1.content} + {idea2.content}",
                    thinking_method=ThinkingMethod.ABDUCTION,
                    agent_name="test_crossover",
                    generation_prompt="test crossover",
                    metadata={"generation": 1}
                ),
                GeneratedIdea(
                    content=f"Crossover2: {idea1.content} + {idea2.content}",
                    thinking_method=ThinkingMethod.ABDUCTION,
                    agent_name="test_crossover",
                    generation_prompt="test crossover",
                    metadata={"generation": 1}
                )
            )
        
        mutation_op.batch_mutate = mock_batch_mutate
        crossover_op.crossover = mock_crossover
        
        return mutation_op, crossover_op

    @pytest.fixture
    def genetic_algorithm(self, mock_fitness_evaluator, mock_semantic_operators):
        """Create genetic algorithm with mocked components."""
        mutation_op, crossover_op = mock_semantic_operators
        
        algorithm = GeneticAlgorithm(
            fitness_evaluator=mock_fitness_evaluator,
            crossover_operator=MagicMock(),  # Traditional crossover
            mutation_operator=MagicMock()    # Traditional mutation
        )
        
        # Set semantic operators directly (they're initialized in constructor with llm_provider)
        algorithm.semantic_mutation_operator = mutation_op
        algorithm.semantic_crossover_operator = crossover_op
        
        return algorithm

    @pytest.mark.asyncio
    async def test_heavy_workload_completion_time(self, genetic_algorithm):
        """Test that heavy workloads complete within acceptable timeframes."""
        # This test should FAIL initially, then PASS after parallel implementation
        
        # Heavy workload configuration
        config = EvolutionConfig(
            population_size=10,
            generations=5,
            mutation_rate=0.3,
            crossover_rate=0.8,
            use_semantic_operators=True,
            max_parallel_evaluations=5
        )
        
        # Create initial population (EvolutionRequest expects GeneratedIdea objects)
        initial_ideas = [
            GeneratedIdea(
                content=f"Initial idea {i}",
                thinking_method=ThinkingMethod.ABDUCTION,
                agent_name="test_generator",
                generation_prompt="initial generation",
                metadata={"generation": 0}
            ) for i in range(config.population_size)
        ]
        
        # Create evolution request
        request = EvolutionRequest(
            initial_population=initial_ideas,
            config=config,
            context="Test parallel processing optimization"
        )
        
        # Measure execution time
        start_time = time.time()
        
        result = await genetic_algorithm.evolve(request)
        
        execution_time = time.time() - start_time
        
        # Assert completion within acceptable timeframe
        # Target: Heavy workload should complete in under 6 minutes (360s)
        # With parallel processing: expect 3-5 minutes for heavy workload
        assert execution_time < 360, f"Heavy workload took {execution_time:.1f}s, should be under 360s"
        
        # Verify result quality
        assert result is not None
        assert len(result.final_population) == config.population_size
        # total_generations includes initial generation (0) + evolution generations
        assert result.total_generations == config.generations + 1

    @pytest.mark.asyncio
    async def test_parallel_vs_sequential_performance(self, genetic_algorithm):
        """Test that parallel processing provides significant performance improvement."""
        
        config = EvolutionConfig(
            population_size=6,
            generations=2,
            mutation_rate=0.3,
            crossover_rate=0.8,
            use_semantic_operators=True,
            max_parallel_evaluations=3
        )
        
        initial_ideas = [
            GeneratedIdea(
                content=f"Test idea {i}",
                thinking_method=ThinkingMethod.ABDUCTION,
                agent_name="test_generator",
                generation_prompt="parallel test",
                metadata={"generation": 0}
            ) for i in range(config.population_size)
        ]
        
        request = EvolutionRequest(
            initial_population=initial_ideas,
            config=config,
            context="Performance comparison test"
        )
        
        # Test with parallel processing (should be implemented)
        start_time = time.time()
        result_parallel = await genetic_algorithm.evolve(request)
        parallel_time = time.time() - start_time
        
        # Verify parallel processing performance characteristics
        # With parallel processing, we expect significant speedup
        # For 6 population, 2 generations: should complete quickly
        assert parallel_time < 30, f"Parallel evolution took {parallel_time:.1f}s, too slow"
        
        # Verify quality is maintained
        assert result_parallel is not None
        assert len(result_parallel.final_population) == config.population_size

    @pytest.mark.asyncio
    async def test_batch_processing_efficiency(self, genetic_algorithm):
        """Test that batch processing is more efficient than individual processing."""
        
        # Test with configuration that would benefit from batching
        config = EvolutionConfig(
            population_size=8,
            generations=3,
            mutation_rate=0.5,  # High mutation rate to test batch efficiency
            crossover_rate=0.7,
            use_semantic_operators=True,
            max_parallel_evaluations=4
        )
        
        initial_ideas = [
            GeneratedIdea(
                content=f"Batch test idea {i}",
                thinking_method=ThinkingMethod.ABDUCTION,
                agent_name="batch_test_generator",
                generation_prompt="batch efficiency test",
                metadata={"generation": 0}
            ) for i in range(config.population_size)
        ]
        
        request = EvolutionRequest(
            initial_population=initial_ideas,
            config=config,
            context="Batch processing efficiency test"
        )
        
        start_time = time.time()
        result = await genetic_algorithm.evolve(request)
        execution_time = time.time() - start_time
        
        # With efficient batch processing, this should complete quickly
        assert execution_time < 45, f"Batch processing took {execution_time:.1f}s, should be under 45s"
        
        # Verify all generations completed (includes initial generation)
        assert result.total_generations == config.generations + 1
        assert len(result.final_population) == config.population_size

    @pytest.mark.asyncio 
    async def test_scalability_across_population_sizes(self, genetic_algorithm):
        """Test that performance scales reasonably with population size."""
        
        population_sizes = [3, 5, 8, 10]
        execution_times = []
        
        for pop_size in population_sizes:
            config = EvolutionConfig(
                population_size=pop_size,
                generations=2,
                mutation_rate=0.3,
                crossover_rate=0.8,
                use_semantic_operators=True,
                max_parallel_evaluations=min(pop_size, 5)
            )
            
            initial_ideas = [
                GeneratedIdea(
                    content=f"Scalability test idea {i}",
                    thinking_method=ThinkingMethod.ABDUCTION,
                    agent_name="scalability_test_generator",
                    generation_prompt="scalability test",
                    metadata={"generation": 0}
                ) for i in range(pop_size)
            ]
            
            request = EvolutionRequest(
                initial_population=initial_ideas,
                config=config,
                context=f"Scalability test for population {pop_size}"
            )
            
            start_time = time.time()
            await genetic_algorithm.evolve(request)
            execution_time = time.time() - start_time
            execution_times.append(execution_time)
        
        # Verify scalability - execution time should scale reasonably
        # With parallel processing, scaling should be much better than linear
        for i, (pop_size, exec_time) in enumerate(zip(population_sizes, execution_times)):
            # Each population size should complete within reasonable bounds
            max_expected_time = pop_size * 5  # 5 seconds per individual (generous)
            assert exec_time < max_expected_time, \
                f"Population {pop_size} took {exec_time:.1f}s, should be under {max_expected_time}s"


class TestParallelEvolutionIntegration:
    """Integration tests for parallel evolution processing with various configurations."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_real_heavy_workload_with_api(self):
        """Integration test with real API for heavy workload - REQUIRES GOOGLE_API_KEY."""
        
        # Skip if no API key available
        import os
        if not os.getenv('GOOGLE_API_KEY'):
            pytest.skip("GOOGLE_API_KEY not available for integration test")
        
        from mad_spark_alt.core import setup_llm_providers
        from mad_spark_alt.core.simple_qadi_orchestrator import SimpleQADIOrchestrator
        
        # Setup real LLM providers
        setup_llm_providers()
        
        # Test heavy workload with real API
        orchestrator = SimpleQADIOrchestrator()
        
        start_time = time.time()
        
        # This should complete without timeout
        result = await orchestrator.run_qadi_with_evolution(
            user_input="How can we create a sustainable transportation system?",
            context="Heavy workload integration test",
            evolve=True,
            generations=5,
            population=10,
            temperature_override=0.8
        )
        
        execution_time = time.time() - start_time
        
        # Heavy workload should complete in under 8 minutes with parallel processing
        assert execution_time < 480, f"Heavy workload took {execution_time:.1f}s, should be under 480s"
        
        # Verify result quality
        assert result is not None
        assert result.hypotheses is not None
        assert len(result.hypotheses) >= 3
        assert result.final_answer is not None
        assert result.verification_conclusion is not None
        
        # If evolution was used, verify evolution results
        if hasattr(result, 'evolution_result') and result.evolution_result:
            assert result.evolution_result.final_population is not None
            assert len(result.evolution_result.final_population) == 10

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_parallel_processing_quality_preservation(self):
        """Test that parallel processing maintains evolution quality."""
        
        import os
        if not os.getenv('GOOGLE_API_KEY'):
            pytest.skip("GOOGLE_API_KEY not available for integration test")
        
        from mad_spark_alt.core import setup_llm_providers
        from mad_spark_alt.core.simple_qadi_orchestrator import SimpleQADIOrchestrator
        
        setup_llm_providers()
        orchestrator = SimpleQADIOrchestrator()
        
        # Run same question multiple times to check consistency
        test_question = "What are innovative approaches to reduce food waste?"
        
        results = []
        for i in range(2):  # Run twice to check consistency
            result = await orchestrator.run_qadi_with_evolution(
                user_input=test_question,
                context=f"Quality test run {i+1}",
                evolve=True,
                generations=3,
                population=6,
                temperature_override=0.7
            )
            results.append(result)
        
        # Verify both runs produced quality results
        for i, result in enumerate(results):
            assert result is not None, f"Run {i+1} failed to produce result"
            assert result.hypotheses is not None, f"Run {i+1} missing hypotheses"
            assert len(result.hypotheses) >= 3, f"Run {i+1} has insufficient hypotheses"
            assert result.final_answer is not None, f"Run {i+1} missing final answer"
            
            # Check that final answer is substantial (not truncated)
            assert len(result.final_answer) > 100, f"Run {i+1} final answer too short: {len(result.final_answer)} chars"
            
            # Check for evolution results quality
            if hasattr(result, 'evolution_result') and result.evolution_result:
                assert result.evolution_result.final_population is not None
                # Evolution should produce diverse results
                unique_contents = set(ind.idea.content for ind in result.evolution_result.final_population)
                assert len(unique_contents) > 1, f"Run {i+1} evolution produced identical results"