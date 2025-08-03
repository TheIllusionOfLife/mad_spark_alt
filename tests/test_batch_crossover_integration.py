"""
Integration tests for batch crossover in genetic algorithm.

This module tests that the batch crossover operator is properly integrated
into the genetic algorithm's parallel processing pipeline.
"""

import asyncio
from typing import List
from unittest.mock import AsyncMock, MagicMock, patch
import json

import pytest

from mad_spark_alt.core.interfaces import GeneratedIdea, ThinkingMethod
from mad_spark_alt.evolution.genetic_algorithm import GeneticAlgorithm
from mad_spark_alt.evolution.interfaces import (
    EvolutionConfig,
    EvolutionRequest,
    IndividualFitness,
)


@pytest.fixture
def mock_llm_provider():
    """Create mock LLM provider."""
    provider = MagicMock()
    provider.generate = AsyncMock()
    return provider


@pytest.fixture
def initial_population() -> List[GeneratedIdea]:
    """Create initial population for testing."""
    ideas = []
    for i in range(5):
        idea = GeneratedIdea(
            content=f"Initial idea {i}: Test content for evolution",
            thinking_method=ThinkingMethod.QUESTIONING,
            agent_name="TestAgent",
            generation_prompt="Test prompt",
            metadata={"seed": i}
        )
        ideas.append(idea)
    return ideas


@pytest.fixture
def evolution_config():
    """Create evolution config for testing."""
    return EvolutionConfig(
        population_size=6,
        generations=2,
        mutation_rate=0.5,
        crossover_rate=0.8,
        elite_size=1,
        use_semantic_operators=True,
        enable_llm_operators=True,
        parallel_evaluation=True,
        max_parallel_evaluations=3
    )


class TestBatchCrossoverIntegration:
    """Test batch crossover integration with genetic algorithm."""

    @pytest.mark.asyncio
    async def test_parallel_generation_uses_batch_crossover(
        self, mock_llm_provider, initial_population, evolution_config
    ):
        """Test that parallel generation correctly uses batch crossover."""
        # Create mock fitness evaluator
        mock_fitness_evaluator = MagicMock()
        mock_fitness_evaluator.evaluate_population = AsyncMock(
            return_value=[
                IndividualFitness(idea=idea, overall_fitness=0.5 + i * 0.1)
                for i, idea in enumerate(initial_population)
            ]
        )
        mock_fitness_evaluator.calculate_population_diversity = AsyncMock(return_value=0.7)
        
        # Create mock operators
        mock_crossover = MagicMock()
        mock_crossover.crossover = AsyncMock(
            side_effect=lambda p1, p2, ctx: (p1, p2)
        )
        
        mock_mutation = MagicMock()
        mock_mutation.mutate = AsyncMock(
            side_effect=lambda idea, rate, ctx: idea
        )
        
        # Set up genetic algorithm
        ga = GeneticAlgorithm(
            fitness_evaluator=mock_fitness_evaluator,
            crossover_operator=mock_crossover,
            mutation_operator=mock_mutation,
            llm_provider=mock_llm_provider,
        )
        
        # Set up mock responses for batch operations
        def generate_side_effect(request):
            # Check if it's a batch crossover request
            if "Generate crossover offspring for the following" in request.user_prompt:
                # Count how many pairs are in the request
                num_pairs = request.user_prompt.count("Pair")
                
                # Return batch crossover response
                crossovers = []
                for i in range(num_pairs):
                    crossovers.append({
                        "pair_id": i,
                        "offspring1": f"Batch crossover offspring {i*2}",
                        "offspring2": f"Batch crossover offspring {i*2+1}"
                    })
                
                return MagicMock(
                    content=json.dumps({"crossovers": crossovers}),
                    cost=0.005
                )
            # Check if it's a batch mutation request
            elif "Perform batch semantic mutation" in request.user_prompt:
                # Count how many ideas to mutate
                num_ideas = request.user_prompt.count("Idea ")
                
                mutations = []
                for i in range(num_ideas):
                    mutations.append({
                        "id": i,
                        "content": f"Batch mutated idea {i}"
                    })
                
                return MagicMock(
                    content=json.dumps({"mutations": mutations}),
                    cost=0.003
                )
            else:
                # Default response for other requests
                return MagicMock(content="Default response", cost=0.001)
        
        mock_llm_provider.generate.side_effect = generate_side_effect
        
        # Create evolution request
        request = EvolutionRequest(
            initial_population=initial_population,
            config=evolution_config,
            context="Test evolution"
        )
        
        # Run one generation
        result = await ga.evolve(request)
        
        # Verify result
        assert result.success
        assert len(result.generation_snapshots) == evolution_config.generations + 1  # Initial + 2 generations
        
        # Check that batch crossover was used
        # With population 6, elite 1, we need 5 new offspring
        # With crossover rate 0.8, we expect ~2 pairs to crossover
        # Should see 1 batch crossover call (not multiple sequential calls)
        
        # Count crossover-related calls
        crossover_calls = sum(
            1 for call in mock_llm_provider.generate.call_args_list
            if "crossover" in str(call)
        )
        
        # Count batch vs sequential crossover calls
        batch_crossover_calls = sum(
            1 for call in mock_llm_provider.generate.call_args_list
            if "Generate crossover offspring for the following" in str(call)
        )
        sequential_crossover_calls = sum(
            1 for call in mock_llm_provider.generate.call_args_list
            if "crossover" in str(call) and "Generate crossover offspring for the following" not in str(call)
        )
        
        # For 2 generations with batch processing, we expect mostly batch calls
        # Batch calls should be significantly less than what sequential would require
        print(f"Batch crossover calls: {batch_crossover_calls}, Sequential: {sequential_crossover_calls}")
        
        # With population 6, elite 1, we need 5 offspring per generation = ~3 pairs
        # Without batching: 2 generations × 3 pairs = 6 crossover calls
        # With batching: 2 generations × 1 batch = 2 batch calls (maybe 1-2 sequential for fallback)
        assert batch_crossover_calls <= 3, f"Too many batch crossover calls: {batch_crossover_calls}"
        assert crossover_calls < 6, f"Total crossover calls {crossover_calls} not much better than sequential"
        
        # Verify metrics show batch processing
        metrics = ga.semantic_operator_metrics
        assert metrics['semantic_crossovers'] > 0
        # LLM calls should be much less than crossovers due to batching
        assert metrics['semantic_llm_calls'] < metrics['semantic_crossovers'] + metrics['semantic_mutations']

    @pytest.mark.asyncio
    async def test_batch_crossover_fallback_to_sequential(
        self, mock_llm_provider, initial_population, evolution_config
    ):
        """Test that batch crossover falls back to sequential on error."""
        # Create mock fitness evaluator
        mock_fitness_evaluator = MagicMock()
        mock_fitness_evaluator.evaluate_population = AsyncMock(
            return_value=[
                IndividualFitness(idea=idea, overall_fitness=0.5)
                for idea in initial_population
            ]
        )
        mock_fitness_evaluator.calculate_population_diversity = AsyncMock(return_value=0.7)
        
        # Create mock operators
        mock_crossover = MagicMock()
        mock_crossover.crossover = AsyncMock(
            side_effect=lambda p1, p2, ctx: (p1, p2)
        )
        
        mock_mutation = MagicMock()
        mock_mutation.mutate = AsyncMock(
            side_effect=lambda idea, rate, ctx: idea
        )
        
        # Set up genetic algorithm
        ga = GeneticAlgorithm(
            fitness_evaluator=mock_fitness_evaluator,
            crossover_operator=mock_crossover,
            mutation_operator=mock_mutation,
            llm_provider=mock_llm_provider,
        )
        
        # Set up mock to fail on batch but succeed on sequential
        call_count = 0
        
        def generate_side_effect(request):
            nonlocal call_count
            call_count += 1
            
            # Fail the first batch crossover attempt
            if "Generate crossover offspring for the following" in request.user_prompt and call_count == 1:
                raise Exception("Batch crossover failed")
            
            # Succeed on sequential crossover attempts
            if "First Approach:" in request.user_prompt:
                return MagicMock(
                    content=json.dumps({
                        "offspring_1": f"Sequential offspring 1 (call {call_count})",
                        "offspring_2": f"Sequential offspring 2 (call {call_count})"
                    }),
                    cost=0.001
                )
            
            # Handle mutations
            if "mutation" in request.user_prompt.lower():
                return MagicMock(
                    content=json.dumps({"mutations": [{"id": 0, "content": "Mutated"}]}),
                    cost=0.001
                )
            
            return MagicMock(content="Default", cost=0.001)
        
        mock_llm_provider.generate.side_effect = generate_side_effect
        
        # Create evolution request
        request = EvolutionRequest(
            initial_population=initial_population,
            config=evolution_config,
            context="Test evolution"
        )
        
        # Run evolution - should not crash despite batch failure
        result = await ga.evolve(request)
        
        # Verify it succeeded using fallback
        assert result.success
        assert call_count > 2  # Should have made multiple sequential calls after batch failed

    @pytest.mark.asyncio 
    async def test_batch_crossover_performance_improvement(
        self, mock_llm_provider, evolution_config
    ):
        """Test that batch crossover reduces LLM calls significantly."""
        # Create larger population for meaningful test
        large_population = []
        for i in range(10):
            idea = GeneratedIdea(
                content=f"Idea {i}",
                thinking_method=ThinkingMethod.QUESTIONING,
                agent_name="TestAgent",
                generation_prompt="Test",
                metadata={"id": i}
            )
            large_population.append(idea)
        
        # Update config for larger population
        evolution_config.population_size = 10
        evolution_config.generations = 3
        evolution_config.elite_size = 2
        
        # Create mock fitness evaluator
        mock_fitness_evaluator = MagicMock()
        mock_fitness_evaluator.evaluate_population = AsyncMock(
            side_effect=lambda ideas, *args: [
                IndividualFitness(idea=idea, overall_fitness=0.5)
                for idea in ideas
            ]
        )
        mock_fitness_evaluator.calculate_population_diversity = AsyncMock(return_value=0.7)
        
        # Create mock operators
        mock_crossover = MagicMock()
        mock_crossover.crossover = AsyncMock(
            side_effect=lambda p1, p2, ctx: (p1, p2)
        )
        
        mock_mutation = MagicMock()
        mock_mutation.mutate = AsyncMock(
            side_effect=lambda idea, rate, ctx: idea
        )
        
        # Set up genetic algorithm
        ga = GeneticAlgorithm(
            fitness_evaluator=mock_fitness_evaluator,
            crossover_operator=mock_crossover,
            mutation_operator=mock_mutation,
            llm_provider=mock_llm_provider,
        )
        
        # Track different types of LLM calls
        crossover_batch_calls = 0
        mutation_batch_calls = 0
        other_calls = 0
        
        def generate_side_effect(request):
            nonlocal crossover_batch_calls, mutation_batch_calls, other_calls
            
            if "Generate crossover offspring for the following" in request.user_prompt:
                crossover_batch_calls += 1
                num_pairs = request.user_prompt.count("Pair")
                crossovers = [
                    {
                        "pair_id": i,
                        "offspring1": f"Batch offspring {i*2}",
                        "offspring2": f"Batch offspring {i*2+1}"
                    }
                    for i in range(num_pairs)
                ]
                return MagicMock(
                    content=json.dumps({"crossovers": crossovers}),
                    cost=0.01
                )
            elif "Perform batch semantic mutation" in request.user_prompt:
                mutation_batch_calls += 1
                num_ideas = request.user_prompt.count("Idea ")
                mutations = [
                    {"id": i, "content": f"Mutated {i}"}
                    for i in range(num_ideas)
                ]
                return MagicMock(
                    content=json.dumps({"mutations": mutations}),
                    cost=0.005
                )
            else:
                other_calls += 1
                return MagicMock(content="Other", cost=0.001)
        
        mock_llm_provider.generate.side_effect = generate_side_effect
        
        # Run evolution
        request = EvolutionRequest(
            initial_population=large_population,
            config=evolution_config
        )
        
        result = await ga.evolve(request)
        
        # Verify performance improvement
        assert result.success
        
        # With 3 generations, population 10, elite 2:
        # Each generation needs 8 new offspring = 4 parent pairs
        # Without batching: 3 generations × 4 pairs = 12 crossover LLM calls
        # With batching: 3 generations × 1 batch call = 3 crossover LLM calls
        
        assert crossover_batch_calls <= 3, f"Expected at most 3 batch crossover calls, got {crossover_batch_calls}"
        assert mutation_batch_calls <= 3, f"Expected at most 3 batch mutation calls, got {mutation_batch_calls}"
        
        # Total LLM calls should be much less than individual operations
        total_llm_calls = crossover_batch_calls + mutation_batch_calls + other_calls
        expected_operations = evolution_config.generations * (evolution_config.population_size - evolution_config.elite_size)
        
        assert total_llm_calls < expected_operations / 2, (
            f"Batch processing should reduce LLM calls significantly. "
            f"Got {total_llm_calls} calls for {expected_operations} operations"
        )