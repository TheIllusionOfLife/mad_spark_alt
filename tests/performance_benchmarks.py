"""
Performance benchmarking tests for Mad Spark Alt.

These tests measure execution time and memory usage for key operations.
"""

import asyncio
import time

import pytest

from mad_spark_alt.core import (
    EvaluationRequest,
    ModelOutput,
    OutputType,
    SmartQADIOrchestrator,
    ThinkingMethod,
)
from mad_spark_alt.evolution import (
    EvolutionConfig,
    EvolutionRequest,
    GeneticAlgorithm,
    SelectionStrategy,
)
from mad_spark_alt.layers.quantitative import DiversityEvaluator, QualityEvaluator


class TestPerformanceBenchmarks:
    """Performance benchmarking tests."""

    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Setup and teardown for each test."""
        # Setup: Clear any caches or state
        yield
        # Teardown: Clean up resources
        import gc

        gc.collect()

    @pytest.mark.asyncio
    async def test_qadi_cycle_performance(self) -> None:
        """Benchmark QADI cycle execution time."""
        orchestrator = SmartQADIOrchestrator()

        test_problems = [
            "How can we reduce plastic waste?",
            "What are innovative solutions for remote work?",
            "How might we improve urban transportation?",
        ]

        execution_times = []

        for problem in test_problems:
            start_time = time.time()
            result = await orchestrator.run_qadi_cycle(
                problem_statement=problem, cycle_config={"max_ideas_per_method": 2}
            )
            execution_time = time.time() - start_time
            execution_times.append(execution_time)

            # Basic assertions
            assert result is not None
            assert len(result.phases) > 0

        avg_time = sum(execution_times) / len(execution_times)
        max_time = max(execution_times)

        # Performance targets (adjust based on requirements)
        assert avg_time < 180  # Average should be under 3 minutes
        assert max_time < 240  # Max should be under 4 minutes

        print("\nQADI Cycle Performance:")
        print(f"  Average time: {avg_time:.2f}s")
        print(f"  Max time: {max_time:.2f}s")
        print(f"  Times: {[f'{t:.2f}s' for t in execution_times]}")

    @pytest.mark.asyncio
    async def test_parallel_generation_performance(self) -> None:
        """Benchmark parallel idea generation."""
        orchestrator = SmartQADIOrchestrator()

        methods = [
            ThinkingMethod.QUESTIONING,
            ThinkingMethod.ABDUCTION,
            ThinkingMethod.DEDUCTION,
            ThinkingMethod.INDUCTION,
        ]

        # Test different parallel configurations
        configurations = [
            (2, "Two methods"),
            (4, "All methods"),
        ]

        for num_methods, desc in configurations:
            start_time = time.time()
            results = await orchestrator.run_parallel_generation(
                problem_statement="Test problem for benchmarking",
                thinking_methods=methods[:num_methods],
                config={"max_ideas_per_method": 3},
            )
            execution_time = time.time() - start_time

            assert len(results) > 0

            print(f"\nParallel Generation ({desc}):")
            print(f"  Time: {execution_time:.2f}s")
            print(f"  Methods completed: {len(results)}")

            # Performance target
            assert execution_time < 60 * num_methods  # Should scale sub-linearly

    @pytest.mark.asyncio
    async def test_evaluation_performance(self) -> None:
        """Benchmark evaluation performance."""
        # Create test outputs
        outputs = [
            ModelOutput(
                content=f"This is test output {i} with some creative content about innovation and problem solving.",
                output_type=OutputType.TEXT,
                model_name="test_model",
                metadata={"index": i},
            )
            for i in range(10)
        ]

        evaluators = [DiversityEvaluator(), QualityEvaluator()]

        for evaluator in evaluators:
            request = EvaluationRequest(
                request_id=f"perf_test_{evaluator.name}",
                outputs=outputs,
                context="Performance testing",
            )

            start_time = time.time()
            results = await evaluator.evaluate(request)
            execution_time = time.time() - start_time

            assert len(results) == len(outputs)

            print(f"\n{evaluator.name} Performance:")
            print(f"  Time for {len(outputs)} outputs: {execution_time:.2f}s")
            print(f"  Time per output: {execution_time/len(outputs):.3f}s")

            # Performance target
            assert execution_time < 10  # Should evaluate 10 outputs in under 10s

    @pytest.mark.asyncio
    async def test_evolution_performance(self) -> None:
        """Benchmark genetic evolution performance."""
        # Create initial population
        initial_ideas = [
            ModelOutput(
                content=f"Idea {i}: Solution involving technology and sustainability",
                output_type=OutputType.TEXT,
                model_name="test_model",
                metadata={"generation": 0},
            )
            for i in range(10)
        ]

        config = EvolutionConfig(
            population_size=10,
            generations=3,
            mutation_rate=0.3,
            crossover_rate=0.7,
            selection_strategy=SelectionStrategy.TOURNAMENT,
            tournament_size=3,
            elite_size=2,
        )

        ga = GeneticAlgorithm()

        # Create evolution request
        evolution_request = EvolutionRequest(
            initial_population=initial_ideas,
            config=config,
            context="Test evolution for performance",
        )

        start_time = time.time()
        result = await ga.evolve(evolution_request)
        execution_time = time.time() - start_time

        assert len(result.final_population) == config.population_size

        print("\nEvolution Performance:")
        print(f"  Time for {config.generations} generations: {execution_time:.2f}s")
        print(f"  Population size: {config.population_size}")
        print(f"  Time per generation: {execution_time/config.generations:.2f}s")

        # Performance target
        assert execution_time < 60  # 3 generations should complete in under 1 minute

    @pytest.mark.asyncio
    async def test_memory_usage_qadi_cycle(self) -> None:
        """Test memory usage during QADI cycle."""
        # This is a simple memory tracking test
        # For detailed profiling, use memory_profiler decorators

        import tracemalloc

        tracemalloc.start()

        orchestrator = SmartQADIOrchestrator()

        # Get baseline
        snapshot1 = tracemalloc.take_snapshot()

        # Run QADI cycle
        result = await orchestrator.run_qadi_cycle(
            problem_statement="Memory test problem",
            cycle_config={"max_ideas_per_method": 5},
        )

        # Get memory after
        snapshot2 = tracemalloc.take_snapshot()

        top_stats = snapshot2.compare_to(snapshot1, "lineno")

        # Calculate total memory increase
        total_memory = sum(stat.size_diff for stat in top_stats)
        total_memory_mb = total_memory / 1024 / 1024

        print("\nMemory Usage (QADI Cycle):")
        print(f"  Total increase: {total_memory_mb:.2f} MB")
        print(f"  Ideas generated: {len(result.synthesized_ideas)}")

        # Memory target
        assert total_memory_mb < 100  # Should use less than 100MB

        tracemalloc.stop()

    def test_concurrent_request_handling(self) -> None:
        """Test handling multiple concurrent requests."""

        async def run_concurrent_test() -> None:
            orchestrator = SmartQADIOrchestrator()

            # Create multiple concurrent requests
            tasks = []
            num_concurrent = 5

            for i in range(num_concurrent):
                task = orchestrator.run_parallel_generation(
                    problem_statement=f"Concurrent test {i}",
                    thinking_methods=[
                        ThinkingMethod.QUESTIONING,
                        ThinkingMethod.ABDUCTION,
                    ],
                    config={"max_ideas_per_method": 2},
                )
                tasks.append(task)

            start_time = time.time()
            results = await asyncio.gather(*tasks)
            execution_time = time.time() - start_time

            assert len(results) == num_concurrent

            print("\nConcurrent Request Performance:")
            print(f"  Requests: {num_concurrent}")
            print(f"  Total time: {execution_time:.2f}s")
            print(f"  Average time per request: {execution_time/num_concurrent:.2f}s")

            # Should handle concurrent requests efficiently
            assert execution_time < 120  # 5 concurrent requests in under 2 minutes

        asyncio.run(run_concurrent_test())


if __name__ == "__main__":
    # Run benchmarks directly
    import sys

    pytest.main([__file__, "-v", "-s"] + sys.argv[1:])
