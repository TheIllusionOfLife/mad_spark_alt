"""
Performance benchmarking suite for evolution system.

This module provides comprehensive benchmarking tools for
measuring and analyzing evolution system performance.
"""

import asyncio
import gc
import json
import time
import tracemalloc
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import psutil  # type: ignore

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    psutil = None

from mad_spark_alt.core.interfaces import GeneratedIdea
from mad_spark_alt.evolution.genetic_algorithm import GeneticAlgorithm
from mad_spark_alt.evolution.interfaces import (
    EvolutionConfig,
    EvolutionRequest,
    EvolutionResult,
    IndividualFitness,
)


@dataclass
class BenchmarkMetrics:
    """Metrics collected during benchmark run."""

    name: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Timing metrics
    total_time: float = 0.0
    generation_times: List[float] = field(default_factory=list)
    evaluation_times: List[float] = field(default_factory=list)

    # Memory metrics
    peak_memory_mb: float = 0.0
    memory_snapshots: List[Dict[str, float]] = field(default_factory=list)

    # Performance metrics
    total_evaluations: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    llm_calls: int = 0
    llm_tokens: int = 0
    llm_cost: float = 0.0

    # Quality metrics
    fitness_progression: List[float] = field(default_factory=list)
    diversity_progression: List[float] = field(default_factory=list)
    final_best_fitness: float = 0.0

    # System metrics
    cpu_usage: List[float] = field(default_factory=list)

    @property
    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0

    @property
    def avg_generation_time(self) -> float:
        """Average time per generation."""
        return (
            sum(self.generation_times) / len(self.generation_times)
            if self.generation_times
            else 0.0
        )

    @property
    def avg_evaluation_time(self) -> float:
        """Average time per evaluation."""
        return (
            sum(self.evaluation_times) / len(self.evaluation_times)
            if self.evaluation_times
            else 0.0
        )

    @property
    def evaluations_per_second(self) -> float:
        """Evaluation throughput."""
        return self.total_evaluations / self.total_time if self.total_time > 0 else 0.0


class EvolutionBenchmark:
    """
    Comprehensive benchmarking suite for evolution system.

    Measures performance, memory usage, and quality metrics
    to identify optimization opportunities.
    """

    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize benchmark suite.

        Args:
            output_dir: Directory for benchmark results
        """
        self.output_dir = output_dir or Path("benchmarks")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_benchmark(
        self,
        config: EvolutionConfig,
        initial_ideas: List[GeneratedIdea],
        name: str = "benchmark",
        profile_memory: bool = True,
        profile_cpu: bool = True,
    ) -> Dict[str, Any]:
        """
        Run a single benchmark.

        Args:
            config: Evolution configuration
            initial_ideas: Initial population
            name: Benchmark name
            profile_memory: Whether to profile memory usage
            profile_cpu: Whether to profile CPU usage

        Returns:
            Benchmark results
        """
        metrics = BenchmarkMetrics(name=name)

        # Start profiling
        if profile_memory:
            tracemalloc.start()

        process = psutil.Process() if (profile_cpu and HAS_PSUTIL) else None

        # Run benchmark
        start_time = time.time()

        try:
            # Run evolution with instrumentation
            result = asyncio.run(
                self._run_instrumented_evolution(
                    config, initial_ideas, metrics, process
                )
            )

            metrics.total_time = time.time() - start_time
            metrics.final_best_fitness = (
                max(ind.overall_fitness for ind in result.final_population)
                if result.final_population
                else 0.0
            )

        finally:
            # Stop profiling
            if profile_memory:
                current, peak = tracemalloc.get_traced_memory()
                metrics.peak_memory_mb = peak / 1024 / 1024
                tracemalloc.stop()

        # Compile results
        results = self._compile_results(metrics)

        # Save results
        self._save_results(results)

        return results

    async def _run_instrumented_evolution(
        self,
        config: EvolutionConfig,
        initial_ideas: List[GeneratedIdea],
        metrics: BenchmarkMetrics,
        process: Optional[Any],  # psutil.Process when available
    ) -> EvolutionResult:
        """Run evolution with instrumentation."""
        # Create instrumented algorithm
        algorithm = InstrumentedGeneticAlgorithm(metrics, process)

        # Create request
        request = EvolutionRequest(
            initial_population=initial_ideas[: config.population_size],
            config=config,
            context="Benchmark run",
        )

        # Run evolution
        result = await algorithm.evolve(request)

        return result

    def run_benchmark_suite(
        self,
        configs: List[EvolutionConfig],
        initial_ideas: List[GeneratedIdea],
        name_prefix: str = "suite",
    ) -> List[Dict[str, Any]]:
        """
        Run multiple benchmarks.

        Args:
            configs: List of configurations to benchmark
            initial_ideas: Initial population
            name_prefix: Prefix for benchmark names

        Returns:
            List of benchmark results
        """
        results = []

        for i, config in enumerate(configs):
            name = f"{name_prefix}_{i}"
            print(f"Running benchmark: {name}")

            # Run garbage collection between benchmarks
            gc.collect()

            result = self.run_benchmark(config, initial_ideas, name)
            results.append(result)

            # Brief pause between benchmarks
            time.sleep(1)

        return results

    def _compile_results(self, metrics: BenchmarkMetrics) -> Dict[str, Any]:
        """Compile metrics into results dictionary."""
        return {
            "name": metrics.name,
            "timestamp": metrics.timestamp.isoformat(),
            "execution_time": metrics.total_time,
            "memory_usage": {
                "peak_mb": metrics.peak_memory_mb,
                "snapshots": metrics.memory_snapshots,
            },
            "cache_performance": {
                "hits": metrics.cache_hits,
                "misses": metrics.cache_misses,
                "hit_rate": metrics.cache_hit_rate,
            },
            "fitness_progression": metrics.fitness_progression,
            "diversity_progression": metrics.diversity_progression,
            "final_best_fitness": metrics.final_best_fitness,
            "llm_usage": {
                "calls": metrics.llm_calls,
                "tokens": metrics.llm_tokens,
                "cost": metrics.llm_cost,
            },
            "performance": {
                "total_evaluations": metrics.total_evaluations,
                "evaluations_per_second": metrics.evaluations_per_second,
                "avg_generation_time": metrics.avg_generation_time,
                "avg_evaluation_time": metrics.avg_evaluation_time,
            },
            "system": {
                "cpu_usage": metrics.cpu_usage,
                "avg_cpu_usage": (
                    sum(metrics.cpu_usage) / len(metrics.cpu_usage)
                    if metrics.cpu_usage
                    else 0
                ),
            },
        }

    def _save_results(self, results: Dict[str, Any]) -> None:
        """Save benchmark results to file."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"{results['name']}_{timestamp}.json"
        filepath = self.output_dir / filename

        with open(filepath, "w") as f:
            json.dump(results, f, indent=2)

        print(f"Benchmark results saved to: {filepath}")

    def compare_benchmarks(
        self,
        benchmark_files: List[str],
        output_file: str = "comparison.json",
    ) -> Dict[str, Any]:
        """
        Compare multiple benchmark results.

        Args:
            benchmark_files: List of benchmark result files
            output_file: Output file for comparison

        Returns:
            Comparison results
        """
        benchmarks = []

        # Load all benchmarks
        for file in benchmark_files:
            filepath = self.output_dir / file
            if filepath.exists():
                with open(filepath) as f:
                    benchmarks.append(json.load(f))

        if not benchmarks:
            return {"error": "No benchmarks found"}

        # Compare key metrics
        comparison = {
            "benchmarks": len(benchmarks),
            "execution_times": [b["execution_time"] for b in benchmarks],
            "peak_memory": [b["memory_usage"]["peak_mb"] for b in benchmarks],
            "final_fitness": [b["final_best_fitness"] for b in benchmarks],
            "cache_hit_rates": [b["cache_performance"]["hit_rate"] for b in benchmarks],
            "evaluations_per_second": [
                b["performance"]["evaluations_per_second"] for b in benchmarks
            ],
            "llm_costs": [b["llm_usage"]["cost"] for b in benchmarks],
        }

        # Calculate statistics
        import numpy as np

        for metric, values in comparison.items():
            if metric != "benchmarks" and values:
                comparison[f"{metric}_avg"] = np.mean(values)
                comparison[f"{metric}_std"] = np.std(values)
                comparison[f"{metric}_min"] = np.min(values)
                comparison[f"{metric}_max"] = np.max(values)

        # Save comparison
        output_path = self.output_dir / output_file
        with open(output_path, "w") as f:
            json.dump(comparison, f, indent=2)

        return comparison


class InstrumentedGeneticAlgorithm(GeneticAlgorithm):
    """
    Instrumented version of genetic algorithm for benchmarking.

    Collects detailed metrics during evolution.
    """

    def __init__(
        self,
        metrics: BenchmarkMetrics,
        process: Optional[Any] = None,  # psutil.Process when available
    ):
        """
        Initialize instrumented algorithm.

        Args:
            metrics: Metrics object to populate
            process: Process for CPU monitoring
        """
        super().__init__()
        self.metrics = metrics
        self.process = process
        self._generation_start_time: Optional[float] = None

    async def evolve(
        self,
        request: EvolutionRequest,
    ) -> EvolutionResult:
        """Evolve with instrumentation."""
        # Override to add instrumentation
        result = await super().evolve(request)

        # Capture final metrics
        if hasattr(self, "_cache") and self._cache:
            cache_stats = self._cache.get_stats()
            self.metrics.cache_hits = cache_stats.get("hits", 0)
            self.metrics.cache_misses = cache_stats.get("misses", 0)

        return result

    async def _evolve_generation(
        self,
        population: List[IndividualFitness],
        config: EvolutionConfig,
        context: Optional[str],
        generation: int,
    ) -> List[IndividualFitness]:
        """Instrumented generation evolution."""
        self._generation_start_time = time.time()

        # Capture memory snapshot
        if tracemalloc.is_tracing():
            current, peak = tracemalloc.get_traced_memory()
            self.metrics.memory_snapshots.append(
                {
                    "generation": generation,
                    "current_mb": current / 1024 / 1024,
                    "peak_mb": peak / 1024 / 1024,
                }
            )

        # Capture CPU usage
        if self.process:
            cpu_percent = self.process.cpu_percent(interval=0.1)
            self.metrics.cpu_usage.append(cpu_percent)

        # Run generation
        result = await super()._evolve_generation(
            population, config, context, generation
        )

        # Record generation time
        if self._generation_start_time is not None:
            generation_time = time.time() - self._generation_start_time
            self.metrics.generation_times.append(generation_time)

        # Record fitness progression
        if result:
            best_fitness = max(ind.overall_fitness for ind in result)
            self.metrics.fitness_progression.append(best_fitness)

        return result
