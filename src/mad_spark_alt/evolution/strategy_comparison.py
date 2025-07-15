"""
Strategy comparison tools for evolution system.

This module provides tools for comparing different evolution
strategies to identify optimal configurations.
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from matplotlib import pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from mad_spark_alt.core.interfaces import GeneratedIdea
from mad_spark_alt.evolution.genetic_algorithm import GeneticAlgorithm
from mad_spark_alt.evolution.interfaces import (
    EvolutionConfig,
    EvolutionRequest,
    EvolutionResult,
    SelectionStrategy,
)


@dataclass
class StrategyResult:
    """Result from testing a single strategy."""

    config: EvolutionConfig
    runs: List[EvolutionResult] = field(default_factory=list)

    @property
    def avg_fitness(self) -> float:
        """Average best fitness across runs."""
        if not self.runs:
            return 0.0
        return float(np.mean([max(ind.overall_fitness for ind in run.final_population) for run in self.runs if run.final_population]))

    @property
    def std_fitness(self) -> float:
        """Standard deviation of best fitness."""
        if not self.runs:
            return 0.0
        return float(np.std([max(ind.overall_fitness for ind in run.final_population) for run in self.runs if run.final_population]))

    @property
    def avg_convergence_generation(self) -> float:
        """Average generation where fitness plateaued."""
        if not self.runs:
            return 0.0
        convergence_gens = []
        for run in self.runs:
            # Use total_generations as a proxy for convergence
            # In future, we could add fitness_history to EvolutionResult
            convergence_gens.append(run.total_generations)

        return float(np.mean(convergence_gens))

    @property
    def avg_diversity_score(self) -> float:
        """Average final population diversity."""
        if not self.runs:
            return 0.0
        return float(np.mean([run.evolution_metrics.get('population_diversity', 0.5) for run in self.runs]))

    @property
    def convergence_rate(self) -> float:
        """Rate of convergence (lower is faster)."""
        total_gens = self.config.generations
        return self.avg_convergence_generation / total_gens if total_gens > 0 else 1.0

    @property
    def avg_execution_time(self) -> float:
        """Average execution time per run."""
        if not self.runs:
            return 0.0
        return float(np.mean([run.execution_time for run in self.runs]))

    @property
    def efficiency_score(self) -> float:
        """Combined efficiency metric (fitness per second)."""
        if self.avg_execution_time > 0:
            return self.avg_fitness / self.avg_execution_time
        return 0.0


class StrategyComparator:
    """
    Compare different evolution strategies.

    This comparator runs multiple strategies with identical
    initial conditions to fairly evaluate their performance.
    """

    def __init__(self) -> None:
        """Initialize strategy comparator."""
        self._results: Dict[str, StrategyResult] = {}

    async def compare_strategies(
        self,
        strategies: List[EvolutionConfig],
        initial_ideas: List[GeneratedIdea],
        runs_per_strategy: int = 5,
        parallel_runs: bool = True,
    ) -> List[Dict]:
        """
        Compare multiple evolution strategies.

        Args:
            strategies: List of configurations to test
            initial_ideas: Initial population (same for all)
            runs_per_strategy: Number of runs per strategy
            parallel_runs: Whether to run tests in parallel

        Returns:
            Comparison results for each strategy
        """
        results = []

        for i, config in enumerate(strategies):
            strategy_name = self._get_strategy_name(config, i)

            # Run multiple times
            strategy_result = StrategyResult(config=config)

            if parallel_runs:
                # Run all iterations in parallel
                tasks = [
                    self._run_single_evolution(config, initial_ideas, run_id=j)
                    for j in range(runs_per_strategy)
                ]
                run_results = await asyncio.gather(*tasks)
                strategy_result.runs = run_results
            else:
                # Run sequentially
                for j in range(runs_per_strategy):
                    result = await self._run_single_evolution(
                        config, initial_ideas, run_id=j
                    )
                    strategy_result.runs.append(result)

            self._results[strategy_name] = strategy_result

            # Compile results
            results.append(
                {
                    "strategy_name": strategy_name,
                    "config": config,
                    "avg_fitness": strategy_result.avg_fitness,
                    "std_fitness": strategy_result.std_fitness,
                    "convergence_rate": strategy_result.convergence_rate,
                    "diversity_score": strategy_result.avg_diversity_score,
                    "avg_execution_time": strategy_result.avg_execution_time,
                    "efficiency_score": strategy_result.efficiency_score,
                    "runs_completed": len(strategy_result.runs),
                }
            )

        # Sort by average fitness (descending)
        results.sort(key=lambda x: x["avg_fitness"], reverse=True)

        return results

    async def _run_single_evolution(
        self,
        config: EvolutionConfig,
        initial_ideas: List[GeneratedIdea],
        run_id: int = 0,
    ) -> EvolutionResult:
        """Run a single evolution with timing."""
        start_time = time.time()

        # Set random seed for reproducibility within strategy
        config.random_seed = run_id

        # Create algorithm and request
        algorithm = GeneticAlgorithm()
        request = EvolutionRequest(
            initial_population=initial_ideas[: config.population_size],
            config=config,
            context="Strategy comparison test",
        )

        # Run evolution
        result = await algorithm.evolve(request)

        # Add execution time
        result.execution_time = time.time() - start_time

        return result

    def _get_strategy_name(self, config: EvolutionConfig, index: int) -> str:
        """Generate descriptive name for strategy."""
        parts = [
            f"Strategy_{index}",
            f"{config.selection_strategy.value}",
            f"pop{config.population_size}",
            f"gen{config.generations}",
        ]

        if config.adaptive_mutation:
            parts.append("adaptive")

        if config.enable_llm_operators:
            parts.append("llm")

        return "_".join(parts)

    def plot_comparison(
        self,
        save_path: Optional[str] = None,
        metrics: Optional[List[str]] = None,
    ) -> None:
        """
        Plot strategy comparison results.

        Args:
            save_path: Path to save plot
            metrics: Metrics to plot (default: all)
        """
        if not HAS_MATPLOTLIB:
            print("Matplotlib not available. Install with: pip install matplotlib")
            return

        if not self._results:
            print("No results to plot")
            return

        if metrics is None:
            metrics = [
                "avg_fitness",
                "convergence_rate",
                "diversity_score",
                "efficiency_score",
            ]

        n_strategies = len(self._results)
        n_metrics = len(metrics)

        fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 6))
        if n_metrics == 1:
            axes = [axes]

        strategy_names = list(self._results.keys())

        for i, metric in enumerate(metrics):
            ax = axes[i]

            # Get metric values
            values = []
            errors = []

            for name in strategy_names:
                result = self._results[name]

                if metric == "avg_fitness":
                    values.append(result.avg_fitness)
                    errors.append(result.std_fitness)
                elif metric == "convergence_rate":
                    values.append(result.convergence_rate)
                    errors.append(0)
                elif metric == "diversity_score":
                    values.append(result.avg_diversity_score)
                    errors.append(0)
                elif metric == "efficiency_score":
                    values.append(result.efficiency_score)
                    errors.append(0)

            # Create bar plot
            x_pos = np.arange(n_strategies)
            ax.bar(x_pos, values, yerr=errors if any(errors) else None, capsize=5)
            ax.set_xlabel("Strategy")
            ax.set_ylabel(metric.replace("_", " ").title())
            ax.set_title(f'{metric.replace("_", " ").title()} Comparison')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(strategy_names, rotation=45, ha="right")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

    def plot_fitness_progression(
        self,
        save_path: Optional[str] = None,
        show_all_runs: bool = False,
    ) -> None:
        """
        Plot fitness progression over generations.

        Args:
            save_path: Path to save plot
            show_all_runs: Show individual runs or just averages
        """
        if not HAS_MATPLOTLIB:
            print("Matplotlib not available. Install with: pip install matplotlib")
            return

        if not self._results:
            print("No results to plot")
            return

        plt.figure(figsize=(10, 6))

        for strategy_name, result in self._results.items():
            if not result.runs:
                continue

            # Get fitness histories from generation snapshots
            all_histories = []
            for run in result.runs:
                if run.generation_snapshots:
                    history = [snap.best_fitness for snap in run.generation_snapshots]
                    all_histories.append(history)
                else:
                    all_histories.append([run.final_best_fitness] if hasattr(run, 'final_best_fitness') else [0.0])
            max_len = max(len(h) for h in all_histories)

            # Pad histories to same length
            padded_histories = []
            for history in all_histories:
                padded = history + [history[-1]] * (max_len - len(history))
                padded_histories.append(padded)

            # Convert to numpy array
            histories_array = np.array(padded_histories)

            # Plot
            generations = np.arange(max_len)

            if show_all_runs:
                # Plot individual runs
                for i, history in enumerate(histories_array):
                    plt.plot(
                        generations,
                        history,
                        alpha=0.3,
                        label=f"{strategy_name}_run{i}" if i == 0 else "",
                    )
            else:
                # Plot average with confidence interval
                mean_fitness = np.mean(histories_array, axis=0)
                std_fitness = np.std(histories_array, axis=0)

                plt.plot(generations, mean_fitness, label=strategy_name, linewidth=2)
                plt.fill_between(
                    generations,
                    mean_fitness - std_fitness,
                    mean_fitness + std_fitness,
                    alpha=0.2,
                )

        plt.xlabel("Generation")
        plt.ylabel("Best Fitness")
        plt.title("Fitness Progression Comparison")
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

    def get_best_strategy(self) -> Tuple[str, EvolutionConfig]:
        """
        Get the best performing strategy.

        Returns:
            Tuple of (strategy_name, config)
        """
        if not self._results:
            raise ValueError("No results available")

        best_name = max(
            self._results.keys(),
            key=lambda name: self._results[name].avg_fitness,
        )

        return best_name, self._results[best_name].config

    def export_results(self, path: str) -> None:
        """
        Export comparison results to file.

        Args:
            path: Path to export file (JSON or CSV)
        """
        import json
        import csv

        data = []

        for name, result in self._results.items():
            data.append(
                {
                    "strategy": name,
                    "avg_fitness": result.avg_fitness,
                    "std_fitness": result.std_fitness,
                    "convergence_rate": result.convergence_rate,
                    "avg_convergence_generation": result.avg_convergence_generation,
                    "diversity_score": result.avg_diversity_score,
                    "avg_execution_time": result.avg_execution_time,
                    "efficiency_score": result.efficiency_score,
                    "population_size": result.config.population_size,
                    "generations": result.config.generations,
                    "selection_strategy": result.config.selection_strategy.value,
                    "mutation_rate": result.config.mutation_rate,
                    "crossover_rate": result.config.crossover_rate,
                    "elite_size": result.config.elite_size,
                    "adaptive_mutation": result.config.adaptive_mutation,
                    "enable_llm_operators": result.config.enable_llm_operators,
                }
            )

        if path.endswith(".json"):
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
        elif path.endswith(".csv"):
            if data:
                with open(path, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=data[0].keys())
                    writer.writeheader()
                    writer.writerows(data)
        else:
            raise ValueError("Unsupported format. Use .json or .csv")
