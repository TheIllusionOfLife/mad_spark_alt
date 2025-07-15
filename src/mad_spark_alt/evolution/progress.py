"""
Progress tracking system for evolution.

This module provides real-time progress callbacks and monitoring
for the genetic evolution process.
"""

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

ProgressCallback = Callable[[Dict[str, Any]], None]


@dataclass
class GenerationStats:
    """Statistics for a single generation."""

    generation: int
    evaluations_completed: int = 0
    evaluations_failed: int = 0
    best_fitness: float = 0.0
    avg_fitness: float = 0.0
    diversity_score: float = 0.0
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None

    @property
    def duration(self) -> float:
        """Get generation duration in seconds."""
        if self.end_time:
            return self.end_time - self.start_time
        return time.time() - self.start_time


class EvolutionProgressTracker:
    """
    Tracks and reports evolution progress in real-time.

    This tracker manages callbacks and provides detailed progress
    information during the evolution process.
    """

    def __init__(self) -> None:
        """Initialize progress tracker."""
        self._callbacks: List[ProgressCallback] = []
        self._start_time: Optional[float] = None
        self._total_generations: int = 0
        self._population_size: int = 0
        self._current_generation: Optional[GenerationStats] = None
        self._completed_generations: List[GenerationStats] = []
        self._total_evaluations: int = 0
        self._total_llm_calls: int = 0
        self._total_cost: float = 0.0

    def add_callback(self, callback: ProgressCallback) -> None:
        """
        Add a progress callback.

        Args:
            callback: Function that receives progress updates
        """
        self._callbacks.append(callback)

    def remove_callback(self, callback: ProgressCallback) -> None:
        """
        Remove a progress callback.

        Args:
            callback: Callback to remove
        """
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    def start_evolution(self, total_generations: int, population_size: int) -> None:
        """
        Start tracking a new evolution.

        Args:
            total_generations: Total number of generations
            population_size: Size of population
        """
        self._start_time = time.time()
        self._total_generations = total_generations
        self._population_size = population_size
        self._completed_generations.clear()
        self._total_evaluations = 0
        self._total_llm_calls = 0
        self._total_cost = 0.0

        self._notify_callbacks(
            {
                "event": "evolution_started",
                "total_generations": total_generations,
                "population_size": population_size,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

    def start_generation(self, generation: int) -> None:
        """
        Start tracking a new generation.

        Args:
            generation: Generation number (1-based)
        """
        self._current_generation = GenerationStats(generation=generation)

        self._notify_callbacks(
            {
                "event": "generation_started",
                "current_generation": generation,
                "total_generations": self._total_generations,
                "elapsed_time": self._get_elapsed_time(),
                "estimated_time_remaining": self._estimate_time_remaining(generation),
            }
        )

    def report_evaluation(
        self,
        individual_index: int,
        success: bool,
        fitness: Optional[float] = None,
        llm_calls: int = 0,
        cost: float = 0.0,
    ) -> None:
        """
        Report an individual evaluation result.

        Args:
            individual_index: Index of evaluated individual
            success: Whether evaluation succeeded
            fitness: Fitness score if successful
            llm_calls: Number of LLM calls made
            cost: Cost of evaluation
        """
        if not self._current_generation:
            return

        if success:
            self._current_generation.evaluations_completed += 1
        else:
            self._current_generation.evaluations_failed += 1

        self._total_evaluations += 1
        self._total_llm_calls += llm_calls
        self._total_cost += cost

        evaluations_in_generation = (
            self._current_generation.evaluations_completed
            + self._current_generation.evaluations_failed
        )

        self._notify_callbacks(
            {
                "event": "evaluation_completed",
                "current_generation": self._current_generation.generation,
                "individual_index": individual_index,
                "success": success,
                "fitness": fitness,
                "evaluations_completed": evaluations_in_generation,
                "evaluations_total": self._population_size,
                "progress_percent": (evaluations_in_generation / self._population_size)
                * 100,
                "total_llm_calls": self._total_llm_calls,
                "total_cost": self._total_cost,
            }
        )

    def complete_generation(
        self,
        generation: int,
        best_fitness: float,
        avg_fitness: float,
        diversity_score: float = 0.0,
    ) -> None:
        """
        Complete tracking for a generation.

        Args:
            generation: Generation number
            best_fitness: Best fitness in generation
            avg_fitness: Average fitness in generation
            diversity_score: Population diversity score
        """
        if (
            not self._current_generation
            or self._current_generation.generation != generation
        ):
            return

        self._current_generation.end_time = time.time()
        self._current_generation.best_fitness = best_fitness
        self._current_generation.avg_fitness = avg_fitness
        self._current_generation.diversity_score = diversity_score

        self._completed_generations.append(self._current_generation)

        self._notify_callbacks(
            {
                "event": "generation_completed",
                "current_generation": generation,
                "total_generations": self._total_generations,
                "best_fitness": best_fitness,
                "avg_fitness": avg_fitness,
                "diversity_score": diversity_score,
                "generation_duration": self._current_generation.duration,
                "evaluations_completed": self._current_generation.evaluations_completed,
                "evaluations_failed": self._current_generation.evaluations_failed,
                "elapsed_time": self._get_elapsed_time(),
                "estimated_time_remaining": self._estimate_time_remaining(
                    generation + 1
                ),
                "total_evaluations": self._total_evaluations,
                "total_llm_calls": self._total_llm_calls,
                "total_cost": self._total_cost,
            }
        )

        self._current_generation = None

    def complete_evolution(self, final_best_fitness: float) -> None:
        """
        Complete tracking for the entire evolution.

        Args:
            final_best_fitness: Best fitness achieved
        """
        total_duration = self._get_elapsed_time()

        self._notify_callbacks(
            {
                "event": "evolution_completed",
                "total_generations": len(self._completed_generations),
                "final_best_fitness": final_best_fitness,
                "total_duration": total_duration,
                "total_evaluations": self._total_evaluations,
                "total_llm_calls": self._total_llm_calls,
                "total_cost": self._total_cost,
                "avg_generation_time": (
                    total_duration / len(self._completed_generations)
                    if self._completed_generations
                    else 0
                ),
            }
        )

    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of evolution progress.

        Returns:
            Dictionary with progress summary
        """
        if not self._completed_generations:
            return {
                "status": "not_started",
                "generations_completed": 0,
                "total_generations": self._total_generations,
            }

        fitness_progression = [g.best_fitness for g in self._completed_generations]

        return {
            "status": "in_progress" if self._current_generation else "completed",
            "generations_completed": len(self._completed_generations),
            "total_generations": self._total_generations,
            "current_generation": (
                self._current_generation.generation
                if self._current_generation
                else None
            ),
            "best_fitness": max(fitness_progression) if fitness_progression else 0.0,
            "fitness_progression": fitness_progression,
            "total_evaluations": self._total_evaluations,
            "total_llm_calls": self._total_llm_calls,
            "total_cost": self._total_cost,
            "elapsed_time": self._get_elapsed_time(),
            "estimated_time_remaining": self._estimate_time_remaining(
                len(self._completed_generations) + 1
            ),
        }

    def _notify_callbacks(self, progress_data: Dict) -> None:
        """
        Notify all registered callbacks with progress data.

        Args:
            progress_data: Progress information to send
        """
        for callback in self._callbacks:
            try:
                callback(progress_data)
            except Exception as e:
                # Log error but don't crash on callback failure
                import logging

                logging.error(f"Progress callback error: {e}")

    def _get_elapsed_time(self) -> float:
        """Get elapsed time since evolution started."""
        if self._start_time:
            return time.time() - self._start_time
        return 0.0

    def _estimate_time_remaining(self, next_generation: int) -> float:
        """
        Estimate time remaining based on average generation time.

        Args:
            next_generation: Next generation number

        Returns:
            Estimated seconds remaining
        """
        if not self._completed_generations or next_generation > self._total_generations:
            return 0.0

        avg_generation_time = sum(
            g.duration for g in self._completed_generations
        ) / len(self._completed_generations)

        remaining_generations = self._total_generations - next_generation + 1
        return avg_generation_time * remaining_generations
