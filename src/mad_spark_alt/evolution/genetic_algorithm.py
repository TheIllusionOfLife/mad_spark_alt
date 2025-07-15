"""
Genetic Algorithm implementation for idea evolution.

This module orchestrates the genetic evolution process, coordinating
fitness evaluation, selection, crossover, and mutation operations.
"""

import asyncio
import logging
import random
import time
from typing import Any, Dict, List, Optional, Tuple, Union

from mad_spark_alt.core.interfaces import GeneratedIdea
from mad_spark_alt.evolution.cached_fitness import CachedFitnessEvaluator
from mad_spark_alt.evolution.checkpointing import (
    EvolutionCheckpoint,
    EvolutionCheckpointer,
)
from mad_spark_alt.evolution.fitness import FitnessEvaluator
from mad_spark_alt.evolution.interfaces import (
    EvolutionConfig,
    EvolutionRequest,
    EvolutionResult,
    IndividualFitness,
    PopulationSnapshot,
    SelectionStrategy,
)
from mad_spark_alt.evolution.operators import (
    CrossoverOperator,
    EliteSelection,
    MutationOperator,
    RandomSelection,
    RankSelection,
    RouletteWheelSelection,
    TournamentSelection,
)

logger = logging.getLogger(__name__)


class GeneticAlgorithm:
    """
    Main genetic algorithm orchestrator for evolving ideas.

    This class coordinates the evolutionary process through multiple generations,
    applying selection, crossover, and mutation to evolve a population of ideas
    toward higher fitness scores.
    """

    def __init__(
        self,
        fitness_evaluator: Optional[FitnessEvaluator] = None,
        crossover_operator: Optional[CrossoverOperator] = None,
        mutation_operator: Optional[MutationOperator] = None,
        use_cache: bool = True,
        cache_ttl: int = 3600,
        checkpoint_dir: Optional[str] = None,
        checkpoint_interval: int = 0,
    ):
        """
        Initialize genetic algorithm.

        Args:
            fitness_evaluator: Optional custom fitness evaluator
            crossover_operator: Optional custom crossover operator
            mutation_operator: Optional custom mutation operator
            use_cache: Whether to enable fitness caching (default: True)
            cache_ttl: Cache time-to-live in seconds (default: 3600)
            checkpoint_dir: Directory for saving checkpoints (None to disable)
            checkpoint_interval: Save checkpoint every N generations (0 to disable)
        """
        # Use cached evaluator by default for performance
        if use_cache and fitness_evaluator is None:
            self.fitness_evaluator: Union[FitnessEvaluator, CachedFitnessEvaluator] = (
                CachedFitnessEvaluator(cache_ttl=cache_ttl)
            )
        else:
            self.fitness_evaluator = fitness_evaluator or FitnessEvaluator()
        self.crossover_operator = crossover_operator or CrossoverOperator()
        self.mutation_operator = mutation_operator or MutationOperator()

        # Initialize selection operators
        self.selection_operators = {
            SelectionStrategy.TOURNAMENT: TournamentSelection(),
            SelectionStrategy.ELITE: EliteSelection(),
            SelectionStrategy.ROULETTE: RouletteWheelSelection(),
            SelectionStrategy.RANK: RankSelection(),
            SelectionStrategy.RANDOM: RandomSelection(),
        }

        # Initialize checkpointing
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_interval = checkpoint_interval
        self.checkpointer = (
            EvolutionCheckpointer(checkpoint_dir) if checkpoint_dir else None
        )

    async def _run_evolution_loop(
        self,
        initial_population: List[IndividualFitness],
        request: EvolutionRequest,
        start_generation: int = 0,
        initial_snapshots: Optional[List[PopulationSnapshot]] = None,
        initial_evaluations: int = 0,
    ) -> Tuple[List[IndividualFitness], List[PopulationSnapshot], int]:
        """
        Run the main evolution loop.

        Args:
            initial_population: Starting population
            request: Evolution request configuration
            start_generation: Generation to start from (for resume)
            initial_snapshots: Existing snapshots (for resume)
            initial_evaluations: Initial evaluation count (for resume)

        Returns:
            Tuple of (final_population, generation_snapshots, total_evaluations)
        """
        current_population = initial_population
        generation_snapshots = initial_snapshots.copy() if initial_snapshots else []
        total_evaluations = initial_evaluations

        # Evolution loop
        for generation in range(start_generation, request.config.generations):
            logger.info(
                f"{'Resuming' if start_generation > 0 else 'Starting'} generation {generation + 1}/{request.config.generations}"
            )

            # Evolve to next generation
            current_population = await self._evolve_generation(
                current_population, request.config, request.context, generation
            )

            # Create snapshot of current generation (after evolution)
            snapshot = PopulationSnapshot.from_population(
                generation + 1, current_population
            )

            # Calculate population diversity
            snapshot.diversity_score = (
                await self.fitness_evaluator.calculate_population_diversity(
                    current_population
                )
            )

            generation_snapshots.append(snapshot)

            # Count evaluations for new offspring
            total_evaluations += len(current_population)

            # Save checkpoint if enabled
            if (
                self.checkpointer
                and self.checkpoint_interval > 0
                and (generation + 1) % self.checkpoint_interval == 0
            ):
                checkpoint = EvolutionCheckpoint(
                    generation=generation + 1,
                    population=current_population,
                    config=request.config,
                    generation_snapshots=generation_snapshots.copy(),
                    context=request.context,
                    metadata={"total_evaluations": total_evaluations},
                )
                checkpoint_id = self.checkpointer.save_checkpoint(checkpoint)
                logger.info(
                    f"Saved checkpoint at generation {generation + 1}: {checkpoint_id}"
                )

            # Check termination criteria
            if self._should_terminate(snapshot, request):
                logger.info(f"Early termination at generation {generation + 1}")
                break

            # Adaptive mutation rate (if enabled)
            if request.config.adaptive_mutation:
                request.config.mutation_rate = self._adapt_mutation_rate(
                    snapshot, request.config.mutation_rate
                )

        return current_population, generation_snapshots, total_evaluations

    def _prepare_evolution_result(
        self,
        population: List[IndividualFitness],
        generation_snapshots: List[PopulationSnapshot],
        total_evaluations: int,
        start_time: float,
        evolution_metrics: dict,
    ) -> EvolutionResult:
        """
        Prepare the final evolution result.

        Args:
            population: Final population
            generation_snapshots: All generation snapshots
            total_evaluations: Total number of evaluations performed
            start_time: Evolution start time
            evolution_metrics: Calculated evolution metrics

        Returns:
            Complete evolution result
        """
        # Extract best ideas
        best_ideas = self._extract_best_ideas(population, n=10)

        # Get cache stats if available
        cache_stats: Dict[str, Any] = {}
        if hasattr(self.fitness_evaluator, "cache") and self.fitness_evaluator.cache:
            cache_stats = self.fitness_evaluator.cache.get_stats()

        return EvolutionResult(
            final_population=population,
            best_ideas=best_ideas,
            generation_snapshots=generation_snapshots,
            total_generations=(
                len(generation_snapshots) if generation_snapshots else 0
            ),  # Total number of generations including initial
            execution_time=time.time() - start_time,
            evolution_metrics={
                **evolution_metrics,
                "cache_stats": cache_stats,
                "total_evaluations": total_evaluations,
            },
            error_message=None,
        )

    async def evolve(self, request: EvolutionRequest) -> EvolutionResult:
        """
        Run the genetic evolution process.

        Args:
            request: Evolution request with initial population and config

        Returns:
            EvolutionResult with final population and metrics
        """
        start_time = time.time()

        # Set random seed for reproducibility if provided
        if request.config.random_seed is not None:
            import random

            random.seed(request.config.random_seed)

        # Validate request
        if not request.validate():
            return EvolutionResult(
                final_population=[],
                best_ideas=[],
                generation_snapshots=[],
                total_generations=0,
                execution_time=0.0,
                error_message="Invalid evolution request",
            )

        try:
            # Initialize population
            current_population = await self._initialize_population(
                request.initial_population, request.config
            )

            # Create initial snapshot (generation 0) for the starting population
            initial_snapshot = PopulationSnapshot.from_population(0, current_population)
            initial_snapshot.diversity_score = (
                await self.fitness_evaluator.calculate_population_diversity(
                    current_population
                )
            )

            # Run evolution loop using helper method
            total_evaluations = len(current_population)  # Count initial evaluation
            current_population, generation_snapshots, total_evaluations = (
                await self._run_evolution_loop(
                    current_population,
                    request,
                    start_generation=0,
                    initial_snapshots=[initial_snapshot],
                    initial_evaluations=total_evaluations,
                )
            )

            # Calculate evolution metrics
            evolution_metrics = self._calculate_evolution_metrics(
                generation_snapshots,
                request.initial_population,
                self._extract_best_ideas(
                    current_population, n=10
                ),  # Extract actual best ideas by fitness
                request.config,
            )

            # Log cache statistics if using cached evaluator
            if isinstance(self.fitness_evaluator, CachedFitnessEvaluator):
                cache_stats = self.fitness_evaluator.get_cache_stats()
                logger.info(
                    f"Evolution cache statistics - Hits: {cache_stats['hits']}, "
                    f"Misses: {cache_stats['misses']}, Hit rate: {cache_stats['hit_rate']:.2%}"
                )

            # Prepare and return result using helper method
            return self._prepare_evolution_result(
                current_population,
                generation_snapshots,
                total_evaluations,
                start_time,
                evolution_metrics,
            )

        except Exception as e:
            logger.error(f"Evolution failed: {e}")
            return EvolutionResult(
                final_population=[],
                best_ideas=[],
                generation_snapshots=[],
                total_generations=0,
                execution_time=time.time() - start_time,
                error_message=str(e),
            )

    async def resume_evolution(self, checkpoint_id: str) -> EvolutionResult:
        """
        Resume evolution from a saved checkpoint.

        Args:
            checkpoint_id: ID of the checkpoint to resume from

        Returns:
            EvolutionResult with the completed evolution
        """
        if not self.checkpointer:
            raise ValueError("Checkpointing not enabled for this GA instance")

        # Load checkpoint
        checkpoint = self.checkpointer.load_checkpoint(checkpoint_id)
        if not checkpoint:
            raise ValueError(f"Checkpoint {checkpoint_id} not found")

        logger.info(f"Resuming evolution from generation {checkpoint.generation}")

        # Create request from checkpoint
        request = EvolutionRequest(
            initial_population=[ind.idea for ind in checkpoint.population],
            config=checkpoint.config,
            context=checkpoint.context,
        )

        # Resume evolution
        start_time = time.time()

        try:
            # Start from checkpoint state
            current_population = checkpoint.population
            initial_snapshots = checkpoint.generation_snapshots.copy()
            total_evaluations = checkpoint.metadata.get("total_evaluations", 0)

            # Continue evolution using helper method
            current_population, generation_snapshots, total_evaluations = (
                await self._run_evolution_loop(
                    current_population,
                    request,
                    start_generation=checkpoint.generation + 1,
                    initial_snapshots=initial_snapshots,
                    initial_evaluations=total_evaluations,
                )
            )

            # Calculate evolution metrics
            # Extract initial ideas from the first generation snapshot for accurate metrics
            initial_ideas = (
                [ind.idea for ind in generation_snapshots[0].population]
                if generation_snapshots
                else []
            )
            evolution_metrics = self._calculate_evolution_metrics(
                generation_snapshots,
                initial_ideas,
                self._extract_best_ideas(
                    current_population, n=10
                ),  # Extract actual best ideas by fitness
                request.config,
            )

            # Log cache statistics if using cached evaluator
            if isinstance(self.fitness_evaluator, CachedFitnessEvaluator):
                cache_stats = self.fitness_evaluator.get_cache_stats()
                logger.info(
                    f"Evolution cache statistics - Hits: {cache_stats['hits']}, "
                    f"Misses: {cache_stats['misses']}, Hit rate: {cache_stats['hit_rate']:.2%}"
                )

            # Prepare and return result using helper method
            return self._prepare_evolution_result(
                current_population,
                generation_snapshots,
                total_evaluations,
                start_time,
                evolution_metrics,
            )

        except Exception as e:
            logger.error(f"Evolution resumption failed: {e}")
            return EvolutionResult(
                final_population=[],
                best_ideas=[],
                generation_snapshots=[],
                total_generations=0,
                execution_time=time.time() - start_time,
                error_message=str(e),
            )

    async def _initialize_population(
        self, initial_ideas: List[GeneratedIdea], config: EvolutionConfig
    ) -> List[IndividualFitness]:
        """Initialize population with fitness evaluation."""
        # Ensure we have enough individuals
        population = initial_ideas.copy()

        # If initial population is smaller than configured size, duplicate randomly
        while len(population) < config.population_size:
            population.append(random.choice(initial_ideas))

        # If larger, truncate
        population = population[: config.population_size]

        # Evaluate initial population
        logger.info(f"Evaluating initial population of {len(population)} individuals")
        return await self.fitness_evaluator.evaluate_population(population, config)

    async def _evolve_generation(
        self,
        population: List[IndividualFitness],
        config: EvolutionConfig,
        context: Optional[str],
        generation: int,
    ) -> List[IndividualFitness]:
        """Evolve one generation to the next."""
        new_population = []

        # Elite preservation
        if config.elite_size > 0:
            elite_selector = self.selection_operators[SelectionStrategy.ELITE]
            elite = await elite_selector.select(population, config.elite_size, config)
            new_population.extend(elite)
            logger.info(f"Preserved {len(elite)} elite individuals")

        # Generate offspring to fill population
        selector = self.selection_operators[config.selection_strategy]

        while len(new_population) < config.population_size:
            # Selection
            parents = await selector.select(population, 2, config)

            # Crossover
            if random.random() < config.crossover_rate:
                offspring1, offspring2 = await self.crossover_operator.crossover(
                    parents[0].idea, parents[1].idea, context
                )
            else:
                # If no crossover, just copy parents
                offspring1, offspring2 = parents[0].idea, parents[1].idea

            # Mutation
            offspring1 = await self.mutation_operator.mutate(
                offspring1, config.mutation_rate, context
            )
            offspring2 = await self.mutation_operator.mutate(
                offspring2, config.mutation_rate, context
            )

            # Add generation info to metadata
            offspring1.metadata["generation"] = generation + 1
            offspring2.metadata["generation"] = generation + 1

            # Add to new population (as unevaluated ideas)
            new_ideas = [offspring1]
            if len(new_population) + 1 < config.population_size:
                new_ideas.append(offspring2)

            # Evaluate new offspring
            evaluated_offspring = await self.fitness_evaluator.evaluate_population(
                new_ideas, config, context
            )
            new_population.extend(evaluated_offspring)

        # Ensure population size
        new_population = new_population[: config.population_size]

        # Apply diversity pressure if configured
        if config.diversity_pressure > 0:
            new_population = await self._apply_diversity_pressure(
                new_population, config.diversity_pressure
            )

        return new_population

    def _should_terminate(
        self, snapshot: PopulationSnapshot, request: EvolutionRequest
    ) -> bool:
        """Check if evolution should terminate early."""
        # Check if target metrics are met
        if request.target_metrics:
            if snapshot.best_fitness >= request.target_metrics.get("min_fitness", 1.0):
                return True

        # Check for convergence (low diversity)
        if snapshot.diversity_score < 0.1:
            logger.warning("Population has converged (low diversity)")
            return True

        return False

    def _adapt_mutation_rate(
        self, snapshot: PopulationSnapshot, current_rate: float
    ) -> float:
        """Adapt mutation rate based on population state."""
        # Increase mutation if diversity is low
        if snapshot.diversity_score < 0.3:
            return min(current_rate * 1.2, 0.5)
        # Decrease mutation if diversity is high
        elif snapshot.diversity_score > 0.7:
            return max(current_rate * 0.8, 0.01)
        return current_rate

    async def _apply_diversity_pressure(
        self, population: List[IndividualFitness], pressure: float
    ) -> List[IndividualFitness]:
        """Apply diversity pressure to maintain genetic diversity."""
        # Calculate diversity-adjusted fitness
        diversity_score = await self.fitness_evaluator.calculate_population_diversity(
            population
        )

        for individual in population:
            # Boost fitness of unique individuals
            diversity_bonus = pressure * (1.0 - diversity_score)
            individual.overall_fitness *= 1.0 + diversity_bonus

        return population

    def _extract_best_ideas(
        self, population: List[IndividualFitness], n: int
    ) -> List[GeneratedIdea]:
        """Extract top N ideas from population."""
        sorted_pop = sorted(population, key=lambda x: x.overall_fitness, reverse=True)
        return [ind.idea for ind in sorted_pop[:n]]

    def _calculate_evolution_metrics(
        self,
        snapshots: List[PopulationSnapshot],
        initial_ideas: List[GeneratedIdea],
        best_ideas: List[GeneratedIdea],
        config: EvolutionConfig,
    ) -> dict:
        """Calculate metrics about the evolution process."""
        if not snapshots:
            return {}

        initial_snapshot = snapshots[0]
        final_snapshot = snapshots[-1]

        # Calculate improvement
        fitness_improvement = (
            (final_snapshot.best_fitness - initial_snapshot.best_fitness)
            / initial_snapshot.best_fitness
            * 100
            if initial_snapshot.best_fitness > 0
            else 0
        )

        # Find generation with best fitness
        best_generation = max(
            range(len(snapshots)), key=lambda i: snapshots[i].best_fitness
        )

        return {
            "fitness_improvement_percent": fitness_improvement,
            "initial_best_fitness": initial_snapshot.best_fitness,
            "final_best_fitness": final_snapshot.best_fitness,
            "initial_avg_fitness": initial_snapshot.average_fitness,
            "final_avg_fitness": final_snapshot.average_fitness,
            "best_fitness_generation": best_generation,
            "diversity_trend": [s.diversity_score for s in snapshots],
            "fitness_trend": [s.best_fitness for s in snapshots],
            "total_ideas_evaluated": config.population_size
            + (len(snapshots) - 1) * (config.population_size - config.elite_size),
        }
