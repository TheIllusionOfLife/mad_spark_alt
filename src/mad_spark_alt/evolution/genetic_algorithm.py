"""
Genetic Algorithm implementation for idea evolution.

This module orchestrates the genetic evolution process, coordinating
fitness evaluation, selection, crossover, and mutation operations.
"""

import asyncio
import logging
import random
import time
from dataclasses import replace
from typing import Any, Dict, List, Optional, Tuple, Union

from mad_spark_alt.core.interfaces import GeneratedIdea
from mad_spark_alt.core.system_constants import CONSTANTS
from mad_spark_alt.evolution.cached_fitness import CachedFitnessEvaluator
from mad_spark_alt.evolution.checkpointing import (
    EvolutionCheckpoint,
    EvolutionCheckpointer,
)
from mad_spark_alt.evolution.constants import (
    HIGH_DIVERSITY_THRESHOLD,
    LOW_DIVERSITY_THRESHOLD,
    MAX_MUTATION_RATE,
    MIN_MUTATION_RATE,
    MODERATE_DIVERSITY_THRESHOLD,
    MUTATION_RATE_DECREASE_FACTOR,
    MUTATION_RATE_INCREASE_FACTOR,
    ZERO_SCORE,
)
from mad_spark_alt.evolution.fitness import FitnessEvaluator
from mad_spark_alt.evolution.interfaces import (
    EvaluationContext,
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
from mad_spark_alt.evolution.semantic_operators import (
    BatchSemanticCrossoverOperator,
    BatchSemanticMutationOperator,
    SemanticCrossoverOperator,
)
# SmartOperatorSelector removed - using simplified semantic operator selection

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
        llm_provider: Optional[Any] = None,
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
            llm_provider: Optional LLM provider for semantic operators
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

        # Initialize semantic operators if LLM provider is available
        self.llm_provider = llm_provider
        self.semantic_mutation_operator: Optional[BatchSemanticMutationOperator] = None
        self.semantic_crossover_operator: Optional[BatchSemanticCrossoverOperator] = None
        # SmartOperatorSelector removed - using simplified semantic operator selection
        
        if llm_provider is not None:
            self.semantic_mutation_operator = BatchSemanticMutationOperator(
                llm_provider, cache_ttl=cache_ttl
            )
            self.semantic_crossover_operator = BatchSemanticCrossoverOperator(
                llm_provider, cache_ttl=cache_ttl
            )
            logger.info("Semantic operators initialized with LLM provider")

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
        
        # Track semantic operator usage
        self.semantic_operator_metrics = {
            'semantic_mutations': 0,
            'semantic_crossovers': 0,
            'traditional_mutations': 0,
            'traditional_crossovers': 0,
            'semantic_llm_calls': 0
        }

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
                current_population, request.config, request.context, generation, request.evaluation_context
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

            # Count evaluations for new offspring only (excluding preserved elite)
            # Elite individuals are carried over without re-evaluation
            new_evaluations = len(current_population) - request.config.elite_size
            total_evaluations += new_evaluations

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

    async def _apply_mutation_with_operator_selection(
        self,
        offspring: GeneratedIdea,
        config: EvolutionConfig,
        context: Optional[str] = None,
        evaluation_context: Optional[EvaluationContext] = None,
    ) -> Tuple[GeneratedIdea, Dict[str, int]]:
        """
        Apply mutation to an offspring with semantic operator selection.

        Args:
            offspring: The offspring idea to mutate
            config: Evolution configuration
            context: Optional context for mutation
            evaluation_context: Optional evaluation context for targeted evolution

        Returns:
            Tuple of (mutated offspring, mutation statistics)
        """
        mutation_stats = {
            'semantic_mutations': 0,
            'traditional_mutations': 0,
            'semantic_llm_calls': 0,
        }

        # Determine whether to use semantic mutation (simplified logic)
        use_semantic = (
            self.semantic_mutation_operator is not None and
            config.use_semantic_operators
        )

        # Apply appropriate mutation type
        if use_semantic:
            assert self.semantic_mutation_operator is not None
            # Use evaluation context if available, otherwise fall back to string context
            operator_context = evaluation_context if evaluation_context else context
            mutated_offspring = await self.semantic_mutation_operator.mutate(
                offspring, config.mutation_rate, operator_context
            )
            # Only count if mutation actually occurred (content changed)
            if mutated_offspring.content != offspring.content:
                mutation_stats['semantic_mutations'] = 1
                self.semantic_operator_metrics['semantic_mutations'] += 1
                
            # Track LLM calls based on metadata (cache hits won't have llm_cost)
            if mutated_offspring.metadata.get('llm_cost', 0) > 0:
                mutation_stats['semantic_llm_calls'] = 1
                self.semantic_operator_metrics['semantic_llm_calls'] += 1
        else:
            mutated_offspring = await self.mutation_operator.mutate(
                offspring, config.mutation_rate, context
            )
            # Only count if mutation actually occurred (content changed)
            if mutated_offspring.content != offspring.content:
                mutation_stats['traditional_mutations'] = 1
                self.semantic_operator_metrics['traditional_mutations'] += 1

        return mutated_offspring, mutation_stats

    async def evolve(self, request: EvolutionRequest) -> EvolutionResult:
        """
        Run the genetic evolution process.

        Args:
            request: Evolution request with initial population and config

        Returns:
            EvolutionResult with final population and metrics
        """
        start_time = time.time()
        
        # Reset semantic operator metrics for this evolution
        self.semantic_operator_metrics = {
            'semantic_mutations': 0,
            'semantic_crossovers': 0,
            'traditional_mutations': 0,
            'traditional_crossovers': 0,
            'semantic_llm_calls': 0
        }

        # Set random seed for reproducibility if provided
        if request.config.random_seed is not None:
            import random

            random.seed(request.config.random_seed)
            
        # Configure diversity calculator based on config
        if hasattr(self.fitness_evaluator, 'configure_diversity_method'):
            self.fitness_evaluator.configure_diversity_method(
                request.config.diversity_method,
                self.llm_provider
            )

        # Validate request
        if not request.validate():
            return EvolutionResult(
                final_population=[],
                best_ideas=[],
                generation_snapshots=[],
                total_generations=0,
                execution_time=ZERO_SCORE,
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

        logger.info(
            f"Resuming evolution from generation {checkpoint.generation} (continuing from checkpoint saved after generation {checkpoint.generation - 1})"
        )

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
            # Note: checkpoint.generation is the generation number after the last evolution step
            # The population in the checkpoint has already evolved through generation (checkpoint.generation - 1)
            # So we resume from checkpoint.generation to continue the evolution
            current_population, generation_snapshots, total_evaluations = (
                await self._run_evolution_loop(
                    current_population,
                    request,
                    start_generation=checkpoint.generation,
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
        evaluation_context: Optional["EvaluationContext"] = None,
    ) -> List[IndividualFitness]:
        """Evolve one generation to the next."""
        new_population = []
        
        # Track operator usage
        semantic_mutations = 0
        semantic_crossovers = 0
        traditional_mutations = 0
        traditional_crossovers = 0
        semantic_llm_calls = 0

        # SmartOperatorSelector removed - using simplified semantic operator selection

        # Elite preservation
        if config.elite_size > 0:
            elite_selector = self.selection_operators[SelectionStrategy.ELITE]
            elite = await elite_selector.select(population, config.elite_size, config)
            new_population.extend(elite)
            logger.info(f"Preserved {len(elite)} elite individuals")

        # Generate offspring to fill population
        selector = self.selection_operators[config.selection_strategy]

        # Calculate population diversity for smart operator selection
        population_diversity = await self.fitness_evaluator.calculate_population_diversity(population)

        # Calculate how many offspring we need
        num_offspring_needed = config.population_size - len(new_population)
        
        # Use parallel processing if semantic operators with batch capability are available
        if (config.use_semantic_operators and 
            self.semantic_mutation_operator is not None and
            hasattr(self.semantic_mutation_operator, 'mutate_batch') and
            num_offspring_needed > CONSTANTS.EVOLUTION.MIN_BATCH_SIZE_FOR_PARALLEL):  # Only use parallel for larger populations
            
            logger.info(f"Using parallel processing for {num_offspring_needed} offspring")
            try:
                # Generate all offspring in parallel with batch operations
                parallel_offspring = await self._generate_offspring_parallel(
                    population, config, context, generation, selector, 
                    num_offspring_needed, evaluation_context
                )
                new_population.extend(parallel_offspring)
                
            except Exception as e:
                logger.warning(f"Parallel processing failed, falling back to sequential: {e}")
                # Fall back to sequential processing
                sequential_offspring = await self._generate_offspring_sequential(
                    population, config, context, generation, selector,
                    new_population, num_offspring_needed, evaluation_context
                )
                new_population = sequential_offspring
        else:
            # Use sequential processing for small populations or when batch not available
            logger.info(f"Using sequential processing for {num_offspring_needed} offspring")
            sequential_offspring = await self._generate_offspring_sequential(
                population, config, context, generation, selector,
                new_population, num_offspring_needed, evaluation_context
            )
            new_population = sequential_offspring

        # Apply diversity pressure if configured
        if config.diversity_pressure > 0:
            new_population = await self._apply_diversity_pressure(
                new_population, config.diversity_pressure
            )

        return new_population

    async def _generate_offspring_sequential(
        self,
        population: List[IndividualFitness],
        config: EvolutionConfig,
        context: Optional[str],
        generation: int,
        selector: Any,
        new_population: List[IndividualFitness],
        num_offspring_needed: int,
        evaluation_context: Optional["EvaluationContext"] = None,
    ) -> List[IndividualFitness]:
        """Generate offspring using the original sequential approach."""
        unevaluated_offspring: List[GeneratedIdea] = []
        
        # Optimization: Collect all offspring first, then evaluate in batch
        while len(new_population) + len(unevaluated_offspring) < config.population_size:
            # Selection
            parents = await selector.select(population, 2, config)

            # Crossover (with smart semantic selection)
            if random.random() < config.crossover_rate:
                # Use simplified logic to decide between semantic and traditional crossover
                use_semantic = (
                    self.semantic_crossover_operator is not None and
                    config.use_semantic_operators
                )
                
                if use_semantic:
                    assert self.semantic_crossover_operator is not None
                    # Use evaluation context if available, otherwise fall back to string context
                    operator_context = evaluation_context if evaluation_context else context
                    offspring1, offspring2 = await self.semantic_crossover_operator.crossover(
                        parents[0].idea, parents[1].idea, operator_context
                    )
                    self.semantic_operator_metrics['semantic_crossovers'] += 1
                    self.semantic_operator_metrics['semantic_llm_calls'] += 1
                else:
                    offspring1, offspring2 = await self.crossover_operator.crossover(
                        parents[0].idea, parents[1].idea, context
                    )
                    self.semantic_operator_metrics['traditional_crossovers'] += 1
            else:
                # If no crossover, just copy parents
                offspring1, offspring2 = parents[0].idea, parents[1].idea

            # Apply mutation to both offspring
            offspring1, mutation_stats1 = await self._apply_mutation_with_operator_selection(
                offspring1, config, context, evaluation_context
            )
            self.semantic_operator_metrics['semantic_mutations'] += mutation_stats1['semantic_mutations']
            self.semantic_operator_metrics['traditional_mutations'] += mutation_stats1['traditional_mutations']
            self.semantic_operator_metrics['semantic_llm_calls'] += mutation_stats1['semantic_llm_calls']

            offspring2, mutation_stats2 = await self._apply_mutation_with_operator_selection(
                offspring2, config, context, evaluation_context
            )
            self.semantic_operator_metrics['semantic_mutations'] += mutation_stats2['semantic_mutations']
            self.semantic_operator_metrics['traditional_mutations'] += mutation_stats2['traditional_mutations']
            self.semantic_operator_metrics['semantic_llm_calls'] += mutation_stats2['semantic_llm_calls']

            # Add generation info to metadata
            offspring1.metadata["generation"] = generation + 1
            offspring2.metadata["generation"] = generation + 1

            # Add to buffer (as unevaluated ideas)
            unevaluated_offspring.append(offspring1)

            # Check if we still need more (in case we only needed 1 more)
            if len(new_population) + len(unevaluated_offspring) < config.population_size:
                unevaluated_offspring.append(offspring2)

        # Batch evaluation of all collected offspring
        if unevaluated_offspring:
            logger.info(f"Batch evaluating {len(unevaluated_offspring)} offspring")
            evaluated_offspring = await self.fitness_evaluator.evaluate_population(
                unevaluated_offspring, config, context
            )
            new_population.extend(evaluated_offspring)

        # Ensure population size
        new_population = new_population[: config.population_size]

        return new_population

    async def _generate_offspring_parallel(
        self,
        population: List[IndividualFitness],
        config: EvolutionConfig,
        context: Optional[str],
        generation: int,
        selector: Any,
        num_offspring_needed: int,
        evaluation_context: Optional["EvaluationContext"] = None,
    ) -> List[IndividualFitness]:
        """Generate offspring using parallel processing for maximum efficiency.
        
        This method eliminates the sequential bottleneck by:
        1. Generating all parent pairs upfront
        2. Batch processing ALL crossovers in a single LLM call (if using semantic)
        3. Batch processing ALL mutations in a single LLM call
        4. Batch processing ALL evaluations
        """
        try:
            # Step 1: Generate all parent pairs upfront
            parent_pairs_for_crossover = []
            parent_pairs_no_crossover = []
            semantic_pairs = []
            traditional_pairs = []
            
            pairs_needed = (num_offspring_needed + 1) // 2  # Round up for odd numbers
            
            for _ in range(pairs_needed):
                # Select parents
                parents = await selector.select(population, 2, config)
                
                # Decide on crossover
                if random.random() < config.crossover_rate:
                    parent_pairs_for_crossover.append((parents[0], parents[1]))
                    
                    # Decide semantic vs traditional
                    use_semantic = (
                        self.semantic_crossover_operator is not None and
                        config.use_semantic_operators
                    )
                    
                    if use_semantic:
                        semantic_pairs.append((parents[0].idea, parents[1].idea))
                    else:
                        traditional_pairs.append((parents[0].idea, parents[1].idea))
                else:
                    parent_pairs_no_crossover.append((parents[0], parents[1]))
            
            # Step 2: Process all crossovers
            offspring_for_mutation: List[GeneratedIdea] = []
            
            # Process semantic crossovers (BATCH if possible)
            if semantic_pairs:
                assert self.semantic_crossover_operator is not None
                operator_context = evaluation_context if evaluation_context else context
                
                # Check if batch crossover is available
                if hasattr(self.semantic_crossover_operator, 'crossover_batch'):
                    logger.info(f"Batch processing {len(semantic_pairs)} semantic crossovers in single LLM call")
                    # CRITICAL OPTIMIZATION: Single LLM call for ALL crossovers!
                    crossover_results = await self.semantic_crossover_operator.crossover_batch(
                        semantic_pairs, operator_context
                    )
                    
                    # Flatten results
                    for offspring1, offspring2 in crossover_results:
                        offspring_for_mutation.append(offspring1)
                        if len(offspring_for_mutation) < num_offspring_needed:
                            offspring_for_mutation.append(offspring2)
                    
                    # Track metrics
                    self.semantic_operator_metrics['semantic_crossovers'] += len(semantic_pairs)
                    self.semantic_operator_metrics['semantic_llm_calls'] += 1  # Single batch call!
                else:
                    # Fall back to sequential if batch not available
                    for parent1, parent2 in semantic_pairs:
                        offspring1, offspring2 = await self.semantic_crossover_operator.crossover(
                            parent1, parent2, operator_context
                        )
                        offspring_for_mutation.append(offspring1)
                        if len(offspring_for_mutation) < num_offspring_needed:
                            offspring_for_mutation.append(offspring2)
                        
                        self.semantic_operator_metrics['semantic_crossovers'] += 1
                        self.semantic_operator_metrics['semantic_llm_calls'] += 1
            
            # Process traditional crossovers (still sequential)
            if traditional_pairs:
                for parent1, parent2 in traditional_pairs:
                    offspring1, offspring2 = await self.crossover_operator.crossover(
                        parent1, parent2, context
                    )
                    offspring_for_mutation.append(offspring1)
                    if len(offspring_for_mutation) < num_offspring_needed:
                        offspring_for_mutation.append(offspring2)
                    
                    self.semantic_operator_metrics['traditional_crossovers'] += 1
            
            # Add no-crossover pairs
            for parent1, parent2 in parent_pairs_no_crossover:
                offspring_for_mutation.append(parent1.idea)
                if len(offspring_for_mutation) < num_offspring_needed:
                    offspring_for_mutation.append(parent2.idea)
            
            # Ensure exact number of offspring
            offspring_for_mutation = offspring_for_mutation[:num_offspring_needed]
            
            # Count the crossovers for logging
            semantic_crossovers = len(semantic_pairs)
            traditional_crossovers = len(traditional_pairs)
            
            # Step 3: CRITICAL OPTIMIZATION - Batch process ALL mutations in a single LLM call
            if (config.use_semantic_operators and 
                self.semantic_mutation_operator is not None and 
                hasattr(self.semantic_mutation_operator, 'mutate_batch')):
                
                # Determine which offspring to mutate
                offspring_to_mutate = [
                    idea for idea in offspring_for_mutation
                    if random.random() < config.mutation_rate
                ]
                
                if offspring_to_mutate:
                    # This is the KEY OPTIMIZATION: Single batch LLM call for ALL mutations!
                    operator_context = evaluation_context if evaluation_context else context
                    logger.debug(f"Batch mutating {len(offspring_to_mutate)} offspring in single LLM call")
                    
                    mutated_offspring = await self.semantic_mutation_operator.mutate_batch(
                        offspring_to_mutate, operator_context
                    )
                    
                    # Create index-based mapping to preserve order
                    mutation_indices = []
                    for i, idea in enumerate(offspring_for_mutation):
                        if idea in offspring_to_mutate:
                            mutation_indices.append(i)
                    
                    # Replace mutated ideas in the offspring list
                    final_offspring = offspring_for_mutation.copy()
                    for idx, mutated_idea in zip(mutation_indices, mutated_offspring):
                        final_offspring[idx] = mutated_idea
                    
                    # Track metrics - key insight: 1 LLM call for multiple mutations!
                    self.semantic_operator_metrics['semantic_mutations'] += len(mutated_offspring)
                    self.semantic_operator_metrics['semantic_llm_calls'] += 1  # Single batch call
                else:
                    final_offspring = offspring_for_mutation
            else:
                # Fall back to individual mutations if batch not available
                logger.debug("Falling back to individual mutations (batch not available)")
                final_offspring = []
                for idea in offspring_for_mutation:
                    if random.random() < config.mutation_rate:
                        mutated_idea, mutation_stats = await self._apply_mutation_with_operator_selection(
                            idea, config, context, evaluation_context
                        )
                        final_offspring.append(mutated_idea)
                        # Update metrics
                        self.semantic_operator_metrics['semantic_mutations'] += mutation_stats['semantic_mutations']
                        self.semantic_operator_metrics['traditional_mutations'] += mutation_stats['traditional_mutations']
                        self.semantic_operator_metrics['semantic_llm_calls'] += mutation_stats['semantic_llm_calls']
                    else:
                        final_offspring.append(idea)
            
            # Step 3: Add generation metadata
            for idea in final_offspring:
                idea.metadata["generation"] = generation + 1
            
            # Step 4: ANOTHER KEY OPTIMIZATION - Batch evaluate ALL offspring
            logger.debug(f"Batch evaluating {len(final_offspring)} offspring")
            evaluated_offspring = await self.fitness_evaluator.evaluate_population(
                final_offspring, config, context
            )
            
            logger.info(f"Parallel generation: {semantic_crossovers} semantic crossovers, "
                       f"{traditional_crossovers} traditional crossovers, "
                       f"batch processed {len(final_offspring)} offspring")
            
            return evaluated_offspring
            
        except Exception as e:
            logger.error(f"Parallel offspring generation failed: {e}", exc_info=True)
            # Don't fall back here - let the caller handle fallback
            raise

    def _should_terminate(
        self, snapshot: PopulationSnapshot, request: EvolutionRequest
    ) -> bool:
        """Check if evolution should terminate early."""
        # Check if target metrics are met
        if request.target_metrics:
            if snapshot.best_fitness >= request.target_metrics.get("min_fitness", 1.0):
                return True

        # Check for convergence (low diversity)
        if snapshot.diversity_score < LOW_DIVERSITY_THRESHOLD:
            logger.warning("Population has converged (low diversity)")
            return True

        return False

    def _adapt_mutation_rate(
        self, snapshot: PopulationSnapshot, current_rate: float
    ) -> float:
        """Adapt mutation rate based on population state."""
        # Increase mutation if diversity is low
        if snapshot.diversity_score < MODERATE_DIVERSITY_THRESHOLD:
            return min(current_rate * MUTATION_RATE_INCREASE_FACTOR, MAX_MUTATION_RATE)
        # Decrease mutation if diversity is high
        elif snapshot.diversity_score > HIGH_DIVERSITY_THRESHOLD:
            return max(current_rate * MUTATION_RATE_DECREASE_FACTOR, MIN_MUTATION_RATE)
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
        """Extract top N unique ideas from population."""
        sorted_pop = sorted(population, key=lambda x: x.overall_fitness, reverse=True)
        
        # Deduplicate based on content
        seen_contents = set()
        unique_ideas = []
        
        for ind in sorted_pop:
            # Normalize content for comparison (strip whitespace)
            normalized_content = ind.idea.content.strip()
            
            if normalized_content not in seen_contents:
                seen_contents.add(normalized_content)
                unique_ideas.append(ind.idea)
                
                if len(unique_ideas) >= n:
                    break
        
        return unique_ideas

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
            "semantic_operators_enabled": self.llm_provider is not None,
            **self.semantic_operator_metrics,  # Add all semantic operator metrics
        }
