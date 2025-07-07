"""
Genetic Algorithm implementation for idea evolution.

This module orchestrates the genetic evolution process, coordinating
fitness evaluation, selection, crossover, and mutation operations.
"""

import asyncio
import logging
import random
import time
from typing import List, Optional, Tuple

from mad_spark_alt.core.interfaces import GeneratedIdea
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
    ):
        """
        Initialize genetic algorithm.
        
        Args:
            fitness_evaluator: Optional custom fitness evaluator
            crossover_operator: Optional custom crossover operator
            mutation_operator: Optional custom mutation operator
        """
        self.fitness_evaluator = fitness_evaluator or FitnessEvaluator()
        self.crossover_operator = crossover_operator or CrossoverOperator()
        self.mutation_operator = mutation_operator or MutationOperator()
        
        # Initialize selection operators
        self.selection_operators = {
            SelectionStrategy.TOURNAMENT: TournamentSelection(),
            SelectionStrategy.ELITE: EliteSelection(),
        }
    
    async def evolve(self, request: EvolutionRequest) -> EvolutionResult:
        """
        Run the genetic evolution process.
        
        Args:
            request: Evolution request with initial population and config
            
        Returns:
            EvolutionResult with final population and metrics
        """
        start_time = time.time()
        
        # Validate request
        if not request.validate():
            return EvolutionResult(
                final_population=[],
                best_ideas=[],
                generation_snapshots=[],
                total_generations=0,
                execution_time=0.0,
                error_message="Invalid evolution request"
            )
        
        try:
            # Initialize population
            current_population = await self._initialize_population(
                request.initial_population,
                request.config
            )
            
            generation_snapshots = []
            
            # Evolution loop
            for generation in range(request.config.generations):
                logger.info(f"Starting generation {generation + 1}/{request.config.generations}")
                
                # Create snapshot of current generation
                snapshot = PopulationSnapshot.from_population(generation, current_population)
                
                # Calculate population diversity
                snapshot.diversity_score = await self.fitness_evaluator.calculate_population_diversity(
                    current_population
                )
                
                generation_snapshots.append(snapshot)
                
                # Check termination criteria
                if self._should_terminate(snapshot, request):
                    logger.info(f"Early termination at generation {generation + 1}")
                    break
                
                # Evolve to next generation
                current_population = await self._evolve_generation(
                    current_population,
                    request.config,
                    request.context,
                    generation
                )
                
                # Adaptive mutation rate (if enabled)
                if request.config.adaptive_mutation:
                    request.config.mutation_rate = self._adapt_mutation_rate(
                        snapshot,
                        request.config.mutation_rate
                    )
            
            # Final population snapshot
            final_snapshot = PopulationSnapshot.from_population(
                len(generation_snapshots),
                current_population
            )
            final_snapshot.diversity_score = await self.fitness_evaluator.calculate_population_diversity(
                current_population
            )
            generation_snapshots.append(final_snapshot)
            
            # Extract best ideas
            best_ideas = self._extract_best_ideas(current_population, n=10)
            
            # Calculate evolution metrics
            evolution_metrics = self._calculate_evolution_metrics(
                generation_snapshots,
                request.initial_population,
                best_ideas
            )
            
            execution_time = time.time() - start_time
            
            return EvolutionResult(
                final_population=current_population,
                best_ideas=best_ideas,
                generation_snapshots=generation_snapshots,
                total_generations=len(generation_snapshots),
                execution_time=execution_time,
                evolution_metrics=evolution_metrics
            )
            
        except Exception as e:
            logger.error(f"Evolution failed: {e}")
            return EvolutionResult(
                final_population=[],
                best_ideas=[],
                generation_snapshots=[],
                total_generations=0,
                execution_time=time.time() - start_time,
                error_message=str(e)
            )
    
    async def _initialize_population(
        self,
        initial_ideas: List[GeneratedIdea],
        config: EvolutionConfig
    ) -> List[IndividualFitness]:
        """Initialize population with fitness evaluation."""
        # Ensure we have enough individuals
        population = initial_ideas.copy()
        
        # If initial population is smaller than configured size, duplicate randomly
        while len(population) < config.population_size:
            population.append(random.choice(initial_ideas))
        
        # If larger, truncate
        population = population[:config.population_size]
        
        # Evaluate initial population
        logger.info(f"Evaluating initial population of {len(population)} individuals")
        return await self.fitness_evaluator.evaluate_population(population, config)
    
    async def _evolve_generation(
        self,
        population: List[IndividualFitness],
        config: EvolutionConfig,
        context: Optional[str],
        generation: int
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
                    parents[0].idea,
                    parents[1].idea,
                    context
                )
            else:
                # If no crossover, just copy parents
                offspring1, offspring2 = parents[0].idea, parents[1].idea
            
            # Mutation
            offspring1 = await self.mutation_operator.mutate(
                offspring1,
                config.mutation_rate,
                context
            )
            offspring2 = await self.mutation_operator.mutate(
                offspring2,
                config.mutation_rate,
                context
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
                new_ideas,
                config,
                context
            )
            new_population.extend(evaluated_offspring)
        
        # Ensure population size
        new_population = new_population[:config.population_size]
        
        # Apply diversity pressure if configured
        if config.diversity_pressure > 0:
            new_population = await self._apply_diversity_pressure(
                new_population,
                config.diversity_pressure
            )
        
        return new_population
    
    def _should_terminate(self, snapshot: PopulationSnapshot, request: EvolutionRequest) -> bool:
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
    
    def _adapt_mutation_rate(self, snapshot: PopulationSnapshot, current_rate: float) -> float:
        """Adapt mutation rate based on population state."""
        # Increase mutation if diversity is low
        if snapshot.diversity_score < 0.3:
            return min(current_rate * 1.2, 0.5)
        # Decrease mutation if diversity is high
        elif snapshot.diversity_score > 0.7:
            return max(current_rate * 0.8, 0.01)
        return current_rate
    
    async def _apply_diversity_pressure(
        self,
        population: List[IndividualFitness],
        pressure: float
    ) -> List[IndividualFitness]:
        """Apply diversity pressure to maintain genetic diversity."""
        # Calculate diversity-adjusted fitness
        diversity_score = await self.fitness_evaluator.calculate_population_diversity(population)
        
        for individual in population:
            # Boost fitness of unique individuals
            diversity_bonus = pressure * (1.0 - diversity_score)
            individual.overall_fitness *= (1.0 + diversity_bonus)
        
        return population
    
    def _extract_best_ideas(self, population: List[IndividualFitness], n: int) -> List[GeneratedIdea]:
        """Extract top N ideas from population."""
        sorted_pop = sorted(population, key=lambda x: x.overall_fitness, reverse=True)
        return [ind.idea for ind in sorted_pop[:n]]
    
    def _calculate_evolution_metrics(
        self,
        snapshots: List[PopulationSnapshot],
        initial_ideas: List[GeneratedIdea],
        best_ideas: List[GeneratedIdea]
    ) -> dict:
        """Calculate metrics about the evolution process."""
        if not snapshots:
            return {}
        
        initial_snapshot = snapshots[0]
        final_snapshot = snapshots[-1]
        
        # Calculate improvement
        fitness_improvement = (
            (final_snapshot.best_fitness - initial_snapshot.best_fitness) / 
            initial_snapshot.best_fitness * 100
            if initial_snapshot.best_fitness > 0 else 0
        )
        
        # Find generation with best fitness
        best_generation = max(range(len(snapshots)), key=lambda i: snapshots[i].best_fitness)
        
        return {
            "fitness_improvement_percent": fitness_improvement,
            "initial_best_fitness": initial_snapshot.best_fitness,
            "final_best_fitness": final_snapshot.best_fitness,
            "initial_avg_fitness": initial_snapshot.average_fitness,
            "final_avg_fitness": final_snapshot.average_fitness,
            "best_fitness_generation": best_generation,
            "diversity_trend": [s.diversity_score for s in snapshots],
            "fitness_trend": [s.best_fitness for s in snapshots],
            "total_ideas_evaluated": len(snapshots) * len(initial_ideas),
        }