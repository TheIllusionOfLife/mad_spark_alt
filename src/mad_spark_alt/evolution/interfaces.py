"""
Interfaces for the genetic evolution system.

This module defines the core interfaces and data structures for evolving ideas
using genetic algorithms.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from mad_spark_alt.core.interfaces import GeneratedIdea
from mad_spark_alt.evolution.constants import (
    DEFAULT_CREATIVITY_WEIGHT,
    DEFAULT_CROSSOVER_RATE,
    DEFAULT_DIVERSITY_PRESSURE,
    DEFAULT_DIVERSITY_WEIGHT,
    DEFAULT_MUTATION_RATE,
    DEFAULT_QUALITY_WEIGHT,
    EQUAL_WEIGHT_CREATIVITY,
    EQUAL_WEIGHT_DIVERSITY,
    EQUAL_WEIGHT_QUALITY,
    EVALUATION_CRITERIA,
    ZERO_SCORE,
)

# Default weight for each QADI criterion when calculating overall fitness
DEFAULT_QADI_CRITERION_WEIGHT = 0.2


@dataclass
class EvaluationContext:
    """Context information for guiding evolution towards specific improvements."""
    
    original_question: str
    current_best_scores: Dict[str, float] = field(default_factory=dict)
    target_improvements: List[str] = field(default_factory=list)
    evaluation_criteria: List[str] = field(default_factory=lambda: EVALUATION_CRITERIA.copy())


class SelectionStrategy(Enum):
    """Selection strategies for genetic algorithms."""

    TOURNAMENT = "tournament"
    ROULETTE = "roulette"
    RANK = "rank"
    ELITE = "elite"
    RANDOM = "random"


class DiversityMethod(Enum):
    """Diversity calculation methods for population analysis."""
    
    JACCARD = "jaccard"  # Word-based similarity (fast, O(n²))
    SEMANTIC = "semantic"  # Embedding-based similarity (slower, API calls)


@dataclass
class EvolutionConfig:
    """Configuration for genetic evolution process."""

    population_size: int = 5  # Changed default from 50 to 5
    generations: int = 3  # Changed default from 10 to 3
    mutation_rate: float = DEFAULT_MUTATION_RATE
    crossover_rate: float = DEFAULT_CROSSOVER_RATE
    elite_size: int = 1  # Changed default from 2 to 1
    tournament_size: int = 2  # Changed default from 3 to 2 to work with smaller populations
    selection_strategy: SelectionStrategy = SelectionStrategy.TOURNAMENT

    # Fitness evaluation config
    fitness_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "creativity_score": DEFAULT_CREATIVITY_WEIGHT,
            "diversity_score": DEFAULT_DIVERSITY_WEIGHT,
            "quality_score": DEFAULT_QUALITY_WEIGHT,
        }
    )

    # Advanced options
    adaptive_mutation: bool = False
    diversity_pressure: float = DEFAULT_DIVERSITY_PRESSURE
    parallel_evaluation: bool = True
    max_parallel_evaluations: int = 3  # Reduced from 5 to work with smaller populations
    random_seed: Optional[int] = None

    # Timeout configuration
    timeout_seconds: Optional[float] = None
    adaptive_timeout: bool = False

    # Progress tracking
    enable_progress_tracking: bool = False
    enable_cost_estimation: bool = False

    # Advanced caching
    enable_semantic_cache: bool = False

    # LLM operators
    enable_llm_operators: bool = False
    
    # Semantic operator configuration
    use_semantic_operators: bool = True
    semantic_operator_threshold: float = 0.8  # Increased from 0.5 to make semantic operators trigger more often
    semantic_batch_size: int = 5
    semantic_cache_ttl: int = 3600
    
    # Diversity calculation method
    diversity_method: DiversityMethod = DiversityMethod.JACCARD  # Default to fast Jaccard method

    def validate(self) -> bool:
        """Validate configuration parameters."""
        # Population size must be between 2 and 10
        if self.population_size < 2 or self.population_size > 10:
            return False
        
        # Generations must be between 2 and 5
        if self.generations < 2 or self.generations > 5:
            return False
            
        if not 0 <= self.mutation_rate <= 1:
            return False
        if not 0 <= self.crossover_rate <= 1:
            return False
        if self.elite_size >= self.population_size:
            return False
        if self.tournament_size > self.population_size:
            return False
        # Max parallel evaluations should not exceed population size
        if self.max_parallel_evaluations > self.population_size:
            return False
        
        # Validate semantic operator config
        if self.use_semantic_operators:
            if not 0 <= self.semantic_operator_threshold <= 1:
                return False
            if self.semantic_batch_size < 1:
                return False
            if self.semantic_cache_ttl < 0:
                return False
                
        return True

    def get_random_state(self) -> Any:
        """Get seeded random state for reproducibility."""
        import random

        if self.random_seed is not None:
            return random.Random(self.random_seed)
        return random


@dataclass
class IndividualFitness:
    """Fitness scores for an individual idea using QADI criteria."""

    idea: GeneratedIdea
    # QADI scoring criteria
    impact: float = ZERO_SCORE
    feasibility: float = ZERO_SCORE
    accessibility: float = ZERO_SCORE
    sustainability: float = ZERO_SCORE
    scalability: float = ZERO_SCORE
    overall_fitness: float = ZERO_SCORE
    evaluation_metadata: Dict[str, Any] = field(default_factory=dict)
    evaluated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        """Set timestamp after initialization."""
        # evaluated_at is now automatically set via default_factory
        
    def get_scores_dict(self) -> Dict[str, float]:
        """Get scores as a dictionary."""
        return {
            "impact": self.impact,
            "feasibility": self.feasibility,
            "accessibility": self.accessibility,
            "sustainability": self.sustainability,
            "scalability": self.scalability,
            "overall": self.overall_fitness
        }

    def calculate_overall_fitness(self, weights: Dict[str, float]) -> float:
        """Calculate weighted overall fitness score using QADI criteria."""
        # Default equal weights for QADI criteria
        impact_weight = weights.get("impact", DEFAULT_QADI_CRITERION_WEIGHT)
        feasibility_weight = weights.get("feasibility", DEFAULT_QADI_CRITERION_WEIGHT)
        accessibility_weight = weights.get("accessibility", DEFAULT_QADI_CRITERION_WEIGHT)
        sustainability_weight = weights.get("sustainability", DEFAULT_QADI_CRITERION_WEIGHT)
        scalability_weight = weights.get("scalability", DEFAULT_QADI_CRITERION_WEIGHT)
        
        self.overall_fitness = (
            impact_weight * self.impact
            + feasibility_weight * self.feasibility
            + accessibility_weight * self.accessibility
            + sustainability_weight * self.sustainability
            + scalability_weight * self.scalability
        )
        return self.overall_fitness


@dataclass
class PopulationSnapshot:
    """Snapshot of population state at a specific generation."""

    generation: int
    population: List[IndividualFitness]
    best_fitness: float
    average_fitness: float
    worst_fitness: float
    diversity_score: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        """Set timestamp after initialization."""
        # timestamp is now automatically set via default_factory

    @classmethod
    def from_population(
        cls, generation: int, population: List[IndividualFitness]
    ) -> "PopulationSnapshot":
        """Create snapshot from current population."""
        fitness_scores = [ind.overall_fitness for ind in population]
        return cls(
            generation=generation,
            population=population,
            best_fitness=max(fitness_scores) if fitness_scores else ZERO_SCORE,
            average_fitness=(
                sum(fitness_scores) / len(fitness_scores)
                if fitness_scores
                else ZERO_SCORE
            ),
            worst_fitness=min(fitness_scores) if fitness_scores else ZERO_SCORE,
            diversity_score=ZERO_SCORE,  # Will be calculated by diversity metrics
        )


@dataclass
class EvolutionRequest:
    """Request for evolving a population of ideas."""

    initial_population: List[GeneratedIdea]
    config: EvolutionConfig = field(default_factory=EvolutionConfig)
    context: Optional[str] = None
    constraints: Optional[List[str]] = None
    target_metrics: Optional[Dict[str, float]] = None
    evaluation_context: Optional[EvaluationContext] = None

    def validate(self) -> bool:
        """Validate the evolution request."""
        if not self.initial_population:
            return False
        if not self.config.validate():
            return False
        return True


@dataclass
class EvolutionResult:
    """Result of genetic evolution process."""

    final_population: List[IndividualFitness]
    best_ideas: List[GeneratedIdea]
    generation_snapshots: List[PopulationSnapshot]
    total_generations: int
    execution_time: float
    evolution_metrics: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None

    @property
    def success(self) -> bool:
        """Check if evolution completed successfully."""
        return self.error_message is None and len(self.final_population) > 0
        
    def get_all_individuals(self) -> List[IndividualFitness]:
        """Get all individuals from all generations."""
        all_individuals = []
        for snapshot in self.generation_snapshots:
            all_individuals.extend(snapshot.population)
        return all_individuals

    def get_top_ideas(self, n: int = 5) -> List[GeneratedIdea]:
        """Get top N ideas by fitness."""
        sorted_population = sorted(
            self.final_population, key=lambda x: x.overall_fitness, reverse=True
        )
        return [ind.idea for ind in sorted_population[:n]]


class GeneticOperatorInterface(ABC):
    """Interface for genetic operators."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the genetic operator."""
        pass

    @property
    @abstractmethod
    def operator_type(self) -> str:
        """Type of operator (crossover, mutation, selection)."""
        pass

    @abstractmethod
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate operator configuration."""
        pass


class CrossoverInterface(GeneticOperatorInterface):
    """Interface for crossover operators."""

    @property
    def operator_type(self) -> str:
        return "crossover"

    @abstractmethod
    async def crossover(
        self,
        parent1: GeneratedIdea,
        parent2: GeneratedIdea,
        context: Optional[str] = None,
    ) -> Tuple[GeneratedIdea, GeneratedIdea]:
        """Perform crossover between two parent ideas."""
        pass


class MutationInterface(GeneticOperatorInterface):
    """Interface for mutation operators."""

    @property
    def operator_type(self) -> str:
        return "mutation"

    @abstractmethod
    async def mutate(
        self, idea: GeneratedIdea, mutation_rate: float, context: Optional[str] = None
    ) -> GeneratedIdea:
        """Mutate an idea."""
        pass


class SelectionInterface(GeneticOperatorInterface):
    """Interface for selection operators."""

    @property
    def operator_type(self) -> str:
        return "selection"

    @abstractmethod
    async def select(
        self,
        population: List[IndividualFitness],
        num_selected: int,
        config: EvolutionConfig,
    ) -> List[IndividualFitness]:
        """Select individuals from population."""
        pass
