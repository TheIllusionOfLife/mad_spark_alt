"""
Interfaces for the genetic evolution system.

This module defines the core interfaces and data structures for evolving ideas
using genetic algorithms.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from mad_spark_alt.core.interfaces import GeneratedIdea


class SelectionStrategy(Enum):
    """Selection strategies for genetic algorithms."""
    
    TOURNAMENT = "tournament"
    ROULETTE = "roulette"
    RANK = "rank"
    ELITE = "elite"
    RANDOM = "random"


@dataclass
class EvolutionConfig:
    """Configuration for genetic evolution process."""
    
    population_size: int = 50
    generations: int = 10
    mutation_rate: float = 0.1
    crossover_rate: float = 0.7
    elite_size: int = 2
    tournament_size: int = 3
    selection_strategy: SelectionStrategy = SelectionStrategy.TOURNAMENT
    
    # Fitness evaluation config
    fitness_weights: Dict[str, float] = field(default_factory=lambda: {
        "creativity_score": 0.4,
        "diversity_score": 0.3,
        "quality_score": 0.3,
    })
    
    # Advanced options
    adaptive_mutation: bool = False
    diversity_pressure: float = 0.1
    parallel_evaluation: bool = True
    max_parallel_evaluations: int = 10
    
    def validate(self) -> bool:
        """Validate configuration parameters."""
        if self.population_size < 2:
            return False
        if not 0 <= self.mutation_rate <= 1:
            return False
        if not 0 <= self.crossover_rate <= 1:
            return False
        if self.elite_size >= self.population_size:
            return False
        if self.tournament_size > self.population_size:
            return False
        return True


@dataclass
class IndividualFitness:
    """Fitness scores for an individual idea."""
    
    idea: GeneratedIdea
    creativity_score: float = 0.0
    diversity_score: float = 0.0
    quality_score: float = 0.0
    overall_fitness: float = 0.0
    evaluation_metadata: Dict[str, Any] = field(default_factory=dict)
    evaluated_at: datetime = field(default_factory=datetime.now)
    
    def calculate_overall_fitness(self, weights: Dict[str, float]) -> float:
        """Calculate weighted overall fitness score."""
        self.overall_fitness = (
            weights.get("creativity_score", 0.33) * self.creativity_score +
            weights.get("diversity_score", 0.33) * self.diversity_score +
            weights.get("quality_score", 0.34) * self.quality_score
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
    timestamp: datetime = field(default_factory=datetime.now)
    
    @classmethod
    def from_population(cls, generation: int, population: List[IndividualFitness]) -> "PopulationSnapshot":
        """Create snapshot from current population."""
        fitness_scores = [ind.overall_fitness for ind in population]
        return cls(
            generation=generation,
            population=population,
            best_fitness=max(fitness_scores) if fitness_scores else 0.0,
            average_fitness=sum(fitness_scores) / len(fitness_scores) if fitness_scores else 0.0,
            worst_fitness=min(fitness_scores) if fitness_scores else 0.0,
            diversity_score=0.0,  # Will be calculated by diversity metrics
        )


@dataclass
class EvolutionRequest:
    """Request for evolving a population of ideas."""
    
    initial_population: List[GeneratedIdea]
    config: EvolutionConfig = field(default_factory=EvolutionConfig)
    context: Optional[str] = None
    constraints: Optional[List[str]] = None
    target_metrics: Optional[Dict[str, float]] = None
    
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
    
    def get_top_ideas(self, n: int = 5) -> List[GeneratedIdea]:
        """Get top N ideas by fitness."""
        sorted_population = sorted(
            self.final_population, 
            key=lambda x: x.overall_fitness, 
            reverse=True
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
        context: Optional[str] = None
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
        self, 
        idea: GeneratedIdea,
        mutation_rate: float,
        context: Optional[str] = None
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
        config: EvolutionConfig
    ) -> List[IndividualFitness]:
        """Select individuals from population."""
        pass