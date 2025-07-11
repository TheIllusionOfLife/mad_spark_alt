"""
Evolution module for Mad Spark Alt.

This module implements genetic algorithms for evolving and optimizing ideas
generated by the QADI thinking agents.
"""

from mad_spark_alt.evolution.interfaces import (
    EvolutionRequest,
    EvolutionResult,
    GeneticOperatorInterface,
    SelectionStrategy,
    PopulationSnapshot,
    EvolutionConfig,
)
from mad_spark_alt.evolution.genetic_algorithm import GeneticAlgorithm
from mad_spark_alt.evolution.fitness import FitnessEvaluator
from mad_spark_alt.evolution.operators import (
    CrossoverOperator,
    MutationOperator,
    TournamentSelection,
    EliteSelection,
    RouletteWheelSelection,
    RankSelection,
    RandomSelection,
)

__all__ = [
    # Interfaces
    "EvolutionRequest",
    "EvolutionResult",
    "GeneticOperatorInterface",
    "SelectionStrategy",
    "PopulationSnapshot",
    "EvolutionConfig",
    # Core Components
    "GeneticAlgorithm",
    "FitnessEvaluator",
    # Operators
    "CrossoverOperator",
    "MutationOperator",
    "TournamentSelection",
    "EliteSelection",
    "RouletteWheelSelection",
    "RankSelection",
    "RandomSelection",
]
