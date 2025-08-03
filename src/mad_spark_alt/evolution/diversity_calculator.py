"""
Abstract base class for diversity calculation strategies.

This module defines the interface that all diversity calculators must implement,
enabling different approaches to measuring population diversity in genetic algorithms.
"""

from abc import ABC, abstractmethod
from typing import List

from .interfaces import IndividualFitness


class DiversityCalculator(ABC):
    """
    Abstract base class for diversity calculation strategies.
    
    Diversity is a measure of how different individuals are from each other
    in a population. Higher diversity (closer to 1.0) indicates more variety,
    while lower diversity (closer to 0.0) indicates more similarity.
    """
    
    @abstractmethod
    async def calculate_diversity(self, population: List[IndividualFitness]) -> float:
        """
        Calculate diversity score for a population.
        
        Args:
            population: List of individuals to calculate diversity for
            
        Returns:
            Diversity score between 0.0 (all identical) and 1.0 (maximum diversity)
        """
        pass