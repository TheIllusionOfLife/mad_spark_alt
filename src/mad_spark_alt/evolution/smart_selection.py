"""
Smart operator selection for genetic evolution.

This module implements logic to intelligently choose between semantic (LLM-powered)
and traditional operators based on population diversity and individual performance.
"""

import random
from typing import Optional

from mad_spark_alt.evolution.interfaces import EvolutionConfig, IndividualFitness


class SmartOperatorSelector:
    """
    Intelligently selects when to use semantic operators.
    
    Uses multiple factors to decide:
    - Population diversity (low diversity triggers semantic operators)
    - Individual fitness (only high performers get semantic operators)
    - Generation number (later generations have higher probability)
    """
    
    def __init__(self, config: Optional[EvolutionConfig] = None):
        """
        Initialize smart selector.
        
        Args:
            config: Evolution configuration with semantic operator settings
        """
        self.config = config or EvolutionConfig()
        # Make semantic operators much more aggressive for better diversity
        self.base_mutation_probability = 0.7  # Increased from 0.3
        self.base_crossover_probability = 0.8  # Increased from 0.5
        self.generation_boost_factor = 0.2     # Increased from 0.1
        self.min_fitness_threshold = 0.3       # Lowered from 0.4
        
    def should_use_semantic_mutation(
        self, 
        individual: IndividualFitness,
        population_diversity: float,
        generation: int
    ) -> bool:
        """
        Determine if semantic mutation should be used for an individual.
        
        Args:
            individual: The individual to mutate
            population_diversity: Current population diversity score (0-1)
            generation: Current generation number
            
        Returns:
            True if semantic mutation should be used
        """
        # Check if semantic operators are enabled
        if not self.config.use_semantic_operators:
            return False
            
        # Check diversity threshold
        if population_diversity >= self.config.semantic_operator_threshold:
            return False
            
        # Check individual fitness threshold
        if individual.overall_fitness < self.min_fitness_threshold:
            return False
            
        # Calculate probability based on generation
        generation_boost = min(generation * self.generation_boost_factor, 0.5)
        total_probability = self.base_mutation_probability + generation_boost
        
        # Make probabilistic decision
        return random.random() < total_probability
        
    def should_use_semantic_crossover(
        self,
        parent1: IndividualFitness,
        parent2: IndividualFitness,
        population_diversity: float
    ) -> bool:
        """
        Determine if semantic crossover should be used for a parent pair.
        
        Args:
            parent1: First parent
            parent2: Second parent
            population_diversity: Current population diversity score (0-1)
            
        Returns:
            True if semantic crossover should be used
        """
        # Check if semantic operators are enabled
        if not self.config.use_semantic_operators:
            return False
            
        # Check diversity threshold
        if population_diversity >= self.config.semantic_operator_threshold:
            return False
            
        # Both parents need minimum fitness
        if (parent1.overall_fitness < self.min_fitness_threshold or 
            parent2.overall_fitness < self.min_fitness_threshold):
            return False
            
        # Make probabilistic decision
        return random.random() < self.base_crossover_probability