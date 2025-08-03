"""
Jaccard similarity-based diversity calculator.

This module implements diversity calculation using Jaccard similarity,
which measures the overlap of words between ideas. This is the original
implementation extracted from FitnessEvaluator for backward compatibility.
"""

import logging
from typing import List

from .diversity_calculator import DiversityCalculator
from .interfaces import IndividualFitness

logger = logging.getLogger(__name__)


class JaccardDiversityCalculator(DiversityCalculator):
    """
    Calculate diversity using Jaccard similarity on word sets.
    
    This implementation has O(n²) complexity but is simple and doesn't
    require external API calls. It measures surface-level similarity
    based on word overlap.
    """
    
    async def calculate_diversity(self, population: List[IndividualFitness]) -> float:
        """
        Calculate diversity score using Jaccard similarity.
        
        Args:
            population: List of individuals to calculate diversity for
            
        Returns:
            Diversity score between 0.0 (all identical) and 1.0 (maximum diversity)
        """
        if len(population) < 2:
            return 1.0  # Maximum diversity for empty or single-item populations
            
        # Calculate pairwise Jaccard similarities
        total_similarity = 0.0
        comparisons = 0
        
        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                idea1 = population[i].idea.content.lower().strip()
                idea2 = population[j].idea.content.lower().strip()
                
                # Simple similarity based on shared words
                words1 = set(idea1.split())
                words2 = set(idea2.split())
                
                if len(words1) == 0 and len(words2) == 0:
                    similarity = 1.0  # Both empty
                elif len(words1) == 0 or len(words2) == 0:
                    similarity = 0.0  # One empty
                else:
                    # Jaccard similarity
                    intersection = len(words1.intersection(words2))
                    union = len(words1.union(words2))
                    similarity = intersection / union if union > 0 else 0.0
                
                total_similarity += similarity
                comparisons += 1
        
        if comparisons > 0:
            avg_similarity = total_similarity / comparisons
            # Diversity is inverse of similarity
            diversity = 1.0 - avg_similarity
        else:
            diversity = 1.0
            
        logger.debug(
            f"Jaccard diversity for {len(population)} individuals: {diversity:.3f}"
        )
        
        return diversity