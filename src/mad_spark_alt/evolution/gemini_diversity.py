"""
Gemini embedding-based diversity calculator.

This module implements diversity calculation using semantic embeddings
from Google's Gemini API, providing true semantic understanding of ideas.
"""

import hashlib
import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity  # type: ignore[import-untyped]

from .diversity_calculator import DiversityCalculator
from .interfaces import IndividualFitness
from ..core.llm_provider import LLMProviderInterface, EmbeddingRequest

logger = logging.getLogger(__name__)


class GeminiDiversityCalculator(DiversityCalculator):
    """
    Calculate diversity using semantic embeddings from Gemini API.
    
    This implementation has O(n) API complexity plus O(n²) for cosine similarity,
    but the cosine calculation is very fast with numpy. It provides true semantic
    understanding of idea diversity.
    """
    
    def __init__(self, llm_provider: LLMProviderInterface):
        """
        Initialize with an LLM provider that supports embeddings.
        
        Args:
            llm_provider: Provider instance with get_embeddings support
        """
        self.llm_provider = llm_provider
        self._cache: Dict[str, List[float]] = {}  # Cache embeddings by content hash
        
    def _get_content_hash(self, content: str) -> str:
        """Get a hash of content for caching."""
        return hashlib.md5(content.encode()).hexdigest()
        
    async def calculate_diversity(self, population: List[IndividualFitness]) -> float:
        """
        Calculate diversity score using semantic embeddings.
        
        Args:
            population: List of individuals to calculate diversity for
            
        Returns:
            Diversity score between 0.0 (all semantically identical) and 1.0 (maximum diversity)
        """
        if len(population) < 2:
            return 1.0  # Maximum diversity for empty or single-item populations
            
        # Extract unique texts and track mapping
        unique_texts: List[str] = []
        text_to_idx = {}
        idx_to_population = []
        
        for i, individual in enumerate(population):
            content = individual.idea.content.strip()
            content_hash = self._get_content_hash(content)
            
            if content_hash not in text_to_idx:
                text_to_idx[content_hash] = len(unique_texts)
                unique_texts.append(content)
            
            idx_to_population.append(text_to_idx[content_hash])
        
        # Get embeddings for unique texts
        embeddings = await self._get_embeddings_with_cache(unique_texts)
        
        # Map embeddings back to population order
        population_embeddings = np.array([embeddings[idx] for idx in idx_to_population])
        
        # Calculate pairwise cosine similarities
        similarity_matrix = cosine_similarity(population_embeddings)
        
        # Extract upper triangle (excluding diagonal)
        upper_triangle = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]
        
        # Average similarity
        avg_similarity = np.mean(upper_triangle) if len(upper_triangle) > 0 else 0.0
        
        # Diversity is inverse of similarity
        diversity = 1.0 - avg_similarity
        
        logger.debug(
            f"Semantic diversity for {len(population)} individuals "
            f"({len(unique_texts)} unique): {diversity:.3f}"
        )
        
        return float(diversity)
        
    async def _get_embeddings_with_cache(self, texts: List[str]) -> np.ndarray[Any, np.dtype[Any]]:
        """
        Get embeddings for texts, using cache where possible.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            Numpy array of embeddings
        """
        # Check cache and identify texts needing API calls
        needed_texts = []
        needed_indices = []
        embeddings: List[Optional[List[float]]] = [None] * len(texts)
        
        for i, text in enumerate(texts):
            text_hash = self._get_content_hash(text)
            if text_hash in self._cache:
                embeddings[i] = self._cache[text_hash]
            else:
                needed_texts.append(text)
                needed_indices.append(i)
        
        # Call API for uncached texts
        if needed_texts:
            request = EmbeddingRequest(
                texts=needed_texts,
                model="models/text-embedding-004",
                task_type="SEMANTIC_SIMILARITY",
                output_dimensionality=768
            )
            
            response = await self.llm_provider.get_embeddings(request)
            
            # Cache and store results
            for i, (text, embedding) in enumerate(zip(needed_texts, response.embeddings)):
                text_hash = self._get_content_hash(text)
                self._cache[text_hash] = embedding
                embeddings[needed_indices[i]] = embedding
        
        return np.array([emb for emb in embeddings if emb is not None], dtype=float)  # type: ignore[no-any-return]