"""
Diversity and novelty evaluation metrics.
"""

import asyncio
import numpy as np
from typing import Dict, List, Any, Set, Optional, cast
from collections import Counter
import logging
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from ...core.interfaces import (
    EvaluatorInterface,
    EvaluationRequest,
    EvaluationResult,
    EvaluationLayer,
    OutputType,
    ModelOutput,
    AsyncEvaluatorMixin,
    CacheableEvaluatorMixin,
)

logger = logging.getLogger(__name__)


class DiversityEvaluator(EvaluatorInterface, AsyncEvaluatorMixin, CacheableEvaluatorMixin):
    """Evaluates diversity and novelty of AI outputs."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._embedding_model: Optional[SentenceTransformer] = None
        self._embedding_cache: Dict[str, np.ndarray] = {}
    
    @property
    def name(self) -> str:
        return "diversity_evaluator"
    
    @property
    def layer(self) -> EvaluationLayer:
        return EvaluationLayer.QUANTITATIVE
    
    @property
    def supported_output_types(self) -> List[OutputType]:
        return [OutputType.TEXT, OutputType.CODE]
    
    @property
    def embedding_model(self) -> SentenceTransformer:
        """Lazy load the embedding model."""
        if self._embedding_model is None:
            self._embedding_model = SentenceTransformer(self.model_name)
        return cast(SentenceTransformer, self._embedding_model)
    
    async def evaluate(self, request: EvaluationRequest) -> List[EvaluationResult]:
        """Evaluate diversity metrics for the given outputs."""
        if len(request.outputs) < 2:
            # Single output - return basic metrics
            return [self._evaluate_single_output(request.outputs[0])]
        
        # Multiple outputs - calculate diversity metrics
        return await self._evaluate_multiple_outputs(request.outputs)
    
    async def evaluate_batch(
        self, 
        requests: List[EvaluationRequest]
    ) -> List[List[EvaluationResult]]:
        """Evaluate multiple requests in batch."""
        tasks = [self.evaluate(request) for request in requests]
        return await asyncio.gather(*tasks)
    
    def _evaluate_single_output(self, output: ModelOutput) -> EvaluationResult:
        """Evaluate a single output."""
        content = str(output.content)
        
        scores = {
            "distinct_1": self._calculate_distinct_n(content, 1),
            "distinct_2": self._calculate_distinct_n(content, 2),
            "distinct_3": self._calculate_distinct_n(content, 3),
            "lexical_diversity": self._calculate_lexical_diversity(content),
        }
        
        explanations = {
            "distinct_1": "Ratio of unique unigrams to total unigrams",
            "distinct_2": "Ratio of unique bigrams to total bigrams", 
            "distinct_3": "Ratio of unique trigrams to total trigrams",
            "lexical_diversity": "Type-token ratio (unique words / total words)",
        }
        
        return EvaluationResult(
            evaluator_name=self.name,
            layer=self.layer,
            scores=scores,
            explanations=explanations,
            metadata={"single_output": True}
        )
    
    async def _evaluate_multiple_outputs(
        self, 
        outputs: List[ModelOutput]
    ) -> List[EvaluationResult]:
        """Evaluate multiple outputs for diversity."""
        contents = [str(output.content) for output in outputs]
        
        # Calculate embeddings for semantic similarity
        embeddings = await self._get_embeddings(contents)
        
        # Calculate pairwise similarities
        similarity_matrix = cosine_similarity(embeddings)
        
        results = []
        for i, output in enumerate(outputs):
            content = contents[i]
            
            # Individual diversity metrics
            scores = {
                "distinct_1": self._calculate_distinct_n(content, 1),
                "distinct_2": self._calculate_distinct_n(content, 2),
                "distinct_3": self._calculate_distinct_n(content, 3),
                "lexical_diversity": self._calculate_lexical_diversity(content),
            }
            
            # Semantic diversity metrics (relative to other outputs)
            if len(outputs) > 1:
                # Average similarity to other outputs (lower = more diverse)
                other_similarities = [
                    similarity_matrix[i][j] 
                    for j in range(len(outputs)) 
                    if j != i
                ]
                avg_similarity = float(np.mean(other_similarities))
                scores["semantic_uniqueness"] = 1.0 - avg_similarity
                scores["min_similarity"] = 1.0 - float(max(other_similarities))
            
            # Calculate novelty score (combination of metrics)
            novelty_components = [
                scores["distinct_2"],
                scores["lexical_diversity"],
                scores.get("semantic_uniqueness", 0.5)
            ]
            scores["novelty_score"] = float(np.mean(novelty_components))
            
            explanations = {
                "distinct_1": "Ratio of unique unigrams to total unigrams",
                "distinct_2": "Ratio of unique bigrams to total bigrams",
                "distinct_3": "Ratio of unique trigrams to total trigrams", 
                "lexical_diversity": "Type-token ratio (unique words / total words)",
                "semantic_uniqueness": "1 - average semantic similarity to other outputs",
                "min_similarity": "1 - maximum similarity to any other output",
                "novelty_score": "Combined novelty metric across all dimensions",
            }
            
            metadata = {
                "total_outputs": len(outputs),
                "output_index": i,
                "content_length": len(content),
                "embedding_model": self.model_name,
            }
            
            results.append(EvaluationResult(
                evaluator_name=self.name,
                layer=self.layer,
                scores=scores,
                explanations=explanations,
                metadata=metadata
            ))
        
        return results
    
    async def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get embeddings for texts, using cache when possible."""
        embeddings = []
        uncached_texts = []
        uncached_indices = []
        
        # Check cache first
        for i, text in enumerate(texts):
            cache_key = self._get_text_cache_key(text)
            if cache_key in self._embedding_cache:
                embeddings.append(self._embedding_cache[cache_key])
            else:
                embeddings.append(None)
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        # Compute embeddings for uncached texts
        if uncached_texts:
            new_embeddings = self.embedding_model.encode(uncached_texts)
            
            # Update cache and results
            for i, embedding in enumerate(new_embeddings):
                original_index = uncached_indices[i]
                text = uncached_texts[i]
                cache_key = self._get_text_cache_key(text)
                
                self._embedding_cache[cache_key] = embedding
                embeddings[original_index] = embedding
        
        return np.array([e for e in embeddings if e is not None])
    
    def _calculate_distinct_n(self, text: str, n: int) -> float:
        """Calculate distinct n-gram ratio."""
        words = text.lower().split()
        if len(words) < n:
            return 1.0
        
        ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
        unique_ngrams = set(ngrams)
        
        return len(unique_ngrams) / len(ngrams) if ngrams else 0.0
    
    def _calculate_lexical_diversity(self, text: str) -> float:
        """Calculate lexical diversity (type-token ratio)."""
        words = text.lower().split()
        if not words:
            return 0.0
        
        unique_words = set(words)
        return len(unique_words) / len(words)
    
    def _get_text_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        return f"{hash(text)}_{len(text)}"
    
    def get_cache_key(self, request: EvaluationRequest) -> str:
        """Generate cache key for request."""
        content_hashes = [
            hash(str(output.content)) for output in request.outputs
        ]
        return f"diversity_{hash(tuple(content_hashes))}"
    
    def is_cacheable(self, request: EvaluationRequest) -> bool:
        """Check if request results should be cached."""
        return True  # Diversity metrics are deterministic
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate configuration."""
        return True  # This evaluator has no required config