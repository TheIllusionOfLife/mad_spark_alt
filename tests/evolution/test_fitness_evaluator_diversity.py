"""
Test suite for FitnessEvaluator with configurable diversity calculators.

This module tests that FitnessEvaluator can work with different diversity
calculator implementations.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from mad_spark_alt.core import GeneratedIdea, ThinkingMethod
from mad_spark_alt.core.unified_evaluator import HypothesisEvaluation, UnifiedEvaluator
from mad_spark_alt.evolution.fitness import FitnessEvaluator
from mad_spark_alt.evolution.interfaces import EvolutionConfig, IndividualFitness
from mad_spark_alt.evolution.jaccard_diversity import JaccardDiversityCalculator
from mad_spark_alt.evolution.gemini_diversity import GeminiDiversityCalculator


class TestFitnessEvaluatorDiversity:
    """Test FitnessEvaluator with different diversity calculators."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create mock unified evaluator
        self.mock_unified_evaluator = MagicMock(spec=UnifiedEvaluator)
        self.mock_unified_evaluator.evaluate_multiple = AsyncMock()
        
        # Mock evaluation result
        mock_evaluation = HypothesisEvaluation(
            content="test hypothesis",
            overall_score=0.7,
            scores={"impact": 0.8, "feasibility": 0.6, "accessibility": 0.7, 
                   "sustainability": 0.7, "scalability": 0.7},
            explanations={"impact": "test explanation"},
            metadata={"llm_cost": 0.001}
        )
        
        # Return list of evaluations for evaluate_multiple
        self.mock_unified_evaluator.evaluate_multiple.return_value = [
            mock_evaluation, mock_evaluation, mock_evaluation
        ]
        
        # Create test configuration
        self.config = EvolutionConfig(
            population_size=5,
            generations=3,
        )
        
    def _create_individual(self, content: str, fitness: float = 0.5) -> IndividualFitness:
        """Helper to create an IndividualFitness instance."""
        idea = GeneratedIdea(
            content=content,
            thinking_method=ThinkingMethod.QUESTIONING,
            agent_name="test_agent",
            generation_prompt="Test prompt"
        )
        return IndividualFitness(
            idea=idea,
            impact=fitness,
            feasibility=fitness,
            accessibility=fitness,
            sustainability=fitness,
            scalability=fitness,
            overall_fitness=fitness
        )
        
    @pytest.mark.asyncio
    async def test_fitness_evaluator_with_jaccard_diversity(self):
        """Test FitnessEvaluator with JaccardDiversityCalculator."""
        jaccard_calculator = JaccardDiversityCalculator()
        fitness_evaluator = FitnessEvaluator(
            unified_evaluator=self.mock_unified_evaluator,
            diversity_calculator=jaccard_calculator
        )
        
        # Create test population
        population = [
            self._create_individual("machine learning model"),
            self._create_individual("deep learning network"),
            self._create_individual("quantum computing system")
        ]
        
        # Evaluate population
        evaluated_population = await fitness_evaluator.evaluate_population(
            [idea.idea for idea in population], self.config
        )
        
        # Calculate diversity separately 
        diversity_score = await fitness_evaluator.calculate_population_diversity(evaluated_population)
        
        # Should have diversity score
        assert 0 < diversity_score < 1
        
    @pytest.mark.asyncio
    async def test_fitness_evaluator_with_gemini_diversity(self):
        """Test FitnessEvaluator with GeminiDiversityCalculator."""
        # Mock LLM provider
        mock_provider = MagicMock()
        mock_provider.get_embeddings = AsyncMock()
        
        # Mock embeddings that show semantic similarity
        import numpy as np
        embeddings = np.array([
            [0.9, 0.1, 0.0],  # "machine learning"
            [0.85, 0.15, 0.0],  # "deep learning" (similar)
            [0.0, 0.0, 1.0]   # "quantum computing" (different)
        ])
        
        from mad_spark_alt.core.llm_provider import EmbeddingResponse
        mock_provider.get_embeddings.return_value = EmbeddingResponse(
            embeddings=embeddings.tolist(),
            model="gemini-embedding-001",
            usage={"total_tokens": 100}
        )
        
        gemini_calculator = GeminiDiversityCalculator(llm_provider=mock_provider)
        fitness_evaluator = FitnessEvaluator(
            unified_evaluator=self.mock_unified_evaluator,
            diversity_calculator=gemini_calculator
        )
        
        # Create test population with semantically similar ideas
        population = [
            self._create_individual("machine learning model"),
            self._create_individual("deep learning network"),  # Similar to first
            self._create_individual("quantum computing system")  # Different
        ]
        
        # Evaluate population
        evaluated_population = await fitness_evaluator.evaluate_population(
            [idea.idea for idea in population], self.config
        )
        
        # Calculate diversity separately 
        diversity_score = await fitness_evaluator.calculate_population_diversity(evaluated_population)
        
        # Should have diversity score
        assert 0 < diversity_score < 1
        # API should have been called
        mock_provider.get_embeddings.assert_called_once()
        
    @pytest.mark.asyncio
    async def test_fitness_evaluator_default_diversity(self):
        """Test FitnessEvaluator uses JaccardDiversityCalculator by default."""
        fitness_evaluator = FitnessEvaluator(
            unified_evaluator=self.mock_unified_evaluator
            # No diversity_calculator provided
        )
        
        # Should have a default calculator
        assert fitness_evaluator.diversity_calculator is not None
        assert isinstance(fitness_evaluator.diversity_calculator, JaccardDiversityCalculator)
        
    @pytest.mark.asyncio
    async def test_fitness_evaluator_fallback_on_api_failure(self):
        """Test FitnessEvaluator handles API failure gracefully."""
        # Mock LLM provider that fails
        mock_provider = MagicMock()
        mock_provider.get_embeddings = AsyncMock(side_effect=Exception("API Error"))
        
        # Create fallback calculator
        fallback_calculator = JaccardDiversityCalculator()
        
        gemini_calculator = GeminiDiversityCalculator(llm_provider=mock_provider)
        fitness_evaluator = FitnessEvaluator(
            unified_evaluator=self.mock_unified_evaluator,
            diversity_calculator=gemini_calculator,
            fallback_diversity_calculator=fallback_calculator
        )
        
        # Create test population
        population = [
            self._create_individual("idea one"),
            self._create_individual("idea two"),
            self._create_individual("idea three")
        ]
        
        # Evaluate population - should use fallback
        evaluated_population = await fitness_evaluator.evaluate_population(
            [idea.idea for idea in population], self.config
        )
        
        # Calculate diversity - should use fallback
        diversity_score = await fitness_evaluator.calculate_population_diversity(evaluated_population)
        
        # Should still have diversity score (from fallback)
        assert 0 < diversity_score <= 1
        # API should have been attempted
        mock_provider.get_embeddings.assert_called_once()