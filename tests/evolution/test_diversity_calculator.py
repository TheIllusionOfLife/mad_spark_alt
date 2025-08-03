"""
Test suite for diversity calculator implementations.

This module tests the DiversityCalculator interface and its implementations,
ensuring they correctly calculate population diversity using different strategies.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from typing import List

from mad_spark_alt.core import GeneratedIdea, ThinkingMethod
from mad_spark_alt.evolution.interfaces import IndividualFitness
from mad_spark_alt.evolution.diversity_calculator import DiversityCalculator
from mad_spark_alt.evolution.jaccard_diversity import JaccardDiversityCalculator
from mad_spark_alt.evolution.gemini_diversity import GeminiDiversityCalculator


class TestDiversityCalculatorInterface:
    """Test the DiversityCalculator interface compliance."""
    
    @pytest.mark.asyncio
    @pytest.mark.asyncio
    async def test_interface_is_abstract(self):
        """Test that DiversityCalculator cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            DiversityCalculator()
    
    @pytest.mark.asyncio
    @pytest.mark.asyncio
    async def test_implementations_follow_interface(self):
        """Test that all implementations properly implement the interface."""
        # JaccardDiversityCalculator should implement calculate_diversity
        jaccard = JaccardDiversityCalculator()
        assert hasattr(jaccard, 'calculate_diversity')
        assert callable(jaccard.calculate_diversity)
        
        # GeminiDiversityCalculator should implement calculate_diversity
        mock_provider = MagicMock()
        gemini = GeminiDiversityCalculator(llm_provider=mock_provider)
        assert hasattr(gemini, 'calculate_diversity')
        assert callable(gemini.calculate_diversity)


class TestJaccardDiversityCalculator:
    """Test the Jaccard-based diversity calculator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.calculator = JaccardDiversityCalculator()
        
    def _create_individual(self, content: str) -> IndividualFitness:
        """Helper to create an IndividualFitness instance."""
        idea = GeneratedIdea(
            content=content,
            thinking_method=ThinkingMethod.QUESTIONING,
            agent_name="test_agent",
            generation_prompt="Test prompt"
        )
        return IndividualFitness(
            idea=idea,
            impact=0.5,
            feasibility=0.5,
            accessibility=0.5,
            sustainability=0.5,
            scalability=0.5,
            overall_fitness=0.5
        )
    
    @pytest.mark.asyncio
    @pytest.mark.asyncio
    async def test_empty_population_returns_one(self):
        """Test that empty population has maximum diversity."""
        result = await self.calculator.calculate_diversity([])
        assert result == 1.0
    
    @pytest.mark.asyncio
    @pytest.mark.asyncio
    async def test_single_individual_returns_one(self):
        """Test that single individual has maximum diversity."""
        population = [self._create_individual("test idea")]
        result = await self.calculator.calculate_diversity(population)
        assert result == 1.0
    
    @pytest.mark.asyncio
    async def test_identical_ideas_return_zero(self):
        """Test that identical ideas have zero diversity."""
        population = [
            self._create_individual("same idea"),
            self._create_individual("same idea"),
            self._create_individual("same idea")
        ]
        result = await self.calculator.calculate_diversity(population)
        assert result == 0.0
    
    @pytest.mark.asyncio
    async def test_completely_different_ideas_return_one(self):
        """Test that completely different ideas have maximum diversity."""
        population = [
            self._create_individual("quantum computing breakthrough"),
            self._create_individual("sustainable agriculture methods"),
            self._create_individual("underwater city construction")
        ]
        result = await self.calculator.calculate_diversity(population)
        assert result > 0.9  # Should be close to 1.0
    
    @pytest.mark.asyncio
    async def test_partial_overlap_returns_intermediate_value(self):
        """Test that partial overlap returns intermediate diversity."""
        population = [
            self._create_individual("machine learning for healthcare"),
            self._create_individual("deep learning for medical diagnosis"),
            self._create_individual("blockchain for supply chain")
        ]
        result = await self.calculator.calculate_diversity(population)
        assert 0.3 < result < 0.85  # Some overlap but not identical


class TestGeminiDiversityCalculator:
    """Test the Gemini embedding-based diversity calculator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_provider = MagicMock()
        self.mock_provider.get_embeddings = AsyncMock()
        self.calculator = GeminiDiversityCalculator(llm_provider=self.mock_provider)
        
    def _create_individual(self, content: str) -> IndividualFitness:
        """Helper to create an IndividualFitness instance."""
        idea = GeneratedIdea(
            content=content,
            thinking_method=ThinkingMethod.QUESTIONING,
            agent_name="test_agent",
            generation_prompt="Test prompt"
        )
        return IndividualFitness(
            idea=idea,
            impact=0.5,
            feasibility=0.5,
            accessibility=0.5,
            sustainability=0.5,
            scalability=0.5,
            overall_fitness=0.5
        )
    
    @pytest.mark.asyncio
    @pytest.mark.asyncio
    async def test_empty_population_returns_one(self):
        """Test that empty population has maximum diversity."""
        result = await self.calculator.calculate_diversity([])
        assert result == 1.0
        # Should not call API for empty population
        self.mock_provider.get_embeddings.assert_not_called()
    
    @pytest.mark.asyncio
    @pytest.mark.asyncio
    async def test_single_individual_returns_one(self):
        """Test that single individual has maximum diversity."""
        population = [self._create_individual("test idea")]
        result = await self.calculator.calculate_diversity(population)
        assert result == 1.0
        # Should not call API for single individual
        self.mock_provider.get_embeddings.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_semantic_similarity_detection(self):
        """Test that semantically similar ideas are detected."""
        # Create ideas that are semantically similar but use different words
        population = [
            self._create_individual("reduce carbon emissions through renewable energy"),
            self._create_individual("lower greenhouse gases using sustainable power"),
            self._create_individual("quantum computing for cryptography")
        ]
        
        # Mock embeddings: first two are similar, third is different
        import numpy as np
        embeddings = np.array([
            [0.9, 0.1, 0.0],  # First idea
            [0.85, 0.15, 0.0],  # Second idea (similar to first)
            [0.0, 0.0, 1.0]   # Third idea (different)
        ])
        
        from mad_spark_alt.core.llm_provider import EmbeddingResponse
        self.mock_provider.get_embeddings.return_value = EmbeddingResponse(
            embeddings=embeddings.tolist(),
            model="text-embedding-004",
            usage={"total_tokens": 100}
        )
        
        result = await self.calculator.calculate_diversity(population)
        # Should detect that first two are similar
        assert 0.4 < result < 0.7
        
    @pytest.mark.asyncio
    async def test_caching_prevents_duplicate_api_calls(self):
        """Test that embeddings are cached to prevent duplicate API calls."""
        population = [
            self._create_individual("test idea 1"),
            self._create_individual("test idea 2"),
            self._create_individual("test idea 1")  # Duplicate
        ]
        
        # Mock embeddings
        import numpy as np
        embeddings = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0]  # Same as first
        ])
        
        from mad_spark_alt.core.llm_provider import EmbeddingResponse
        self.mock_provider.get_embeddings.return_value = EmbeddingResponse(
            embeddings=embeddings[:2].tolist(),  # Only return 2 embeddings
            model="text-embedding-004",
            usage={"total_tokens": 50}
        )
        
        result = await self.calculator.calculate_diversity(population)
        
        # Should only call API once with 2 unique texts
        self.mock_provider.get_embeddings.assert_called_once()
        call_args = self.mock_provider.get_embeddings.call_args[0][0]
        assert len(call_args.texts) == 2  # Only unique texts
        
    @pytest.mark.asyncio
    async def test_long_text_handling(self):
        """Test handling of very long texts."""
        # Create a very long text (but under 2048 token limit)
        long_text = " ".join(["sustainable development"] * 500)
        population = [
            self._create_individual(long_text),
            self._create_individual("short idea")
        ]
        
        import numpy as np
        embeddings = np.array([
            [1.0, 0.0],
            [0.0, 1.0]
        ])
        
        from mad_spark_alt.core.llm_provider import EmbeddingResponse
        self.mock_provider.get_embeddings.return_value = EmbeddingResponse(
            embeddings=embeddings.tolist(),
            model="text-embedding-004",
            usage={"total_tokens": 2000}
        )
        
        result = await self.calculator.calculate_diversity(population)
        assert 0.9 < result <= 1.0  # Different ideas
        
    @pytest.mark.asyncio
    async def test_api_failure_raises_exception(self):
        """Test that API failures are propagated."""
        population = [
            self._create_individual("idea 1"),
            self._create_individual("idea 2")
        ]
        
        # Mock API failure
        self.mock_provider.get_embeddings.side_effect = Exception("API Error")
        
        with pytest.raises(Exception, match="API Error"):
            await self.calculator.calculate_diversity(population)