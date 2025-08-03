"""
Test suite for diversity configuration in the evolution system.

This module tests that the evolution system correctly configures diversity
calculators based on the EvolutionConfig.diversity_method setting.
"""

import pytest
from unittest.mock import MagicMock

from mad_spark_alt.evolution.fitness import FitnessEvaluator, create_diversity_calculator
from mad_spark_alt.evolution.interfaces import DiversityMethod
from mad_spark_alt.evolution.jaccard_diversity import JaccardDiversityCalculator
from mad_spark_alt.evolution.gemini_diversity import GeminiDiversityCalculator


class TestDiversityCalculatorFactory:
    """Test the diversity calculator factory function."""
    
    def test_create_jaccard_calculator(self):
        """Test creating Jaccard diversity calculator."""
        calculator = create_diversity_calculator(DiversityMethod.JACCARD)
        assert isinstance(calculator, JaccardDiversityCalculator)
        
    def test_create_semantic_calculator_with_provider(self):
        """Test creating semantic diversity calculator with LLM provider."""
        mock_provider = MagicMock()
        calculator = create_diversity_calculator(DiversityMethod.SEMANTIC, mock_provider)
        assert isinstance(calculator, GeminiDiversityCalculator)
        assert calculator.llm_provider is mock_provider
        
    def test_create_semantic_calculator_without_provider_raises_error(self):
        """Test that semantic calculator requires LLM provider."""
        with pytest.raises(ValueError, match="LLM provider required"):
            create_diversity_calculator(DiversityMethod.SEMANTIC)
            
    def test_create_unknown_method_raises_error(self):
        """Test that unknown method raises error."""
        with pytest.raises(ValueError, match="Unknown diversity method"):
            create_diversity_calculator("unknown_method")


class TestFitnessEvaluatorConfiguration:
    """Test FitnessEvaluator diversity configuration."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.fitness_evaluator = FitnessEvaluator()
        
    def test_default_diversity_calculator(self):
        """Test that default diversity calculator is Jaccard."""
        assert isinstance(self.fitness_evaluator.diversity_calculator, JaccardDiversityCalculator)
        assert self.fitness_evaluator.fallback_diversity_calculator is None
        
    def test_configure_jaccard_method(self):
        """Test configuring Jaccard diversity method."""
        self.fitness_evaluator.configure_diversity_method(DiversityMethod.JACCARD)
        
        assert isinstance(self.fitness_evaluator.diversity_calculator, JaccardDiversityCalculator)
        assert self.fitness_evaluator.fallback_diversity_calculator is None
        
    def test_configure_semantic_method(self):
        """Test configuring semantic diversity method."""
        mock_provider = MagicMock()
        self.fitness_evaluator.configure_diversity_method(DiversityMethod.SEMANTIC, mock_provider)
        
        assert isinstance(self.fitness_evaluator.diversity_calculator, GeminiDiversityCalculator)
        assert isinstance(self.fitness_evaluator.fallback_diversity_calculator, JaccardDiversityCalculator)
        
    def test_configure_semantic_method_without_provider_raises_error(self):
        """Test that semantic method requires LLM provider."""
        with pytest.raises(ValueError, match="LLM provider required"):
            self.fitness_evaluator.configure_diversity_method(DiversityMethod.SEMANTIC)