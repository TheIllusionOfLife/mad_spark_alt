"""
Tests for smart operator selection logic.

This module tests the logic that determines when to use semantic (LLM-powered)
operators vs traditional operators based on population diversity and performance.
"""

import pytest
from mad_spark_alt.evolution.interfaces import IndividualFitness, EvolutionConfig
from mad_spark_alt.evolution.smart_selection import SmartOperatorSelector
from mad_spark_alt.core.interfaces import GeneratedIdea


class TestSmartOperatorSelector:
    """Test smart operator selection logic."""

    @pytest.fixture
    def sample_individual(self):
        """Create a sample individual for testing."""
        idea = GeneratedIdea(
            content="Test idea content",
            thinking_method="test_method",
            agent_name="test_agent",
            generation_prompt="test prompt",
            confidence_score=0.8
        )
        return IndividualFitness(
            idea=idea,
            creativity_score=0.7,
            diversity_score=0.6,
            quality_score=0.8,
            overall_fitness=0.7
        )

    def test_low_diversity_triggers_semantic_operators(self, sample_individual):
        """Test that low population diversity triggers semantic operators."""
        selector = SmartOperatorSelector()
        
        # Low diversity should trigger semantic operators
        should_use = selector.should_use_semantic_mutation(
            individual=sample_individual,
            population_diversity=0.3,  # Below threshold
            generation=1
        )
        assert should_use is True

    def test_high_diversity_avoids_semantic_operators(self, sample_individual):
        """Test that high population diversity avoids semantic operators."""
        selector = SmartOperatorSelector()
        
        # High diversity should not trigger semantic operators
        should_use = selector.should_use_semantic_mutation(
            individual=sample_individual,
            population_diversity=0.8,  # Above threshold
            generation=1
        )
        assert should_use is False

    def test_diversity_threshold_boundary(self, sample_individual):
        """Test behavior at diversity threshold boundary."""
        selector = SmartOperatorSelector()
        
        # At threshold (0.5)
        should_use = selector.should_use_semantic_mutation(
            individual=sample_individual,
            population_diversity=0.5,  # At threshold
            generation=1
        )
        assert should_use is False  # Should not use at exact threshold

    def test_low_fitness_individuals_ignored(self):
        """Test that low fitness individuals don't use semantic operators."""
        # Create low fitness individual
        idea = GeneratedIdea(
            content="Poor idea",
            thinking_method="test_method",
            agent_name="test_agent",
            generation_prompt="test prompt",
            confidence_score=0.3
        )
        low_fitness_ind = IndividualFitness(
            idea=idea,
            overall_fitness=0.3  # Below performance threshold
        )
        
        selector = SmartOperatorSelector()
        should_use = selector.should_use_semantic_mutation(
            individual=low_fitness_ind,
            population_diversity=0.3,  # Low diversity
            generation=1
        )
        assert should_use is False

    def test_generation_based_probability_increase(self, sample_individual):
        """Test that probability increases with generation number."""
        selector = SmartOperatorSelector()
        
        # Test with low diversity to ensure base conditions are met
        uses_in_early_gen = []
        uses_in_late_gen = []
        
        # Run multiple times to test probability
        for _ in range(100):
            early = selector.should_use_semantic_mutation(
                individual=sample_individual,
                population_diversity=0.3,
                generation=1
            )
            uses_in_early_gen.append(early)
            
            late = selector.should_use_semantic_mutation(
                individual=sample_individual,
                population_diversity=0.3,
                generation=4
            )
            uses_in_late_gen.append(late)
        
        # Later generations should have higher probability
        early_rate = sum(uses_in_early_gen) / len(uses_in_early_gen)
        late_rate = sum(uses_in_late_gen) / len(uses_in_late_gen)
        
        # Allow for some randomness, but late should be higher
        assert late_rate >= early_rate

    def test_custom_configuration(self, sample_individual):
        """Test selector with custom configuration."""
        config = EvolutionConfig(
            semantic_operator_threshold=0.7,  # Higher threshold
            use_semantic_operators=True
        )
        
        selector = SmartOperatorSelector(config=config)
        
        # Diversity below new threshold
        should_use = selector.should_use_semantic_mutation(
            individual=sample_individual,
            population_diversity=0.6,  # Below 0.7 threshold
            generation=1
        )
        assert should_use is True
        
        # Diversity above new threshold
        should_use = selector.should_use_semantic_mutation(
            individual=sample_individual,
            population_diversity=0.8,  # Above 0.7 threshold
            generation=1
        )
        assert should_use is False

    def test_semantic_operators_disabled(self, sample_individual):
        """Test when semantic operators are disabled in config."""
        config = EvolutionConfig(use_semantic_operators=False)
        selector = SmartOperatorSelector(config=config)
        
        # Should never use semantic operators when disabled
        should_use = selector.should_use_semantic_mutation(
            individual=sample_individual,
            population_diversity=0.1,  # Very low diversity
            generation=5  # Late generation
        )
        assert should_use is False

    def test_should_use_semantic_crossover(self):
        """Test crossover selection logic."""
        selector = SmartOperatorSelector()
        
        # Create two high-fitness individuals
        idea1 = GeneratedIdea(
            content="Idea 1",
            thinking_method="method1",
            agent_name="agent1",
            generation_prompt="prompt1"
        )
        idea2 = GeneratedIdea(
            content="Idea 2", 
            thinking_method="method2",
            agent_name="agent2",
            generation_prompt="prompt2"
        )
        
        parent1 = IndividualFitness(idea=idea1, overall_fitness=0.8)
        parent2 = IndividualFitness(idea=idea2, overall_fitness=0.7)
        
        # Low diversity should trigger semantic crossover
        should_use = selector.should_use_semantic_crossover(
            parent1=parent1,
            parent2=parent2,
            population_diversity=0.3
        )
        assert should_use is True
        
        # High diversity should not trigger
        should_use = selector.should_use_semantic_crossover(
            parent1=parent1,
            parent2=parent2,
            population_diversity=0.8
        )
        assert should_use is False

    def test_minimum_fitness_threshold_for_crossover(self):
        """Test that both parents need minimum fitness for semantic crossover."""
        selector = SmartOperatorSelector()
        
        # One low fitness parent
        good_parent = IndividualFitness(
            idea=GeneratedIdea("Good", "method", "agent", "prompt"),
            overall_fitness=0.8
        )
        poor_parent = IndividualFitness(
            idea=GeneratedIdea("Poor", "method", "agent", "prompt"),
            overall_fitness=0.2
        )
        
        should_use = selector.should_use_semantic_crossover(
            parent1=good_parent,
            parent2=poor_parent,
            population_diversity=0.3
        )
        assert should_use is False

    def test_probability_based_selection(self, sample_individual):
        """Test that selection is probability-based, not deterministic."""
        selector = SmartOperatorSelector()
        
        # Run multiple times with conditions that give ~50% probability
        results = []
        for _ in range(100):
            result = selector.should_use_semantic_mutation(
                individual=sample_individual,
                population_diversity=0.4,  # Below threshold
                generation=2  # Middle generation
            )
            results.append(result)
        
        # Should have mix of True and False
        true_count = sum(results)
        assert 20 < true_count < 80  # Roughly 20-80% range