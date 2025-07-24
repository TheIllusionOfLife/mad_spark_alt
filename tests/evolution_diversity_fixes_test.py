"""
Tests for evolution system diversity and semantic operator fixes.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from mad_spark_alt.evolution.smart_selection import SmartOperatorSelector
from mad_spark_alt.evolution.interfaces import EvolutionConfig, IndividualFitness
from mad_spark_alt.evolution.fitness import FitnessEvaluator
from mad_spark_alt.core.interfaces import GeneratedIdea, ThinkingMethod


@pytest.fixture
def sample_population():
    """Create a sample population with varying content."""
    return [
        IndividualFitness(
            idea=GeneratedIdea(
                content="Create educational campaigns about sustainability",
                thinking_method=ThinkingMethod.ABDUCTION,
                agent_name="test",
                generation_prompt="test",
            ),
            overall_fitness=0.8,
        ),
        IndividualFitness(
            idea=GeneratedIdea(
                content="Build green infrastructure for cities",
                thinking_method=ThinkingMethod.ABDUCTION,
                agent_name="test", 
                generation_prompt="test",
            ),
            overall_fitness=0.7,
        ),
        IndividualFitness(
            idea=GeneratedIdea(
                content="Develop renewable energy systems",
                thinking_method=ThinkingMethod.ABDUCTION,
                agent_name="test",
                generation_prompt="test",
            ),
            overall_fitness=0.9,
        ),
        IndividualFitness(
            idea=GeneratedIdea(
                content="Create educational campaigns about environment",  # Similar to first
                thinking_method=ThinkingMethod.ABDUCTION,
                agent_name="test",
                generation_prompt="test",
            ),
            overall_fitness=0.75,
        ),
    ]


def test_smart_selector_more_aggressive():
    """Test that smart selector has more aggressive settings for semantic operators."""
    config = EvolutionConfig()
    selector = SmartOperatorSelector(config)
    
    # Check that the new aggressive settings are in place
    assert selector.base_mutation_probability == 0.7, f"Expected 0.7, got {selector.base_mutation_probability}"
    assert selector.base_crossover_probability == 0.8, f"Expected 0.8, got {selector.base_crossover_probability}" 
    assert selector.min_fitness_threshold == 0.3, f"Expected 0.3, got {selector.min_fitness_threshold}"
    assert selector.generation_boost_factor == 0.2, f"Expected 0.2, got {selector.generation_boost_factor}"


def test_semantic_threshold_increased():
    """Test that semantic operator threshold has been increased."""
    config = EvolutionConfig()
    
    # The threshold should now be 0.8 instead of 0.5
    assert config.semantic_operator_threshold == 0.8, f"Expected 0.8, got {config.semantic_operator_threshold}"


def test_smart_selector_semantic_trigger_conditions():
    """Test that semantic operators are triggered more often with new settings."""
    config = EvolutionConfig()
    selector = SmartOperatorSelector(config)
    
    # Create an individual with moderate fitness (should now qualify)
    individual = IndividualFitness(
        idea=GeneratedIdea(
            content="Test idea",
            thinking_method=ThinkingMethod.ABDUCTION,
            agent_name="test",
            generation_prompt="test",
        ),
        overall_fitness=0.4,  # Above new threshold of 0.3
    )
    
    # With low diversity (0.2), semantic operators should be triggered
    low_diversity = 0.2
    
    # Test mutation - should have high probability with new settings
    mutation_triggers = 0
    for _ in range(100):
        if selector.should_use_semantic_mutation(individual, low_diversity, generation=1):
            mutation_triggers += 1
    
    # With 70% base probability + 20% generation boost, expect ~90% trigger rate
    assert mutation_triggers > 80, f"Expected >80 triggers out of 100, got {mutation_triggers}"
    
    # Test crossover - should have high probability 
    crossover_triggers = 0
    for _ in range(100):
        if selector.should_use_semantic_crossover(individual, individual, low_diversity):
            crossover_triggers += 1
    
    # With 80% base probability, expect ~80% trigger rate
    assert crossover_triggers > 70, f"Expected >70 triggers out of 100, got {crossover_triggers}"


@pytest.mark.asyncio
async def test_population_diversity_calculation_improved(sample_population):
    """Test that the new diversity calculation works better than the old one."""
    evaluator = FitnessEvaluator()
    
    # Test with diverse population
    diversity_score = await evaluator.calculate_population_diversity(sample_population)
    
    # Should be between 0 and 1
    assert 0.0 <= diversity_score <= 1.0, f"Diversity score {diversity_score} out of range"
    
    # Should detect that the population has some diversity
    # (even though two ideas are similar, the others are different)
    assert diversity_score > 0.3, f"Expected diversity > 0.3, got {diversity_score}"


@pytest.mark.asyncio
async def test_identical_population_diversity():
    """Test diversity calculation with identical ideas."""
    # Create population with identical content
    identical_population = [
        IndividualFitness(
            idea=GeneratedIdea(
                content="Same idea repeated",
                thinking_method=ThinkingMethod.ABDUCTION,
                agent_name="test",
                generation_prompt="test",
            ),
            overall_fitness=0.8,
        ),
        IndividualFitness(
            idea=GeneratedIdea(
                content="Same idea repeated",
                thinking_method=ThinkingMethod.ABDUCTION,
                agent_name="test",
                generation_prompt="test",
            ),
            overall_fitness=0.7,
        ),
    ]
    
    evaluator = FitnessEvaluator()
    diversity_score = await evaluator.calculate_population_diversity(identical_population)
    
    # Should be very low diversity for identical content
    assert diversity_score < 0.1, f"Expected diversity < 0.1 for identical ideas, got {diversity_score}"


def test_evolution_config_better_defaults():
    """Test that evolution config has better default settings for diversity."""
    config = EvolutionConfig()
    
    # Check that default elite size is reasonable for small populations
    assert config.elite_size == 1, f"Expected elite_size=1, got {config.elite_size}"
    
    # Check that semantic operators are enabled by default
    assert config.use_semantic_operators == True, f"Expected semantic operators enabled"


def test_deduplication_logic():
    """Test the deduplication logic used in CLI results."""
    # Simulate the deduplication logic from CLI
    ideas = [
        "Create educational campaigns about sustainability",
        "Build green infrastructure for cities", 
        "Create educational campaigns about environment",  # Similar to first
        "Develop renewable energy systems",
        "Educational campaigns for sustainability awareness",  # Similar to first
    ]
    
    unique_contents = []
    seen_contents = []
    
    for content in ideas:
        content_lower = content.lower().strip()
        
        # Check similarity
        is_duplicate = False
        for seen_content in seen_contents:
            words1 = set(content_lower.split())
            words2 = set(seen_content.split())
            
            if len(words1) > 0 and len(words2) > 0:
                intersection = len(words1.intersection(words2))
                union = len(words1.union(words2))
                similarity = intersection / union
                
                if similarity > 0.6:
                    is_duplicate = True
                    break
        
        if not is_duplicate:
            unique_contents.append(content)
            seen_contents.append(content_lower)
    
    # Should deduplicate similar ideas
    assert len(unique_contents) == 4, f"Expected 4 unique ideas, got {len(unique_contents)}"
    assert "Create educational campaigns about sustainability" in unique_contents
    assert "Build green infrastructure for cities" in unique_contents  
    assert "Develop renewable energy systems" in unique_contents
    assert "Educational campaigns for sustainability awareness" in unique_contents


@pytest.mark.asyncio 
async def test_semantic_operators_trigger_with_new_threshold():
    """Test that semantic operators trigger more often with the new higher threshold."""
    config = EvolutionConfig()
    selector = SmartOperatorSelector(config)
    
    # Create individuals with moderate fitness
    individual = IndividualFitness(
        idea=GeneratedIdea(
            content="Test idea",
            thinking_method=ThinkingMethod.ABDUCTION,
            agent_name="test",
            generation_prompt="test",
        ),
        overall_fitness=0.4,
    )
    
    # Test with diversity that would have disabled semantic operators before (0.6)
    # but should now allow them (threshold is 0.8) 
    medium_diversity = 0.6
    
    # Should now trigger semantic operators
    mutation_allowed = selector.should_use_semantic_mutation(individual, medium_diversity, generation=1)
    crossover_allowed = selector.should_use_semantic_crossover(individual, individual, medium_diversity)
    
    # With the probabilistic nature, we can't guarantee it will always trigger,
    # but it should be possible now (before it would always be False)
    # Run multiple times to check that it can trigger
    can_trigger_mutation = False
    can_trigger_crossover = False
    
    for _ in range(50):
        if selector.should_use_semantic_mutation(individual, medium_diversity, generation=1):
            can_trigger_mutation = True
        if selector.should_use_semantic_crossover(individual, individual, medium_diversity):
            can_trigger_crossover = True
            
        if can_trigger_mutation and can_trigger_crossover:
            break
    
    assert can_trigger_mutation, "Semantic mutation should be able to trigger with diversity=0.6"
    assert can_trigger_crossover, "Semantic crossover should be able to trigger with diversity=0.6"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])