"""
Tests for evolution diversity and mutation fixes.

This module tests that the genetic evolution system produces diverse results
and properly handles mutation to avoid duplicate ideas.
"""

import pytest
from datetime import datetime, timezone

from mad_spark_alt.core.interfaces import GeneratedIdea, ThinkingMethod
from mad_spark_alt.evolution.operators import MutationOperator
from mad_spark_alt.evolution.genetic_algorithm import GeneticAlgorithm
from mad_spark_alt.evolution.interfaces import (
    EvolutionConfig,
    EvolutionRequest,
    IndividualFitness,
)


@pytest.fixture
def sample_idea():
    """Create a sample idea for testing."""
    return GeneratedIdea(
        content="Test idea for reducing plastic waste through community action",
        thinking_method=ThinkingMethod.QUESTIONING,
        agent_name="TestAgent",
        generation_prompt="Test prompt",
        confidence_score=0.8,
        reasoning="Test reasoning",
        parent_ideas=[],
        metadata={"generation": 0},
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@pytest.mark.asyncio
async def test_mutation_always_creates_new_object(sample_idea):
    """Test that mutation operator always returns a new object."""
    operator = MutationOperator()
    
    # Test with 0% mutation rate (no mutation should occur)
    mutated = await operator.mutate(sample_idea, mutation_rate=0.0)
    
    # Should be different object
    assert mutated is not sample_idea
    assert id(mutated) != id(sample_idea)
    
    # Content should be the same
    assert mutated.content == sample_idea.content
    
    # Generation should be incremented
    assert mutated.metadata.get("generation") == 1
    assert mutated.metadata.get("mutation_type") == "none"


@pytest.mark.asyncio
async def test_mutation_with_high_rate_changes_content(sample_idea):
    """Test that mutation with high rate actually changes content."""
    operator = MutationOperator()
    
    # Test with 100% mutation rate
    mutated = await operator.mutate(sample_idea, mutation_rate=1.0)
    
    # Should be different object
    assert mutated is not sample_idea
    
    # Generation should be incremented
    assert mutated.metadata.get("generation") == 1
    
    # Should have a mutation type
    assert mutated.metadata.get("mutation_type") != "none"
    assert mutated.metadata.get("mutation_type") in [
        "word_substitution",
        "phrase_reordering",
        "concept_addition",
        "concept_removal",
        "emphasis_change",
    ]


def test_extract_best_ideas_deduplicates():
    """Test that _extract_best_ideas removes duplicates."""
    ga = GeneticAlgorithm()
    
    # Create population with duplicates
    idea1 = GeneratedIdea(
        content="Idea about plastic reduction",
        thinking_method=ThinkingMethod.QUESTIONING,
        agent_name="Agent1",
        generation_prompt="",
        confidence_score=0.9,
        reasoning="",
        parent_ideas=[],
        metadata={},
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
    
    idea2 = GeneratedIdea(
        content="Idea about plastic reduction",  # Same content
        thinking_method=ThinkingMethod.ABDUCTION,
        agent_name="Agent2",
        generation_prompt="",
        confidence_score=0.8,
        reasoning="",
        parent_ideas=[],
        metadata={},
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
    
    idea3 = GeneratedIdea(
        content="Different idea about recycling",
        thinking_method=ThinkingMethod.DEDUCTION,
        agent_name="Agent3",
        generation_prompt="",
        confidence_score=0.7,
        reasoning="",
        parent_ideas=[],
        metadata={},
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
    
    # Create fitness scores
    population = [
        IndividualFitness(
            idea=idea1,
            overall_fitness=0.9,
            evaluation_metadata={},
        ),
        IndividualFitness(
            idea=idea2,
            overall_fitness=0.85,  # Different fitness but same content
            evaluation_metadata={},
        ),
        IndividualFitness(
            idea=idea3,
            overall_fitness=0.7,
            evaluation_metadata={},
        ),
    ]
    
    # Extract best ideas
    best_ideas = ga._extract_best_ideas(population, n=3)
    
    # Should only have 2 unique ideas
    assert len(best_ideas) == 2
    assert best_ideas[0].content == "Idea about plastic reduction"
    assert best_ideas[1].content == "Different idea about recycling"


