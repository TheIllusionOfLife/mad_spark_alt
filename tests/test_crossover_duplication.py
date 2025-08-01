"""
Tests for semantic crossover duplication bug.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from difflib import SequenceMatcher

from mad_spark_alt.core.interfaces import GeneratedIdea
from mad_spark_alt.evolution.semantic_operators import SemanticCrossoverOperator
from mad_spark_alt.core.llm_provider import GoogleProvider, LLMResponse


class TestCrossoverDuplication:
    """Test that semantic crossover produces distinct offspring."""
    
    def test_offspring_similarity_detection(self):
        """Test helper function to detect similar content between offspring."""
        # Test content that has high similarity (problematic)
        offspring1 = """
        Resource management becomes a fascinating challenge as supply chains, 
        transportation networks, and environmental effects seamlessly wrap around 
        the entire surface. For example, pollution generated on one perceived "side" 
        will eventually loop back to affect the "other" side.
        """
        
        offspring2 = """
        Resource management becomes a fascinating challenge as supply chains, 
        transportation networks, and environmental effects seamlessly wrap around 
        the entire surface. For example, pollution generated on one perceived "side" 
        will eventually loop back to affect the "other" side. This could also 
        incorporate user feedback mechanisms.
        """
        
        # Calculate similarity
        similarity = SequenceMatcher(None, offspring1.strip(), offspring2.strip()).ratio()
        
        # High similarity indicates duplication problem
        assert similarity > 0.8  # More than 80% similar
        
        # Test content that should have low similarity (good)
        distinct_offspring1 = """
        This approach focuses on individual user control and customization, 
        allowing each person to tailor the system to their specific needs 
        through advanced configuration options and personal preferences.
        """
        
        distinct_offspring2 = """
        Community-driven development emphasizes collaborative decision-making 
        and shared resources, building consensus through democratic processes 
        and collective ownership of outcomes.
        """
        
        distinct_similarity = SequenceMatcher(None, distinct_offspring1.strip(), distinct_offspring2.strip()).ratio()
        
        # Low similarity indicates good diversity
        assert distinct_similarity < 0.4  # Less than 40% similar
        
    @pytest.mark.asyncio
    async def test_crossover_produces_distinct_offspring(self):
        """Test that semantic crossover produces distinct offspring."""
        mock_provider = Mock(spec=GoogleProvider)
        
        # Mock response with distinct offspring
        crossover_response = LLMResponse(
            content="""{
                "offspring_1": "Individual-focused solution: This approach empowers each user with personalized tools and customizable interfaces, allowing them to adapt the system to their unique workflow and preferences. Implementation involves creating modular components, user preference systems, and flexible APIs that can be configured per user.",
                "offspring_2": "Community-based framework: This solution leverages collective intelligence and shared resources, building platforms for collaboration and knowledge sharing. Implementation requires developing group management tools, consensus mechanisms, and shared resource allocation systems."
            }""",
            cost=0.002,
            model="gemini-pro", 
            provider="google"
        )
        
        mock_provider.generate = AsyncMock(return_value=crossover_response)
        
        # Create test parents
        parent1 = GeneratedIdea(
            content="Personal productivity system with individual customization",
            thinking_method="abduction",
            agent_name="test",
            generation_prompt="test"
        )
        
        parent2 = GeneratedIdea(
            content="Community platform for shared knowledge and collaboration",
            thinking_method="deduction", 
            agent_name="test",
            generation_prompt="test"
        )
        
        # Create crossover operator
        operator = SemanticCrossoverOperator(mock_provider)
        
        # Perform crossover
        offspring1, offspring2 = await operator.crossover(parent1, parent2, "productivity")
        
        # Check that offspring are distinct
        similarity = SequenceMatcher(
            None, 
            offspring1.content.strip(), 
            offspring2.content.strip()
        ).ratio()
        
        # Should be less than 50% similar
        assert similarity < 0.5, f"Offspring too similar: {similarity:.2f}"
        
        # Check that both offspring have substantial content
        assert len(offspring1.content) > 100
        assert len(offspring2.content) > 100
        
        # Check that both offspring integrate concepts from both parents
        assert any(concept in offspring1.content.lower() for concept in ["individual", "personal", "user"])
        assert any(concept in offspring2.content.lower() for concept in ["community", "shared", "collaboration"])
        
    @pytest.mark.asyncio
    async def test_crossover_with_duplicated_content_fallback(self):
        """Test that crossover handles LLM responses with duplicated content."""
        mock_provider = Mock(spec=GoogleProvider)
        
        # Mock response with problematic duplication
        duplicated_response = LLMResponse(
            content="""{
                "offspring_1": "Resource management becomes a challenge as supply chains wrap around the surface. Enhanced variation exploring alternative approaches.",
                "offspring_2": "Resource management becomes a challenge as supply chains wrap around the surface. This could also incorporate user feedback mechanisms."
            }""",
            cost=0.002,
            model="gemini-pro",
            provider="google"
        )
        
        mock_provider.generate = AsyncMock(return_value=duplicated_response)
        
        parent1 = GeneratedIdea(
            content="City building game on mobius strip",
            thinking_method="abduction",
            agent_name="test",
            generation_prompt="test"
        )
        
        parent2 = GeneratedIdea(
            content="Rhythm game with musical elements",
            thinking_method="deduction",
            agent_name="test", 
            generation_prompt="test"
        )
        
        operator = SemanticCrossoverOperator(mock_provider)
        
        # Perform crossover - should detect duplication and handle it
        offspring1, offspring2 = await operator.crossover(parent1, parent2, "gaming")
        
        # Calculate similarity
        similarity = SequenceMatcher(
            None,
            offspring1.content.strip(),
            offspring2.content.strip()
        ).ratio()
        
        # If similarity is too high, the operator should have used fallback
        if similarity > 0.7:
            # Check that at least one offspring used fallback text
            assert ("[FALLBACK TEXT]" in offspring1.content or 
                   "[FALLBACK TEXT]" in offspring2.content)
        
    def test_detect_content_duplication(self):
        """Test function to detect content duplication between offspring."""
        
        def has_excessive_duplication(content1: str, content2: str, threshold: float = 0.7) -> bool:
            """Check if two content strings have excessive duplication."""
            # Normalize content
            norm1 = content1.strip().lower()
            norm2 = content2.strip().lower()
            
            # Calculate similarity ratio
            similarity = SequenceMatcher(None, norm1, norm2).ratio()
            
            return similarity > threshold
        
        # Test with problematic duplication
        duplicate_content1 = "Resource management becomes a fascinating challenge as supply chains wrap around."
        duplicate_content2 = "Resource management becomes a fascinating challenge as supply chains wrap around. Additional features."
        
        assert has_excessive_duplication(duplicate_content1, duplicate_content2)
        
        # Test with acceptable diversity
        diverse_content1 = "Focus on individual customization and personal workflow optimization."
        diverse_content2 = "Emphasize community collaboration and shared resource management."
        
        assert not has_excessive_duplication(diverse_content1, diverse_content2)