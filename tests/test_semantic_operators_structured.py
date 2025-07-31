"""Tests for semantic operators with proper structured output."""

import pytest
import json
from unittest.mock import AsyncMock, MagicMock
from mad_spark_alt.evolution.semantic_operators import (
    SemanticCrossoverOperator, 
    BatchSemanticMutationOperator,
    get_crossover_schema,
    get_mutation_schema
)
from mad_spark_alt.core.interfaces import GeneratedIdea, ThinkingMethod
from mad_spark_alt.core.llm_provider import LLMResponse


class TestSemanticOperatorsStructured:
    """Test semantic operators with proper Gemini structured output."""
    
    def test_crossover_schema_format(self):
        """Test that crossover schema matches Gemini format."""
        schema = get_crossover_schema()
        
        # Verify schema structure
        assert schema["type"] == "OBJECT"
        assert "properties" in schema
        assert "offspring_1" in schema["properties"]
        assert "offspring_2" in schema["properties"]
        assert schema["properties"]["offspring_1"]["type"] == "STRING"
        assert schema["properties"]["offspring_2"]["type"] == "STRING"
        assert "required" in schema
        assert "offspring_1" in schema["required"]
        assert "offspring_2" in schema["required"]
    
    def test_mutation_schema_format(self):
        """Test that mutation schema matches Gemini format."""
        schema = get_mutation_schema()
        
        # Verify schema structure
        assert schema["type"] == "OBJECT"
        assert "properties" in schema
        assert "mutations" in schema["properties"]
        assert schema["properties"]["mutations"]["type"] == "ARRAY"
        assert "items" in schema["properties"]["mutations"]
        assert schema["properties"]["mutations"]["items"]["type"] == "OBJECT"
    
    @pytest.mark.asyncio
    async def test_crossover_with_proper_json_response(self):
        """Test crossover with properly formatted JSON response."""
        mock_llm = MagicMock()
        operator = SemanticCrossoverOperator(llm_provider=mock_llm)
        
        # Create proper JSON response
        json_response = {
            "offspring_1": "This is a detailed first offspring that combines elements from both parents. It integrates the game mechanics from parent 1 with the narrative structure from parent 2, creating a unique hybrid approach. The implementation would involve building a modular system that allows for dynamic story progression based on player actions within the Mobius strip environment. This creates an emergent gameplay experience.",
            "offspring_2": "The second offspring takes a different approach by emphasizing the visual aspects from parent 1 and the puzzle mechanics from parent 2. This creates a visually stunning puzzle game where the Mobius strip itself becomes the canvas for solving increasingly complex challenges. Players would manipulate the strip's properties to unlock new areas and reveal hidden story elements throughout their journey."
        }
        
        mock_response = LLMResponse(
            content=json.dumps(json_response),
            model="gemini-pro",
            provider="google", 
            cost=0.001
        )
        
        mock_llm.generate = AsyncMock(return_value=mock_response)
        
        # Create parent ideas
        parent1 = GeneratedIdea(
            content="Parent 1: Visual game concept",
            thinking_method=ThinkingMethod.QUESTIONING,
            agent_name="test_agent",
            generation_prompt="test",
            confidence_score=0.8
        )
        parent2 = GeneratedIdea(
            content="Parent 2: Puzzle mechanics",
            thinking_method=ThinkingMethod.ABDUCTION,
            agent_name="test_agent",
            generation_prompt="test",
            confidence_score=0.7
        )
        
        # Run crossover
        offspring1, offspring2 = await operator.crossover(parent1, parent2)
        
        # Verify results
        assert offspring1.content == json_response["offspring_1"]
        assert offspring2.content == json_response["offspring_2"]
        assert "[FALLBACK TEXT]" not in offspring1.content
        assert "[FALLBACK TEXT]" not in offspring2.content
        assert len(offspring1.content) > 150
        assert len(offspring2.content) > 150
    
    @pytest.mark.asyncio
    async def test_crossover_fallback_only_on_failure(self):
        """Test that fallback is only used when parsing actually fails."""
        mock_llm = MagicMock()
        operator = SemanticCrossoverOperator(llm_provider=mock_llm)
        
        # Test with malformed JSON
        mock_response = LLMResponse(
            content="This is not JSON at all",
            model="gemini-pro",
            provider="google",
            cost=0.001
        )
        
        mock_llm.generate = AsyncMock(return_value=mock_response)
        
        parent1 = GeneratedIdea(
            content="Parent 1 content",
            thinking_method=ThinkingMethod.QUESTIONING,
            agent_name="test_agent", 
            generation_prompt="test",
            confidence_score=0.8
        )
        parent2 = GeneratedIdea(
            content="Parent 2 content",
            thinking_method=ThinkingMethod.ABDUCTION,
            agent_name="test_agent",
            generation_prompt="test", 
            confidence_score=0.7
        )
        
        # Run crossover
        offspring1, offspring2 = await operator.crossover(parent1, parent2)
        
        # Should use fallback
        assert "[FALLBACK TEXT]" in offspring1.content
        assert "[FALLBACK TEXT]" in offspring2.content
    
    @pytest.mark.asyncio
    async def test_mutation_with_structured_output(self):
        """Test mutation with proper structured output."""
        mock_llm = MagicMock()
        operator = BatchSemanticMutationOperator(llm_provider=mock_llm)
        
        # Create structured response
        json_response = {
            "mutations": [
                {
                    "idea_id": 1,
                    "mutated_content": "This is a detailed mutation that transforms the original concept by shifting perspective from individual gameplay to community-driven experiences. The Mobius strip becomes a shared space where multiple players can interact and influence each other's progress. Implementation involves creating a persistent world state that tracks collective actions and emergent behaviors across all connected players."
                }
            ]
        }
        
        mock_response = LLMResponse(
            content=json.dumps(json_response),
            model="gemini-pro",
            provider="google",
            cost=0.001
        )
        
        mock_llm.generate = AsyncMock(return_value=mock_response)
        
        # Create idea to mutate
        original_idea = GeneratedIdea(
            content="Original game concept focusing on single player",
            thinking_method=ThinkingMethod.QUESTIONING,
            agent_name="test_agent",
            generation_prompt="test",
            confidence_score=0.8
        )
        
        # Run mutation
        ideas = [original_idea]
        mutated = await operator.mutate_batch(ideas)
        
        # Verify results
        assert len(mutated) == 1
        assert mutated[0].content == json_response["mutations"][0]["mutated_content"]
        assert "[FALLBACK TEXT]" not in mutated[0].content
        assert len(mutated[0].content) > 150
    
    @pytest.mark.asyncio 
    async def test_crossover_with_gemini_style_response(self):
        """Test crossover with response formatted as Gemini would return it."""
        mock_llm = MagicMock()
        operator = SemanticCrossoverOperator(llm_provider=mock_llm)
        
        # Gemini might return with some formatting
        gemini_response = """
{
  "offspring_1": "Revolutionary hybrid concept: This offspring brilliantly merges the immersive storytelling elements from the first parent with the innovative gameplay mechanics from the second. The result is a narrative-driven experience where player choices dynamically reshape the Mobius strip world itself. Each decision creates ripples that affect both the physical topology and the unfolding story, leading to a truly unique journey for every player.",
  "offspring_2": "Alternative fusion approach: Taking the technical framework from parent one and the artistic vision from parent two, this offspring creates a meditative exploration game. Players traverse a living Mobius strip that responds to their emotional state, detected through gameplay patterns. The environment evolves based on how players interact with it, creating a deeply personal and reflective gaming experience unlike anything seen before."
}
"""
        
        mock_response = LLMResponse(
            content=gemini_response.strip(),
            model="gemini-pro",
            provider="google",
            cost=0.001
        )
        
        mock_llm.generate = AsyncMock(return_value=mock_response)
        
        parent1 = GeneratedIdea(
            content="Technical game framework",
            thinking_method=ThinkingMethod.QUESTIONING,
            agent_name="test_agent",
            generation_prompt="test",
            confidence_score=0.8
        )
        parent2 = GeneratedIdea(
            content="Artistic game vision", 
            thinking_method=ThinkingMethod.ABDUCTION,
            agent_name="test_agent",
            generation_prompt="test",
            confidence_score=0.7
        )
        
        # Run crossover
        offspring1, offspring2 = await operator.crossover(parent1, parent2)
        
        # Should parse successfully
        assert "Revolutionary hybrid concept" in offspring1.content
        assert "Alternative fusion approach" in offspring2.content
        assert "[FALLBACK TEXT]" not in offspring1.content
        assert "[FALLBACK TEXT]" not in offspring2.content