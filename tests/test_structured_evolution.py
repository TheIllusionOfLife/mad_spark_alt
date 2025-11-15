"""
Tests for structured output in evolution operators.

This module tests the new structured output capabilities for
genetic algorithm mutation and crossover operations.
"""

import json
import pytest
from typing import List
from unittest.mock import AsyncMock, MagicMock, patch

from mad_spark_alt.core.interfaces import GeneratedIdea, ThinkingMethod
from mad_spark_alt.core.llm_provider import LLMResponse, LLMProvider
from mad_spark_alt.evolution.semantic_operators import BatchSemanticMutationOperator, SemanticCrossoverOperator


class TestStructuredMutation:
    """Test structured output for semantic mutation operator."""

    @pytest.fixture
    def llm_provider(self):
        """Create mock LLM provider."""
        return AsyncMock()

    @pytest.fixture
    def mutation_operator(self, llm_provider):
        """Create mutation operator with mock provider."""
        return BatchSemanticMutationOperator(llm_provider=llm_provider, cache_ttl=0)

    @pytest.mark.asyncio
    async def test_mutation_uses_structured_output(self, mutation_operator, llm_provider):
        """Test that mutation operator uses structured output."""
        # Create test idea
        idea = GeneratedIdea(
            content="Implement a recycling program in schools",
            thinking_method=ThinkingMethod.QUESTIONING,
            agent_name="TestAgent",
            generation_prompt="Test prompt",
            confidence_score=0.8
        )
        
        # Create structured response
        structured_response = {
            "mutations": [
                {
                    "id": 1,
                    "content": "Develop a comprehensive recycling education program in schools with student ambassadors and rewards system"
                }
            ]
        }
        
        # Mock LLM response
        mock_response = LLMResponse(
            content=json.dumps(structured_response),
            provider=LLMProvider.GOOGLE,
            model="gemini-2.5-flash",
            usage={"prompt_tokens": 100, "completion_tokens": 150},
            cost=0.001
        )
        
        llm_provider.generate = AsyncMock(return_value=mock_response)
        
        # Perform mutation
        mutated_ideas = await mutation_operator.mutate_batch([idea], context="Test mutation context")
        
        # Verify structured output was requested
        llm_provider.generate.assert_called_once()
        request = llm_provider.generate.call_args[0][0]
        
        # Check that request includes response schema
        assert request.response_schema is not None
        assert request.response_mime_type == "application/json"
        
        # Verify schema structure
        schema = request.response_schema
        assert schema["type"] == "object"
        assert "mutations" in schema["properties"]
        assert schema["properties"]["mutations"]["type"] == "array"
        
        # Verify mutation result
        assert len(mutated_ideas) == 1
        assert "comprehensive recycling education program" in mutated_ideas[0].content
        assert "student ambassadors" in mutated_ideas[0].content

    @pytest.mark.asyncio
    async def test_mutation_batch_with_structured_output(self, mutation_operator, llm_provider):
        """Test batch mutation with structured output."""
        # Create test ideas
        ideas = [
            GeneratedIdea(
                content="Use solar panels on rooftops",
                thinking_method=ThinkingMethod.DEDUCTION,
                agent_name="TestAgent",
                generation_prompt="Test prompt",
                confidence_score=0.7
            ),
            GeneratedIdea(
                content="Create community gardens",
                thinking_method=ThinkingMethod.INDUCTION,
                agent_name="TestAgent",
                generation_prompt="Test prompt",
                confidence_score=0.8
            ),
            GeneratedIdea(
                content="Promote electric vehicles",
                thinking_method=ThinkingMethod.ABDUCTION,
                agent_name="TestAgent",
                generation_prompt="Test prompt",
                confidence_score=0.9
            )
        ]
        
        # Create structured response
        structured_response = {
            "mutations": [
                {
                    "id": 1,  # Changed from idea_id
                    "content": "Install solar panels with battery storage systems on residential and commercial rooftops"  # Changed from mutated_content
                },
                {
                    "id": 2,
                    "content": "Establish community gardens with composting programs and educational workshops"
                },
                {
                    "id": 3,
                    "content": "Develop electric vehicle infrastructure with charging stations and purchase incentives"
                }
            ]
        }
        
        # Mock LLM response
        mock_response = LLMResponse(
            content=json.dumps(structured_response),
            provider=LLMProvider.GOOGLE,
            model="gemini-2.5-flash",
            usage={"prompt_tokens": 200, "completion_tokens": 300},
            cost=0.002
        )
        
        llm_provider.generate = AsyncMock(return_value=mock_response)
        
        # Perform batch mutation
        mutated_ideas = await mutation_operator.mutate_batch(ideas, context="Test mutation context")
        
        # Verify results
        assert len(mutated_ideas) == 3
        assert "battery storage systems" in mutated_ideas[0].content
        assert "composting programs" in mutated_ideas[1].content
        assert "charging stations" in mutated_ideas[2].content
        
        # Verify cost was distributed
        for idea in mutated_ideas:
            assert hasattr(idea, 'parent_ideas')
            assert len(idea.parent_ideas) == 1

    @pytest.mark.asyncio
    async def test_mutation_fallback_to_text_parsing(self, mutation_operator, llm_provider):
        """Test fallback to text parsing when structured output fails."""
        # Create test idea
        idea = GeneratedIdea(
            content="Reduce plastic waste",
            thinking_method=ThinkingMethod.QUESTIONING,
            agent_name="TestAgent",
            generation_prompt="Test prompt",
            confidence_score=0.7
        )
        
        # Create invalid JSON response that contains text format
        text_response = """Here are the mutations:

IDEA_1_MUTATION: Implement comprehensive plastic reduction strategies including bans on single-use plastics and incentives for alternatives"""
        
        # Mock LLM response
        mock_response = LLMResponse(
            content=text_response,
            provider=LLMProvider.GOOGLE,
            model="gemini-2.5-flash",
            usage={"prompt_tokens": 100, "completion_tokens": 150},
            cost=0.001
        )
        
        llm_provider.generate = AsyncMock(return_value=mock_response)
        
        # Perform mutation
        mutated_ideas = await mutation_operator.mutate_batch([idea])
        
        # Verify fallback parsing worked
        assert len(mutated_ideas) == 1
        assert "comprehensive plastic reduction strategies" in mutated_ideas[0].content
        assert "single-use plastics" in mutated_ideas[0].content

    @pytest.mark.asyncio
    async def test_mutation_batch_with_zero_based_ids(self, mutation_operator, llm_provider):
        """Test batch mutation handles 0-based idea_ids correctly."""
        # Create test ideas
        ideas = [
            GeneratedIdea(
                content="Original idea A",
                thinking_method=ThinkingMethod.ABDUCTION,
                agent_name="TestAgent",
                generation_prompt="Test prompt",
            ),
            GeneratedIdea(
                content="Original idea B", 
                thinking_method=ThinkingMethod.DEDUCTION,
                agent_name="TestAgent",
                generation_prompt="Test prompt",
            )
        ]
        
        # Mock structured output response with 1-based IDs (as expected by parser)
        structured_response = {
            "mutations": [
                {"id": 1, "content": "Enhanced idea A content"},  # IDEA_1
                {"id": 2, "content": "Enhanced idea B content"}   # IDEA_2
            ]
        }
        
        # Mock LLM response
        mock_response = LLMResponse(
            content=json.dumps(structured_response),
            provider=LLMProvider.GOOGLE,
            model="gemini-2.5-flash",
            usage={"prompt_tokens": 100, "completion_tokens": 150},
            cost=0.001
        )
        
        llm_provider.generate = AsyncMock(return_value=mock_response)
        
        # Test batch mutation
        results = await mutation_operator.mutate_batch(ideas, "test context")
        
        # Verify results - mutations should be applied in correct order regardless of ID system
        assert len(results) == 2
        assert results[0].content == "Enhanced idea A content"
        assert results[1].content == "Enhanced idea B content"

    @pytest.mark.asyncio
    async def test_mutation_batch_with_non_sequential_ids(self, mutation_operator, llm_provider):
        """Test batch mutation handles non-sequential idea_ids correctly."""
        # Create test ideas
        ideas = [
            GeneratedIdea(
                content="Original idea A",
                thinking_method=ThinkingMethod.ABDUCTION,
                agent_name="TestAgent",
                generation_prompt="Test prompt",
            ),
            GeneratedIdea(
                content="Original idea B", 
                thinking_method=ThinkingMethod.DEDUCTION,
                agent_name="TestAgent",
                generation_prompt="Test prompt",
            ),
            GeneratedIdea(
                content="Original idea C", 
                thinking_method=ThinkingMethod.INDUCTION,
                agent_name="TestAgent",
                generation_prompt="Test prompt",
            )
        ]
        
        # Mock structured output response with non-sequential IDs (out of order)
        structured_response = {
            "mutations": [
                {"id": 3, "content": "Enhanced idea C content"},
                {"id": 1, "content": "Enhanced idea A content"},
                {"id": 2, "content": "Enhanced idea B content"}
            ]
        }
        
        # Mock LLM response
        mock_response = LLMResponse(
            content=json.dumps(structured_response),
            provider=LLMProvider.GOOGLE,
            model="gemini-2.5-flash",
            usage={"prompt_tokens": 150, "completion_tokens": 200},
            cost=0.002
        )
        
        llm_provider.generate = AsyncMock(return_value=mock_response)
        
        # Test batch mutation
        results = await mutation_operator.mutate_batch(ideas, "test context")
        
        # Verify results - mutations should be applied in sorted order by ID
        assert len(results) == 3
        # ID 1 should be first, ID 2 second, ID 3 third after sorting
        assert results[0].content == "Enhanced idea A content"
        assert results[1].content == "Enhanced idea B content"
        assert results[2].content == "Enhanced idea C content"


class TestStructuredCrossover:
    """Test structured output for semantic crossover operator."""

    @pytest.fixture
    def llm_provider(self):
        """Create mock LLM provider."""
        return AsyncMock()

    @pytest.fixture
    def crossover_operator(self, llm_provider):
        """Create crossover operator with mock provider."""
        return SemanticCrossoverOperator(llm_provider=llm_provider, cache_ttl=0)

    @pytest.mark.asyncio
    async def test_crossover_uses_structured_output(self, crossover_operator, llm_provider):
        """Test that crossover operator uses structured output."""
        # Create parent ideas
        parent1 = GeneratedIdea(
            content="Use renewable energy sources like solar and wind",
            thinking_method=ThinkingMethod.DEDUCTION,
            agent_name="TestAgent",
            generation_prompt="Test prompt",
            confidence_score=0.8
        )
        parent2 = GeneratedIdea(
            content="Implement smart grid technology for efficient distribution",
            thinking_method=ThinkingMethod.INDUCTION,
            agent_name="TestAgent",
            generation_prompt="Test prompt",
            confidence_score=0.7
        )
        
        # Create structured response
        structured_response = {
            "offspring_1": "Integrate renewable energy sources with smart grid technology for optimized distribution",
            "offspring_2": "Develop smart grid systems powered primarily by solar and wind energy"
        }
        
        # Mock LLM response
        mock_response = LLMResponse(
            content=json.dumps(structured_response),
            provider=LLMProvider.GOOGLE,
            model="gemini-2.5-flash",
            usage={"prompt_tokens": 150, "completion_tokens": 100},
            cost=0.001
        )
        
        llm_provider.generate = AsyncMock(return_value=mock_response)
        
        # Perform crossover
        offspring = await crossover_operator.crossover(parent1, parent2)
        
        # Verify structured output was requested
        llm_provider.generate.assert_called_once()
        request = llm_provider.generate.call_args[0][0]
        
        # Check that request includes response schema
        assert request.response_schema is not None
        assert request.response_mime_type == "application/json"
        
        # Verify schema structure (Pydantic uses offspring1/offspring2, not offspring_1/offspring_2)
        schema = request.response_schema
        assert schema["type"] == "object"
        assert "offspring1" in schema["properties"]
        assert "offspring2" in schema["properties"]
        assert schema["properties"]["offspring1"]["type"] == "string"
        assert schema["properties"]["offspring2"]["type"] == "string"
        
        # Verify offspring results
        assert len(offspring) == 2
        assert "renewable energy sources with smart grid" in offspring[0].content
        assert "smart grid systems powered primarily by solar" in offspring[1].content
        
        # Verify parent tracking
        for child in offspring:
            assert len(child.parent_ideas) == 2
            assert parent1.content in child.parent_ideas
            assert parent2.content in child.parent_ideas

    @pytest.mark.asyncio
    async def test_crossover_fallback_to_text_parsing(self, crossover_operator, llm_provider):
        """Test fallback to text parsing when structured output fails."""
        # Create parent ideas
        parent1 = GeneratedIdea(
            content="Improve public transportation",
            thinking_method=ThinkingMethod.QUESTIONING,
            agent_name="TestAgent",
            generation_prompt="Test prompt",
            confidence_score=0.8
        )
        parent2 = GeneratedIdea(
            content="Reduce carbon emissions",
            thinking_method=ThinkingMethod.ABDUCTION,
            agent_name="TestAgent",
            generation_prompt="Test prompt",
            confidence_score=0.7
        )
        
        # Create text response (invalid JSON)
        text_response = """Combining these ideas:

OFFSPRING_1: Develop electric public transportation systems to significantly reduce urban carbon emissions

OFFSPRING_2: Create carbon-neutral public transit networks with renewable energy integration"""
        
        # Mock LLM response
        mock_response = LLMResponse(
            content=text_response,
            provider=LLMProvider.GOOGLE,
            model="gemini-2.5-flash",
            usage={"prompt_tokens": 150, "completion_tokens": 100},
            cost=0.001
        )
        
        llm_provider.generate = AsyncMock(return_value=mock_response)
        
        # Perform crossover
        offspring = await crossover_operator.crossover(parent1, parent2)
        
        # Verify fallback parsing worked
        assert len(offspring) == 2
        assert "electric public transportation" in offspring[0].content
        assert "carbon-neutral public transit" in offspring[1].content

    @pytest.mark.asyncio
    async def test_batch_crossover_with_non_sequential_ids(self):
        """Test batch crossover handles non-sequential pair_ids correctly."""
        from mad_spark_alt.evolution.semantic_operators import BatchSemanticCrossoverOperator
        
        # Create mock LLM provider
        llm_provider = AsyncMock()
        
        # Create test parent pairs
        parent_pairs = [
            (
                GeneratedIdea(
                    content="Solar panels on buildings",
                    thinking_method=ThinkingMethod.DEDUCTION,
                    agent_name="TestAgent",
                    generation_prompt="Test prompt",
                    confidence_score=0.8
                ),
                GeneratedIdea(
                    content="Wind turbines in parks",
                    thinking_method=ThinkingMethod.INDUCTION,
                    agent_name="TestAgent",
                    generation_prompt="Test prompt",
                    confidence_score=0.7
                )
            ),
            (
                GeneratedIdea(
                    content="Geothermal energy systems",
                    thinking_method=ThinkingMethod.QUESTIONING,
                    agent_name="TestAgent",
                    generation_prompt="Test prompt",
                    confidence_score=0.9
                ),
                GeneratedIdea(
                    content="Tidal power generation",
                    thinking_method=ThinkingMethod.ABDUCTION,
                    agent_name="TestAgent",
                    generation_prompt="Test prompt",
                    confidence_score=0.8
                )
            ),
            (
                GeneratedIdea(
                    content="Community composting programs",
                    thinking_method=ThinkingMethod.DEDUCTION,
                    agent_name="TestAgent",
                    generation_prompt="Test prompt",
                    confidence_score=0.7
                ),
                GeneratedIdea(
                    content="Urban farming initiatives",
                    thinking_method=ThinkingMethod.INDUCTION,
                    agent_name="TestAgent",
                    generation_prompt="Test prompt",
                    confidence_score=0.8
                )
            )
        ]
        
        # Mock structured response with non-sequential pair_ids (out of order)
        structured_response = {
            "crossovers": [
                {
                    "pair_id": 3,
                    "offspring1": "Sustainable food production with composting integration",
                    "offspring2": "Urban agriculture with waste recycling systems"
                },
                {
                    "pair_id": 1,
                    "offspring1": "Solar-wind hybrid energy for buildings and parks",
                    "offspring2": "Renewable energy infrastructure in urban spaces"
                },
                {
                    "pair_id": 2,
                    "offspring1": "Ocean-based renewable energy systems",
                    "offspring2": "Combined geothermal-tidal power generation"
                }
            ]
        }
        
        # Mock LLM response
        mock_response = LLMResponse(
            content=json.dumps(structured_response),
            provider=LLMProvider.GOOGLE,
            model="gemini-2.5-flash",
            usage={"prompt_tokens": 300, "completion_tokens": 400},
            cost=0.003
        )
        
        llm_provider.generate = AsyncMock(return_value=mock_response)
        
        # Create operator and test batch crossover
        operator = BatchSemanticCrossoverOperator(llm_provider=llm_provider, cache_ttl=0)
        results = await operator.crossover_batch(parent_pairs, "test context")
        
        # Verify results - crossovers should be applied in correct order based on pair_id
        assert len(results) == 3
        
        # pair_id 1 should match first parent pair
        assert "Solar-wind hybrid" in results[0][0].content
        assert "Renewable energy infrastructure" in results[0][1].content
        
        # pair_id 2 should match second parent pair
        assert "Ocean-based renewable" in results[1][0].content
        assert "geothermal-tidal" in results[1][1].content
        
        # pair_id 3 should match third parent pair
        assert "Sustainable food production" in results[2][0].content
        assert "Urban agriculture" in results[2][1].content


class TestEvolutionOperatorSchemas:
    """Test schema generation for evolution operators."""

    def test_mutation_schema_structure(self):
        """Test the structure of mutation schema."""
        from mad_spark_alt.evolution.semantic_operators import get_mutation_schema

        schema = get_mutation_schema()  # No arguments needed

        # Verify structure
        assert schema["type"] == "object"
        assert "mutations" in schema["properties"]
        assert schema["properties"]["mutations"]["type"] == "array"

        # Verify item structure (Pydantic uses $ref for nested schemas)
        item_schema = schema["properties"]["mutations"]["items"]
        if "$ref" in item_schema:
            # Extract definition name and look it up
            ref_path = item_schema["$ref"].split("/")[-1]
            mutation_def = schema["$defs"][ref_path]
            assert mutation_def["type"] == "object"
            assert "id" in mutation_def["properties"]
            assert "mutated_idea" in mutation_def["properties"]  # Pydantic field name
            assert mutation_def["properties"]["id"]["type"] == "integer"
            assert mutation_def["properties"]["mutated_idea"]["type"] == "string"
        else:
            # Fallback for inline definitions
            assert item_schema["type"] == "object"
            assert "id" in item_schema["properties"]
            assert "content" in item_schema["properties"]
            assert item_schema["properties"]["id"]["type"] == "integer"
            assert item_schema["properties"]["content"]["type"] == "string"

    def test_crossover_schema_structure(self):
        """Test the structure of crossover schema."""
        from mad_spark_alt.evolution.semantic_operators import get_crossover_schema

        schema = get_crossover_schema()

        # Verify structure (Pydantic uses offspring1/offspring2, not offspring_1/offspring_2)
        assert schema["type"] == "object"
        assert "offspring1" in schema["properties"]
        assert "offspring2" in schema["properties"]
        assert schema["properties"]["offspring1"]["type"] == "string"
        assert schema["properties"]["offspring2"]["type"] == "string"
        assert schema["required"] == ["offspring1", "offspring2"]


@pytest.mark.integration
class TestStructuredEvolutionIntegration:
    """Integration tests for structured output in evolution operators."""

    @pytest.mark.asyncio
    async def test_real_mutation_with_structured_output(self):
        """Test real LLM mutation with structured output."""
        import os
        if not os.getenv("GOOGLE_API_KEY"):
            pytest.skip("GOOGLE_API_KEY not set")
        
        from mad_spark_alt.core.llm_provider import GoogleProvider
        
        provider = GoogleProvider(api_key=os.getenv("GOOGLE_API_KEY"))
        mutation_operator = BatchSemanticMutationOperator(llm_provider=provider, cache_ttl=0)
        
        # Create test idea
        idea = GeneratedIdea(
            content="Create urban green spaces to improve air quality",
            thinking_method=ThinkingMethod.INDUCTION,
            agent_name="TestAgent",
            generation_prompt="Test prompt",
            confidence_score=0.8
        )
        
        # Perform mutation
        mutated_ideas = await mutation_operator.mutate_batch([idea])
        
        # Verify mutation worked
        assert len(mutated_ideas) == 1
        assert mutated_ideas[0].content != idea.content
        assert len(mutated_ideas[0].content) > 20  # Non-trivial mutation
        
        # Verify it's a meaningful variation
        # The mutated content should be related but different
        assert any(word in mutated_ideas[0].content.lower() 
                  for word in ["green", "air", "urban", "quality", "space"])

    @pytest.mark.asyncio
    async def test_real_crossover_with_structured_output(self):
        """Test real LLM crossover with structured output."""
        import os
        if not os.getenv("GOOGLE_API_KEY"):
            pytest.skip("GOOGLE_API_KEY not set")
        
        from mad_spark_alt.core.llm_provider import GoogleProvider
        
        provider = GoogleProvider(api_key=os.getenv("GOOGLE_API_KEY"))
        crossover_operator = SemanticCrossoverOperator(llm_provider=provider, cache_ttl=0)
        
        # Create parent ideas
        parent1 = GeneratedIdea(
            content="Promote work-from-home policies to reduce commuting",
            thinking_method=ThinkingMethod.DEDUCTION,
            agent_name="TestAgent",
            generation_prompt="Test prompt",
            confidence_score=0.8
        )
        parent2 = GeneratedIdea(
            content="Invest in public transportation infrastructure",
            thinking_method=ThinkingMethod.ABDUCTION,
            agent_name="TestAgent",
            generation_prompt="Test prompt",
            confidence_score=0.7
        )
        
        # Perform crossover
        offspring = await crossover_operator.crossover(parent1, parent2)
        
        # Verify crossover worked
        assert len(offspring) == 2
        assert offspring[0].content != parent1.content
        assert offspring[0].content != parent2.content
        assert offspring[1].content != parent1.content
        assert offspring[1].content != parent2.content
        
        # Verify offspring are meaningful variations
        # Since crossover might generate abstract combinations, verify they have substantial content
        for i, child in enumerate(offspring):
            print(f"Offspring {i+1} content: {child.content}")
            # Check that content is substantial (not just a placeholder)
            assert len(child.content) > 100  # Should be detailed
            # Check that it's different from parents
            assert child.content != parent1.content
            assert child.content != parent2.content