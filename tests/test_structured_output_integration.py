"""
Integration tests for structured output implementation.

These tests verify that LLM responses using structured output are properly parsed
and that fallback mechanisms work correctly when structured output fails.
"""

import pytest
import os
import json
from unittest.mock import Mock, AsyncMock, patch

from mad_spark_alt.core.llm_provider import LLMResponse
from mad_spark_alt.core.simple_qadi_orchestrator import SimpleQADIOrchestrator
from mad_spark_alt.core.phase_logic import (
    get_hypothesis_generation_schema,
    get_deduction_schema,
)
from mad_spark_alt.evolution.semantic_operators import (
    BatchSemanticMutationOperator,
    SemanticCrossoverOperator,
    get_mutation_schema,
    get_crossover_schema,
)
from mad_spark_alt.core.interfaces import GeneratedIdea, ThinkingMethod


class TestStructuredOutputSchemas:
    """Test that all schemas are properly defined and valid."""
    
    def test_hypothesis_generation_schema_structure(self):
        """Test hypothesis generation schema has correct structure."""
        schema = get_hypothesis_generation_schema()
        
        assert schema["type"] == "OBJECT"
        assert "properties" in schema
        assert "hypotheses" in schema["properties"]
        
        hypotheses_schema = schema["properties"]["hypotheses"]
        assert hypotheses_schema["type"] == "ARRAY"
        assert "items" in hypotheses_schema
        
        item_schema = hypotheses_schema["items"]
        assert item_schema["type"] == "OBJECT"
        assert "id" in item_schema["properties"]
        assert "content" in item_schema["properties"]
    
    def test_deduction_schema_structure(self):
        """Test deduction (score evaluation) schema has correct structure."""
        schema = get_deduction_schema()
        
        assert schema["type"] == "OBJECT"
        assert "properties" in schema
        assert "evaluations" in schema["properties"]
        
        evaluations_schema = schema["properties"]["evaluations"]
        assert evaluations_schema["type"] == "ARRAY"
        
        # Check that schema includes all required fields
        item_schema = evaluations_schema["items"]
        assert "hypothesis_id" in item_schema["properties"]
        assert "scores" in item_schema["properties"]
        
        # Check score fields are in the nested scores object
        scores_schema = item_schema["properties"]["scores"]
        score_fields = ["impact", "feasibility", "accessibility", 
                       "sustainability", "scalability"]
        
        for field in score_fields:
            assert field in scores_schema["properties"]
    
    def test_mutation_schema_structure(self):
        """Test mutation schema has correct structure."""
        schema = get_mutation_schema()
        
        assert schema["type"] == "OBJECT"
        assert "properties" in schema
        assert "mutations" in schema["properties"]
        
        mutations_schema = schema["properties"]["mutations"]
        assert mutations_schema["type"] == "ARRAY"
        
        item_schema = mutations_schema["items"]
        assert "id" in item_schema["properties"]
        assert item_schema["properties"]["id"]["type"] == "INTEGER"
        assert "content" in item_schema["properties"]
    
    def test_crossover_schema_structure(self):
        """Test crossover schema has correct structure."""
        schema = get_crossover_schema()
        
        assert schema["type"] == "OBJECT"
        assert "properties" in schema
        assert "offspring_1" in schema["properties"]
        assert "offspring_2" in schema["properties"]


class TestStructuredOutputParsing:
    """Test parsing of structured output responses."""
    
    @pytest.mark.asyncio
    async def test_hypothesis_generation_with_valid_json(self):
        """Test hypothesis generation with properly structured JSON response."""
        mock_response = LLMResponse(
            content=json.dumps({
                "hypotheses": [
                    {"id": "1", "content": "First hypothesis about reducing waste"},
                    {"id": "2", "content": "Second hypothesis about recycling"},
                    {"id": "3", "content": "Third hypothesis about circular economy"}
                ]
            }),
            provider="google",
            model="gemini-2.0-flash",
            cost=0.001
        )
        
        with patch('mad_spark_alt.core.llm_provider.llm_manager.generate', 
                   new_callable=AsyncMock, return_value=mock_response):
            orchestrator = SimpleQADIOrchestrator()
            hypotheses, cost = await orchestrator._run_abduction_phase(
                "How can we reduce plastic waste?",
                "How can we reduce plastic waste?",
                max_retries=0
            )
            
            assert len(hypotheses) == 3
            assert all("hypothesis" in h.lower() for h in hypotheses)
            assert cost == 0.001
    
    @pytest.mark.asyncio
    async def test_hypothesis_generation_fallback_to_text_parsing(self):
        """Test fallback to text parsing when JSON parsing fails."""
        # Simulate text response that needs regex parsing
        mock_response = LLMResponse(
            content="H1: First hypothesis about waste reduction strategies\nH2: Second hypothesis about recycling improvements\nH3: Third hypothesis about circular economy principles",
            provider="google",
            model="gemini-2.0-flash",
            cost=0.001
        )
        
        with patch('mad_spark_alt.core.llm_provider.llm_manager.generate',
                   new_callable=AsyncMock, return_value=mock_response):
            orchestrator = SimpleQADIOrchestrator()
            hypotheses, cost = await orchestrator._run_abduction_phase(
                "How can we reduce plastic waste?",
                "How can we reduce plastic waste?",
                max_retries=0
            )
            
            assert len(hypotheses) == 3
            assert "waste reduction" in hypotheses[0]
            assert "recycling" in hypotheses[1]
            assert "circular economy" in hypotheses[2]
    
    @pytest.mark.asyncio
    async def test_score_evaluation_with_valid_json(self):
        """Test score evaluation with properly structured JSON response."""
        mock_response = LLMResponse(
            content=json.dumps({
                "evaluations": [
                    {
                        "hypothesis_id": "1",
                        "scores": {
                            "impact": 0.8,
                            "feasibility": 0.7,
                            "accessibility": 0.9,
                            "sustainability": 0.85,
                            "scalability": 0.75
                        }
                    }
                ]
            }),
            provider="google",
            model="gemini-2.0-flash",
            cost=0.001
        )
        
        with patch('mad_spark_alt.core.llm_provider.llm_manager.generate',
                   new_callable=AsyncMock, return_value=mock_response):
            orchestrator = SimpleQADIOrchestrator()
            deduction_result = await orchestrator._run_deduction_phase(
                "How can we reduce plastic waste?",
                "How can we reduce plastic waste?",
                ["Test hypothesis"],
                max_retries=0
            )
            
            scores = deduction_result["scores"]
            cost = deduction_result["cost"]
            
            assert len(scores) == 1
            assert scores[0].impact == 0.8
            assert cost == 0.001
    
    @pytest.mark.asyncio
    async def test_mutation_with_valid_json(self):
        """Test mutation with properly structured JSON response."""
        mock_response = LLMResponse(
            content=json.dumps({
                "mutations": [
                    {
                        "id": 1,
                        "content": "Enhanced version of the idea"
                    }
                ]
            }),
            provider="google",
            model="gemini-2.0-flash",
            cost=0.001
        )
        
        mock_llm = Mock()
        mock_llm.generate = AsyncMock(return_value=mock_response)
        
        operator = BatchSemanticMutationOperator(mock_llm)
        idea = GeneratedIdea(
            content="Original idea",
            thinking_method=ThinkingMethod.ABDUCTION,
            agent_name="test",
            generation_prompt="test"
        )
        
        # BatchSemanticMutationOperator expects a list
        mutated_list = await operator.mutate_batch([idea], "reduce waste")
        mutated = mutated_list[0]
        assert mutated.content == "Enhanced version of the idea"
    
    @pytest.mark.asyncio
    async def test_crossover_with_valid_json(self):
        """Test crossover with properly structured JSON response."""
        mock_response = LLMResponse(
            content=json.dumps({
                "offspring_1": "First combined idea focusing on technical implementation with advanced algorithms",
                "offspring_2": "Alternative approach emphasizing social collaboration and community engagement"
            }),
            provider="google",
            model="gemini-2.0-flash",
            cost=0.001
        )
        
        mock_llm = Mock()
        mock_llm.generate = AsyncMock(return_value=mock_response)
        
        operator = SemanticCrossoverOperator(mock_llm)
        parent1 = GeneratedIdea(
            content="Parent 1 idea",
            thinking_method=ThinkingMethod.ABDUCTION,
            agent_name="test",
            generation_prompt="test"
        )
        parent2 = GeneratedIdea(
            content="Parent 2 idea",
            thinking_method=ThinkingMethod.ABDUCTION,
            agent_name="test",
            generation_prompt="test"
        )
        
        offspring = await operator.crossover(parent1, parent2, "reduce waste")
        assert len(offspring) == 2
        assert offspring[0].content == "First combined idea focusing on technical implementation with advanced algorithms"
        assert offspring[1].content == "Alternative approach emphasizing social collaboration and community engagement"


@pytest.mark.integration
class TestRealLLMStructuredOutput:
    """Test structured output with real LLM calls."""
    
    @pytest.mark.asyncio
    async def test_real_hypothesis_generation(self):
        """Test hypothesis generation with real Gemini API."""
        if not os.getenv("GOOGLE_API_KEY"):
            pytest.skip("Integration test requires GOOGLE_API_KEY")
        
        # Setup LLM providers first
        from mad_spark_alt.core import setup_llm_providers
        await setup_llm_providers(google_api_key=os.getenv("GOOGLE_API_KEY"))
        
        orchestrator = SimpleQADIOrchestrator()
        hypotheses, cost = await orchestrator._run_abduction_phase(
            "How can we make cities more sustainable?",
            "How can we make cities more sustainable?",
            max_retries=1
        )
        
        # Verify we got valid hypotheses
        assert len(hypotheses) >= 3
        assert all(len(h) > 20 for h in hypotheses)  # Non-trivial content
        assert cost > 0
        
        # Verify no parsing artifacts
        for h in hypotheses:
            assert "H1:" not in h
            assert "Hypothesis 1:" not in h
            assert not h.startswith("- ")
    
    @pytest.mark.asyncio
    async def test_real_score_evaluation(self):
        """Test score evaluation with real Gemini API."""
        if not os.getenv("GOOGLE_API_KEY"):
            pytest.skip("Integration test requires GOOGLE_API_KEY")
        
        # Setup LLM providers first
        from mad_spark_alt.core import setup_llm_providers
        await setup_llm_providers(google_api_key=os.getenv("GOOGLE_API_KEY"))
        
        orchestrator = SimpleQADIOrchestrator()
        test_hypotheses = [
            "Implement comprehensive public transportation networks",
            "Create urban green spaces and vertical gardens",
            "Develop smart grid systems for energy efficiency"
        ]
        
        deduction_result = await orchestrator._run_deduction_phase(
            "How can we make cities more sustainable?",
            "How can we make cities more sustainable?",
            test_hypotheses,
            max_retries=1
        )
        
        # Extract scores from result
        scores = deduction_result["scores"]
        cost = deduction_result["cost"]
        
        # Verify we got valid scores
        assert len(scores) == 3
        assert cost > 0
        
        for score in scores:
            # All scores should be between 0 and 1
            assert 0 <= score.impact <= 1
            assert 0 <= score.feasibility <= 1
            assert 0 <= score.accessibility <= 1
            assert 0 <= score.sustainability <= 1
            assert 0 <= score.scalability <= 1
            assert 0 <= score.overall <= 1
            
            # Scores should not all be 0.5 (default fallback)
            score_values = [score.impact, score.feasibility, score.accessibility,
                           score.sustainability, score.scalability]
            assert not all(v == 0.5 for v in score_values)
    
    @pytest.mark.asyncio
    async def test_real_semantic_mutation(self):
        """Test semantic mutation with real Gemini API."""
        if not os.getenv("GOOGLE_API_KEY"):
            pytest.skip("Integration test requires GOOGLE_API_KEY")
        
        from mad_spark_alt.core.llm_provider import get_google_provider
        
        llm_provider = get_google_provider()
        operator = BatchSemanticMutationOperator(llm_provider)
        
        original_idea = GeneratedIdea(
            content="Create community composting programs in neighborhoods",
            thinking_method=ThinkingMethod.ABDUCTION,
            agent_name="test",
            generation_prompt="sustainability"
        )
        
        # BatchSemanticMutationOperator expects a list
        mutated_list = await operator.mutate_batch([original_idea], "urban sustainability")
        mutated = mutated_list[0]
        
        # Verify mutation produced valid content
        assert mutated.content != original_idea.content
        assert len(mutated.content) > 20
        assert "Parent" not in mutated.content  # No parent references
    
    @pytest.mark.asyncio
    async def test_real_semantic_crossover(self):
        """Test semantic crossover with real Gemini API."""
        if not os.getenv("GOOGLE_API_KEY"):
            pytest.skip("Integration test requires GOOGLE_API_KEY")
        
        from mad_spark_alt.core.llm_provider import get_google_provider
        
        llm_provider = get_google_provider()
        operator = SemanticCrossoverOperator(llm_provider)
        
        parent1 = GeneratedIdea(
            content="Develop renewable energy microgrids",
            thinking_method=ThinkingMethod.ABDUCTION,
            agent_name="test",
            generation_prompt="sustainability"
        )
        parent2 = GeneratedIdea(
            content="Create urban farming initiatives",
            thinking_method=ThinkingMethod.ABDUCTION,
            agent_name="test",
            generation_prompt="sustainability"
        )
        
        offspring = await operator.crossover(parent1, parent2, "urban sustainability")
        
        # Verify crossover produced valid offspring
        assert len(offspring) == 2
        assert all(len(o.content) > 20 for o in offspring)
        assert all("Parent" not in o.content for o in offspring)
        
        # Offspring should be different from each other
        assert offspring[0].content != offspring[1].content


class TestErrorHandling:
    """Test error handling in structured output parsing."""
    
    @pytest.mark.asyncio
    async def test_empty_response_handling(self):
        """Test handling of empty responses."""
        mock_response = LLMResponse(
            content="",
            provider="google",
            model="gemini-2.0-flash",
            cost=0.001
        )
        
        with patch('mad_spark_alt.core.llm_provider.llm_manager.generate',
                   new_callable=AsyncMock, return_value=mock_response):
            orchestrator = SimpleQADIOrchestrator()
            
            # Should handle empty response gracefully
            hypotheses, cost = await orchestrator._run_abduction_phase(
                "Test question",
                "Test question",
                max_retries=0
            )
            
            # Should return empty list or raise appropriate error
            assert isinstance(hypotheses, list)
    
    @pytest.mark.asyncio
    async def test_malformed_json_handling(self):
        """Test handling of malformed JSON responses."""
        mock_response = LLMResponse(
            content='{"hypotheses": [{"id": "1", "content": "Test"',  # Incomplete JSON
            provider="google",
            model="gemini-2.0-flash",
            cost=0.001
        )
        
        with patch('mad_spark_alt.core.llm_provider.llm_manager.generate',
                   new_callable=AsyncMock, return_value=mock_response):
            orchestrator = SimpleQADIOrchestrator()
            
            # Should fall back to text parsing
            hypotheses, cost = await orchestrator._run_abduction_phase(
                "Test question",
                "Test question", 
                max_retries=0
            )
            
            # Should handle gracefully
            assert isinstance(hypotheses, list)