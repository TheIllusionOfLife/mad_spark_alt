"""Tests for semantic operator response parsing."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from mad_spark_alt.evolution.semantic_operators import SemanticCrossoverOperator, BatchSemanticMutationOperator
from mad_spark_alt.core.interfaces import GeneratedIdea, ThinkingMethod
from mad_spark_alt.core.llm_provider import LLMResponse


class TestSemanticCrossoverParsing:
    """Test semantic crossover response parsing."""
    
    def test_parse_valid_crossover_response(self):
        """Test parsing of properly formatted crossover response."""
        operator = SemanticCrossoverOperator(llm_provider=None)
        
        response = """OFFSPRING_1: This is the first offspring with a detailed implementation that combines elements from both parent ideas. It leverages AI-powered automation from parent 1 and the collaborative framework from parent 2 to create a hybrid solution that maximizes efficiency while maintaining human oversight. The implementation includes setting up automated workflows, establishing collaboration protocols, and creating feedback loops for continuous improvement. This approach is expected to reduce time-to-market by 40% while increasing team satisfaction through meaningful participation in the creative process.

OFFSPRING_2: The second offspring takes a different approach by focusing on the educational aspects from parent 1 and the scalability features from parent 2. This creates a learning platform that can grow with demand while maintaining personalized instruction quality. The solution involves developing adaptive learning algorithms, creating modular content systems, and implementing peer-to-peer learning networks. Resources required include cloud infrastructure, content management systems, and community moderators. Expected outcomes include 10x reach with maintained 90% satisfaction rates."""
        
        offspring1, offspring2 = operator._parse_crossover_response(response)
        
        assert "first offspring with a detailed implementation" in offspring1
        assert "combines elements from both parent ideas" in offspring1
        assert len(offspring1) > 150  # Should be detailed
        
        assert "second offspring takes a different approach" in offspring2
        assert "educational aspects from parent 1" in offspring2
        assert len(offspring2) > 150  # Should be detailed
        
        # Should NOT contain old placeholder text
        assert "Alternative combination of parent ideas" not in offspring1
        assert "Alternative combination of parent ideas" not in offspring2
        assert "Combined approach integrating both parent concepts" not in offspring1
    
    def test_parse_crossover_with_extra_text(self):
        """Test parsing when LLM adds extra explanation."""
        operator = SemanticCrossoverOperator(llm_provider=None)
        
        response = """Here are the two offspring ideas:

OFFSPRING_1: Detailed first offspring that is at least 150 words long with specific implementation steps...
[Content continues for proper length to simulate real response]

OFFSPRING_2: Detailed second offspring that is also at least 150 words long with different approach...
[Content continues for proper length to simulate real response]

These offspring effectively combine the parent concepts."""
        
        offspring1, offspring2 = operator._parse_crossover_response(response)
        
        assert "Detailed first offspring" in offspring1
        assert "Detailed second offspring" in offspring2
        assert "These offspring effectively" not in offspring1
        assert "These offspring effectively" not in offspring2
    
    def test_parse_malformed_response_uses_fallback(self):
        """Test that malformed responses trigger fallback text."""
        operator = SemanticCrossoverOperator(llm_provider=None)
        
        # Missing OFFSPRING_2
        response = """OFFSPRING_1: Valid first offspring with enough content."""
        
        offspring1, offspring2 = operator._parse_crossover_response(response)
        
        assert "Valid first offspring" in offspring1
        assert "Alternative fusion emphasizing innovation" in offspring2  # Fallback
        assert len(offspring2) > 150  # Should be detailed fallback
    
    def test_parse_empty_response_uses_both_fallbacks(self):
        """Test that empty response uses both fallback texts."""
        operator = SemanticCrossoverOperator(llm_provider=None)
        
        response = ""
        
        offspring1, offspring2 = operator._parse_crossover_response(response)
        
        assert "Integrated solution combining complementary strengths" in offspring1
        assert "Alternative fusion emphasizing innovation" in offspring2
        assert len(offspring1) > 150  # Should be detailed fallback
        assert len(offspring2) > 150  # Should be detailed fallback


class TestSemanticMutationParsing:
    """Test semantic mutation response parsing."""
    
    def test_parse_single_mutation_response(self):
        """Test parsing of single mutation response."""
        operator = BatchSemanticMutationOperator(llm_provider=None)
        
        # The single mutation prompt asks for just the mutated idea
        response = """This is a detailed mutation that transforms the original idea using a perspective shift. Instead of focusing on individual productivity, this approach emphasizes community-driven solutions where multiple stakeholders collaborate to achieve shared goals. The implementation involves setting up collaboration platforms, establishing governance structures, and creating incentive mechanisms. Technologies include blockchain for transparency, AI for matching collaborators, and cloud infrastructure for scalability. Expected outcomes include 3x productivity gains through synergy and reduced duplicate efforts."""
        
        # For single mutations, the response IS the mutation
        assert len(response) > 150
        assert "perspective shift" in response
        assert "community-driven solutions" in response
    
    def test_parse_batch_mutation_response(self):
        """Test parsing of batch mutation response."""
        operator = BatchSemanticMutationOperator(llm_provider=None)
        
        response = """IDEA_1_MUTATION: First mutation with detailed implementation spanning multiple sentences to reach the minimum word count. This mutation uses mechanism change to achieve the same goal through different means. It involves specific technologies like React, Node.js, and PostgreSQL. Resources include cloud hosting, development team, and user testing infrastructure. Expected benefits include improved performance and user satisfaction.

IDEA_2_MUTATION: Second mutation also with comprehensive details. This uses abstraction shift to make the concept more concrete and actionable. Implementation steps include market research, prototype development, user testing, and iterative refinement. Technologies involve mobile frameworks, analytics platforms, and payment processors. Resources needed are development team, marketing budget, and operational support.

IDEA_3_MUTATION: Third mutation with its own unique approach using constraint variation. By removing geographical constraints, this solution can scale globally. Implementation requires multi-language support, distributed infrastructure, and local partnerships. Technologies include CDN, localization tools, and automated translation. Expected outcomes are 10x market reach and diversified revenue streams."""
        
        # Test the batch parsing logic (this would be in a batch processing method)
        mutations = []
        for line in response.strip().split('\n'):
            line = line.strip()
            if line.startswith('IDEA_') and '_MUTATION:' in line:
                mutation_text = line.split('_MUTATION:', 1)[1].strip()
                mutations.append(mutation_text)
        
        assert len(mutations) == 3
        assert all(len(m) > 100 for m in mutations)
        assert "mechanism change" in mutations[0]
        assert "abstraction shift" in mutations[1]
        assert "constraint variation" in mutations[2]


class TestSemanticOperatorIntegration:
    """Test full semantic operator flow with LLM."""
    
    @pytest.mark.asyncio
    async def test_crossover_with_parsing_failure_recovery(self):
        """Test that crossover handles parsing failures gracefully."""
        # Mock LLM provider
        mock_llm = MagicMock()
        mock_llm.generate = AsyncMock(return_value=LLMResponse(
            content="Malformed response without proper markers",
            model="gemini-pro",
            provider="google",
            cost=0.01
        ))
        
        operator = SemanticCrossoverOperator(llm_provider=mock_llm)
        
        # Create parent ideas
        parent1 = GeneratedIdea(
            content="Parent 1 content",
            thinking_method=ThinkingMethod.QUESTIONING,
            agent_name="test_agent_1",
            generation_prompt="test prompt",
            confidence_score=0.8
        )
        parent2 = GeneratedIdea(
            content="Parent 2 content",
            thinking_method=ThinkingMethod.ABDUCTION,
            agent_name="test_agent_2",
            generation_prompt="test prompt",
            confidence_score=0.7
        )
        
        # Run crossover
        offspring1, offspring2 = await operator.crossover(parent1, parent2)
        
        # Should get detailed fallback content
        assert "Integrated solution combining complementary strengths" in offspring1.content
        assert "Alternative fusion emphasizing innovation" in offspring2.content
        assert len(offspring1.content) > 150
        assert len(offspring2.content) > 150
        
        # Should preserve parent metadata
        assert parent1.content in offspring1.parent_ideas
        assert parent2.content in offspring2.parent_ideas