"""
Integration tests for SimpleQADIOrchestrator with real LLM calls.

These tests validate the actual parsing of LLM responses and prevent
mock-reality divergence issues discovered in PR #46.
"""

import os
import pytest
from mad_spark_alt.core.simple_qadi_orchestrator import SimpleQADIOrchestrator, SimpleQADIResult, HypothesisScore
from mad_spark_alt.core.interfaces import GeneratedIdea, ThinkingMethod
from mad_spark_alt.core.llm_provider import setup_llm_providers


@pytest.mark.integration
class TestSimpleQADIIntegration:
    """Integration tests requiring real Google API key."""

    async def _setup_llm_providers(self):
        """Setup LLM providers for integration tests."""
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            pytest.skip("GOOGLE_API_KEY not available")
        
        # Setup LLM providers for integration tests
        await setup_llm_providers(google_api_key)

    @pytest.mark.asyncio
    async def test_complete_qadi_cycle_integration(self):
        """Test complete QADI cycle with real LLM calls."""
        await self._setup_llm_providers()
        orchestrator = SimpleQADIOrchestrator()
        
        result = await orchestrator.run_qadi_cycle(
            user_input="How can we reduce plastic waste in our daily lives?",
            context="Focus on practical, actionable solutions for individuals"
        )
        
        # Verify result structure
        assert isinstance(result, SimpleQADIResult)
        assert result.core_question != ""
        assert len(result.hypotheses) == 3
        assert len(result.hypothesis_scores) == 3
        assert result.final_answer != ""
        assert len(result.action_plan) >= 1
        assert len(result.verification_examples) >= 1
        assert result.verification_conclusion != ""
        assert result.total_llm_cost > 0
        
        # Verify synthesized ideas for evolution compatibility
        assert len(result.synthesized_ideas) == 3
        assert all(isinstance(idea, GeneratedIdea) for idea in result.synthesized_ideas)
        assert all(idea.thinking_method == ThinkingMethod.ABDUCTION for idea in result.synthesized_ideas)
        
        # Verify hypothesis scores have all required fields
        for score in result.hypothesis_scores:
            assert isinstance(score, HypothesisScore)
            assert 0.0 <= score.impact <= 1.0
            assert 0.0 <= score.feasibility <= 1.0
            assert 0.0 <= score.accessibility <= 1.0
            assert 0.0 <= score.sustainability <= 1.0
            assert 0.0 <= score.scalability <= 1.0
            assert 0.0 <= score.overall <= 1.0

    @pytest.mark.asyncio
    async def test_hypothesis_score_parsing_real_llm(self):
        """Test hypothesis score parsing with real LLM responses."""
        await self._setup_llm_providers()
        orchestrator = SimpleQADIOrchestrator()
        
        # Test with business question to get predictable hypothesis structure
        result = await orchestrator.run_qadi_cycle(
            user_input="How can small businesses improve customer retention?",
            context="Focus on cost-effective strategies"
        )
        
        # Verify all scores were parsed correctly (not default 0.5)
        assert len(result.hypothesis_scores) == 3
        
        # At least some scores should be different from default 0.5
        # (This would fail if parsing reverted to all defaults)
        all_scores = []
        for score in result.hypothesis_scores:
            all_scores.extend([score.impact, score.feasibility, score.accessibility, score.sustainability, score.scalability])
        
        # Should have variety in scores, not all 0.5
        unique_scores = set(all_scores)
        assert len(unique_scores) > 1, f"All scores are identical: {unique_scores}"
        
        # Should have reasonable score distribution
        non_default_scores = [s for s in all_scores if abs(s - 0.5) > 0.1]
        assert len(non_default_scores) >= 3, "Too many scores are default 0.5, parsing may be failing"

    @pytest.mark.asyncio  
    async def test_temperature_override_integration(self):
        """Test temperature override with real LLM calls."""
        await self._setup_llm_providers()
        
        # Low temperature should be more conservative
        orchestrator_low = SimpleQADIOrchestrator(temperature_override=0.2)
        result_low = await orchestrator_low.run_qadi_cycle(
            user_input="Creative solutions for urban gardening",
            context="Think of innovative approaches"
        )
        
        # High temperature should be more creative
        orchestrator_high = SimpleQADIOrchestrator(temperature_override=1.5)
        result_high = await orchestrator_high.run_qadi_cycle(
            user_input="Creative solutions for urban gardening", 
            context="Think of innovative approaches"
        )
        
        # Both should work and produce different results
        assert result_low.core_question != ""
        assert result_high.core_question != ""
        assert len(result_low.hypotheses) == 3
        assert len(result_high.hypotheses) == 3
        
        # Results should be different (high chance with different temperatures)
        assert result_low.final_answer != result_high.final_answer

    @pytest.mark.asyncio
    async def test_error_resilience_integration(self):
        """Test system resilience with edge case inputs."""
        await self._setup_llm_providers()
        orchestrator = SimpleQADIOrchestrator()
        
        # Test with very short input
        result = await orchestrator.run_qadi_cycle(
            user_input="Fix climate change",
            context=""
        )
        
        assert isinstance(result, SimpleQADIResult)
        assert result.core_question != ""
        assert len(result.hypotheses) == 3
        
        # Test with philosophical question that might confuse scoring
        result = await orchestrator.run_qadi_cycle(
            user_input="What is the meaning of consciousness in AI?",
            context="Explore philosophical implications"
        )
        
        assert isinstance(result, SimpleQADIResult)
        assert result.core_question != ""
        assert len(result.hypotheses) == 3
        # Should still parse scores even for abstract topics
        assert all(isinstance(score, HypothesisScore) for score in result.hypothesis_scores)

    @pytest.mark.asyncio
    async def test_cost_tracking_integration(self):
        """Test that cost tracking works with real API calls."""
        await self._setup_llm_providers()
        orchestrator = SimpleQADIOrchestrator()
        
        result = await orchestrator.run_qadi_cycle(
            user_input="Improve work-life balance",
            context="Focus on remote work scenarios"
        )
        
        # Should have non-zero cost from real API calls
        assert result.total_llm_cost > 0
        assert result.total_llm_cost < 1.0  # Sanity check - shouldn't be extremely expensive
        
        # Cost should be reasonable for 4 phases (Q-A-D-I)
        # Typical cost should be $0.01-0.10 range for moderate prompts
        assert 0.001 <= result.total_llm_cost <= 0.5

    @pytest.mark.asyncio
    async def test_content_quality_integration(self):
        """Test that generated content meets quality standards."""
        await self._setup_llm_providers()
        orchestrator = SimpleQADIOrchestrator()
        
        result = await orchestrator.run_qadi_cycle(
            user_input="How can we make cities more sustainable?",
            context="Consider environmental, social, and economic factors"
        )
        
        # Core question should be well-formed
        assert result.core_question.strip().endswith("?")
        assert len(result.core_question) > 20
        
        # Hypotheses should be substantial and different
        for hypothesis in result.hypotheses:
            assert len(hypothesis) > 30, f"Hypothesis too short: {hypothesis}"
            assert hypothesis.strip() != ""
        
        # Should have different hypotheses (not duplicated)
        assert len(set(result.hypotheses)) == 3, "Hypotheses should be unique"
        
        # Final answer should be coherent
        assert len(result.final_answer) > 50
        assert result.final_answer.strip() != ""
        
        # Action plan should have actionable steps
        assert len(result.action_plan) >= 2
        for action in result.action_plan:
            assert len(action) > 15, f"Action step too short: {action}"
        
        # Verification examples should be real-world
        assert len(result.verification_examples) >= 2
        for example in result.verification_examples:
            assert len(example) > 30, f"Example too short: {example}"

    @pytest.mark.asyncio
    async def test_different_question_types_integration(self):
        """Test QADI with different types of questions."""
        await self._setup_llm_providers()
        orchestrator = SimpleQADIOrchestrator()
        
        # Technical question
        result_tech = await orchestrator.run_qadi_cycle(
            user_input="How can we optimize database performance?",
            context="Focus on scalable solutions"
        )
        
        # Personal question  
        result_personal = await orchestrator.run_qadi_cycle(
            user_input="How can I improve my productivity while working from home?",
            context="Consider work-life balance"
        )
        
        # Business question
        result_business = await orchestrator.run_qadi_cycle(
            user_input="How should a startup approach market validation?",
            context="Limited budget and resources"
        )
        
        # All should produce valid results
        for result in [result_tech, result_personal, result_business]:
            assert isinstance(result, SimpleQADIResult)
            assert len(result.hypotheses) == 3
            assert len(result.hypothesis_scores) == 3
            assert result.final_answer != ""
            assert result.total_llm_cost > 0
        
        # Results should be contextually different
        tech_keywords = ["database", "performance", "query", "index", "optimize"]
        personal_keywords = ["productivity", "focus", "time", "routine", "balance"]
        business_keywords = ["market", "customer", "validation", "feedback", "product"]
        
        assert any(keyword in result_tech.final_answer.lower() for keyword in tech_keywords)
        assert any(keyword in result_personal.final_answer.lower() for keyword in personal_keywords)  
        assert any(keyword in result_business.final_answer.lower() for keyword in business_keywords)

    @pytest.mark.asyncio
    async def test_num_hypotheses_parameter_integration(self):
        """Test that num_hypotheses parameter works correctly."""
        await self._setup_llm_providers()
        
        # Test with default 3 hypotheses
        orchestrator_default = SimpleQADIOrchestrator()
        result_default = await orchestrator_default.run_qadi_cycle(
            user_input="How can we reduce food waste?",
            context="Focus on household level solutions"
        )
        
        assert len(result_default.hypotheses) == 3
        assert len(result_default.hypothesis_scores) == 3
        
        # Test with custom number of hypotheses
        orchestrator_custom = SimpleQADIOrchestrator(num_hypotheses=4)
        result_custom = await orchestrator_custom.run_qadi_cycle(
            user_input="How can we reduce food waste?",
            context="Focus on household level solutions"
        )
        
        assert len(result_custom.hypotheses) == 4
        assert len(result_custom.hypothesis_scores) == 4

    @pytest.mark.asyncio
    async def test_parsing_robustness_integration(self):
        """Test parsing robustness with questions that might produce varied LLM formats."""
        await self._setup_llm_providers()
        orchestrator = SimpleQADIOrchestrator()
        
        # Question that might produce bullet points, numbers, or other formats
        result = await orchestrator.run_qadi_cycle(
            user_input="What are the best strategies for learning a new programming language?",
            context="Consider different learning styles and time constraints"
        )
        
        # Should successfully parse despite format variations
        assert isinstance(result, SimpleQADIResult)
        assert len(result.hypotheses) == 3
        assert len(result.hypothesis_scores) == 3
        
        # All hypothesis scores should be parsed (not defaults)
        score_values = []
        for score in result.hypothesis_scores:
            score_values.extend([score.impact, score.feasibility, score.accessibility, 
                               score.sustainability, score.scalability])
        
        # Should have score variation, not all 0.5 defaults
        unique_scores = len(set(score_values))
        assert unique_scores >= 5, f"Too little score variation: {set(score_values)}"
        
        # Verify all fields are within valid range
        for score_val in score_values:
            assert 0.0 <= score_val <= 1.0, f"Score out of range: {score_val}"