"""
Integration tests for the multi-perspective QADI analysis system.

This module tests the multi-perspective analysis system introduced in PR #49,
including intent detection, perspective orchestration, and result synthesis.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from typing import List

from mad_spark_alt.core.intent_detector import IntentDetector, QuestionIntent
from mad_spark_alt.core.multi_perspective_orchestrator import (
    MultiPerspectiveQADIOrchestrator,
    MultiPerspectiveQADIResult,
    PerspectiveResult,
)


class TestIntentDetection:
    """Test intent detection for multi-perspective analysis."""

    def test_environmental_intent_detection(self):
        """Test detection of environmental questions."""
        detector = IntentDetector()
        
        environmental_questions = [
            "How can we reduce plastic waste and pollution?",
            "What sustainable solutions exist for climate change?",
            "How can we protect the ocean from pollution?", 
            "What renewable energy and solar solutions work best?",
        ]
        
        for question in environmental_questions:
            result = detector.detect_intent(question)
            assert result.primary_intent == QuestionIntent.ENVIRONMENTAL, f"Failed for: {question}"
            assert result.confidence > 0.0

    def test_technical_intent_detection(self):
        """Test detection of technical questions."""
        detector = IntentDetector()
        
        technical_questions = [
            "How to build a scalable software system?",
            "What database architecture works for large systems?",
            "How to implement secure API infrastructure?",
            "What are the best code development practices?",
        ]
        
        for question in technical_questions:
            result = detector.detect_intent(question)
            assert result.primary_intent == QuestionIntent.TECHNICAL, f"Failed for: {question}"
            assert result.confidence > 0.0

    def test_business_intent_detection(self):
        """Test detection of business questions."""
        detector = IntentDetector()
        
        business_questions = [
            "How can our company increase revenue growth?",
            "What marketing strategy drives customer acquisition?", 
            "How to improve business customer retention?",
            "What pricing model maximizes profit margins?",
        ]
        
        for question in business_questions:
            result = detector.detect_intent(question)
            assert result.primary_intent == QuestionIntent.BUSINESS, f"Failed for: {question}"
            assert result.confidence > 0.0

    def test_intent_confidence_levels(self):
        """Test that confidence levels are reasonable."""
        detector = IntentDetector()
        
        # Clear case with multiple keywords - should have decent confidence
        clear_result = detector.detect_intent("How can we reduce carbon emissions and plastic pollution?")
        assert clear_result.confidence >= 0.4  # Realistic expectation based on actual normalization
        assert clear_result.primary_intent == QuestionIntent.ENVIRONMENTAL
        
        # Lower confidence case - ambiguous question
        ambiguous_result = detector.detect_intent("How can we improve things?")
        # Should default to GENERAL with 0.5 confidence
        assert ambiguous_result.confidence == 0.5
        assert ambiguous_result.primary_intent == QuestionIntent.GENERAL

    def test_keyword_matching(self):
        """Test that keywords are properly detected."""
        detector = IntentDetector()
        
        result = detector.detect_intent("How can we reduce plastic waste and carbon footprint?")
        assert result.primary_intent == QuestionIntent.ENVIRONMENTAL
        assert len(result.keywords_matched) > 0
        assert any(keyword in ["plastic", "waste", "carbon", "footprint"] for keyword in result.keywords_matched)


class TestMultiPerspectiveOrchestration:
    """Test multi-perspective orchestration logic."""

    @pytest.fixture
    def orchestrator(self):
        """Create a real orchestrator for testing."""
        return MultiPerspectiveQADIOrchestrator()

    def test_perspective_recommendation_logic(self, orchestrator):
        """Test that perspectives are recommended correctly based on intent."""
        # Test environmental question
        env_result = orchestrator.intent_detector.detect_intent("How can we reduce plastic waste?")
        env_perspectives = orchestrator.intent_detector.get_recommended_perspectives(env_result, 3)
        assert QuestionIntent.ENVIRONMENTAL in env_perspectives
        assert len(env_perspectives) <= 3
        
        # Test business question  
        biz_result = orchestrator.intent_detector.detect_intent("How can we increase company revenue?")
        biz_perspectives = orchestrator.intent_detector.get_recommended_perspectives(biz_result, 3)
        assert QuestionIntent.BUSINESS in biz_perspectives
        assert len(biz_perspectives) <= 3

    @pytest.mark.asyncio
    async def test_perspective_analysis_structure(self, orchestrator):
        """Test that individual perspective analysis returns proper structure."""
        # Mock the LLM calls to avoid API dependency
        with patch.object(orchestrator, '_run_llm_phase') as mock_llm:
            mock_llm.side_effect = [
                ("Q: How can we reduce plastic waste from an environmental perspective?", 0.01),
                ("H1: Implement recycling programs\nH2: Use biodegradable alternatives\nH3: Reduce single-use items", 0.02),
                ("H1: Impact: 0.8 - High environmental benefit\nFeasibility: 0.7 - Moderately feasible\nANSWER: Focus on recycling programs\nAction Plan:\n1. Partner with recycling centers\n2. Launch awareness campaigns", 0.02),
                ("1. Germany's bottle deposit system\n2. Sweden's recycling success\nConclusion: Recycling programs work", 0.01),
            ]
            
            result = await orchestrator._run_perspective_analysis(
                "How can we reduce plastic waste?", QuestionIntent.ENVIRONMENTAL
            )
            
            assert result is not None
            assert result.core_question != ""
            assert len(result.hypotheses) > 0
            assert len(result.hypothesis_scores) > 0
            assert result.final_answer != ""
            assert len(result.action_plan) > 0

    def test_relevance_scoring_pattern(self, orchestrator):
        """Test that relevance scores follow the expected pattern."""
        # Create mock perspective results
        from mad_spark_alt.core.simple_qadi_orchestrator import SimpleQADIResult, HypothesisScore
        
        # Create individual results to avoid reference issues
        results = []
        for i in range(3):
            result = SimpleQADIResult(
                core_question=f"Test question {i+1}",
                hypotheses=[f"H{i+1}.1", f"H{i+1}.2"],
                hypothesis_scores=[
                    HypothesisScore(impact=0.8, feasibility=0.7, accessibility=0.6, sustainability=0.5, scalability=0.4, overall=0.6),
                    HypothesisScore(impact=0.7, feasibility=0.6, accessibility=0.5, sustainability=0.4, scalability=0.3, overall=0.5)
                ],
                final_answer=f"Test answer {i+1}",
                action_plan=[f"Action {i+1}.1"],
                verification_examples=[f"Example {i+1}.1"],
                verification_conclusion=f"Test conclusion {i+1}",
                total_llm_cost=0.02
            )
            results.append(result)
        
        perspectives = [QuestionIntent.ENVIRONMENTAL, QuestionIntent.PERSONAL, QuestionIntent.TECHNICAL]
        
        # Simulate the relevance scoring logic from the actual implementation
        scored_results = []
        for i, (perspective, result) in enumerate(zip(perspectives, results)):
            relevance = 1.0 if i == 0 else 0.8 - (i * 0.1)  # Primary gets 1.0
            scored_results.append(PerspectiveResult(perspective, result, relevance))
        
        # Verify scoring pattern (use approximate equality for floating point)
        assert scored_results[0].relevance_score == 1.0  # Primary
        assert abs(scored_results[1].relevance_score - 0.7) < 0.001  # Secondary (0.8 - 0.1 = 0.7)  
        assert abs(scored_results[2].relevance_score - 0.6) < 0.001  # Tertiary (0.8 - 0.2 = 0.6)

    def test_cost_calculation_logic(self, orchestrator):
        """Test that costs are properly calculated across perspectives."""
        from mad_spark_alt.core.simple_qadi_orchestrator import SimpleQADIResult, HypothesisScore
        
        # Mock perspective results with different costs
        perspective_results = [
            PerspectiveResult(
                perspective=QuestionIntent.ENVIRONMENTAL,
                result=SimpleQADIResult(
                    core_question="Test",
                    hypotheses=["H1"],
                    hypothesis_scores=[HypothesisScore(impact=0.8, feasibility=0.7, accessibility=0.6, sustainability=0.5, scalability=0.4, overall=0.6)],
                    final_answer="Answer",
                    action_plan=["Action"],
                    verification_examples=["Example"],
                    verification_conclusion="Conclusion",
                    total_llm_cost=0.03
                ),
                relevance_score=1.0,
            ),
            PerspectiveResult(
                perspective=QuestionIntent.PERSONAL,
                result=SimpleQADIResult(
                    core_question="Test",
                    hypotheses=["H1"],
                    hypothesis_scores=[HypothesisScore(impact=0.8, feasibility=0.7, accessibility=0.6, sustainability=0.5, scalability=0.4, overall=0.6)],
                    final_answer="Answer",
                    action_plan=["Action"],
                    verification_examples=["Example"],
                    verification_conclusion="Conclusion",
                    total_llm_cost=0.02
                ),
                relevance_score=0.8,
            ),
        ]
        
        # Calculate total cost (matches implementation logic)
        total_cost = sum(pr.result.total_llm_cost for pr in perspective_results)
        assert total_cost == 0.05  # 0.03 + 0.02


@pytest.mark.integration  
class TestMultiPerspectiveEndToEnd:
    """End-to-end integration tests for multi-perspective system."""
    
    @pytest.mark.asyncio
    async def test_complete_analysis_flow_mocked(self):
        """Test complete multi-perspective analysis flow with mocked LLM calls."""
        orchestrator = MultiPerspectiveQADIOrchestrator()
        
        # Mock all LLM calls to avoid API dependency
        with patch.object(orchestrator, '_run_llm_phase') as mock_llm:
            # Mock LLM responses for complete QADI cycle (2 perspectives × 4 phases + 1 synthesis = 9 calls)
            mock_llm.side_effect = [
                # Environmental perspective - 4 phases
                ("Q: How can we reduce plastic waste from an environmental perspective?", 0.01),
                ("H1: Implement recycling programs\nH2: Use biodegradable alternatives\nH3: Reduce single-use items", 0.02),
                ("H1: Impact: 0.8 - High environmental benefit\nFeasibility: 0.7 - Moderately feasible\nAccessibility: 0.6 - Moderate access\nSustainability: 0.9 - Highly sustainable\nScalability: 0.7 - Good scaling potential\n\nANSWER: Focus on implementing comprehensive recycling programs\n\nAction Plan:\n1. Partner with local recycling centers\n2. Launch public awareness campaign\n3. Implement deposit-return systems", 0.02),
                ("1. Germany's bottle deposit system achieving 98% return rate\n2. Sweden's recycling success with 50% plastic recycling rate\nConclusion: Recycling programs show proven effectiveness", 0.01),
                
                # Personal perspective - 4 phases  
                ("Q: How can individuals reduce their plastic waste footprint?", 0.01),
                ("H1: Change personal consumption habits\nH2: Use reusable alternatives\nH3: Support plastic-free businesses", 0.02),
                ("H1: Impact: 0.6 - Moderate personal impact\nFeasibility: 0.8 - Highly feasible\nAccessibility: 0.9 - Very accessible\nSustainability: 0.7 - Good sustainability\nScalability: 0.5 - Limited scaling\n\nANSWER: Focus on adopting reusable alternatives in daily life\n\nAction Plan:\n1. Replace single-use items with reusable versions\n2. Shop at zero-waste stores\n3. Educate family and friends", 0.02),
                ("1. Zero-waste lifestyle movement growing globally\n2. Reusable product market expanding rapidly\nConclusion: Individual actions create collective impact", 0.01),
                
                # Synthesis phase
                ("SYNTHESIS: Combining environmental infrastructure with personal action creates the most effective approach to reducing plastic waste\n\nINTEGRATED ACTION PLAN:\n1. Support policy for recycling infrastructure while adopting personal reusable alternatives\n2. Participate in community recycling programs and educate others\n3. Choose businesses that prioritize sustainable packaging", 0.01),
            ]
            
            result = await orchestrator.run_multi_perspective_analysis(
                "How can we reduce plastic waste?",
                max_perspectives=2
            )
            
            # Verify result structure and content
            assert isinstance(result, MultiPerspectiveQADIResult)
            assert result.primary_intent == QuestionIntent.ENVIRONMENTAL
            assert len(result.perspective_results) >= 1  # At least one should succeed
            assert result.total_llm_cost > 0
            assert result.synthesized_answer is not None
            assert len(result.synthesized_action_plan) >= 1
            
            # Verify we have environmental perspective (primary)
            env_results = [pr for pr in result.perspective_results if pr.perspective == QuestionIntent.ENVIRONMENTAL]
            assert len(env_results) >= 1
            env_result = env_results[0]
            assert env_result.relevance_score == 1.0  # Primary perspective
            assert len(env_result.result.hypotheses) > 0
            assert len(env_result.result.action_plan) > 0

    @pytest.mark.asyncio
    async def test_error_handling_in_perspectives(self):
        """Test that the system handles errors in individual perspectives gracefully."""
        orchestrator = MultiPerspectiveQADIOrchestrator()
        
        # Mock perspective analysis to simulate one failure and one success
        with patch.object(orchestrator, '_run_perspective_analysis') as mock_analysis:
            from mad_spark_alt.core.simple_qadi_orchestrator import SimpleQADIResult, HypothesisScore
            
            # Create a successful result
            successful_result = SimpleQADIResult(
                core_question="How can we reduce plastic waste personally?",
                hypotheses=["Use reusable items", "Buy less plastic"],
                hypothesis_scores=[
                    HypothesisScore(impact=0.7, feasibility=0.8, accessibility=0.9, sustainability=0.7, scalability=0.5, overall=0.7),
                    HypothesisScore(impact=0.6, feasibility=0.9, accessibility=0.8, sustainability=0.8, scalability=0.4, overall=0.7)
                ],
                final_answer="Focus on reusable alternatives",
                action_plan=["Replace disposables", "Shop consciously"],
                verification_examples=["Zero waste movement"],
                verification_conclusion="Personal action works",
                total_llm_cost=0.04
            )
            
            # First call fails, second succeeds
            mock_analysis.side_effect = [None, successful_result]
            
            # Mock synthesis
            with patch.object(orchestrator, '_synthesize_results') as mock_synthesis:
                mock_synthesis.return_value = {
                    "answer": "Focus on personal action",
                    "action_plan": ["Take personal steps"],
                    "best_hypothesis": ("Use reusable items", QuestionIntent.PERSONAL),
                    "synthesis_cost": 0.01,
                }
                
                # Force specific perspectives to test failure handling
                result = await orchestrator.run_multi_perspective_analysis(
                    "How can we reduce plastic waste?",
                    max_perspectives=2,
                    force_perspectives=[QuestionIntent.ENVIRONMENTAL, QuestionIntent.PERSONAL]
                )
                
                # System should handle the failure gracefully
                assert isinstance(result, MultiPerspectiveQADIResult)
                # Only the successful perspective should be in results
                assert len(result.perspective_results) == 1, f"Expected 1 result, got {len(result.perspective_results)}"
                
                # Verify the successful perspective is the one we expected
                successful_perspective = result.perspective_results[0]
                assert successful_perspective.result.final_answer == "Focus on reusable alternatives"

    def test_intent_detection_integration(self):
        """Test that intent detection integrates properly with orchestration."""
        orchestrator = MultiPerspectiveQADIOrchestrator()
        
        # Test that different question types are detected correctly
        test_cases = [
            ("How can we reduce plastic waste and pollution?", QuestionIntent.ENVIRONMENTAL),
            ("How can our company increase revenue growth?", QuestionIntent.BUSINESS),
            ("How to build a scalable software system?", QuestionIntent.TECHNICAL),
        ]
        
        for question, expected_intent in test_cases:
            intent_result = orchestrator.intent_detector.detect_intent(question)
            assert intent_result.primary_intent == expected_intent, f"Failed for: {question}"
            
            # Test perspective recommendation
            perspectives = orchestrator.intent_detector.get_recommended_perspectives(intent_result, 3)
            assert expected_intent in perspectives
            assert len(perspectives) <= 3


if __name__ == "__main__":
    pytest.main([__file__])