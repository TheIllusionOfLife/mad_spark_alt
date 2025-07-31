"""
Tests for structured output in SimpleQADIOrchestrator.

This module tests the integration of structured output
with the QADI hypothesis generation and scoring phases.
"""

import asyncio
import json
import pytest
from typing import Dict, Any, List
from unittest.mock import AsyncMock, MagicMock, patch

from mad_spark_alt.core.simple_qadi_orchestrator import (
    SimpleQADIOrchestrator,
    HypothesisScore,
)
from mad_spark_alt.core.llm_provider import LLMResponse, LLMProvider


class TestStructuredHypothesisGeneration:
    """Test structured output for hypothesis generation phase."""

    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator instance."""
        return SimpleQADIOrchestrator(num_hypotheses=3)

    @pytest.mark.asyncio
    async def test_abduction_uses_structured_output(self, orchestrator):
        """Test that abduction phase uses structured output for hypothesis generation."""
        # Mock LLM manager
        with patch("mad_spark_alt.core.simple_qadi_orchestrator.llm_manager") as mock_manager:
            # Create structured response
            structured_response = {
                "hypotheses": [
                    {"id": "H1", "content": "First hypothesis about solving the problem with innovative approach"},
                    {"id": "H2", "content": "Second hypothesis using traditional methods combined with modern technology"},
                    {"id": "H3", "content": "Third hypothesis exploring alternative sustainable solutions"}
                ]
            }
            
            # Mock response
            mock_response = LLMResponse(
                content=json.dumps(structured_response),
                provider=LLMProvider.GOOGLE,
                model="gemini-2.5-flash",
                usage={"prompt_tokens": 100, "completion_tokens": 200},
                cost=0.001
            )
            
            mock_manager.generate = AsyncMock(return_value=mock_response)
            
            # Run abduction phase
            hypotheses, cost = await orchestrator._run_abduction_phase(
                "How can we reduce plastic waste?",
                "Q: What are effective ways to reduce plastic waste in oceans?",
                max_retries=0
            )
            
            # Verify structured output was requested
            mock_manager.generate.assert_called_once()
            request = mock_manager.generate.call_args[0][0]
            
            # Check that request includes response schema
            assert request.response_schema is not None
            assert request.response_mime_type == "application/json"
            
            # Verify schema structure
            schema = request.response_schema
            assert schema["type"] == "OBJECT"
            assert "hypotheses" in schema["properties"]
            assert schema["properties"]["hypotheses"]["type"] == "ARRAY"
            
            # Verify hypotheses were extracted correctly
            assert len(hypotheses) == 3
            assert hypotheses[0] == "First hypothesis about solving the problem with innovative approach"
            assert hypotheses[1] == "Second hypothesis using traditional methods combined with modern technology"
            assert hypotheses[2] == "Third hypothesis exploring alternative sustainable solutions"

    @pytest.mark.asyncio
    async def test_abduction_fallback_to_text_parsing(self, orchestrator):
        """Test fallback to text parsing when structured output is not available."""
        with patch("mad_spark_alt.core.simple_qadi_orchestrator.llm_manager") as mock_manager:
            # Create text response (simulating non-JSON response)
            text_response = """Here are three hypotheses:

H1: First hypothesis about reducing plastic waste through policy changes
H2: Second hypothesis focusing on consumer behavior modification
H3: Third hypothesis developing biodegradable alternatives"""
            
            # Mock response
            mock_response = LLMResponse(
                content=text_response,
                provider=LLMProvider.GOOGLE,
                model="gemini-2.5-flash",
                usage={"prompt_tokens": 100, "completion_tokens": 200},
                cost=0.001
            )
            
            mock_manager.generate = AsyncMock(return_value=mock_response)
            
            # Run abduction phase
            hypotheses, cost = await orchestrator._run_abduction_phase(
                "How can we reduce plastic waste?",
                "Q: What are effective ways to reduce plastic waste?",
                max_retries=0
            )
            
            # Verify hypotheses were extracted using fallback parsing
            assert len(hypotheses) == 3
            assert "policy changes" in hypotheses[0]
            assert "consumer behavior" in hypotheses[1]
            assert "biodegradable alternatives" in hypotheses[2]


class TestStructuredScoreParsing:
    """Test structured output for deduction/scoring phase."""

    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator instance."""
        return SimpleQADIOrchestrator()

    @pytest.mark.asyncio
    async def test_deduction_uses_structured_output(self, orchestrator):
        """Test that deduction phase uses structured output for scoring."""
        # Mock LLM manager
        with patch("mad_spark_alt.core.simple_qadi_orchestrator.llm_manager") as mock_manager:
            # Create structured response
            structured_response = {
                "evaluations": [
                    {
                        "hypothesis_id": "H1",
                        "scores": {
                            "impact": 0.8,
                            "feasibility": 0.7,
                            "accessibility": 0.9,
                            "sustainability": 0.6,
                            "scalability": 0.7
                        }
                    },
                    {
                        "hypothesis_id": "H2",
                        "scores": {
                            "impact": 0.9,
                            "feasibility": 0.6,
                            "accessibility": 0.8,
                            "sustainability": 0.8,
                            "scalability": 0.9
                        }
                    }
                ],
                "answer": "Based on evaluation, H2 provides the best overall solution with high impact and scalability.",
                "action_plan": [
                    "Implement the core strategy from H2",
                    "Start with pilot testing in controlled environment",
                    "Gather feedback and iterate on the approach"
                ]
            }
            
            # Mock response
            mock_response = LLMResponse(
                content=json.dumps(structured_response),
                provider=LLMProvider.GOOGLE,
                model="gemini-2.5-flash",
                usage={"prompt_tokens": 200, "completion_tokens": 300},
                cost=0.002
            )
            
            mock_manager.generate = AsyncMock(return_value=mock_response)
            
            # Test hypotheses
            hypotheses = [
                "First hypothesis about policy changes",
                "Second hypothesis about technology solutions"
            ]
            
            # Run deduction phase
            result = await orchestrator._run_deduction_phase(
                "How can we reduce plastic waste?",
                "Q: What are effective ways to reduce plastic waste?",
                hypotheses,
                max_retries=0
            )
            
            # Verify structured output was requested
            mock_manager.generate.assert_called_once()
            request = mock_manager.generate.call_args[0][0]
            
            # Check that request includes response schema
            assert request.response_schema is not None
            assert request.response_mime_type == "application/json"
            
            # Verify schema structure for deduction
            schema = request.response_schema
            assert schema["type"] == "OBJECT"
            assert all(key in schema["properties"] for key in ["evaluations", "answer", "action_plan"])
            
            # Verify results
            assert len(result["scores"]) == 2
            assert result["scores"][0].impact == 0.8
            assert result["scores"][0].feasibility == 0.7
            assert result["scores"][1].impact == 0.9
            assert result["scores"][1].scalability == 0.9
            
            assert "H2 provides the best" in result["answer"]
            assert len(result["action_plan"]) == 3
            assert result["action_plan"][0] == "Implement the core strategy from H2"

    @pytest.mark.asyncio
    async def test_deduction_handles_invalid_json(self, orchestrator):
        """Test graceful handling when structured output returns invalid JSON."""
        with patch("mad_spark_alt.core.simple_qadi_orchestrator.llm_manager") as mock_manager:
            # Create response that is invalid JSON but contains text format scores
            # This simulates when LLM fails to return proper JSON but still has the content
            invalid_json_with_text = """Invalid JSON but contains scores:
H1:
* Impact: 0.7 - Good potential for reducing waste
* Feasibility: 0.8 - Easy to implement
* Accessibility: 0.9 - Widely accessible
* Sustainability: 0.7 - Long-term viable
* Scalability: 0.6 - Can scale moderately

ANSWER: H1 provides a balanced approach to waste reduction.

Action Plan:
1. Start implementation immediately
2. Monitor progress closely"""
            
            mock_response = LLMResponse(
                content=invalid_json_with_text,
                provider=LLMProvider.GOOGLE,
                model="gemini-2.5-flash",
                usage={"prompt_tokens": 200, "completion_tokens": 300},
                cost=0.002
            )
            
            mock_manager.generate = AsyncMock(return_value=mock_response)
            
            # Run deduction phase
            result = await orchestrator._run_deduction_phase(
                "How can we reduce plastic waste?",
                "Q: What are effective ways to reduce plastic waste?",
                ["First hypothesis"],
                max_retries=0
            )
            
            # Should fall back to text parsing and extract scores from the text
            assert len(result["scores"]) == 1
            assert result["scores"][0].impact == 0.7
            assert result["scores"][0].feasibility == 0.8
            assert result["scores"][0].accessibility == 0.9
            assert result["scores"][0].sustainability == 0.7
            assert result["scores"][0].scalability == 0.6
            assert "balanced approach" in result["answer"]
            assert len(result["action_plan"]) == 2


class TestEndToEndStructuredOutput:
    """Test end-to-end QADI cycle with structured output."""

    @pytest.mark.asyncio
    async def test_full_qadi_cycle_with_structured_output(self):
        """Test complete QADI cycle using structured output."""
        orchestrator = SimpleQADIOrchestrator(num_hypotheses=3)
        
        with patch("mad_spark_alt.core.simple_qadi_orchestrator.llm_manager") as mock_manager:
            # Mock responses for each phase
            responses = []
            
            # 1. Question phase response
            responses.append(LLMResponse(
                content="Q: How can we effectively reduce ocean plastic pollution?",
                provider=LLMProvider.GOOGLE,
                model="gemini-2.5-flash",
                usage={"prompt_tokens": 50, "completion_tokens": 20},
                cost=0.0001
            ))
            
            # 2. Abduction phase response (structured)
            responses.append(LLMResponse(
                content=json.dumps({
                    "hypotheses": [
                        {"id": "H1", "content": "Implement strict regulations on single-use plastics"},
                        {"id": "H2", "content": "Develop advanced ocean cleanup technology"},
                        {"id": "H3", "content": "Create economic incentives for plastic recycling"}
                    ]
                }),
                provider=LLMProvider.GOOGLE,
                model="gemini-2.5-flash",
                usage={"prompt_tokens": 100, "completion_tokens": 150},
                cost=0.001
            ))
            
            # 3. Deduction phase response (structured)
            responses.append(LLMResponse(
                content=json.dumps({
                    "evaluations": [
                        {
                            "hypothesis_id": "H1",
                            "scores": {
                                "impact": 0.9,
                                "feasibility": 0.6,
                                "accessibility": 0.7,
                                "sustainability": 0.8,
                                "scalability": 0.7
                            }
                        },
                        {
                            "hypothesis_id": "H2",
                            "scores": {
                                "impact": 0.8,
                                "feasibility": 0.5,
                                "accessibility": 0.4,
                                "sustainability": 0.7,
                                "scalability": 0.6
                            }
                        },
                        {
                            "hypothesis_id": "H3",
                            "scores": {
                                "impact": 0.7,
                                "feasibility": 0.8,
                                "accessibility": 0.9,
                                "sustainability": 0.9,
                                "scalability": 0.8
                            }
                        }
                    ],
                    "answer": "H1 offers the highest impact for reducing ocean plastic pollution through regulatory measures.",
                    "action_plan": [
                        "Draft comprehensive single-use plastic regulations",
                        "Engage stakeholders for implementation",
                        "Monitor and enforce compliance"
                    ]
                }),
                provider=LLMProvider.GOOGLE,
                model="gemini-2.5-flash",
                usage={"prompt_tokens": 300, "completion_tokens": 400},
                cost=0.003
            ))
            
            # 4. Induction phase response
            responses.append(LLMResponse(
                content="""Example 1: The EU's Single-Use Plastics Directive has shown 35% reduction in ocean plastic waste.

Example 2: California's plastic bag ban reduced beach litter by 72% within two years.

Example 3: Taiwan's comprehensive plastic restrictions led to 95% reduction in plastic straw usage.

Conclusion: The evidence strongly supports that regulatory approaches are highly effective in reducing ocean plastic pollution.""",
                provider=LLMProvider.GOOGLE,
                model="gemini-2.5-flash",
                usage={"prompt_tokens": 200, "completion_tokens": 150},
                cost=0.002
            ))
            
            mock_manager.generate = AsyncMock(side_effect=responses)
            
            # Run full QADI cycle
            result = await orchestrator.run_qadi_cycle(
                "How can we reduce ocean plastic pollution?",
                context="Focus on scalable solutions"
            )
            
            # Verify results
            assert result.core_question == "How can we effectively reduce ocean plastic pollution?"
            assert len(result.hypotheses) == 3
            assert "regulations" in result.hypotheses[0]
            assert "cleanup technology" in result.hypotheses[1]
            assert "economic incentives" in result.hypotheses[2]
            
            # Check scores
            assert len(result.hypothesis_scores) == 3
            assert result.hypothesis_scores[0].impact == 0.9
            assert result.hypothesis_scores[0].feasibility == 0.6
            
            # Check answer and action plan
            assert "H1 offers the highest impact" in result.final_answer
            assert len(result.action_plan) == 3
            assert "Draft comprehensive" in result.action_plan[0]
            
            # Check verification
            assert len(result.verification_examples) == 3
            assert "EU's Single-Use Plastics Directive" in result.verification_examples[0]
            assert "evidence strongly supports" in result.verification_conclusion
            
            # Verify structured output was used for appropriate phases
            assert mock_manager.generate.call_count == 4
            
            # Check abduction call used structured output
            abduction_call = mock_manager.generate.call_args_list[1]
            assert abduction_call[0][0].response_schema is not None
            assert abduction_call[0][0].response_mime_type == "application/json"
            
            # Check deduction call used structured output
            deduction_call = mock_manager.generate.call_args_list[2]
            assert deduction_call[0][0].response_schema is not None
            assert deduction_call[0][0].response_mime_type == "application/json"


@pytest.mark.integration
class TestStructuredOutputIntegrationQADI:
    """Integration tests for structured output in QADI orchestrator."""

    @pytest.mark.asyncio
    async def test_real_qadi_with_structured_output(self):
        """Test real QADI cycle with structured output enabled."""
        import os
        if not os.getenv("GOOGLE_API_KEY"):
            pytest.skip("GOOGLE_API_KEY not set")
        
        # Import and setup
        from mad_spark_alt.core.llm_provider import setup_llm_providers
        await setup_llm_providers(google_api_key=os.getenv("GOOGLE_API_KEY"))
        
        orchestrator = SimpleQADIOrchestrator(num_hypotheses=3)
        
        # Run a simple QADI cycle
        result = await orchestrator.run_qadi_cycle(
            "What are simple ways to save water at home?",
            context="Focus on easy-to-implement solutions"
        )
        
        # Verify structured output was used effectively
        assert result.core_question.startswith("Q:") or "water" in result.core_question.lower()
        assert len(result.hypotheses) >= 3
        assert all(len(h) > 20 for h in result.hypotheses)  # Non-trivial hypotheses
        
        # Check scores are properly extracted
        assert len(result.hypothesis_scores) >= 3
        for score in result.hypothesis_scores:
            assert 0 <= score.impact <= 1
            assert 0 <= score.feasibility <= 1
            assert 0 <= score.accessibility <= 1
            assert 0 <= score.sustainability <= 1
            assert 0 <= score.scalability <= 1
            assert 0 <= score.overall <= 1
        
        # Verify we have a meaningful answer and action plan
        assert len(result.final_answer) > 50
        assert len(result.action_plan) >= 3
        assert all(len(step) > 10 for step in result.action_plan)
        
        # Check total cost is tracked
        assert result.total_llm_cost > 0