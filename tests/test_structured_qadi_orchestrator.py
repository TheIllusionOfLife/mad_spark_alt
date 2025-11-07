"""
Tests for structured output in SimpleQADIOrchestrator.

This module tests the integration of structured output with the QADI cycle.

Note: Individual phase logic tests have been moved to test_phase_logic.py after
the refactoring in PR #111. This file now focuses on end-to-end structured output testing.
"""

import json
import pytest
from unittest.mock import AsyncMock, patch

from mad_spark_alt.core.simple_qadi_orchestrator import SimpleQADIOrchestrator
from mad_spark_alt.core.llm_provider import LLMResponse, LLMProvider


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
