"""
Integration tests for phase_logic.py - Full QADI Cycle

Tests that all phases work together correctly in a complete flow.
"""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from mad_spark_alt.core.llm_provider import LLMProvider, LLMResponse
from mad_spark_alt.core.phase_logic import (
    PhaseInput,
    execute_abduction_phase,
    execute_deduction_phase,
    execute_induction_phase,
    execute_questioning_phase,
)


@pytest.fixture
def mock_llm_manager():
    """Mock LLM manager with realistic responses for full cycle."""
    mock = MagicMock()
    mock.generate = AsyncMock()
    return mock


@pytest.fixture
def phase_input(mock_llm_manager):
    """Standard phase input for integration tests."""
    return PhaseInput(
        user_input="How can we reduce plastic waste in oceans?",
        llm_manager=mock_llm_manager,
        context={},
    )


@pytest.mark.asyncio
async def test_full_qadi_cycle_integration(mock_llm_manager, phase_input):
    """Test complete QADI cycle: Question → Abduction → Deduction → Induction."""

    # Phase 1: Questioning - Mock response
    mock_llm_manager.generate.return_value = LLMResponse(
        content="Q: What innovative technologies can effectively remove microplastics from ocean water at scale?",
        provider=LLMProvider.GOOGLE,
        model="gemini-1.5-flash",
        cost=0.001,
    )

    questioning_result = await execute_questioning_phase(phase_input)
    assert questioning_result.core_question
    assert "microplastics" in questioning_result.core_question.lower()
    total_cost = questioning_result.llm_cost

    # Phase 2: Abduction - Mock response
    mock_llm_manager.generate.return_value = LLMResponse(
        content=json.dumps(
            {
                "hypotheses": [
                    {
                        "id": "1",
                        "content": "Deploy autonomous cleanup drones with advanced filtration technology",
                    },
                    {
                        "id": "2",
                        "content": "Implement biodegradable packaging mandates to prevent new pollution",
                    },
                    {
                        "id": "3",
                        "content": "Create ocean filtration systems at major river mouths to intercept plastics",
                    },
                ]
            }
        ),
        provider=LLMProvider.GOOGLE,
        model="gemini-1.5-flash",
        cost=0.005,
    )

    abduction_result = await execute_abduction_phase(
        phase_input, questioning_result.core_question, num_hypotheses=3
    )
    assert len(abduction_result.hypotheses) == 3
    # Verify hypotheses are non-empty and reasonable length
    assert all(len(h) > 20 for h in abduction_result.hypotheses)
    total_cost += abduction_result.llm_cost

    # Phase 3: Deduction - Mock response
    mock_llm_manager.generate.return_value = LLMResponse(
        content=json.dumps(
            {
                "evaluations": [
                    {
                        "hypothesis_id": "1",
                        "scores": {
                            "impact": 0.8,
                            "feasibility": 0.7,
                            "accessibility": 0.6,
                            "sustainability": 0.75,
                            "scalability": 0.65,
                        },
                    },
                    {
                        "hypothesis_id": "2",
                        "scores": {
                            "impact": 0.9,
                            "feasibility": 0.6,
                            "accessibility": 0.7,
                            "sustainability": 0.9,
                            "scalability": 0.8,
                        },
                    },
                    {
                        "hypothesis_id": "3",
                        "scores": {
                            "impact": 0.85,
                            "feasibility": 0.5,
                            "accessibility": 0.6,
                            "sustainability": 0.7,
                            "scalability": 0.6,
                        },
                    },
                ],
                "answer": "A combination of preventive measures (biodegradable packaging mandates) and active cleanup (river mouth filtration systems) provides the most sustainable solution.",
                "action_plan": [
                    "Conduct feasibility study for river mouth filtration systems",
                    "Develop international biodegradable packaging standards",
                    "Create pilot program in 3 major rivers",
                    "Establish monitoring and evaluation framework",
                ],
            }
        ),
        provider=LLMProvider.GOOGLE,
        model="gemini-1.5-flash",
        cost=0.008,
    )

    deduction_result = await execute_deduction_phase(
        phase_input, questioning_result.core_question, abduction_result.hypotheses
    )
    assert len(deduction_result.hypothesis_scores) == 3
    assert deduction_result.answer
    assert len(deduction_result.action_plan) == 4
    assert all(score.overall > 0 for score in deduction_result.hypothesis_scores)
    total_cost += deduction_result.llm_cost

    # Phase 4: Induction - Mock response
    mock_llm_manager.generate.return_value = LLMResponse(
        content="""
        Example 1: The city of Amsterdam implemented biodigradable packaging mandates in 2020, reducing plastic waste by 40% within 2 years while maintaining economic growth.

        Example 2: Singapore's Marina Bay filtration system successfully intercepts 80% of plastic debris from entering the ocean, demonstrating the viability of river mouth installations.

        Example 3: The European Union's comprehensive approach combining legislation and infrastructure investment has reduced Mediterranean plastic pollution by 35% since 2018.

        Conclusion: The recommended approach has been successfully implemented across multiple contexts, demonstrating both environmental effectiveness and economic viability at scale.
        """,
        provider=LLMProvider.GOOGLE,
        model="gemini-1.5-flash",
        cost=0.004,
    )

    induction_result = await execute_induction_phase(
        phase_input,
        questioning_result.core_question,
        deduction_result.answer,
        abduction_result.hypotheses,
    )
    assert len(induction_result.examples) >= 3
    assert induction_result.conclusion
    assert "recommended approach" in induction_result.conclusion.lower()
    total_cost += induction_result.llm_cost

    # Verify cost accumulation (with floating point tolerance)
    assert abs(total_cost - 0.018) < 0.0001  # Sum of all phase costs

    # Verify data flow
    assert "microplastics" in questioning_result.core_question.lower()
    assert len(abduction_result.hypotheses) == 3
    assert len(deduction_result.hypothesis_scores) == 3
    assert len(induction_result.examples) >= 3


@pytest.mark.asyncio
async def test_phase_error_propagation(mock_llm_manager, phase_input):
    """Test that errors in one phase prevent downstream execution."""

    # Mock a persistent failure in questioning phase
    mock_llm_manager.generate.side_effect = Exception("Persistent API error")

    with pytest.raises(RuntimeError, match="Failed to extract core question"):
        await execute_questioning_phase(phase_input)

    # Verify that we don't proceed to abduction with failed questioning
    # (This is a behavioral test - in practice, the orchestrator handles this)


@pytest.mark.asyncio
async def test_phase_independence(mock_llm_manager, phase_input):
    """Verify each phase can be called independently with appropriate inputs."""

    # Each phase should work independently when given valid inputs

    # Test questioning independently
    mock_llm_manager.generate.return_value = LLMResponse(
        content="Q: Independent question test",
        provider=LLMProvider.GOOGLE,
        model="gemini-1.5-flash",
        cost=0.001,
    )
    q_result = await execute_questioning_phase(phase_input)
    assert q_result.core_question == "Independent question test"

    # Test abduction independently
    mock_llm_manager.generate.return_value = LLMResponse(
        content=json.dumps(
            {
                "hypotheses": [
                    {
                        "id": "1",
                        "content": "Independent hypothesis one for testing extraction",
                    },
                    {
                        "id": "2",
                        "content": "Independent hypothesis two for testing extraction",
                    },
                ]
            }
        ),
        provider=LLMProvider.GOOGLE,
        model="gemini-1.5-flash",
        cost=0.002,
    )
    a_result = await execute_abduction_phase(
        phase_input, "Test question?", num_hypotheses=2
    )
    assert len(a_result.hypotheses) == 2

    # Test deduction independently
    mock_llm_manager.generate.return_value = LLMResponse(
        content=json.dumps(
            {
                "evaluations": [
                    {
                        "hypothesis_id": "1",
                        "scores": {
                            "impact": 0.7,
                            "feasibility": 0.7,
                            "accessibility": 0.7,
                            "sustainability": 0.7,
                            "scalability": 0.7,
                        },
                    },
                ],
                "answer": "Independent answer",
                "action_plan": ["Action 1"],
            }
        ),
        provider=LLMProvider.GOOGLE,
        model="gemini-1.5-flash",
        cost=0.003,
    )
    d_result = await execute_deduction_phase(
        phase_input,
        "Test Q?",
        ["Test hypothesis one with enough length to pass validation"],
    )
    assert d_result.answer == "Independent answer"

    # Test induction independently
    mock_llm_manager.generate.return_value = LLMResponse(
        content="Example 1: Test\nConclusion: Independent conclusion test",
        provider=LLMProvider.GOOGLE,
        model="gemini-1.5-flash",
        cost=0.001,
    )
    i_result = await execute_induction_phase(
        phase_input, "Test Q?", "Test answer", ["H1"]
    )
    assert "Independent conclusion test" in i_result.conclusion
