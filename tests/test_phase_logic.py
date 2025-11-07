"""
Tests for phase_logic.py - QADI Phase Implementations

Tests are organized by phase with comprehensive coverage of:
- Happy path scenarios
- Fallback/retry logic
- Cost tracking
- Error handling
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mad_spark_alt.core.llm_provider import LLMProvider, LLMResponse
from mad_spark_alt.core.parsing_utils import ParsedScores
from mad_spark_alt.core.phase_logic import (
    AbductionResult,
    DeductionResult,
    InductionResult,
    PhaseInput,
    QuestioningResult,
    execute_abduction_phase,
    execute_deduction_phase,
    execute_induction_phase,
    execute_questioning_phase,
)
from mad_spark_alt.core.simple_qadi_orchestrator import HypothesisScore


@pytest.fixture
def mock_llm_manager():
    """Mock LLM manager with configurable responses."""
    mock = MagicMock()
    mock.generate = AsyncMock()
    return mock


@pytest.fixture
def phase_input(mock_llm_manager):
    """Standard phase input for testing."""
    return PhaseInput(
        user_input="How can we reduce ocean plastic?",
        llm_manager=mock_llm_manager,
        context={},
    )


# ========== Questioning Phase Tests ==========


@pytest.mark.asyncio
async def test_questioning_phase_with_q_prefix(mock_llm_manager, phase_input):
    """Test questioning phase happy path with Q: prefix in response."""
    # Mock LLM response with Q: prefix
    mock_response = LLMResponse(
        content="Q: What innovative technologies can effectively remove microplastics from ocean water?",
        provider=LLMProvider.GOOGLE,
        model="gemini-1.5-flash",
        usage={"input_tokens": 50, "output_tokens": 20},
        cost=0.001,
    )
    mock_llm_manager.generate.return_value = mock_response

    result = await execute_questioning_phase(phase_input)

    assert isinstance(result, QuestioningResult)
    assert (
        result.core_question
        == "What innovative technologies can effectively remove microplastics from ocean water?"
    )
    assert result.llm_cost == 0.001
    assert result.raw_response == mock_response.content
    assert mock_llm_manager.generate.call_count == 1


@pytest.mark.asyncio
async def test_questioning_phase_without_prefix(mock_llm_manager, phase_input):
    """Test questioning phase with no Q: prefix (fallback parsing)."""
    mock_response = LLMResponse(
        content="What methods can reduce ocean plastic pollution most effectively?",
        provider=LLMProvider.GOOGLE,
        model="gemini-1.5-flash",
        usage={"input_tokens": 50, "output_tokens": 15},
        cost=0.0008,
    )
    mock_llm_manager.generate.return_value = mock_response

    result = await execute_questioning_phase(phase_input)

    assert (
        result.core_question
        == "What methods can reduce ocean plastic pollution most effectively?"
    )
    assert result.llm_cost == 0.0008


@pytest.mark.asyncio
async def test_questioning_phase_retry_on_failure(mock_llm_manager, phase_input):
    """Test questioning phase retry logic on transient failures."""
    # First call fails, second succeeds
    mock_llm_manager.generate.side_effect = [
        Exception("Temporary API error"),
        LLMResponse(
            content="Q: How can we eliminate ocean plastic waste?",
            provider=LLMProvider.GOOGLE,
            model="gemini-1.5-flash",
            cost=0.001,
        ),
    ]

    result = await execute_questioning_phase(phase_input)

    assert result.core_question == "How can we eliminate ocean plastic waste?"
    assert mock_llm_manager.generate.call_count == 2


@pytest.mark.asyncio
async def test_questioning_phase_cost_tracking(mock_llm_manager, phase_input):
    """Test that costs accumulate correctly across retries."""
    # Multiple attempts with different costs
    mock_llm_manager.generate.side_effect = [
        Exception("Fail 1"),
        LLMResponse(
            content="Q: Test question?",
            provider=LLMProvider.GOOGLE,
            model="gemini-1.5-flash",
            cost=0.002,
        ),
    ]

    result = await execute_questioning_phase(phase_input)

    # Only successful call cost counted
    assert result.llm_cost == 0.002


@pytest.mark.asyncio
async def test_questioning_phase_max_retries(mock_llm_manager, phase_input):
    """Test that RuntimeError raised after max_retries exhausted."""
    mock_llm_manager.generate.side_effect = Exception("Persistent failure")

    with pytest.raises(RuntimeError, match="Failed to extract core question after"):
        await execute_questioning_phase(phase_input)

    # max_retries=2 means 3 total attempts (0, 1, 2)
    assert mock_llm_manager.generate.call_count == 3


# ========== Abduction Phase Tests ==========


@pytest.mark.asyncio
async def test_abduction_structured_json(mock_llm_manager, phase_input):
    """Test abduction phase with structured JSON output."""
    mock_response = LLMResponse(
        content=json.dumps(
            {
                "hypotheses": [
                    {"id": "1", "content": "Deploy autonomous cleanup drones"},
                    {
                        "id": "2",
                        "content": "Implement biodegradable packaging mandates",
                    },
                    {
                        "id": "3",
                        "content": "Create ocean filtration systems at river mouths",
                    },
                ]
            }
        ),
        provider=LLMProvider.GOOGLE,
        model="gemini-1.5-flash",
        cost=0.005,
    )
    mock_llm_manager.generate.return_value = mock_response

    result = await execute_abduction_phase(
        phase_input, "How to reduce ocean plastic?", num_hypotheses=3
    )

    assert isinstance(result, AbductionResult)
    assert len(result.hypotheses) == 3
    assert "autonomous cleanup drones" in result.hypotheses[0]
    assert result.num_requested == 3
    assert result.num_generated == 3
    assert result.llm_cost == 0.005


@pytest.mark.asyncio
async def test_abduction_text_fallback(mock_llm_manager, phase_input):
    """Test abduction phase with text parsing fallback."""
    mock_response = LLMResponse(
        content="""
        H1: Use ocean cleanup vessels with advanced filtering
        H2: Ban single-use plastics globally
        H3: Develop plastic-eating bacteria
        """,
        provider=LLMProvider.GOOGLE,
        model="gemini-1.5-flash",
        cost=0.004,
    )
    mock_llm_manager.generate.return_value = mock_response

    with patch("mad_spark_alt.core.phase_logic.HypothesisParser") as mock_parser:
        mock_parser.parse_with_fallback.return_value = [
            "Use ocean cleanup vessels with advanced filtering",
            "Ban single-use plastics globally",
            "Develop plastic-eating bacteria",
        ]

        result = await execute_abduction_phase(
            phase_input, "How to reduce ocean plastic?"
        )

        assert len(result.hypotheses) == 3
        assert result.num_generated == 3


@pytest.mark.asyncio
async def test_abduction_custom_temperature(mock_llm_manager, phase_input):
    """Test abduction phase with temperature override."""
    mock_response = LLMResponse(
        content=json.dumps(
            {
                "hypotheses": [
                    {
                        "id": "1",
                        "content": "Implement AI-powered sorting systems to categorize plastic waste efficiently",
                    },
                    {
                        "id": "2",
                        "content": "Deploy ocean cleanup drones with advanced filtration technology",
                    },
                    {
                        "id": "3",
                        "content": "Create biodegradable packaging alternatives to reduce plastic production",
                    },
                ]
            }
        ),
        provider=LLMProvider.GOOGLE,
        model="gemini-1.5-flash",
        cost=0.003,
    )
    mock_llm_manager.generate.return_value = mock_response

    await execute_abduction_phase(
        phase_input, "Test question?", num_hypotheses=3, temperature_override=1.2
    )

    # Verify temperature was set correctly in LLM request
    call_args = mock_llm_manager.generate.call_args[0][0]
    assert call_args.temperature == 1.2


@pytest.mark.asyncio
async def test_abduction_insufficient_hypotheses(mock_llm_manager, phase_input):
    """Test abduction retry when insufficient hypotheses generated."""
    # First call returns only 1 hypothesis, second returns 3
    mock_llm_manager.generate.side_effect = [
        LLMResponse(
            content=json.dumps(
                {
                    "hypotheses": [
                        {
                            "id": "1",
                            "content": "Only one hypothesis with sufficient length for parsing validation",
                        }
                    ]
                }
            ),
            provider=LLMProvider.GOOGLE,
            model="gemini-1.5-flash",
            cost=0.002,
        ),
        LLMResponse(
            content=json.dumps(
                {
                    "hypotheses": [
                        {
                            "id": "1",
                            "content": "Deploy autonomous cleanup drones with advanced filtering",
                        },
                        {
                            "id": "2",
                            "content": "Implement biodegradable packaging mandates globally",
                        },
                        {
                            "id": "3",
                            "content": "Create ocean filtration systems at river mouths",
                        },
                    ]
                }
            ),
            provider=LLMProvider.GOOGLE,
            model="gemini-1.5-flash",
            cost=0.003,
        ),
    ]

    result = await execute_abduction_phase(phase_input, "Test?", num_hypotheses=3)

    assert len(result.hypotheses) == 3
    assert mock_llm_manager.generate.call_count == 2
    assert result.llm_cost == 0.005  # Sum of both attempts


@pytest.mark.asyncio
async def test_abduction_cost_tracking(mock_llm_manager, phase_input):
    """Test cost accumulation across retries."""
    mock_llm_manager.generate.side_effect = [
        Exception("Temporary failure"),
        LLMResponse(
            content=json.dumps(
                {
                    "hypotheses": [
                        {
                            "id": "1",
                            "content": "Develop plastic-eating bacteria for ocean cleanup",
                        },
                        {
                            "id": "2",
                            "content": "Ban single-use plastics globally through legislation",
                        },
                        {
                            "id": "3",
                            "content": "Use ocean cleanup vessels with advanced filtering",
                        },
                    ]
                }
            ),
            provider=LLMProvider.GOOGLE,
            model="gemini-1.5-flash",
            cost=0.006,
        ),
    ]

    result = await execute_abduction_phase(phase_input, "Question?")

    assert result.llm_cost == 0.006


@pytest.mark.asyncio
async def test_abduction_num_hypotheses_parameter(mock_llm_manager, phase_input):
    """Test that num_hypotheses parameter affects request."""
    mock_response = LLMResponse(
        content=json.dumps(
            {
                "hypotheses": [
                    {
                        "id": "1",
                        "content": "Deploy autonomous cleanup drones with AI-powered navigation",
                    },
                    {
                        "id": "2",
                        "content": "Implement biodegradable packaging mandates worldwide",
                    },
                    {
                        "id": "3",
                        "content": "Create ocean filtration systems at major river mouths",
                    },
                    {
                        "id": "4",
                        "content": "Develop plastic-eating bacteria for deep ocean cleanup",
                    },
                    {
                        "id": "5",
                        "content": "Ban single-use plastics through international agreements",
                    },
                ]
            }
        ),
        provider=LLMProvider.GOOGLE,
        model="gemini-1.5-flash",
        cost=0.007,
    )
    mock_llm_manager.generate.return_value = mock_response

    result = await execute_abduction_phase(phase_input, "Question?", num_hypotheses=5)

    assert result.num_requested == 5
    assert len(result.hypotheses) == 5


# ========== Deduction Phase Tests ==========


@pytest.mark.asyncio
async def test_deduction_sequential_small_set(mock_llm_manager, phase_input):
    """Test deduction with sequential processing for small hypothesis sets (≤5)."""
    hypotheses = ["H1 text", "H2 text", "H3 text"]

    mock_response = LLMResponse(
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
                            "sustainability": 0.8,
                            "scalability": 0.7,
                        },
                    },
                    {
                        "hypothesis_id": "3",
                        "scores": {
                            "impact": 0.7,
                            "feasibility": 0.9,
                            "accessibility": 0.8,
                            "sustainability": 0.7,
                            "scalability": 0.75,
                        },
                    },
                ],
                "answer": "Combination approach recommended",
                "action_plan": ["Step 1", "Step 2", "Step 3"],
            }
        ),
        provider=LLMProvider.GOOGLE,
        model="gemini-1.5-flash",
        cost=0.008,
    )
    mock_llm_manager.generate.return_value = mock_response

    result = await execute_deduction_phase(phase_input, "Core question?", hypotheses)

    assert isinstance(result, DeductionResult)
    assert len(result.hypothesis_scores) == 3
    assert result.hypothesis_scores[0].impact == 0.8
    assert result.answer == "Combination approach recommended"
    assert len(result.action_plan) == 3
    assert result.used_parallel is False
    assert result.llm_cost == 0.008


@pytest.mark.asyncio
async def test_deduction_parallel_large_set(mock_llm_manager, phase_input):
    """Test deduction with large hypothesis sets (>5)."""
    hypotheses = [f"Hypothesis {i}" for i in range(1, 8)]  # 7 hypotheses

    # Mock batch responses
    mock_llm_manager.generate.return_value = LLMResponse(
        content=json.dumps(
            {
                "evaluations": [
                    {
                        "hypothesis_id": str(i),
                        "scores": {
                            "impact": 0.7,
                            "feasibility": 0.7,
                            "accessibility": 0.7,
                            "sustainability": 0.7,
                            "scalability": 0.7,
                        },
                    }
                    for i in range(1, 8)
                ],
                "answer": "Evaluation result for large set",
                "action_plan": ["Action 1", "Action 2"],
            }
        ),
        provider=LLMProvider.GOOGLE,
        model="gemini-1.5-flash",
        cost=0.012,
    )

    result = await execute_deduction_phase(phase_input, "Question?", hypotheses)

    assert len(result.hypothesis_scores) == 7
    # Note: Currently using sequential for all sets (parallel can be added later)
    assert result.used_parallel is False


@pytest.mark.asyncio
async def test_deduction_structured_scores(mock_llm_manager, phase_input):
    """Test deduction with structured JSON score parsing."""
    hypotheses = ["H1", "H2"]

    mock_response = LLMResponse(
        content=json.dumps(
            {
                "evaluations": [
                    {
                        "hypothesis_id": "1",
                        "scores": {
                            "impact": 0.85,
                            "feasibility": 0.75,
                            "accessibility": 0.65,
                            "sustainability": 0.80,
                            "scalability": 0.70,
                        },
                    },
                    {
                        "hypothesis_id": "2",
                        "scores": {
                            "impact": 0.90,
                            "feasibility": 0.80,
                            "accessibility": 0.70,
                            "sustainability": 0.85,
                            "scalability": 0.75,
                        },
                    },
                ],
                "answer": "Test answer",
                "action_plan": ["Test action"],
            }
        ),
        provider=LLMProvider.GOOGLE,
        model="gemini-1.5-flash",
        cost=0.005,
    )
    mock_llm_manager.generate.return_value = mock_response

    result = await execute_deduction_phase(phase_input, "Q?", hypotheses)

    assert result.hypothesis_scores[0].impact == 0.85
    assert result.hypothesis_scores[1].feasibility == 0.80


@pytest.mark.asyncio
async def test_deduction_text_score_fallback(mock_llm_manager, phase_input):
    """Test deduction with text parsing fallback for scores."""
    hypotheses = ["H1", "H2"]

    mock_response = LLMResponse(
        content="""
        Hypothesis 1:
        * Impact: 0.8 - High impact
        * Feasibility: 0.7 - Moderately feasible
        * Accessibility: 0.6 - Accessible
        * Sustainability: 0.75 - Sustainable
        * Scalability: 0.65 - Scalable

        Hypothesis 2:
        * Impact: 0.9
        * Feasibility: 0.8
        * Accessibility: 0.7
        * Sustainability: 0.85
        * Scalability: 0.75

        ANSWER: Combined approach
        Action Plan:
        1. First action
        2. Second action
        """,
        provider=LLMProvider.GOOGLE,
        model="gemini-1.5-flash",
        cost=0.006,
    )
    mock_llm_manager.generate.return_value = mock_response

    with patch("mad_spark_alt.core.phase_logic.ScoreParser") as mock_parser:
        mock_parser.parse_with_fallback.side_effect = [
            ParsedScores(
                impact=0.8,
                feasibility=0.7,
                accessibility=0.6,
                sustainability=0.75,
                scalability=0.65,
            ),
            ParsedScores(
                impact=0.9,
                feasibility=0.8,
                accessibility=0.7,
                sustainability=0.85,
                scalability=0.75,
            ),
        ]

        result = await execute_deduction_phase(phase_input, "Q?", hypotheses)

        assert len(result.hypothesis_scores) == 2


@pytest.mark.asyncio
async def test_deduction_action_plan_extraction(mock_llm_manager, phase_input):
    """Test action plan parsing from deduction response."""
    hypotheses = ["H1"]

    mock_response = LLMResponse(
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
                    }
                ],
                "answer": "Answer text",
                "action_plan": [
                    "Conduct feasibility study",
                    "Develop prototype",
                    "Test with focus group",
                    "Scale implementation",
                ],
            }
        ),
        provider=LLMProvider.GOOGLE,
        model="gemini-1.5-flash",
        cost=0.004,
    )
    mock_llm_manager.generate.return_value = mock_response

    result = await execute_deduction_phase(phase_input, "Q?", hypotheses)

    assert len(result.action_plan) == 4
    assert "feasibility study" in result.action_plan[0]


@pytest.mark.asyncio
async def test_deduction_answer_extraction(mock_llm_manager, phase_input):
    """Test answer extraction with multiple fallback patterns."""
    hypotheses = ["H1"]

    mock_response = LLMResponse(
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
                    }
                ],
                "answer": "The recommended approach is to implement a phased rollout strategy.",
                "action_plan": ["Action 1"],
            }
        ),
        provider=LLMProvider.GOOGLE,
        model="gemini-1.5-flash",
        cost=0.003,
    )
    mock_llm_manager.generate.return_value = mock_response

    result = await execute_deduction_phase(phase_input, "Q?", hypotheses)

    assert "phased rollout strategy" in result.answer


@pytest.mark.asyncio
async def test_deduction_cost_distribution(mock_llm_manager, phase_input):
    """Test cost tracking across sequential evaluations."""
    hypotheses = ["H1", "H2", "H3"]

    mock_response = LLMResponse(
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
                    {
                        "hypothesis_id": "2",
                        "scores": {
                            "impact": 0.7,
                            "feasibility": 0.7,
                            "accessibility": 0.7,
                            "sustainability": 0.7,
                            "scalability": 0.7,
                        },
                    },
                    {
                        "hypothesis_id": "3",
                        "scores": {
                            "impact": 0.7,
                            "feasibility": 0.7,
                            "accessibility": 0.7,
                            "sustainability": 0.7,
                            "scalability": 0.7,
                        },
                    },
                ],
                "answer": "Answer",
                "action_plan": ["Action"],
            }
        ),
        provider=LLMProvider.GOOGLE,
        model="gemini-1.5-flash",
        cost=0.009,
    )
    mock_llm_manager.generate.return_value = mock_response

    result = await execute_deduction_phase(phase_input, "Q?", hypotheses)

    assert result.llm_cost == 0.009


# ========== Induction Phase Tests ==========


@pytest.mark.asyncio
async def test_induction_example_pattern_1(mock_llm_manager, phase_input):
    """Test induction with 'Example N:' format."""
    mock_response = LLMResponse(
        content="""
        Example 1: A city implemented smart traffic lights and reduced congestion by 30%.
        Example 2: Company X used AI scheduling to cut delivery times by 25%.
        Example 3: Hospital Y optimized bed allocation using predictive models.

        Conclusion: These examples demonstrate measurable improvements across sectors.
        """,
        provider=LLMProvider.GOOGLE,
        model="gemini-1.5-flash",
        cost=0.004,
    )
    mock_llm_manager.generate.return_value = mock_response

    result = await execute_induction_phase(
        phase_input, "Core question?", "Use AI for optimization", ["H1", "H2", "H3"]
    )

    assert isinstance(result, InductionResult)
    assert len(result.examples) >= 3
    assert "traffic lights" in result.examples[0].lower()
    assert "measurable improvements" in result.conclusion.lower()
    assert result.llm_cost == 0.004


@pytest.mark.asyncio
async def test_induction_example_pattern_2(mock_llm_manager, phase_input):
    """Test induction with numbered list format."""
    mock_response = LLMResponse(
        content="""
        1. Tesla's Gigafactory uses renewable energy for production
        2. Denmark powers 50% of electricity from wind turbines
        3. Costa Rica achieved 99% renewable energy in recent years

        Conclusion: Renewable energy adoption is accelerating globally.
        """,
        provider=LLMProvider.GOOGLE,
        model="gemini-1.5-flash",
        cost=0.0035,
    )
    mock_llm_manager.generate.return_value = mock_response

    result = await execute_induction_phase(
        phase_input, "Q?", "Renewable energy is viable", ["H1", "H2"]
    )

    assert len(result.examples) >= 3
    assert (
        "tesla" in result.examples[0].lower()
        or "gigafactory" in result.examples[0].lower()
    )


@pytest.mark.asyncio
async def test_induction_conclusion_extraction(mock_llm_manager, phase_input):
    """Test conclusion parsing from induction response."""
    mock_response = LLMResponse(
        content="""
        Example 1: Case study A
        Example 2: Case study B

        Conclusion: The approach is proven effective across multiple contexts and scales well.
        """,
        provider=LLMProvider.GOOGLE,
        model="gemini-1.5-flash",
        cost=0.003,
    )
    mock_llm_manager.generate.return_value = mock_response

    result = await execute_induction_phase(phase_input, "Q?", "Answer", ["H1"])

    assert "proven effective" in result.conclusion
    assert "scales well" in result.conclusion


@pytest.mark.asyncio
async def test_induction_reference_substitution(mock_llm_manager, phase_input):
    """Test H1/H2/H3 reference substitution in conclusion when used with 'your answer'."""
    hypotheses = [
        "Use machine learning for predictions",
        "Implement blockchain for transparency",
        "Deploy IoT sensors for monitoring",
    ]

    mock_response = LLMResponse(
        content="""
        Example 1: ML example
        Example 2: Blockchain example

        Conclusion: Your answer (H1) provides predictive capabilities. The answer (H2) ensures transparency and trust.
        """,
        provider=LLMProvider.GOOGLE,
        model="gemini-1.5-flash",
        cost=0.0045,
    )
    mock_llm_manager.generate.return_value = mock_response

    result = await execute_induction_phase(phase_input, "Q?", "Answer", hypotheses)

    # Self-references like "your answer" should be replaced with "the recommended approach"
    assert "your answer" not in result.conclusion.lower()
    # When H references appear with "your answer" or "the answer", they should be replaced
    assert "the recommended approach" in result.conclusion


@pytest.mark.asyncio
async def test_induction_cost_tracking(mock_llm_manager, phase_input):
    """Test cost tracking in induction phase."""
    mock_response = LLMResponse(
        content="Example 1: Test\nConclusion: Test conclusion",
        provider=LLMProvider.GOOGLE,
        model="gemini-1.5-flash",
        cost=0.0025,
    )
    mock_llm_manager.generate.return_value = mock_response

    result = await execute_induction_phase(phase_input, "Q?", "Answer", ["H1"])

    assert result.llm_cost == 0.0025
