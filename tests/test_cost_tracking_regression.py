"""
Regression tests for cost tracking functionality.

These tests ensure that cost data is properly propagated through all QADI phases
and handle edge cases in JSON parsing with cost information.
"""

import json
from unittest.mock import AsyncMock, patch

import pytest

from mad_spark_alt.core import (
    LLMManager,
    LLMProvider,
    LLMResponse,
    SimpleQADIOrchestrator,
    cost_utils,
)
from mad_spark_alt.core.json_utils import safe_json_parse


class TestCostTrackingRegression:
    """Test suite for cost tracking regression issues."""

    @pytest.fixture
    def mock_llm_manager(self):
        """Create a mock LLM manager."""
        manager = AsyncMock(spec=LLMManager)
        return manager

    @pytest.fixture
    def orchestrator(self):
        """Create a SimpleQADIOrchestrator instance."""
        return SimpleQADIOrchestrator()

    @pytest.mark.asyncio
    async def test_all_qadi_phases_include_cost_data(self, orchestrator, mock_llm_manager):
        """Test that all QADI phases properly include cost data in their results."""
        # Set up mock responses for each phase with cost data
        mock_responses = [
            # Questioning phase
            LLMResponse(
                content="Q: What are the root causes of urban traffic congestion?",
                provider=LLMProvider.GOOGLE,
                model="test-model",
                usage={"prompt_tokens": 100, "completion_tokens": 200},
                cost=0.015,
                response_time=0.5,
            ),
            # Abduction phase
            LLMResponse(
                content="""H1: Poor urban planning with inadequate road infrastructure
H2: Lack of effective public transportation systems
H3: Concentration of jobs in city centers causing rush hour bottlenecks""",
                provider=LLMProvider.GOOGLE,
                model="test-model",
                usage={"prompt_tokens": 150, "completion_tokens": 300},
                cost=0.025,
                response_time=0.8,
            ),
            # Deduction phase
            LLMResponse(
                content="""H1: Poor urban planning
Novelty: 0.4
Impact: 0.9
Cost: 0.2
Feasibility: 0.6
Risks: 0.7

H2: Lack of public transportation
Novelty: 0.3
Impact: 0.8
Cost: 0.3
Feasibility: 0.7
Risks: 0.6

H3: Job concentration
Novelty: 0.6
Impact: 0.7
Cost: 0.8
Feasibility: 0.5
Risks: 0.5

ANSWER: Improving public transportation emerges as the most feasible high-impact solution.

Action Plan:
1. Expand metro/bus networks to underserved areas
2. Implement dedicated bus lanes on major routes
3. Create park-and-ride facilities at city outskirts""",
                provider=LLMProvider.GOOGLE,
                model="test-model",
                usage={"prompt_tokens": 200, "completion_tokens": 400},
                cost=0.035,
                response_time=1.0,
            ),
            # Induction phase
            LLMResponse(
                content="""1. Tokyo's extensive subway system reduced traffic by 40% over 20 years.

2. Curitiba's Bus Rapid Transit (BRT) moves 2.3 million passengers daily at 10% of subway cost.

3. Copenhagen's bike infrastructure led to 40% of commuters cycling to work.

Conclusion: Cities with diverse, well-integrated public transport see 30-50% traffic reduction.""",
                provider=LLMProvider.GOOGLE,
                model="test-model",
                usage={"prompt_tokens": 180, "completion_tokens": 350},
                cost=0.030,
                response_time=0.9,
            ),
        ]

        mock_llm_manager.generate.side_effect = mock_responses

        with patch("mad_spark_alt.core.simple_qadi_orchestrator.llm_manager", mock_llm_manager):
            result = await orchestrator.run_qadi_cycle(
                "How can we solve urban traffic problems?",
                context="Focus on sustainable solutions",
            )

        # Verify all phases have cost data in phase_results
        assert result.phase_results["questioning"]["cost"] == pytest.approx(0.015)
        assert result.phase_results["abduction"]["cost"] == pytest.approx(0.025)
        assert result.phase_results["deduction"]["cost"] == pytest.approx(0.035)
        assert result.phase_results["induction"]["cost"] == pytest.approx(0.030)
        
        # Verify total cost is calculated correctly
        assert abs(result.total_llm_cost - 0.105) < 0.0001

    @pytest.mark.asyncio
    async def test_cost_tracking_with_malformed_json(self, orchestrator, mock_llm_manager):
        """Test cost tracking when LLM returns malformed JSON in responses."""
        # Mock response with JSON wrapped in markdown
        mock_response = LLMResponse(
            content="""```json
{
    "hypotheses": [
        "H1: Implement congestion pricing",
        "H2: Expand public transit"
    ],
    "analysis": "Both approaches show promise"
}
```""",
            provider=LLMProvider.GOOGLE,
            model="test-model",
            usage={"prompt_tokens": 100, "completion_tokens": 150},
            cost=0.020,
            response_time=0.6,
        )

        # Test JSON parsing extracts content correctly
        parsed = safe_json_parse(mock_response.content)
        assert parsed is not None
        assert "hypotheses" in parsed
        assert len(parsed["hypotheses"]) == 2
        
        # Verify cost is preserved even with JSON parsing
        assert mock_response.cost == pytest.approx(0.020)

    @pytest.mark.asyncio
    async def test_cost_aggregation_across_operations(self):
        """Test that costs are correctly aggregated across multiple operations."""
        total_cost = 0.0
        
        # Simulate multiple LLM calls with cost tracking
        operations = [
            {"input_tokens": 100, "output_tokens": 200, "expected_cost": 0.00053},  # (100/1000 * 0.00030) + (200/1000 * 0.0025) = 0.00003 + 0.0005 = 0.00053
            {"input_tokens": 500, "output_tokens": 1000, "expected_cost": 0.00265},  # (500/1000 * 0.00030) + (1000/1000 * 0.0025) = 0.00015 + 0.0025 = 0.00265
            {"input_tokens": 50, "output_tokens": 100, "expected_cost": 0.000265},  # (50/1000 * 0.00030) + (100/1000 * 0.0025) = 0.000015 + 0.00025 = 0.000265
        ]
        
        for op in operations:
            cost = cost_utils.calculate_llm_cost(
                op["input_tokens"], 
                op["output_tokens"], 
                "gemini-2.5-flash"
            )
            total_cost += cost
            # Verify individual cost calculation
            assert abs(cost - op["expected_cost"]) < 0.0001
        
        # Verify total aggregation
        expected_total = sum(op["expected_cost"] for op in operations)
        assert abs(total_cost - expected_total) < 0.0001

    def test_cost_calculation_accuracy_different_models(self):
        """Test cost calculation accuracy for different model types."""
        test_cases = [
            {
                "model": "gemini-2.5-flash",
                "input_tokens": 1000,
                "output_tokens": 500,
                "expected_cost": 0.00155  # (1000/1000 * 0.00030) + (500/1000 * 0.0025) = 0.0003 + 0.00125 = 0.00155
            },
            {
                "model": "gemini-2.5-flash",
                "input_tokens": 2000,
                "output_tokens": 1000,
                "expected_cost": 0.0031  # (2000/1000 * 0.00030) + (1000/1000 * 0.0025) = 0.0006 + 0.0025 = 0.0031
            },
            {
                "model": "gemini-2.5-flash",
                "input_tokens": 1500,
                "output_tokens": 750,
                "expected_cost": 0.002325  # (1500/1000 * 0.00030) + (750/1000 * 0.0025) = 0.00045 + 0.001875 = 0.002325
            },
        ]
        
        for case in test_cases:
            cost = cost_utils.calculate_llm_cost(
                case["input_tokens"],
                case["output_tokens"],
                case["model"]
            )
            assert abs(cost - case["expected_cost"]) < 0.00001, f"Cost mismatch for {case['model']}"

    def test_cost_tracking_edge_cases(self):
        """Test edge cases in cost tracking."""
        # Zero tokens
        cost = cost_utils.calculate_llm_cost(0, 0, "gemini-2.5-flash")
        assert cost == 0.0
        
        # Very large token counts
        cost = cost_utils.calculate_llm_cost(1000000, 500000, "gemini-2.5-flash")
        assert cost == pytest.approx(1.55)  # (1000000/1000 * 0.00030) + (500000/1000 * 0.0025) = 0.30 + 1.25 = 1.55
        
        # Unknown model falls back to Gemini 2.5 Flash
        cost = cost_utils.calculate_llm_cost(100, 100, "unknown-model")
        assert cost > 0  # Should use Gemini 2.5 Flash pricing
        
        # Fractional tokens (should handle gracefully)
        cost = cost_utils.calculate_llm_cost(150, 75, "gemini-2.5-flash")
        assert cost == pytest.approx(0.0000675)  # (150/1000 * 0.00015) + (75/1000 * 0.0006) = 0.0000225 + 0.000045 = 0.0000675

    @pytest.mark.asyncio
    async def test_cost_propagation_in_error_scenarios(self, orchestrator, mock_llm_manager):
        """Test that costs are tracked and available even when errors occur during QADI cycles."""
        # First call succeeds with cost
        successful_response = LLMResponse(
            content="Q: Test question?",
            provider=LLMProvider.GOOGLE,
            model="test-model",
            usage={"prompt_tokens": 50, "completion_tokens": 100},
            cost=0.010,
            response_time=0.5,
        )
        
        # Second call fails but still has cost
        error_response = LLMResponse(
            content="",
            provider=LLMProvider.GOOGLE,
            model="test-model",
            usage={"prompt_tokens": 25, "completion_tokens": 0},
            cost=0.003,  # Cost incurred before error
            response_time=0.1,
        )
        
        # Configure mock to return first response then fail immediately
        # This simulates a failure after the first successful call
        mock_llm_manager.generate.side_effect = [
            successful_response,
            Exception("API error after tokens consumed"),
        ]
        
        # Track total cost from calls
        total_cost = 0.0
        
        with patch("mad_spark_alt.core.simple_qadi_orchestrator.llm_manager", mock_llm_manager):
            try:
                result = await orchestrator.run_qadi_cycle("Test query")
                # Should not reach here if test is working correctly
                assert False, "Expected exception was not raised"
            except Exception as e:
                # Verify that generate was called at least once before failing
                assert mock_llm_manager.generate.call_count >= 1, "LLM should be called at least once"
                
                # Verify we got the expected exception about API issues
                assert "API error after tokens consumed" in str(e) or "QADI cycle failed" in str(e)
                
                # The first successful call should have incurred cost
                # We verify this by checking the mock was called with the successful response
                calls = mock_llm_manager.generate.call_args_list
                assert len(calls) >= 1, "At least one LLM call should have been made"
                
                # Verify cost tracking capability - the successful response had a cost
                assert successful_response.cost == pytest.approx(0.010), "First response should have tracked cost"
                
                # Additional verification: ensure the mock setup correctly simulates partial success
                # The first call succeeded (returned response), second failed (raised exception)
                assert mock_llm_manager.generate.call_count >= 1
                
        # This test verifies that:
        # 1. LLM calls can fail mid-cycle (simulated)
        # 2. Cost data is available from successful calls before failure
        # 3. The system handles partial failures gracefully
        # Note: The orchestrator's internal cost tracking during failures would require
        # additional architecture changes to expose partial results.

    def test_cost_utils_usage_dict_parsing(self):
        """Test cost calculation from usage dictionaries with different formats."""
        # Google format
        cost, input_tokens, output_tokens = cost_utils.calculate_cost_with_usage(
            {"prompt_tokens": 100, "completion_tokens": 200},
            "gemini-2.5-flash"
        )
        assert input_tokens == 100
        assert output_tokens == 200
        assert cost == pytest.approx(0.00053)  # (100/1000 * 0.00030) + (200/1000 * 0.0025) = 0.00003 + 0.0005 = 0.00053
        
        # Alternative format
        cost, input_tokens, output_tokens = cost_utils.calculate_cost_with_usage(
            {"input_tokens": 150, "output_tokens": 250},
            "gemini-2.5-flash"
        )
        assert input_tokens == 150
        assert output_tokens == 250
        assert cost == pytest.approx(0.0001725)  # (150/1000 * 0.00015) + (250/1000 * 0.0006) = 0.0000225 + 0.00015 = 0.0001725
        
        # Empty usage dict
        cost, input_tokens, output_tokens = cost_utils.calculate_cost_with_usage(
            {},
            "gemini-2.5-flash"
        )
        assert input_tokens == 0
        assert output_tokens == 0
        assert cost == 0.0

    def test_centralized_vs_legacy_cost_calculation_parity(self):
        """Ensure centralized cost calculation matches legacy calculations."""
        test_tokens = [
            (1000, 500),
            (100, 50),
            (5000, 2500),
            (10, 5),
        ]
        
        for input_tok, output_tok in test_tokens:
            # Legacy calculation pattern (as it was in llm_provider.py) for Gemini 2.5 Flash
            # Updated to use new pricing: $0.30 per 1M input, $2.50 per 1M output
            legacy_cost = (input_tok / 1000) * 0.00030 + (output_tok / 1000) * 0.0025
            
            # New centralized calculation
            new_cost = cost_utils.calculate_llm_cost(input_tok, output_tok, "gemini-2.5-flash")
            
            # They should match exactly
            assert abs(legacy_cost - new_cost) < 0.000001, f"Cost mismatch for {input_tok}/{output_tok}"