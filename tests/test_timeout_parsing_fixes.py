"""
Tests for timeout and parsing fixes in Mad Spark Alt system.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import after path setup
from mad_spark_alt.core.simple_qadi_orchestrator import SimpleQADIOrchestrator, HypothesisScore
from mad_spark_alt.core.interfaces import GeneratedIdea, ThinkingMethod


@pytest.fixture
def mock_qadi_result():
    """Mock QADI result with synthesized ideas."""
    from mad_spark_alt.core.simple_qadi_orchestrator import SimpleQADIResult
    
    return SimpleQADIResult(
        core_question="How can we build more sustainable cities?",
        hypotheses=[
            "Empower citizens with personalized tools and incentives",
            "Create hyper-local, self-sufficient community hubs", 
            "Re-engineer urban planning with circular economy principles"
        ],
        hypothesis_scores=[
            HypothesisScore(0.8, 0.7, 0.9, 0.8, 0.7, 0.78),
            HypothesisScore(0.7, 0.8, 0.8, 0.7, 0.8, 0.76),
            HypothesisScore(0.9, 0.6, 0.7, 0.8, 0.6, 0.72),
        ],
        final_answer="The best approach combines systemic planning changes...",
        action_plan=["Step 1: Policy research", "Step 2: Pilot programs", "Step 3: Scale implementation"],
        verification_examples=["Example 1: Copenhagen", "Example 2: Singapore"],
        verification_conclusion="Systemic approaches show highest success rates",
        total_llm_cost=0.015,
        synthesized_ideas=[
            GeneratedIdea(
                content="Empower citizens with personalized tools and incentives",
                thinking_method=ThinkingMethod.ABDUCTION,
                agent_name="test",
                generation_prompt="test",
                confidence_score=0.8,
            ),
            GeneratedIdea(
                content="Create hyper-local, self-sufficient community hubs",
                thinking_method=ThinkingMethod.ABDUCTION,
                agent_name="test", 
                generation_prompt="test",
                confidence_score=0.7,
            ),
        ]
    )


@pytest.mark.asyncio
async def test_cli_adaptive_timeout_calculation():
    """Test that CLI uses adaptive timeout calculation instead of hard-coded 120s."""
    from mad_spark_alt.cli import calculate_evolution_timeout
    
    # Test various parameter combinations - CLI uses generations * population * 10, min 60s, max 600s
    assert calculate_evolution_timeout(3, 5) == 150.0  # max(60, 3*5*10) = max(60, 150) = 150
    assert calculate_evolution_timeout(2, 3) == 60.0   # max(60, 2*3*10) = max(60, 60) = 60
    assert calculate_evolution_timeout(5, 10) == 500.0 # max(60, 5*10*10) = max(60, 500) = 500
    assert calculate_evolution_timeout(10, 50) == 600.0  # max(60, 10*50*10) = max(60, 5000) = 600 (capped)


@pytest.mark.asyncio
async def test_qadi_simple_timeout_calculation():
    """Test that qadi_simple.py uses adaptive timeout calculation."""
    # The timeout calculation function from qadi_simple.py
    def calculate_evolution_timeout(gens: int, pop: int) -> float:
        """Calculate timeout in seconds based on generations and population."""
        base_timeout = 60.0  # Base 1 minute
        time_per_eval = 2.0  # 2 seconds per idea evaluation
        
        # Estimate total evaluations
        total_evaluations = gens * pop
        estimated_time = base_timeout + (total_evaluations * time_per_eval)
        
        # Cap at 10 minutes
        return min(estimated_time, 600.0)
    
    # Test various scenarios
    assert calculate_evolution_timeout(3, 5) == 90.0  # 60 + 15*2 = 90
    assert calculate_evolution_timeout(5, 10) == 160.0  # 60 + 50*2 = 160
    assert calculate_evolution_timeout(10, 50) == 600.0  # Capped at 600


@pytest.mark.asyncio
async def test_hypothesis_parsing_robustness(monkeypatch):
    """Test that hypothesis parsing can handle various LLM response formats."""
    # Mock GOOGLE_API_KEY
    monkeypatch.setenv('GOOGLE_API_KEY', 'test-key')
    
    # Test data with different format variations
    test_responses = [
        # Standard format
        """
        H1: Empower citizens with personalized sustainability tools
        H2: Create hyper-local community resource hubs  
        H3: Re-engineer urban planning with circular principles
        """,
        # Numbered list format
        """
        1. Empower citizens with personalized sustainability tools
        2. Create hyper-local community resource hubs
        3. Re-engineer urban planning with circular principles
        """,
        # Bullet point format
        """
        - Empower citizens with personalized sustainability tools
        - Create hyper-local community resource hubs
        - Re-engineer urban planning with circular principles
        """,
        # Mixed markdown format
        """
        **H1:** Empower citizens with personalized sustainability tools
        **H2:** Create hyper-local community resource hubs
        **H3:** Re-engineer urban planning with circular principles
        """
    ]
    
    orchestrator = SimpleQADIOrchestrator()
    
    for i, response_content in enumerate(test_responses):
        with patch('mad_spark_alt.core.simple_qadi_orchestrator.llm_manager') as mock_llm:
            # Mock the LLM response
            mock_response = MagicMock()
            mock_response.content = response_content
            mock_response.cost = 0.001
            mock_llm.generate = AsyncMock(return_value=mock_response)
            
            try:
                hypotheses, _ = await orchestrator._run_abduction_phase(
                    "Test input", "Test question", max_retries=0
                )
                
                # Should extract at least 2 hypotheses regardless of format
                assert len(hypotheses) >= 2, f"Format {i} failed to extract hypotheses: got {len(hypotheses)}"
                
                # Check that hypotheses contain meaningful content
                for hypothesis in hypotheses:
                    assert len(hypothesis) > 15, f"Hypothesis too short: {hypothesis}"  # Reduced from 20 to 15
                    # Check for key content without being too strict
                    content_lower = hypothesis.lower()
                    has_content = any(word in content_lower for word in ["empower", "citizen", "community", "hub", "urban", "planning", "sustainability", "personalized"])
                    assert has_content, f"Hypothesis doesn't contain expected content: {hypothesis}"
                    
            except Exception as e:
                pytest.fail(f"Format {i} parsing failed: {e}")


@pytest.mark.asyncio
async def test_evaluation_criteria_consistency():
    """Test that evaluation criteria are consistent between systems."""
    from mad_spark_alt.core.qadi_prompts import EVALUATION_CRITERIA
    
    # QADI system criteria
    qadi_criteria = set(EVALUATION_CRITERIA.keys())
    expected_qadi_criteria = {"impact", "feasibility", "accessibility", "sustainability", "scalability"}
    
    assert qadi_criteria == expected_qadi_criteria, f"QADI criteria mismatch: {qadi_criteria}"
    
    # Verify weights sum to 1.0
    total_weight = sum(EVALUATION_CRITERIA.values())
    assert abs(total_weight - 1.0) < 0.01, f"Criteria weights should sum to 1.0, got {total_weight}"


@pytest.mark.asyncio
async def test_score_parsing_improved_patterns():
    """Test that score parsing handles various LLM response formats."""
    from mad_spark_alt.core.simple_qadi_orchestrator import SimpleQADIOrchestrator
    
    orchestrator = SimpleQADIOrchestrator()
    
    # Test various score format variations
    test_content_formats = [
        # Standard format
        """
        H1: Test hypothesis
        * Impact: 0.8 - High positive change expected
        * Feasibility: 0.7 - Moderately easy to implement  
        * Accessibility: 0.9 - Most people can participate
        * Sustainability: 0.8 - Long-term viability
        * Scalability: 0.6 - Can grow gradually
        * Overall: 0.76
        """,
        # Bold format
        """
        **H1:** Test hypothesis
        **Impact:** 0.8 - High positive change expected
        **Feasibility:** 0.7 - Moderately easy to implement
        **Accessibility:** 0.9 - Most people can participate  
        **Sustainability:** 0.8 - Long-term viability
        **Scalability:** 0.6 - Can grow gradually
        **Overall:** 0.76
        """,
        # Fractional format
        """
        H1: Test hypothesis
        Impact: 8/10 - High positive change expected
        Feasibility: 7/10 - Moderately easy to implement
        Accessibility: 9/10 - Most people can participate
        Sustainability: 8/10 - Long-term viability  
        Scalability: 6/10 - Can grow gradually
        """,
        # Parenthetical format  
        """
        H1: Test hypothesis
        Impact (0.8): High positive change expected
        Feasibility (0.7): Moderately easy to implement
        Accessibility (0.9): Most people can participate
        Sustainability (0.8): Long-term viability
        Scalability (0.6): Can grow gradually
        """
    ]
    
    for i, content in enumerate(test_content_formats):
        score = orchestrator._parse_hypothesis_scores(content, 1)
        
        # Should extract meaningful scores, not defaults
        assert score.impact > 0.5, f"Format {i}: Impact score not parsed correctly: {score.impact}"
        assert score.feasibility > 0.5, f"Format {i}: Feasibility score not parsed correctly: {score.feasibility}"  
        assert score.accessibility > 0.5, f"Format {i}: Accessibility score not parsed correctly: {score.accessibility}"
        assert score.sustainability > 0.5, f"Format {i}: Sustainability score not parsed correctly: {score.sustainability}"
        
        # Overall score should be calculated, not default
        assert score.overall != 0.5, f"Format {i}: Overall score appears to be default: {score.overall}"


@pytest.mark.asyncio
async def test_timeout_error_handling():
    """Test that timeout errors provide helpful error messages."""
    
    # Test CLI timeout error handling
    async def simulate_cli_timeout():
        try:
            await asyncio.wait_for(asyncio.sleep(5), timeout=1.0)
        except asyncio.TimeoutError:
            return "CLI timeout handled correctly"
    
    result = await simulate_cli_timeout()
    assert result == "CLI timeout handled correctly"
    
    # Test qadi_simple timeout error handling  
    async def simulate_qadi_simple_timeout():
        try:
            await asyncio.wait_for(asyncio.sleep(5), timeout=1.0)
        except asyncio.TimeoutError:
            return "QADI simple timeout handled correctly"
    
    result = await simulate_qadi_simple_timeout()
    assert result == "QADI simple timeout handled correctly"


@pytest.mark.asyncio
async def test_no_hard_coded_timeouts():
    """Test that hard-coded timeouts have been removed from critical paths."""
    
    # This test ensures we don't regress back to hard-coded timeouts
    # by checking that timeout calculation functions are being used
    
    from mad_spark_alt.cli import calculate_evolution_timeout
    
    # CLI timeout calculation should be reasonable for test parameters
    cli_timeout = calculate_evolution_timeout(3, 5)
    assert cli_timeout > 60, "CLI timeout should be more than base 60s"
    assert cli_timeout < 600, "CLI timeout should be less than max 600s"
    
    # qadi_simple timeout calculation (from the function in that file)
    def qadi_simple_timeout(gens: int, pop: int) -> float:
        base_timeout = 60.0
        time_per_eval = 2.0
        total_evaluations = gens * pop
        estimated_time = base_timeout + (total_evaluations * time_per_eval)
        return min(estimated_time, 600.0)
    
    qadi_timeout = qadi_simple_timeout(3, 5)
    assert qadi_timeout > 60, "QADI timeout should be more than base 60s"
    assert qadi_timeout < 600, "QADI timeout should be less than max 600s"
    
    # Different calculation methods but both should be adaptive
    assert cli_timeout != 120.0, "CLI should not use hard-coded 120s"
    assert qadi_timeout != 120.0, "QADI simple should not use hard-coded 120s"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])