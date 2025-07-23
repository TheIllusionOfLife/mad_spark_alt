"""Integration tests with real LLM calls to verify prompt-parser compatibility."""

import os
import pytest

from mad_spark_alt.core import SimpleQADIOrchestrator, setup_llm_providers


class TestRealLLMIntegration:
    """Integration tests that use real LLM calls to validate the system."""

    @pytest.fixture(scope="class")
    async def llm_setup(self):
        """Setup LLM providers for integration tests."""
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            pytest.skip("GOOGLE_API_KEY not set - skipping real LLM integration tests")
        
        await setup_llm_providers(google_api_key=google_api_key)

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.getenv("GOOGLE_API_KEY"),
        reason="GOOGLE_API_KEY not set - requires real API access"
    )
    async def test_real_qadi_cycle_score_parsing(self, llm_setup) -> None:
        """Test that real LLM responses can be parsed correctly."""
        # llm_setup fixture already handles provider setup
        
        orchestrator = SimpleQADIOrchestrator()
        
        # Use a simple, focused question to get a more predictable response
        user_input = "How to reduce costs?"
        
        try:
            result = await orchestrator.run_qadi_cycle(user_input)
            
            # Verify the basic structure was parsed
            assert result.core_question is not None
            assert len(result.hypotheses) >= 1
            assert result.final_answer is not None
            
            # The key test: verify that scores were parsed (not all defaults)
            # If parsing works, at least some scores should not be exactly 0.5
            all_default_scores = all(
                score.novelty == 0.5 and 
                score.impact == 0.5 and 
                score.cost == 0.5 and 
                score.feasibility == 0.5 and 
                score.risks == 0.5
                for score in result.hypothesis_scores
            )
            
            # Log the actual scores for debugging
            print(f"\nActual scores from LLM:")
            for i, score in enumerate(result.hypothesis_scores):
                print(f"H{i+1}: novelty={score.novelty}, impact={score.impact}, overall={score.overall}")
            
            # This test passes if we successfully parse at least one non-default score
            # If this fails, it means our parser still can't handle real LLM responses
            if all_default_scores:
                pytest.fail(
                    "All scores are default values (0.5), indicating parser failed to "
                    "extract scores from real LLM response. This suggests a Mock-Reality "
                    "Divergence issue."
                )
                
        except RuntimeError as e:
            pytest.fail(f"Real LLM integration test failed: {e}")

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.getenv("GOOGLE_API_KEY"),
        reason="GOOGLE_API_KEY not set - requires real API access"
    )
    async def test_deduction_phase_format_validation(self, llm_setup) -> None:
        """Test that deduction phase returns parseable format."""
        # llm_setup fixture already handles provider setup
        
        orchestrator = SimpleQADIOrchestrator()
        
        # Test the deduction phase specifically
        user_input = "Improve efficiency"
        core_question = "What specific outcome do you want?"
        hypotheses = [
            "Reduce waste by 10%",
            "Automate repetitive tasks", 
            "Improve team communication"
        ]
        
        try:
            result = await orchestrator._run_deduction_phase(
                user_input, core_question, hypotheses, max_retries=1
            )
            
            # Verify we got scores and they're not all defaults
            assert len(result["scores"]) == 3
            assert result["answer"] is not None
            
            # Check if any scores were successfully parsed (not defaults)
            non_default_scores = [
                score for score in result["scores"]
                if not (score.novelty == 0.5 and score.impact == 0.5 and 
                       score.cost == 0.5 and score.feasibility == 0.5 and 
                       score.risks == 0.5)
            ]
            
            print(f"\nDeduction phase results:")
            print(f"Raw content length: {len(result['raw_content'])}")
            print(f"Non-default scores found: {len(non_default_scores)}")
            
            # This validates that our parser can handle real deduction responses
            assert len(non_default_scores) > 0, (
                "No scores were parsed from real deduction response. "
                "This indicates the parser cannot handle the actual LLM format."
            )
            
        except Exception as e:
            pytest.fail(f"Deduction phase format validation failed: {e}")


# Mark these as integration tests that can be run separately
pytestmark = pytest.mark.integration