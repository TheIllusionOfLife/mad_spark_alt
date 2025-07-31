"""Tests for hypothesis format without H+number prefix."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from mad_spark_alt.core.simple_qadi_orchestrator import SimpleQADIOrchestrator
from mad_spark_alt.core.llm_provider import LLMResponse


class TestHypothesisFormat:
    """Test that hypotheses are formatted without H1:, H2: prefixes."""
    
    @pytest.mark.asyncio
    async def test_hypothesis_generation_without_h_prefix(self):
        """Test that generated hypotheses don't include H1:, H2: prefixes."""
        orchestrator = SimpleQADIOrchestrator(num_hypotheses=3)
        
        # Mock LLM response with new format (no H prefix)
        mock_response = LLMResponse(
            content="""
            1. Revolutionary Game Concept Using Mobius Strip
            This is a detailed game concept that uses the Mobius strip as a core mechanic...
            
            2. Time-Loop Adventure on Mobius Surface
            A narrative-driven game where time itself follows a Mobius strip pattern...
            
            3. Spatial Puzzle Platform with Twisted Reality
            Players navigate through levels that are physically structured as Mobius strips...
            """,
            model="gemini-pro",
            provider="google",
            cost=0.001
        )
        
        with patch.object(orchestrator.llm_provider, 'generate', new=AsyncMock(return_value=mock_response)):
            hypotheses, cost = await orchestrator._run_abduction_phase(
                "Create a game concept using Mobius strip",
                "How to create a game with Mobius strip theme?",
                max_retries=0
            )
        
        # Verify hypotheses don't start with H1:, H2:, etc.
        assert len(hypotheses) == 3
        for i, hypothesis in enumerate(hypotheses):
            assert not hypothesis.startswith(f"H{i+1}:")
            assert "Mobius" in hypothesis  # Content is preserved
    
    @pytest.mark.asyncio
    async def test_hypothesis_parsing_handles_both_formats(self):
        """Test that parser handles both old (H1:) and new formats."""
        orchestrator = SimpleQADIOrchestrator(num_hypotheses=2)
        
        # Test with old format (should still parse correctly)
        old_format_response = LLMResponse(
            content="""
            H1: Legacy Format Hypothesis
            This hypothesis uses the old H1: prefix format...
            
            H2: Another Legacy Hypothesis
            This also uses the old format with H2: prefix...
            """,
            model="gemini-pro",
            provider="google",
            cost=0.001
        )
        
        with patch.object(orchestrator.llm_provider, 'generate', new=AsyncMock(return_value=old_format_response)):
            hypotheses, _ = await orchestrator._run_abduction_phase(
                "Test input",
                "Test question?",
                max_retries=0
            )
        
        # Should parse but remove the H prefix
        assert len(hypotheses) == 2
        assert "Legacy Format Hypothesis" in hypotheses[0]
        assert not hypotheses[0].startswith("H1:")
        assert "Another Legacy Hypothesis" in hypotheses[1]
        assert not hypotheses[1].startswith("H2:")
    
    @pytest.mark.asyncio
    async def test_structured_output_format(self):
        """Test that structured output doesn't include H prefix."""
        orchestrator = SimpleQADIOrchestrator(num_hypotheses=2)
        
        # Mock structured JSON response
        json_response = LLMResponse(
            content='{"hypotheses": [{"id": "1", "content": "First hypothesis without H prefix"}, {"id": "2", "content": "Second hypothesis without H prefix"}]}',
            model="gemini-pro",
            provider="google",
            cost=0.001
        )
        
        with patch.object(orchestrator.llm_provider, 'generate', new=AsyncMock(return_value=json_response)):
            hypotheses, _ = await orchestrator._run_abduction_phase(
                "Test input",
                "Test question?",
                max_retries=0
            )
        
        assert len(hypotheses) == 2
        assert hypotheses[0] == "First hypothesis without H prefix"
        assert hypotheses[1] == "Second hypothesis without H prefix"
    
    def test_display_format_shows_approach_number_only(self):
        """Test that display format shows 'Approach 1:' not 'Approach 1: H1:'."""
        from qadi_simple import get_approach_label
        
        # Test various hypothesis texts
        test_cases = [
            ("This is a personal solution", 1, "Personal Approach: "),
            ("A team-based collaborative approach", 2, "Collaborative Approach: "),
            ("Systemic organizational change", 3, "Systemic Approach: "),
            ("Generic hypothesis text", 4, "Approach 4: "),
        ]
        
        for text, index, expected_prefix in test_cases:
            label = get_approach_label(text, index)
            assert label == expected_prefix
            # Ensure no H{number} in the label
            assert f"H{index}" not in label
    
    @pytest.mark.asyncio
    async def test_deduction_phase_references_without_h_prefix(self):
        """Test that deduction phase correctly references hypotheses without H prefix."""
        orchestrator = SimpleQADIOrchestrator(num_hypotheses=2)
        
        # Mock deduction response that references hypotheses
        deduction_response = LLMResponse(
            content='{"evaluations": [{"hypothesis_id": "1", "scores": {"impact": 0.8, "feasibility": 0.7, "accessibility": 0.6, "sustainability": 0.8, "scalability": 0.7}}, {"hypothesis_id": "2", "scores": {"impact": 0.7, "feasibility": 0.8, "accessibility": 0.7, "sustainability": 0.7, "scalability": 0.8}}], "answer": "Based on the evaluation, Approach 1 scores highest", "action_plan": ["Implement Approach 1", "Monitor results"]}',
            model="gemini-pro", 
            provider="google",
            cost=0.001
        )
        
        hypotheses = ["First approach", "Second approach"]
        
        with patch.object(orchestrator.llm_provider, 'generate', new=AsyncMock(return_value=deduction_response)):
            result = await orchestrator._run_deduction_phase(
                "Test input",
                "Test question?", 
                hypotheses,
                max_retries=0
            )
        
        # Answer should reference "Approach 1" not "H1"
        assert "Approach 1" in result["answer"]
        assert "H1" not in result["answer"]