"""
Tests for hypothesis parsing improvements to handle various LLM output formats.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from mad_spark_alt.core.simple_qadi_orchestrator import SimpleQADIOrchestrator
from mad_spark_alt.core.llm_provider import LLMResponse, LLMProvider


class TestHypothesisParsingFix:
    """Test that hypothesis parsing handles various output formats correctly."""

    @pytest.mark.asyncio
    async def test_parse_hypotheses_with_ansi_codes(self):
        """Test parsing hypotheses with ANSI color codes."""
        orchestrator = SimpleQADIOrchestrator()
        
        # Mock LLM manager
        mock_llm_manager = MagicMock()
        mock_llm_manager.generate = AsyncMock()
        
        # Response with ANSI codes (as seen in actual output)
        response_with_ansi = """[1mApproach 1:[0m Create a foundational, highly abstract "cognitive kernel" or "orchestrator" that defines standard interfaces for various "cognitive services." Think of it as building the APIs and internal bus for AGI.

[1mApproach 2:[0m Develop a multi-modal learning system that combines reinforcement learning, transformer architectures, and neurosymbolic reasoning. This system would continuously learn from diverse data sources and adapt its internal representations.

[1mApproach 3:[0m Build a distributed AGI framework using microservices architecture where each service handles specific cognitive functions (perception, reasoning, memory, planning) and communicates through a unified protocol."""
        
        mock_llm_manager.generate.return_value = LLMResponse(
            content=response_with_ansi,
            provider=LLMProvider.GOOGLE,
            model="gemini-1.5-flash",
            usage={"input_tokens": 100, "output_tokens": 200},
            cost=0.001,
            response_time=0.5,
            metadata={"test": True}
        )
        
        # Test hypothesis extraction with patched llm_manager
        with patch("mad_spark_alt.core.simple_qadi_orchestrator.llm_manager", mock_llm_manager):
            hypotheses, cost = await orchestrator._run_abduction_phase("test input", "test question", max_retries=0)
        
        assert len(hypotheses) == 3
        assert "cognitive kernel" in hypotheses[0]
        assert "multi-modal learning system" in hypotheses[1]
        assert "distributed AGI framework" in hypotheses[2]
        
        # Ensure ANSI codes are stripped
        for hypothesis in hypotheses:
            assert "[1m" not in hypothesis
            assert "[0m" not in hypothesis

    @pytest.mark.asyncio
    async def test_parse_hypotheses_with_approach_format(self):
        """Test parsing hypotheses with 'Approach N:' format."""
        orchestrator = SimpleQADIOrchestrator()
        
        # Mock LLM manager
        mock_llm_manager = MagicMock()
        mock_llm_manager.generate = AsyncMock()
        
        # Response with "Approach N:" format
        response_approach_format = """Approach 1: Implement a blockchain-based solution for transparent tracking and verification of all transactions in the supply chain.

Approach 2: Create a machine learning model that predicts demand patterns and optimizes inventory levels across multiple locations.

Approach 3: Develop a collaborative platform that connects suppliers, manufacturers, and retailers in real-time for better coordination."""
        
        mock_llm_manager.generate.return_value = LLMResponse(
            content=response_approach_format,
            provider=LLMProvider.GOOGLE,
            model="gemini-1.5-flash",
            usage={"input_tokens": 100, "output_tokens": 200},
            cost=0.001,
            response_time=0.5,
            metadata={"test": True}
        )
        
        # Test hypothesis extraction with patched llm_manager
        with patch("mad_spark_alt.core.simple_qadi_orchestrator.llm_manager", mock_llm_manager):
            hypotheses, cost = await orchestrator._run_abduction_phase("test input", "test question", max_retries=0)
        
        assert len(hypotheses) == 3
        assert "blockchain-based solution" in hypotheses[0]
        assert "machine learning model" in hypotheses[1]
        assert "collaborative platform" in hypotheses[2]

    @pytest.mark.asyncio
    async def test_parse_hypotheses_with_mixed_formats(self):
        """Test parsing hypotheses with mixed formats."""
        orchestrator = SimpleQADIOrchestrator()
        
        # Mock LLM manager
        mock_llm_manager = MagicMock()
        mock_llm_manager.generate = AsyncMock()
        
        # Response with mixed formats
        response_mixed = """H1: Traditional approach using established methodologies.

**Approach 2:** Modern solution leveraging cloud technologies.

3. Innovative method combining multiple disciplines."""
        
        mock_llm_manager.generate.return_value = LLMResponse(
            content=response_mixed,
            provider=LLMProvider.GOOGLE,
            model="gemini-1.5-flash",
            usage={"input_tokens": 100, "output_tokens": 200},
            cost=0.001,
            response_time=0.5,
            metadata={"test": True}
        )
        
        # Test hypothesis extraction with patched llm_manager
        with patch("mad_spark_alt.core.simple_qadi_orchestrator.llm_manager", mock_llm_manager):
            hypotheses, cost = await orchestrator._run_abduction_phase("test input", "test question", max_retries=0)
        
        assert len(hypotheses) == 3
        assert "Traditional approach" in hypotheses[0]
        assert "Modern solution" in hypotheses[1]
        assert "Innovative method" in hypotheses[2]

    @pytest.mark.asyncio
    async def test_hypotheses_not_truncated_for_evolution(self):
        """Test that full hypotheses are passed to evolution, not truncated versions."""
        orchestrator = SimpleQADIOrchestrator()
        
        # Mock LLM manager
        mock_llm_manager = MagicMock()
        mock_llm_manager.generate = AsyncMock()
        
        # Long hypothesis that would be truncated for display
        long_hypothesis = "A" * 500  # 500 characters
        
        response_long = f"""H1: {long_hypothesis}

H2: Short hypothesis.

H3: Another short one."""
        
        mock_llm_manager.generate.return_value = LLMResponse(
            content=response_long,
            provider=LLMProvider.GOOGLE,
            model="gemini-1.5-flash",
            usage={"input_tokens": 100, "output_tokens": 600},
            cost=0.001,
            response_time=0.5,
            metadata={"test": True}
        )
        
        # Mock all phases for full QADI cycle
        mock_llm_manager.generate.side_effect = [
            # Questioning phase response
            LLMResponse(
                content="Q: How to test hypothesis preservation?",
                provider=LLMProvider.GOOGLE,
                model="gemini-1.5-flash",
                usage={"input_tokens": 50, "output_tokens": 20},
                cost=0.0001,
                response_time=0.1,
                metadata={"test": True}
            ),
            # Abduction phase response (already configured above)
            mock_llm_manager.generate.return_value,
            # Deduction phase response
            LLMResponse(
                content="H1:\n* Impact: 0.8\n* Feasibility: 0.7\n\nANSWER: Test answer",
                provider=LLMProvider.GOOGLE,
                model="gemini-1.5-flash",
                usage={"input_tokens": 200, "output_tokens": 100},
                cost=0.001,
                response_time=0.3,
                metadata={"test": True}
            ),
            # Induction phase response
            LLMResponse(
                content="Example 1: Test example",
                provider=LLMProvider.GOOGLE,
                model="gemini-1.5-flash",
                usage={"input_tokens": 150, "output_tokens": 80},
                cost=0.0005,
                response_time=0.2,
                metadata={"test": True}
            )
        ]
        
        # Run QADI cycle with patched llm_manager
        with patch("mad_spark_alt.core.simple_qadi_orchestrator.llm_manager", mock_llm_manager):
            result = await orchestrator.run_qadi_cycle("test input", "test context")
        
        # Check synthesized ideas contain full content
        assert len(result.synthesized_ideas) >= 1
        assert len(result.synthesized_ideas[0].content) == 500  # Full length preserved
        assert result.synthesized_ideas[0].content == long_hypothesis