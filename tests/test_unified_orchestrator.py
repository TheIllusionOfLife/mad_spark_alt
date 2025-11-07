"""
Comprehensive tests for UnifiedQADIOrchestrator.

Following TDD methodology - tests written BEFORE implementation.
Tests cover all three strategies: Simple, MultiPerspective, Smart.
"""

import pytest
from unittest.mock import AsyncMock, Mock, patch
from typing import List


# =============================================================================
# Simple Strategy Tests (20-25 tests)
# =============================================================================


class TestUnifiedOrchestratorSimple:
    """Tests for UnifiedQADIOrchestrator with Simple strategy."""

    @pytest.fixture
    def simple_config(self):
        """Create simple strategy configuration."""
        from mad_spark_alt.core.orchestrator_config import OrchestratorConfig
        return OrchestratorConfig.simple_config()

    @pytest.fixture
    def fast_config(self):
        """Create fast (parallel) configuration."""
        from mad_spark_alt.core.orchestrator_config import OrchestratorConfig
        return OrchestratorConfig.fast_config()

    def test_init_with_simple_config(self, simple_config):
        """Test initialization with simple configuration."""
        from mad_spark_alt.core.unified_orchestrator import UnifiedQADIOrchestrator

        orchestrator = UnifiedQADIOrchestrator(config=simple_config)

        assert orchestrator.config == simple_config
        assert orchestrator.config.strategy.value == "simple"
        assert orchestrator.config.execution_mode.value == "sequential"

    def test_init_with_default_config(self):
        """Test initialization without config creates default."""
        from mad_spark_alt.core.unified_orchestrator import UnifiedQADIOrchestrator

        orchestrator = UnifiedQADIOrchestrator()

        assert orchestrator.config is not None
        assert orchestrator.config.strategy.value == "simple"

    def test_init_validates_config(self):
        """Test initialization validates the configuration."""
        from mad_spark_alt.core.unified_orchestrator import UnifiedQADIOrchestrator
        from mad_spark_alt.core.orchestrator_config import OrchestratorConfig, Strategy

        # Invalid config: multi-perspective without perspectives
        invalid_config = OrchestratorConfig(
            strategy=Strategy.MULTI_PERSPECTIVE,
            perspectives=None,
            auto_detect_perspectives=False
        )

        with pytest.raises(ValueError, match="Multi-perspective requires perspectives"):
            UnifiedQADIOrchestrator(config=invalid_config)

    def test_init_with_custom_registry(self, simple_config):
        """Test initialization with custom registry."""
        from mad_spark_alt.core.unified_orchestrator import UnifiedQADIOrchestrator
        from mad_spark_alt.core.smart_registry import SmartAgentRegistry

        custom_registry = SmartAgentRegistry()
        orchestrator = UnifiedQADIOrchestrator(
            config=simple_config,
            registry=custom_registry
        )

        assert orchestrator.registry == custom_registry

    def test_init_auto_setup_flag(self, simple_config):
        """Test auto_setup flag is stored."""
        from mad_spark_alt.core.unified_orchestrator import UnifiedQADIOrchestrator

        orch_with_setup = UnifiedQADIOrchestrator(config=simple_config, auto_setup=True)
        orch_without = UnifiedQADIOrchestrator(config=simple_config, auto_setup=False)

        assert orch_with_setup.auto_setup is True
        assert orch_without.auto_setup is False

    @pytest.mark.asyncio
    async def test_simple_sequential_full_cycle(self, simple_config):
        """Test simple sequential QADI cycle completes successfully."""
        from mad_spark_alt.core.unified_orchestrator import UnifiedQADIOrchestrator

        orchestrator = UnifiedQADIOrchestrator(config=simple_config)

        # Mock SimpleQADIOrchestrator
        with patch('mad_spark_alt.core.unified_orchestrator.SimpleQADIOrchestrator') as MockSimple:
            # Create mock result
            mock_result = Mock()
            mock_result.core_question = "What is AI?"
            mock_result.hypotheses = ["H1", "H2", "H3"]
            mock_result.hypothesis_scores = [Mock(), Mock(), Mock()]
            mock_result.final_answer = "AI is..."
            mock_result.action_plan = ["Step 1", "Step 2"]
            mock_result.verification_examples = ["Example 1"]
            mock_result.verification_conclusion = "Verified"
            mock_result.total_llm_cost = 0.005
            mock_result.phase_results = {}
            mock_result.synthesized_ideas = []

            mock_instance = AsyncMock()
            mock_instance.run_qadi_cycle.return_value = mock_result
            MockSimple.return_value = mock_instance

            result = await orchestrator.run_qadi_cycle("What is AI?")

            # Verify result structure
            assert result.core_question == "What is AI?"
            assert len(result.hypotheses) == 3
            assert result.final_answer == "AI is..."
            assert len(result.action_plan) == 2
            assert result.total_llm_cost == 0.005
            assert result.strategy_used.value == "simple"
            assert result.execution_mode.value == "sequential"

    @pytest.mark.asyncio
    async def test_simple_sequential_cost_tracking(self, simple_config):
        """Test cost tracking in simple sequential mode."""
        from mad_spark_alt.core.unified_orchestrator import UnifiedQADIOrchestrator

        orchestrator = UnifiedQADIOrchestrator(config=simple_config)

        with patch('mad_spark_alt.core.unified_orchestrator.SimpleQADIOrchestrator') as MockSimple:
            mock_result = Mock()
            mock_result.core_question = "Q"
            mock_result.hypotheses = ["H1"]
            mock_result.hypothesis_scores = [Mock()]
            mock_result.final_answer = "A"
            mock_result.action_plan = []
            mock_result.verification_examples = []
            mock_result.verification_conclusion = "C"
            mock_result.total_llm_cost = 0.0123
            mock_result.phase_results = {}
            mock_result.synthesized_ideas = []

            mock_instance = AsyncMock()
            mock_instance.run_qadi_cycle.return_value = mock_result
            MockSimple.return_value = mock_instance

            result = await orchestrator.run_qadi_cycle("Question")

            assert result.total_llm_cost == 0.0123

    @pytest.mark.asyncio
    async def test_simple_sequential_hypothesis_count(self, simple_config):
        """Test hypothesis count configuration is respected."""
        from mad_spark_alt.core.unified_orchestrator import UnifiedQADIOrchestrator

        simple_config.num_hypotheses = 5
        orchestrator = UnifiedQADIOrchestrator(config=simple_config)

        with patch('mad_spark_alt.core.unified_orchestrator.SimpleQADIOrchestrator') as MockSimple:
            mock_instance = AsyncMock()
            MockSimple.return_value = mock_instance

            await orchestrator.run_qadi_cycle("Question")

            # Verify SimpleQADI was initialized with correct hypothesis count
            MockSimple.assert_called_once()
            call_kwargs = MockSimple.call_args.kwargs
            assert call_kwargs['num_hypotheses'] == 5

    @pytest.mark.asyncio
    async def test_simple_sequential_temperature_override(self, simple_config):
        """Test temperature override is passed to SimpleQADI."""
        from mad_spark_alt.core.unified_orchestrator import UnifiedQADIOrchestrator

        simple_config.temperature_override = 1.5
        orchestrator = UnifiedQADIOrchestrator(config=simple_config)

        with patch('mad_spark_alt.core.unified_orchestrator.SimpleQADIOrchestrator') as MockSimple:
            mock_instance = AsyncMock()
            MockSimple.return_value = mock_instance

            await orchestrator.run_qadi_cycle("Question")

            # Verify temperature was passed
            call_kwargs = MockSimple.call_args.kwargs
            assert call_kwargs['temperature_override'] == 1.5

    @pytest.mark.asyncio
    async def test_simple_sequential_context_passing(self, simple_config):
        """Test context is passed through to SimpleQADI."""
        from mad_spark_alt.core.unified_orchestrator import UnifiedQADIOrchestrator

        orchestrator = UnifiedQADIOrchestrator(config=simple_config)
        context = "Previous analysis showed X, Y, Z"

        with patch('mad_spark_alt.core.unified_orchestrator.SimpleQADIOrchestrator') as MockSimple:
            mock_instance = AsyncMock()
            MockSimple.return_value = mock_instance

            await orchestrator.run_qadi_cycle("Question", context=context)

            # Verify context was passed
            mock_instance.run_qadi_cycle.assert_called_once()
            call_kwargs = mock_instance.run_qadi_cycle.call_args.kwargs
            assert call_kwargs['context'] == context

    @pytest.mark.asyncio
    async def test_simple_sequential_phase_results(self, simple_config):
        """Test phase results are preserved in unified result."""
        from mad_spark_alt.core.unified_orchestrator import UnifiedQADIOrchestrator

        orchestrator = UnifiedQADIOrchestrator(config=simple_config)

        with patch('mad_spark_alt.core.unified_orchestrator.SimpleQADIOrchestrator') as MockSimple:
            mock_result = Mock()
            mock_result.core_question = "Q"
            mock_result.hypotheses = ["H1"]
            mock_result.hypothesis_scores = [Mock()]
            mock_result.final_answer = "A"
            mock_result.action_plan = []
            mock_result.verification_examples = []
            mock_result.verification_conclusion = "C"
            mock_result.total_llm_cost = 0.005
            mock_result.phase_results = {
                "questioning": {"elapsed": 2.5},
                "abduction": {"elapsed": 3.0}
            }
            mock_result.synthesized_ideas = []

            mock_instance = AsyncMock()
            mock_instance.run_qadi_cycle.return_value = mock_result
            MockSimple.return_value = mock_instance

            result = await orchestrator.run_qadi_cycle("Question")

            assert "questioning" in result.phase_results
            assert "abduction" in result.phase_results

    @pytest.mark.asyncio
    async def test_simple_sequential_error_handling(self, simple_config):
        """Test error handling in simple sequential mode."""
        from mad_spark_alt.core.unified_orchestrator import UnifiedQADIOrchestrator

        orchestrator = UnifiedQADIOrchestrator(config=simple_config)

        with patch('mad_spark_alt.core.unified_orchestrator.SimpleQADIOrchestrator') as MockSimple:
            mock_instance = AsyncMock()
            mock_instance.run_qadi_cycle.side_effect = ValueError("Test error")
            MockSimple.return_value = mock_instance

            with pytest.raises(ValueError, match="Test error"):
                await orchestrator.run_qadi_cycle("Question")

    @pytest.mark.asyncio
    async def test_simple_sequential_retry_logic(self, simple_config):
        """Test retry configuration is passed through."""
        from mad_spark_alt.core.unified_orchestrator import UnifiedQADIOrchestrator

        simple_config.timeout_config.max_retries = 5
        orchestrator = UnifiedQADIOrchestrator(config=simple_config)

        with patch('mad_spark_alt.core.unified_orchestrator.SimpleQADIOrchestrator') as MockSimple:
            mock_instance = AsyncMock()
            MockSimple.return_value = mock_instance

            await orchestrator.run_qadi_cycle("Question")

            # Verify max_retries was passed
            call_kwargs = mock_instance.run_qadi_cycle.call_args.kwargs
            assert call_kwargs['max_retries'] == 5

    @pytest.mark.asyncio
    async def test_simple_parallel_full_cycle(self, fast_config):
        """Test simple parallel QADI cycle (if supported)."""
        from mad_spark_alt.core.unified_orchestrator import UnifiedQADIOrchestrator

        orchestrator = UnifiedQADIOrchestrator(config=fast_config)

        with patch('mad_spark_alt.core.unified_orchestrator.SimpleQADIOrchestrator') as MockSimple:
            mock_result = Mock()
            mock_result.core_question = "Q"
            mock_result.hypotheses = ["H1"]
            mock_result.hypothesis_scores = [Mock()]
            mock_result.final_answer = "A"
            mock_result.action_plan = []
            mock_result.verification_examples = []
            mock_result.verification_conclusion = "C"
            mock_result.total_llm_cost = 0.005
            mock_result.phase_results = {}
            mock_result.synthesized_ideas = []

            mock_instance = AsyncMock()
            mock_instance.run_qadi_cycle.return_value = mock_result
            MockSimple.return_value = mock_instance

            result = await orchestrator.run_qadi_cycle("Question")

            # Note: SimpleQADI doesn't natively support parallel, but unified should handle it
            assert result.execution_mode.value == "parallel"

    @pytest.mark.asyncio
    async def test_result_contains_all_fields(self, simple_config):
        """Test unified result contains all required fields."""
        from mad_spark_alt.core.unified_orchestrator import UnifiedQADIOrchestrator

        orchestrator = UnifiedQADIOrchestrator(config=simple_config)

        with patch('mad_spark_alt.core.unified_orchestrator.SimpleQADIOrchestrator') as MockSimple:
            mock_result = Mock()
            mock_result.core_question = "Q"
            mock_result.hypotheses = ["H1", "H2"]
            mock_result.hypothesis_scores = [Mock(), Mock()]
            mock_result.final_answer = "A"
            mock_result.action_plan = ["Step"]
            mock_result.verification_examples = ["Ex"]
            mock_result.verification_conclusion = "C"
            mock_result.total_llm_cost = 0.005
            mock_result.phase_results = {}
            mock_result.synthesized_ideas = []

            mock_instance = AsyncMock()
            mock_instance.run_qadi_cycle.return_value = mock_result
            MockSimple.return_value = mock_instance

            result = await orchestrator.run_qadi_cycle("Question")

            # Check all required fields
            assert hasattr(result, 'strategy_used')
            assert hasattr(result, 'execution_mode')
            assert hasattr(result, 'core_question')
            assert hasattr(result, 'hypotheses')
            assert hasattr(result, 'final_answer')
            assert hasattr(result, 'action_plan')
            assert hasattr(result, 'total_llm_cost')
            assert hasattr(result, 'synthesized_ideas')
            assert hasattr(result, 'hypothesis_scores')
            assert hasattr(result, 'verification_examples')
            assert hasattr(result, 'verification_conclusion')

    @pytest.mark.asyncio
    async def test_result_backward_compatible(self, simple_config):
        """Test unified result is backward compatible with SimpleQADIResult."""
        from mad_spark_alt.core.unified_orchestrator import UnifiedQADIOrchestrator

        orchestrator = UnifiedQADIOrchestrator(config=simple_config)

        with patch('mad_spark_alt.core.unified_orchestrator.SimpleQADIOrchestrator') as MockSimple:
            mock_result = Mock()
            mock_result.core_question = "Q"
            mock_result.hypotheses = ["H1"]
            mock_result.hypothesis_scores = [Mock()]
            mock_result.final_answer = "A"
            mock_result.action_plan = []
            mock_result.verification_examples = []
            mock_result.verification_conclusion = "C"
            mock_result.total_llm_cost = 0.005
            mock_result.phase_results = {}
            mock_result.synthesized_ideas = []

            mock_instance = AsyncMock()
            mock_instance.run_qadi_cycle.return_value = mock_result
            MockSimple.return_value = mock_instance

            result = await orchestrator.run_qadi_cycle("Question")

            # Should have all SimpleQADIResult fields
            assert result.core_question is not None
            assert result.hypotheses is not None
            assert result.hypothesis_scores is not None
            assert result.final_answer is not None
            assert result.verification_examples is not None
            assert result.verification_conclusion is not None

    @pytest.mark.asyncio
    async def test_result_evolution_compatible(self, simple_config):
        """Test result is compatible with evolution system."""
        from mad_spark_alt.core.unified_orchestrator import UnifiedQADIOrchestrator

        orchestrator = UnifiedQADIOrchestrator(config=simple_config)

        with patch('mad_spark_alt.core.unified_orchestrator.SimpleQADIOrchestrator') as MockSimple:
            from mad_spark_alt.core.interfaces import GeneratedIdea, ThinkingMethod

            mock_ideas = [
                GeneratedIdea(
                    content="Idea 1",
                    thinking_method=ThinkingMethod.ABDUCTION,
                    agent_name="test",
                    generation_prompt="prompt"
                )
            ]

            mock_result = Mock()
            mock_result.core_question = "Q"
            mock_result.hypotheses = ["H1"]
            mock_result.hypothesis_scores = [Mock()]
            mock_result.final_answer = "A"
            mock_result.action_plan = []
            mock_result.verification_examples = []
            mock_result.verification_conclusion = "C"
            mock_result.total_llm_cost = 0.005
            mock_result.phase_results = {}
            mock_result.synthesized_ideas = mock_ideas

            mock_instance = AsyncMock()
            mock_instance.run_qadi_cycle.return_value = mock_result
            MockSimple.return_value = mock_instance

            result = await orchestrator.run_qadi_cycle("Question")

            # Should have synthesized_ideas for evolution
            assert hasattr(result, 'synthesized_ideas')
            assert len(result.synthesized_ideas) == 1
            assert result.synthesized_ideas[0].content == "Idea 1"

    @pytest.mark.asyncio
    async def test_result_cost_aggregation(self, simple_config):
        """Test cost aggregation from phases."""
        from mad_spark_alt.core.unified_orchestrator import UnifiedQADIOrchestrator

        orchestrator = UnifiedQADIOrchestrator(config=simple_config)

        with patch('mad_spark_alt.core.unified_orchestrator.SimpleQADIOrchestrator') as MockSimple:
            mock_result = Mock()
            mock_result.core_question = "Q"
            mock_result.hypotheses = ["H1"]
            mock_result.hypothesis_scores = [Mock()]
            mock_result.final_answer = "A"
            mock_result.action_plan = []
            mock_result.verification_examples = []
            mock_result.verification_conclusion = "C"
            mock_result.total_llm_cost = 0.01234
            mock_result.phase_results = {}
            mock_result.synthesized_ideas = []

            mock_instance = AsyncMock()
            mock_instance.run_qadi_cycle.return_value = mock_result
            MockSimple.return_value = mock_instance

            result = await orchestrator.run_qadi_cycle("Question")

            assert result.total_llm_cost == 0.01234

    @pytest.mark.asyncio
    async def test_result_metadata_tracking(self, simple_config):
        """Test execution metadata is captured."""
        from mad_spark_alt.core.unified_orchestrator import UnifiedQADIOrchestrator

        orchestrator = UnifiedQADIOrchestrator(config=simple_config)

        with patch('mad_spark_alt.core.unified_orchestrator.SimpleQADIOrchestrator') as MockSimple:
            mock_result = Mock()
            mock_result.core_question = "Q"
            mock_result.hypotheses = ["H1"]
            mock_result.hypothesis_scores = [Mock()]
            mock_result.final_answer = "A"
            mock_result.action_plan = []
            mock_result.verification_examples = []
            mock_result.verification_conclusion = "C"
            mock_result.total_llm_cost = 0.005
            mock_result.phase_results = {}
            mock_result.synthesized_ideas = []

            mock_instance = AsyncMock()
            mock_instance.run_qadi_cycle.return_value = mock_result
            MockSimple.return_value = mock_instance

            result = await orchestrator.run_qadi_cycle("Question")

            # Should have execution metadata
            assert hasattr(result, 'execution_metadata')
            assert isinstance(result.execution_metadata, dict)


# Placeholder for future test classes (to be added in next steps)
class TestUnifiedOrchestratorMultiPerspective:
    """Tests for Multi-Perspective strategy - to be implemented."""
    pass


class TestUnifiedOrchestratorSmart:
    """Tests for Smart strategy - to be implemented."""
    pass
