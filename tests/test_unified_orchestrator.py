"""
Comprehensive tests for UnifiedQADIOrchestrator.

Following TDD methodology - tests written BEFORE implementation.
Tests cover Simple and MultiPerspective strategies.
"""

import pytest
from unittest.mock import AsyncMock, Mock, patch
from typing import List

from mad_spark_alt.core.orchestrator_config import OrchestratorConfig, Strategy, ExecutionMode
from mad_spark_alt.core.unified_orchestrator import UnifiedQADIOrchestrator


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


# Multi-Perspective Strategy Tests
class TestUnifiedOrchestratorMultiPerspective:
    """Tests for Multi-Perspective strategy integration."""

    @pytest.mark.asyncio
    async def test_init_with_multi_perspective_config(self):
        """Test initialization with multi-perspective config."""
        config = OrchestratorConfig.multi_perspective_config(
            perspectives=["environmental", "technical"]
        )
        orchestrator = UnifiedQADIOrchestrator(config=config)

        assert orchestrator.config.strategy == Strategy.MULTI_PERSPECTIVE
        assert orchestrator.config.perspectives == ["environmental", "technical"]

    @pytest.mark.asyncio
    async def test_init_validates_multi_perspective_requirements(self):
        """Test that multi-perspective config validation works."""
        # Should fail without perspectives or auto_detect
        config = OrchestratorConfig(
            strategy=Strategy.MULTI_PERSPECTIVE,
            perspectives=None,
            auto_detect_perspectives=False
        )

        with pytest.raises(ValueError, match="Multi-perspective requires"):
            UnifiedQADIOrchestrator(config=config)

    @pytest.mark.asyncio
    @patch('mad_spark_alt.core.unified_orchestrator.MultiPerspectiveQADIOrchestrator')
    async def test_multi_perspective_with_explicit_perspectives(self, MockMultiPerspective):
        """Test multi-perspective execution with explicit perspectives."""
        from mad_spark_alt.core.multi_perspective_orchestrator import (
            MultiPerspectiveQADIResult,
            PerspectiveResult
        )
        from mad_spark_alt.core.intent_detector import QuestionIntent
        from mad_spark_alt.core.simple_qadi_orchestrator import SimpleQADIResult, HypothesisScore

        # Mock perspective results
        simple_result1 = SimpleQADIResult(
            core_question="Environmental question",
            hypotheses=["H1-env", "H2-env", "H3-env"],
            hypothesis_scores=[
                HypothesisScore(impact=0.9, feasibility=0.8, accessibility=0.7, sustainability=0.85, scalability=0.75, overall=0.80),
                HypothesisScore(impact=0.7, feasibility=0.6, accessibility=0.8, sustainability=0.65, scalability=0.70, overall=0.69),
                HypothesisScore(impact=0.6, feasibility=0.7, accessibility=0.6, sustainability=0.65, scalability=0.65, overall=0.64)
            ],
            final_answer="Environmental answer",
            action_plan=["Step 1", "Step 2"],
            verification_examples=["Example 1"],
            verification_conclusion="Verified",
            total_llm_cost=0.01,
            phase_results={},
            synthesized_ideas=[]
        )

        simple_result2 = SimpleQADIResult(
            core_question="Technical question",
            hypotheses=["H1-tech", "H2-tech", "H3-tech"],
            hypothesis_scores=[
                HypothesisScore(impact=0.85, feasibility=0.9, accessibility=0.75, sustainability=0.88, scalability=0.82, overall=0.84),
                HypothesisScore(impact=0.75, feasibility=0.8, accessibility=0.7, sustainability=0.78, scalability=0.75, overall=0.76),
                HypothesisScore(impact=0.65, feasibility=0.7, accessibility=0.65, sustainability=0.68, scalability=0.68, overall=0.67)
            ],
            final_answer="Technical answer",
            action_plan=["Tech step 1", "Tech step 2"],
            verification_examples=["Tech example 1"],
            verification_conclusion="Tech verified",
            total_llm_cost=0.01,
            phase_results={},
            synthesized_ideas=[]
        )

        perspective_result1 = PerspectiveResult(
            perspective=QuestionIntent.ENVIRONMENTAL,
            result=simple_result1,
            relevance_score=1.0
        )

        perspective_result2 = PerspectiveResult(
            perspective=QuestionIntent.TECHNICAL,
            result=simple_result2,
            relevance_score=0.8
        )

        mock_mp_result = MultiPerspectiveQADIResult(
            primary_intent=QuestionIntent.ENVIRONMENTAL,
            intent_confidence=0.9,
            keywords_matched=["climate", "sustainable"],
            perspective_results=[perspective_result1, perspective_result2],
            synthesized_answer="Synthesized answer combining perspectives",
            synthesized_action_plan=["Action 1", "Action 2", "Action 3"],
            best_hypothesis=("H1-env", QuestionIntent.ENVIRONMENTAL),
            total_llm_cost=0.025,
            perspectives_used=[QuestionIntent.ENVIRONMENTAL, QuestionIntent.TECHNICAL],
            synthesized_ideas=[]
        )

        mock_instance = AsyncMock()
        mock_instance.run_multi_perspective_analysis.return_value = mock_mp_result
        MockMultiPerspective.return_value = mock_instance

        config = OrchestratorConfig.multi_perspective_config(
            perspectives=["environmental", "technical"]
        )
        orchestrator = UnifiedQADIOrchestrator(config=config)

        result = await orchestrator.run_qadi_cycle("How to build sustainable cities?")

        # Verify orchestrator was called correctly
        MockMultiPerspective.assert_called_once_with(temperature_override=None)
        mock_instance.run_multi_perspective_analysis.assert_called_once()
        call_args = mock_instance.run_multi_perspective_analysis.call_args
        assert call_args[0][0] == "How to build sustainable cities?"

        # Verify result structure
        assert result.strategy_used == Strategy.MULTI_PERSPECTIVE
        assert result.core_question == "How to build sustainable cities?"
        assert result.final_answer == "Synthesized answer combining perspectives"
        assert result.action_plan == ["Action 1", "Action 2", "Action 3"]
        assert result.total_llm_cost == 0.025
        assert result.perspectives_used == ["environmental", "technical"]
        assert result.synthesized_answer == "Synthesized answer combining perspectives"

    @pytest.mark.asyncio
    @patch('mad_spark_alt.core.unified_orchestrator.MultiPerspectiveQADIOrchestrator')
    async def test_multi_perspective_with_auto_detect(self, MockMultiPerspective):
        """Test multi-perspective with auto-detection enabled."""
        from mad_spark_alt.core.multi_perspective_orchestrator import (
            MultiPerspectiveQADIResult,
            PerspectiveResult
        )
        from mad_spark_alt.core.intent_detector import QuestionIntent
        from mad_spark_alt.core.simple_qadi_orchestrator import SimpleQADIResult, HypothesisScore

        simple_result = SimpleQADIResult(
            core_question="Business question",
            hypotheses=["H1", "H2", "H3"],
            hypothesis_scores=[
                HypothesisScore(impact=0.8, feasibility=0.7, accessibility=0.8, sustainability=0.75, scalability=0.75, overall=0.76)
            ],
            final_answer="Business answer",
            action_plan=["Step 1"],
            verification_examples=[],
            verification_conclusion="OK",
            total_llm_cost=0.01,
            phase_results={},
            synthesized_ideas=[]
        )

        perspective_result = PerspectiveResult(
            perspective=QuestionIntent.BUSINESS,
            result=simple_result,
            relevance_score=1.0
        )

        mock_mp_result = MultiPerspectiveQADIResult(
            primary_intent=QuestionIntent.BUSINESS,
            intent_confidence=0.95,
            keywords_matched=["market", "revenue"],
            perspective_results=[perspective_result],
            synthesized_answer="Business synthesis",
            synthesized_action_plan=["Business action"],
            best_hypothesis=("H1", QuestionIntent.BUSINESS),
            total_llm_cost=0.015,
            perspectives_used=[QuestionIntent.BUSINESS],
            synthesized_ideas=[]
        )

        mock_instance = AsyncMock()
        mock_instance.run_multi_perspective_analysis.return_value = mock_mp_result
        MockMultiPerspective.return_value = mock_instance

        config = OrchestratorConfig(
            strategy=Strategy.MULTI_PERSPECTIVE,
            auto_detect_perspectives=True
        )
        orchestrator = UnifiedQADIOrchestrator(config=config)

        result = await orchestrator.run_qadi_cycle("What's our market strategy?")

        # Verify force_perspectives is None (auto-detect)
        call_args = mock_instance.run_multi_perspective_analysis.call_args
        assert call_args[1].get('force_perspectives') is None

    @pytest.mark.asyncio
    @patch('mad_spark_alt.core.unified_orchestrator.MultiPerspectiveQADIOrchestrator')
    async def test_multi_perspective_temperature_override(self, MockMultiPerspective):
        """Test temperature override is passed to MultiPerspective orchestrator."""

        from mad_spark_alt.core.multi_perspective_orchestrator import MultiPerspectiveQADIResult
        from mad_spark_alt.core.intent_detector import QuestionIntent

        mock_mp_result = MultiPerspectiveQADIResult(
            primary_intent=QuestionIntent.TECHNICAL,
            intent_confidence=0.9,
            keywords_matched=[],
            perspective_results=[],
            synthesized_answer="Answer",
            synthesized_action_plan=[],
            best_hypothesis=("H", QuestionIntent.TECHNICAL),
            total_llm_cost=0.01,
            perspectives_used=[QuestionIntent.TECHNICAL],
            synthesized_ideas=[]
        )

        mock_instance = AsyncMock()
        mock_instance.run_multi_perspective_analysis.return_value = mock_mp_result
        MockMultiPerspective.return_value = mock_instance

        config = OrchestratorConfig(
            strategy=Strategy.MULTI_PERSPECTIVE,
            auto_detect_perspectives=True,
            temperature_override=1.2
        )
        orchestrator = UnifiedQADIOrchestrator(config=config)

        await orchestrator.run_qadi_cycle("Question")

        # Verify temperature was passed
        MockMultiPerspective.assert_called_once_with(temperature_override=1.2)

    @pytest.mark.asyncio
    @patch('mad_spark_alt.core.unified_orchestrator.MultiPerspectiveQADIOrchestrator')
    async def test_multi_perspective_top_n_hypotheses(self, MockMultiPerspective):
        """Test that top N hypotheses are collected across perspectives."""

        from mad_spark_alt.core.multi_perspective_orchestrator import (
            MultiPerspectiveQADIResult,
            PerspectiveResult
        )
        from mad_spark_alt.core.intent_detector import QuestionIntent
        from mad_spark_alt.core.simple_qadi_orchestrator import SimpleQADIResult, HypothesisScore

        # Create results with different scores
        simple_result1 = SimpleQADIResult(
            core_question="Q1",
            hypotheses=["H1-p1", "H2-p1"],
            hypothesis_scores=[
                HypothesisScore(impact=0.9, feasibility=0.9, accessibility=0.9, sustainability=0.90, scalability=0.90, overall=0.90),  # 0.9
                HypothesisScore(impact=0.7, feasibility=0.7, accessibility=0.7, sustainability=0.70, scalability=0.70, overall=0.70)  # 0.7
            ],
            final_answer="A1",
            action_plan=[],
            verification_examples=[],
            verification_conclusion="OK",
            total_llm_cost=0.01,
            phase_results={},
            synthesized_ideas=[]
        )

        simple_result2 = SimpleQADIResult(
            core_question="Q2",
            hypotheses=["H1-p2", "H2-p2"],
            hypothesis_scores=[
                HypothesisScore(impact=0.85, feasibility=0.85, accessibility=0.85, sustainability=0.85, scalability=0.85, overall=0.85),  # 0.85
                HypothesisScore(impact=0.6, feasibility=0.6, accessibility=0.6, sustainability=0.60, scalability=0.60, overall=0.60)  # 0.6
            ],
            final_answer="A2",
            action_plan=[],
            verification_examples=[],
            verification_conclusion="OK",
            total_llm_cost=0.01,
            phase_results={},
            synthesized_ideas=[]
        )

        perspective_result1 = PerspectiveResult(
            perspective=QuestionIntent.ENVIRONMENTAL,
            result=simple_result1,
            relevance_score=1.0
        )

        perspective_result2 = PerspectiveResult(
            perspective=QuestionIntent.TECHNICAL,
            result=simple_result2,
            relevance_score=0.8
        )

        mock_mp_result = MultiPerspectiveQADIResult(
            primary_intent=QuestionIntent.ENVIRONMENTAL,
            intent_confidence=0.9,
            keywords_matched=[],
            perspective_results=[perspective_result1, perspective_result2],
            synthesized_answer="Synthesis",
            synthesized_action_plan=[],
            best_hypothesis=("H1-p1", QuestionIntent.ENVIRONMENTAL),
            total_llm_cost=0.02,
            perspectives_used=[QuestionIntent.ENVIRONMENTAL, QuestionIntent.TECHNICAL],
            synthesized_ideas=[]
        )

        mock_instance = AsyncMock()
        mock_instance.run_multi_perspective_analysis.return_value = mock_mp_result
        MockMultiPerspective.return_value = mock_instance

        config = OrchestratorConfig(
            strategy=Strategy.MULTI_PERSPECTIVE,
            auto_detect_perspectives=True,
            num_hypotheses=3  # Top 3
        )
        orchestrator = UnifiedQADIOrchestrator(config=config)

        result = await orchestrator.run_qadi_cycle("Question")

        # Should have top 3: H1-p1 (0.9), H1-p2 (0.85), H2-p1 (0.7)
        # H2-p2 (0.6) should be excluded
        assert len(result.hypotheses) == 3
        assert "H1-p1" in result.hypotheses
        assert "H1-p2" in result.hypotheses
        assert "H2-p1" in result.hypotheses

    @pytest.mark.asyncio
    @patch('mad_spark_alt.core.unified_orchestrator.MultiPerspectiveQADIOrchestrator')
    async def test_multi_perspective_cost_aggregation(self, MockMultiPerspective):
        """Test that costs are correctly aggregated."""

        from mad_spark_alt.core.multi_perspective_orchestrator import MultiPerspectiveQADIResult
        from mad_spark_alt.core.intent_detector import QuestionIntent

        mock_mp_result = MultiPerspectiveQADIResult(
            primary_intent=QuestionIntent.TECHNICAL,
            intent_confidence=0.9,
            keywords_matched=[],
            perspective_results=[],
            synthesized_answer="Answer",
            synthesized_action_plan=[],
            best_hypothesis=("H", QuestionIntent.TECHNICAL),
            total_llm_cost=0.037,  # Already aggregated in MP orchestrator
            perspectives_used=[QuestionIntent.TECHNICAL],
            synthesized_ideas=[]
        )

        mock_instance = AsyncMock()
        mock_instance.run_multi_perspective_analysis.return_value = mock_mp_result
        MockMultiPerspective.return_value = mock_instance

        config = OrchestratorConfig(
            strategy=Strategy.MULTI_PERSPECTIVE,
            auto_detect_perspectives=True
        )
        orchestrator = UnifiedQADIOrchestrator(config=config)

        result = await orchestrator.run_qadi_cycle("Question")

        assert result.total_llm_cost == 0.037

    @pytest.mark.asyncio
    @patch('mad_spark_alt.core.unified_orchestrator.MultiPerspectiveQADIOrchestrator')
    async def test_multi_perspective_result_structure(self, MockMultiPerspective):
        """Test that all required UnifiedQADIResult fields are populated."""

        from mad_spark_alt.core.multi_perspective_orchestrator import (
            MultiPerspectiveQADIResult,
            PerspectiveResult
        )
        from mad_spark_alt.core.intent_detector import QuestionIntent
        from mad_spark_alt.core.simple_qadi_orchestrator import SimpleQADIResult, HypothesisScore
        from mad_spark_alt.core.interfaces import GeneratedIdea, ThinkingMethod

        idea = GeneratedIdea(
            content="Test idea",
            thinking_method=ThinkingMethod.QUESTIONING,
            agent_name="test_agent",
            generation_prompt="test prompt",
            confidence_score=0.8,
            metadata={}
        )

        simple_result = SimpleQADIResult(
            core_question="Q",
            hypotheses=["H1", "H2"],
            hypothesis_scores=[
                HypothesisScore(impact=0.8, feasibility=0.7, accessibility=0.9, sustainability=0.75, scalability=0.80, overall=0.79),
                HypothesisScore(impact=0.7, feasibility=0.6, accessibility=0.8, sustainability=0.65, scalability=0.70, overall=0.69)
            ],
            final_answer="Answer",
            action_plan=["Step 1", "Step 2"],
            verification_examples=["Ex1"],
            verification_conclusion="Verified",
            total_llm_cost=0.01,
            phase_results={},
            synthesized_ideas=[idea]
        )

        perspective_result = PerspectiveResult(
            perspective=QuestionIntent.ENVIRONMENTAL,
            result=simple_result,
            relevance_score=1.0
        )

        mock_mp_result = MultiPerspectiveQADIResult(
            primary_intent=QuestionIntent.ENVIRONMENTAL,
            intent_confidence=0.95,
            keywords_matched=["climate"],
            perspective_results=[perspective_result],
            synthesized_answer="Synthesized",
            synthesized_action_plan=["Action 1"],
            best_hypothesis=("H1", QuestionIntent.ENVIRONMENTAL),
            total_llm_cost=0.015,
            perspectives_used=[QuestionIntent.ENVIRONMENTAL],
            synthesized_ideas=[idea]
        )

        mock_instance = AsyncMock()
        mock_instance.run_multi_perspective_analysis.return_value = mock_mp_result
        MockMultiPerspective.return_value = mock_instance

        config = OrchestratorConfig(
            strategy=Strategy.MULTI_PERSPECTIVE,
            auto_detect_perspectives=True
        )
        orchestrator = UnifiedQADIOrchestrator(config=config)

        result = await orchestrator.run_qadi_cycle("Test question")

        # Verify all common fields
        assert result.strategy_used == Strategy.MULTI_PERSPECTIVE
        assert result.execution_mode == ExecutionMode.SEQUENTIAL
        assert result.core_question == "Test question"
        assert len(result.hypotheses) > 0
        assert result.final_answer == "Synthesized"
        assert len(result.action_plan) > 0
        assert result.total_llm_cost == 0.015
        assert len(result.synthesized_ideas) == 1

        # Verify optional MultiPerspective-specific fields
        assert result.hypothesis_scores is not None
        assert result.perspectives_used == ["environmental"]
        assert result.synthesized_answer == "Synthesized"
        assert "perspective_count" in result.phase_results
        assert "primary_intent" in result.phase_results
        assert "intent_confidence" in result.phase_results

    @pytest.mark.asyncio
    @patch('mad_spark_alt.core.unified_orchestrator.MultiPerspectiveQADIOrchestrator')
    async def test_multi_perspective_context_passing(self, MockMultiPerspective):
        """Test that context is properly passed (though MP doesn't use it)."""

        from mad_spark_alt.core.multi_perspective_orchestrator import MultiPerspectiveQADIResult
        from mad_spark_alt.core.intent_detector import QuestionIntent

        mock_mp_result = MultiPerspectiveQADIResult(
            primary_intent=QuestionIntent.TECHNICAL,
            intent_confidence=0.9,
            keywords_matched=[],
            perspective_results=[],
            synthesized_answer="Answer",
            synthesized_action_plan=[],
            best_hypothesis=("H", QuestionIntent.TECHNICAL),
            total_llm_cost=0.01,
            perspectives_used=[QuestionIntent.TECHNICAL],
            synthesized_ideas=[]
        )

        mock_instance = AsyncMock()
        mock_instance.run_multi_perspective_analysis.return_value = mock_mp_result
        MockMultiPerspective.return_value = mock_instance

        config = OrchestratorConfig(
            strategy=Strategy.MULTI_PERSPECTIVE,
            auto_detect_perspectives=True
        )
        orchestrator = UnifiedQADIOrchestrator(config=config)

        # Context parameter exists but MP orchestrator doesn't use it
        result = await orchestrator.run_qadi_cycle(
            "Question",
            context="Some previous context"
        )

        assert result is not None

    @pytest.mark.asyncio
    @patch('mad_spark_alt.core.unified_orchestrator.MultiPerspectiveQADIOrchestrator')
    async def test_multi_perspective_perspectives_used_field(self, MockMultiPerspective):
        """Test that perspectives_used is correctly converted to strings."""

        from mad_spark_alt.core.multi_perspective_orchestrator import MultiPerspectiveQADIResult
        from mad_spark_alt.core.intent_detector import QuestionIntent

        mock_mp_result = MultiPerspectiveQADIResult(
            primary_intent=QuestionIntent.ENVIRONMENTAL,
            intent_confidence=0.9,
            keywords_matched=[],
            perspective_results=[],
            synthesized_answer="Answer",
            synthesized_action_plan=[],
            best_hypothesis=("H", QuestionIntent.ENVIRONMENTAL),
            total_llm_cost=0.01,
            perspectives_used=[
                QuestionIntent.ENVIRONMENTAL,
                QuestionIntent.TECHNICAL,
                QuestionIntent.BUSINESS
            ],
            synthesized_ideas=[]
        )

        mock_instance = AsyncMock()
        mock_instance.run_multi_perspective_analysis.return_value = mock_mp_result
        MockMultiPerspective.return_value = mock_instance

        config = OrchestratorConfig(
            strategy=Strategy.MULTI_PERSPECTIVE,
            auto_detect_perspectives=True
        )
        orchestrator = UnifiedQADIOrchestrator(config=config)

        result = await orchestrator.run_qadi_cycle("Question")

        # Should convert QuestionIntent enums to strings
        assert result.perspectives_used == ["environmental", "technical", "business"]

    @pytest.mark.asyncio
    @patch('mad_spark_alt.core.unified_orchestrator.MultiPerspectiveQADIOrchestrator')
    async def test_multi_perspective_synthesized_answer_field(self, MockMultiPerspective):
        """Test that synthesized_answer is properly mapped."""

        from mad_spark_alt.core.multi_perspective_orchestrator import MultiPerspectiveQADIResult
        from mad_spark_alt.core.intent_detector import QuestionIntent

        synthesized_text = "This is a comprehensive answer synthesized from multiple perspectives."

        mock_mp_result = MultiPerspectiveQADIResult(
            primary_intent=QuestionIntent.ENVIRONMENTAL,
            intent_confidence=0.9,
            keywords_matched=[],
            perspective_results=[],
            synthesized_answer=synthesized_text,
            synthesized_action_plan=[],
            best_hypothesis=("H", QuestionIntent.ENVIRONMENTAL),
            total_llm_cost=0.01,
            perspectives_used=[QuestionIntent.ENVIRONMENTAL],
            synthesized_ideas=[]
        )

        mock_instance = AsyncMock()
        mock_instance.run_multi_perspective_analysis.return_value = mock_mp_result
        MockMultiPerspective.return_value = mock_instance

        config = OrchestratorConfig(
            strategy=Strategy.MULTI_PERSPECTIVE,
            auto_detect_perspectives=True
        )
        orchestrator = UnifiedQADIOrchestrator(config=config)

        result = await orchestrator.run_qadi_cycle("Question")

        # Both final_answer and synthesized_answer should have same value
        assert result.final_answer == synthesized_text
        assert result.synthesized_answer == synthesized_text

    @pytest.mark.asyncio
    @patch('mad_spark_alt.core.unified_orchestrator.MultiPerspectiveQADIOrchestrator')
    async def test_multi_perspective_phase_results_metadata(self, MockMultiPerspective):
        """Test that phase_results contains MP-specific metadata."""

        from mad_spark_alt.core.multi_perspective_orchestrator import (
            MultiPerspectiveQADIResult,
            PerspectiveResult
        )
        from mad_spark_alt.core.intent_detector import QuestionIntent
        from mad_spark_alt.core.simple_qadi_orchestrator import SimpleQADIResult, HypothesisScore

        simple_result = SimpleQADIResult(
            core_question="Q",
            hypotheses=["H"],
            hypothesis_scores=[HypothesisScore(impact=0.8, feasibility=0.7, accessibility=0.8, sustainability=0.75, scalability=0.75, overall=0.76)],
            final_answer="A",
            action_plan=[],
            verification_examples=[],
            verification_conclusion="OK",
            total_llm_cost=0.01,
            phase_results={},
            synthesized_ideas=[]
        )

        perspective_results = [
            PerspectiveResult(QuestionIntent.ENVIRONMENTAL, simple_result, 1.0),
            PerspectiveResult(QuestionIntent.TECHNICAL, simple_result, 0.8),
            PerspectiveResult(QuestionIntent.BUSINESS, simple_result, 0.7)
        ]

        mock_mp_result = MultiPerspectiveQADIResult(
            primary_intent=QuestionIntent.ENVIRONMENTAL,
            intent_confidence=0.95,
            keywords_matched=["climate", "sustainable"],
            perspective_results=perspective_results,
            synthesized_answer="Synthesis",
            synthesized_action_plan=[],
            best_hypothesis=("H", QuestionIntent.ENVIRONMENTAL),
            total_llm_cost=0.03,
            perspectives_used=[QuestionIntent.ENVIRONMENTAL, QuestionIntent.TECHNICAL, QuestionIntent.BUSINESS],
            synthesized_ideas=[]
        )

        mock_instance = AsyncMock()
        mock_instance.run_multi_perspective_analysis.return_value = mock_mp_result
        MockMultiPerspective.return_value = mock_instance

        config = OrchestratorConfig(
            strategy=Strategy.MULTI_PERSPECTIVE,
            auto_detect_perspectives=True
        )
        orchestrator = UnifiedQADIOrchestrator(config=config)

        result = await orchestrator.run_qadi_cycle("Question")

        # Verify phase_results metadata
        assert result.phase_results["perspective_count"] == 3
        assert result.phase_results["primary_intent"] == "environmental"
        assert result.phase_results["intent_confidence"] == 0.95

    @pytest.mark.asyncio
    @patch('mad_spark_alt.core.unified_orchestrator.MultiPerspectiveQADIOrchestrator')
    async def test_multi_perspective_backward_compatible(self, MockMultiPerspective):
        """Test that result works with existing code expecting SimpleQADIResult fields."""
        from mad_spark_alt.core.multi_perspective_orchestrator import MultiPerspectiveQADIResult
        from mad_spark_alt.core.intent_detector import QuestionIntent
        from mad_spark_alt.core.interfaces import GeneratedIdea, ThinkingMethod

        idea = GeneratedIdea(
            content="Test",
            thinking_method=ThinkingMethod.QUESTIONING,
            agent_name="test_agent",
            generation_prompt="test prompt",
            confidence_score=0.8,
            metadata={}
        )

        mock_mp_result = MultiPerspectiveQADIResult(
            primary_intent=QuestionIntent.TECHNICAL,
            intent_confidence=0.9,
            keywords_matched=[],
            perspective_results=[],
            synthesized_answer="Answer",
            synthesized_action_plan=["Step 1", "Step 2"],
            best_hypothesis=("H", QuestionIntent.TECHNICAL),
            total_llm_cost=0.015,
            perspectives_used=[QuestionIntent.TECHNICAL],
            synthesized_ideas=[idea]
        )

        mock_instance = AsyncMock()
        mock_instance.run_multi_perspective_analysis.return_value = mock_mp_result
        MockMultiPerspective.return_value = mock_instance

        config = OrchestratorConfig(
            strategy=Strategy.MULTI_PERSPECTIVE,
            auto_detect_perspectives=True
        )
        orchestrator = UnifiedQADIOrchestrator(config=config)

        result = await orchestrator.run_qadi_cycle("Question")

        # These fields must exist for backward compatibility
        assert hasattr(result, 'core_question')
        assert hasattr(result, 'hypotheses')
        assert hasattr(result, 'final_answer')
        assert hasattr(result, 'action_plan')
        assert hasattr(result, 'total_llm_cost')
        assert hasattr(result, 'synthesized_ideas')

        # Verify they have correct values
        assert isinstance(result.hypotheses, list)
        assert isinstance(result.action_plan, list)
        assert len(result.action_plan) == 2
        assert isinstance(result.synthesized_ideas, list)
        assert len(result.synthesized_ideas) == 1
