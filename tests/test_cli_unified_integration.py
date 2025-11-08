"""
CLI Integration Tests for UnifiedQADIOrchestrator.

Verifies that CLI correctly uses UnifiedQADIOrchestrator and maintains backward compatibility.
"""

import pytest
from unittest.mock import AsyncMock, patch
from mad_spark_alt.core.unified_orchestrator import UnifiedQADIOrchestrator, UnifiedQADIResult
from mad_spark_alt.core.orchestrator_config import OrchestratorConfig, Strategy, ExecutionMode
from mad_spark_alt.core.simple_qadi_orchestrator import SimpleQADIResult
from mad_spark_alt.core.interfaces import GeneratedIdea, ThinkingMethod
from mad_spark_alt.core.phase_logic import HypothesisScore


class TestCLIUnifiedIntegration:
    """Test CLI integration with UnifiedQADIOrchestrator."""

    @pytest.fixture
    def mock_simple_result(self):
        """Create mock SimpleQADIResult for testing."""
        return SimpleQADIResult(
            core_question="How can we reduce plastic waste?",
            hypotheses=["H1: Develop biodegradable alternatives", "H2: Improve recycling", "H3: Reduce consumption"],
            hypothesis_scores=[
                HypothesisScore(impact=0.9, feasibility=0.7, accessibility=0.6, sustainability=0.8, scalability=0.7, overall=0.74),
                HypothesisScore(impact=0.8, feasibility=0.8, accessibility=0.7, sustainability=0.7, scalability=0.6, overall=0.72),
                HypothesisScore(impact=0.7, feasibility=0.6, accessibility=0.8, sustainability=0.9, scalability=0.5, overall=0.70),
            ],
            final_answer="Combine biodegradable alternatives with improved recycling systems.",
            action_plan=["Invest in biodegradable material research", "Upgrade recycling infrastructure"],
            verification_examples=["Example 1", "Example 2"],
            verification_conclusion="Approach is viable",
            total_llm_cost=0.0123,
            phase_results={"phase1": "success", "phase2": "success"},
            synthesized_ideas=[
                GeneratedIdea(
                    content="H1: Develop biodegradable alternatives",
                    thinking_method=ThinkingMethod.ABDUCTION,
                    confidence_score=0.74,
                    agent_name="TestAgent",
                    generation_prompt="Test prompt"
                )
            ]
        )

    def test_config_creation_with_defaults(self):
        """Test OrchestratorConfig creation with default values."""
        config = OrchestratorConfig.simple_config()

        assert config.strategy == Strategy.SIMPLE
        assert config.execution_mode == ExecutionMode.SEQUENTIAL
        assert config.num_hypotheses == 3
        assert config.temperature_override is None

    def test_config_temperature_override(self):
        """Test temperature override propagates correctly."""
        config = OrchestratorConfig.simple_config()
        config.temperature_override = 1.5

        assert config.temperature_override == 1.5

    def test_config_hypothesis_count_override(self):
        """Test hypothesis count override for evolution."""
        config = OrchestratorConfig.simple_config()
        config.num_hypotheses = 10

        assert config.num_hypotheses == 10

    @pytest.mark.asyncio
    @patch('mad_spark_alt.core.unified_orchestrator.SimpleQADIOrchestrator')
    async def test_unified_orchestrator_creation(self, MockSimple, mock_simple_result):
        """Test UnifiedQADIOrchestrator creates SimpleQADI with correct config."""
        # Setup mock
        mock_instance = AsyncMock()
        mock_instance.run_qadi_cycle.return_value = mock_simple_result
        MockSimple.return_value = mock_instance

        # Create config and orchestrator
        config = OrchestratorConfig.simple_config()
        config.temperature_override = 1.2
        config.num_hypotheses = 5

        orchestrator = UnifiedQADIOrchestrator(config=config)

        # Run cycle
        await orchestrator.run_qadi_cycle("Test question")

        # Verify SimpleQADI was created with correct parameters
        MockSimple.assert_called_once_with(
            temperature_override=1.2,
            num_hypotheses=5
        )

    @pytest.mark.asyncio
    @patch('mad_spark_alt.core.unified_orchestrator.SimpleQADIOrchestrator')
    async def test_result_structure_matches_expected(self, MockSimple, mock_simple_result):
        """Test result structure matches expected format for CLI."""
        # Setup mock
        mock_instance = AsyncMock()
        mock_instance.run_qadi_cycle.return_value = mock_simple_result
        MockSimple.return_value = mock_instance

        # Create orchestrator
        config = OrchestratorConfig.simple_config()
        orchestrator = UnifiedQADIOrchestrator(config=config)

        # Run cycle
        result = await orchestrator.run_qadi_cycle("Test question")

        # Verify result structure
        assert isinstance(result, UnifiedQADIResult)
        assert result.core_question == "How can we reduce plastic waste?"
        assert len(result.hypotheses) == 3
        assert len(result.action_plan) == 2
        assert result.total_llm_cost == 0.0123

    @pytest.mark.asyncio
    @patch('mad_spark_alt.core.unified_orchestrator.SimpleQADIOrchestrator')
    async def test_evolution_integration_synthesized_ideas(self, MockSimple, mock_simple_result):
        """Test synthesized_ideas field preserved for evolution integration."""
        # Setup mock
        mock_instance = AsyncMock()
        mock_instance.run_qadi_cycle.return_value = mock_simple_result
        MockSimple.return_value = mock_instance

        # Create orchestrator
        config = OrchestratorConfig.simple_config()
        orchestrator = UnifiedQADIOrchestrator(config=config)

        # Run cycle
        result = await orchestrator.run_qadi_cycle("Test question")

        # Verify synthesized_ideas field exists and has correct structure
        assert hasattr(result, 'synthesized_ideas')
        assert isinstance(result.synthesized_ideas, list)
        assert len(result.synthesized_ideas) == 1
        assert result.synthesized_ideas[0].content == "H1: Develop biodegradable alternatives"
        assert result.synthesized_ideas[0].thinking_method == ThinkingMethod.ABDUCTION
        assert result.synthesized_ideas[0].confidence_score == 0.74

    @pytest.mark.asyncio
    @patch('mad_spark_alt.core.unified_orchestrator.SimpleQADIOrchestrator')
    async def test_cost_tracking_accurate(self, MockSimple, mock_simple_result):
        """Test cost tracking is accurate."""
        # Setup mock
        mock_instance = AsyncMock()
        mock_instance.run_qadi_cycle.return_value = mock_simple_result
        MockSimple.return_value = mock_instance

        # Create orchestrator
        config = OrchestratorConfig.simple_config()
        orchestrator = UnifiedQADIOrchestrator(config=config)

        # Run cycle
        result = await orchestrator.run_qadi_cycle("Test question")

        # Verify cost
        assert result.total_llm_cost == 0.0123

    @pytest.mark.asyncio
    @patch('mad_spark_alt.core.unified_orchestrator.SimpleQADIOrchestrator')
    async def test_backward_compatibility_fields(self, MockSimple, mock_simple_result):
        """Test all backward compatibility fields are present."""
        # Setup mock
        mock_instance = AsyncMock()
        mock_instance.run_qadi_cycle.return_value = mock_simple_result
        MockSimple.return_value = mock_instance

        # Create orchestrator
        config = OrchestratorConfig.simple_config()
        orchestrator = UnifiedQADIOrchestrator(config=config)

        # Run cycle
        result = await orchestrator.run_qadi_cycle("Test question")

        # Verify all expected fields exist
        assert hasattr(result, 'core_question')
        assert hasattr(result, 'hypotheses')
        assert hasattr(result, 'hypothesis_scores')
        assert hasattr(result, 'final_answer')
        assert hasattr(result, 'action_plan')
        assert hasattr(result, 'verification_examples')
        assert hasattr(result, 'verification_conclusion')
        assert hasattr(result, 'total_llm_cost')
        assert hasattr(result, 'phase_results')
        assert hasattr(result, 'synthesized_ideas')

    def test_config_validation_temperature_range(self):
        """Test config validates temperature range."""
        config = OrchestratorConfig.simple_config()
        config.temperature_override = 2.5  # Invalid - outside range

        with pytest.raises(ValueError, match=r"temperature_override must be between 0\.0 and 2\.0"):
            config.validate()

    def test_config_validation_num_hypotheses(self):
        """Test config validates num_hypotheses is positive."""
        config = OrchestratorConfig.simple_config()
        config.num_hypotheses = 0  # Invalid

        with pytest.raises(ValueError, match="num_hypotheses must be >= 1"):
            config.validate()
