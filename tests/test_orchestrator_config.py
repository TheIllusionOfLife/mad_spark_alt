"""
Comprehensive tests for OrchestratorConfig module.

Following TDD methodology - tests written BEFORE implementation.
"""

import pytest
from dataclasses import FrozenInstanceError


def test_execution_mode_enum_values():
    """Test ExecutionMode enum has correct values."""
    from mad_spark_alt.core.orchestrator_config import ExecutionMode

    assert ExecutionMode.SEQUENTIAL.value == "sequential"
    assert ExecutionMode.PARALLEL.value == "parallel"
    assert len(ExecutionMode) == 2


def test_strategy_enum_values():
    """Test Strategy enum has correct values."""
    from mad_spark_alt.core.orchestrator_config import Strategy

    assert Strategy.SIMPLE.value == "simple"
    assert Strategy.SMART.value == "smart"
    assert Strategy.MULTI_PERSPECTIVE.value == "multi_perspective"
    assert len(Strategy) == 3


def test_timeout_config_defaults():
    """Test TimeoutConfig has sensible defaults."""
    from mad_spark_alt.core.orchestrator_config import TimeoutConfig

    config = TimeoutConfig()
    assert config.phase_timeout == 90.0
    assert config.total_timeout == 900.0
    assert config.enable_retry is True
    assert config.max_retries == 3


def test_timeout_config_custom_values():
    """Test TimeoutConfig accepts custom values."""
    from mad_spark_alt.core.orchestrator_config import TimeoutConfig

    config = TimeoutConfig(
        phase_timeout=120.0,
        total_timeout=600.0,
        enable_retry=False,
        max_retries=5
    )
    assert config.phase_timeout == 120.0
    assert config.total_timeout == 600.0
    assert config.enable_retry is False
    assert config.max_retries == 5


def test_orchestrator_config_defaults():
    """Test OrchestratorConfig has sensible defaults."""
    from mad_spark_alt.core.orchestrator_config import (
        OrchestratorConfig,
        ExecutionMode,
        Strategy
    )

    config = OrchestratorConfig()
    assert config.execution_mode == ExecutionMode.SEQUENTIAL
    assert config.strategy == Strategy.SIMPLE
    assert config.num_hypotheses == 3
    assert config.enable_scoring is True
    assert config.perspectives is None
    assert config.auto_detect_perspectives is False
    assert config.enable_answer_extraction is False
    assert config.enable_robust_timeout is True
    assert config.timeout_config is not None
    assert config.model_config is None
    assert config.temperature_override is None


def test_orchestrator_config_custom_values():
    """Test OrchestratorConfig accepts all custom parameters."""
    from mad_spark_alt.core.orchestrator_config import (
        OrchestratorConfig,
        ExecutionMode,
        Strategy,
        TimeoutConfig
    )

    timeout_config = TimeoutConfig(phase_timeout=120.0)
    config = OrchestratorConfig(
        execution_mode=ExecutionMode.PARALLEL,
        strategy=Strategy.MULTI_PERSPECTIVE,
        num_hypotheses=5,
        enable_scoring=False,
        perspectives=["technical", "business"],
        auto_detect_perspectives=True,
        enable_answer_extraction=True,
        enable_robust_timeout=False,
        timeout_config=timeout_config,
        temperature_override=1.2
    )

    assert config.execution_mode == ExecutionMode.PARALLEL
    assert config.strategy == Strategy.MULTI_PERSPECTIVE
    assert config.num_hypotheses == 5
    assert config.enable_scoring is False
    assert config.perspectives == ["technical", "business"]
    assert config.auto_detect_perspectives is True
    assert config.enable_answer_extraction is True
    assert config.enable_robust_timeout is False
    assert config.timeout_config == timeout_config
    assert config.temperature_override == 1.2


def test_orchestrator_config_validation_multi_perspective_requires_perspectives():
    """Test validation fails if multi-perspective without perspectives or auto-detect."""
    from mad_spark_alt.core.orchestrator_config import (
        OrchestratorConfig,
        Strategy
    )

    config = OrchestratorConfig(
        strategy=Strategy.MULTI_PERSPECTIVE,
        perspectives=None,
        auto_detect_perspectives=False
    )

    with pytest.raises(ValueError, match="Multi-perspective requires perspectives or auto-detect"):
        config.validate()


def test_orchestrator_config_validation_multi_perspective_with_explicit_perspectives():
    """Test validation passes with explicit perspectives."""
    from mad_spark_alt.core.orchestrator_config import (
        OrchestratorConfig,
        Strategy
    )

    config = OrchestratorConfig(
        strategy=Strategy.MULTI_PERSPECTIVE,
        perspectives=["technical", "business"],
        auto_detect_perspectives=False
    )

    # Should not raise
    config.validate()


def test_orchestrator_config_validation_multi_perspective_with_auto_detect():
    """Test validation passes with auto-detect enabled."""
    from mad_spark_alt.core.orchestrator_config import (
        OrchestratorConfig,
        Strategy
    )

    config = OrchestratorConfig(
        strategy=Strategy.MULTI_PERSPECTIVE,
        perspectives=None,
        auto_detect_perspectives=True
    )

    # Should not raise
    config.validate()


def test_orchestrator_config_validation_num_hypotheses_positive():
    """Test validation fails if num_hypotheses < 1."""
    from mad_spark_alt.core.orchestrator_config import OrchestratorConfig

    config = OrchestratorConfig(num_hypotheses=0)

    with pytest.raises(ValueError, match="num_hypotheses must be >= 1"):
        config.validate()

    config2 = OrchestratorConfig(num_hypotheses=-1)
    with pytest.raises(ValueError, match="num_hypotheses must be >= 1"):
        config2.validate()


def test_orchestrator_config_validation_temperature_range():
    """Test validation checks temperature range."""
    from mad_spark_alt.core.orchestrator_config import OrchestratorConfig

    # Temperature too low
    config = OrchestratorConfig(temperature_override=-0.1)
    with pytest.raises(ValueError, match="temperature_override must be between 0.0 and 2.0"):
        config.validate()

    # Temperature too high
    config2 = OrchestratorConfig(temperature_override=2.1)
    with pytest.raises(ValueError, match="temperature_override must be between 0.0 and 2.0"):
        config2.validate()

    # Valid temperatures
    config3 = OrchestratorConfig(temperature_override=0.0)
    config3.validate()  # Should not raise

    config4 = OrchestratorConfig(temperature_override=2.0)
    config4.validate()  # Should not raise


def test_orchestrator_config_validation_timeout_positive():
    """Test validation checks timeout values are positive."""
    from mad_spark_alt.core.orchestrator_config import (
        OrchestratorConfig,
        TimeoutConfig
    )

    # Negative phase timeout
    timeout = TimeoutConfig(phase_timeout=-1.0)
    config = OrchestratorConfig(timeout_config=timeout)
    with pytest.raises(ValueError, match="phase_timeout must be positive"):
        config.validate()

    # Negative total timeout
    timeout2 = TimeoutConfig(total_timeout=-1.0)
    config2 = OrchestratorConfig(timeout_config=timeout2)
    with pytest.raises(ValueError, match="total_timeout must be positive"):
        config2.validate()


def test_simple_config_factory():
    """Test simple_config factory creates correct configuration."""
    from mad_spark_alt.core.orchestrator_config import (
        OrchestratorConfig,
        ExecutionMode,
        Strategy
    )

    config = OrchestratorConfig.simple_config()

    assert config.execution_mode == ExecutionMode.SEQUENTIAL
    assert config.strategy == Strategy.SIMPLE
    assert config.num_hypotheses == 3
    assert config.enable_scoring is True

    # Should be valid
    config.validate()


def test_fast_config_factory():
    """Test fast_config factory enables parallel execution."""
    from mad_spark_alt.core.orchestrator_config import (
        OrchestratorConfig,
        ExecutionMode,
        Strategy
    )

    config = OrchestratorConfig.fast_config()

    assert config.execution_mode == ExecutionMode.PARALLEL
    assert config.strategy == Strategy.SIMPLE
    assert config.num_hypotheses == 3

    # Should be valid
    config.validate()


def test_multi_perspective_config_factory():
    """Test multi_perspective_config factory creates correct configuration."""
    from mad_spark_alt.core.orchestrator_config import (
        OrchestratorConfig,
        ExecutionMode,
        Strategy
    )

    perspectives = ["technical", "business", "environmental"]
    config = OrchestratorConfig.multi_perspective_config(perspectives)

    assert config.execution_mode == ExecutionMode.SEQUENTIAL
    assert config.strategy == Strategy.MULTI_PERSPECTIVE
    assert config.perspectives == perspectives
    assert config.num_hypotheses == 3

    # Should be valid
    config.validate()


def test_smart_config_factory():
    """Test smart_config factory creates correct configuration."""
    from mad_spark_alt.core.orchestrator_config import (
        OrchestratorConfig,
        ExecutionMode,
        Strategy
    )

    config = OrchestratorConfig.smart_config()

    assert config.execution_mode == ExecutionMode.SEQUENTIAL
    assert config.strategy == Strategy.SMART
    assert config.num_hypotheses == 3
    assert config.enable_robust_timeout is True

    # Should be valid
    config.validate()


def test_orchestrator_config_mutability():
    """Test that config is mutable (allows runtime adjustments)."""
    from mad_spark_alt.core.orchestrator_config import OrchestratorConfig

    config = OrchestratorConfig()

    # Dataclass is mutable, which is appropriate for configuration
    # Verify we can change values
    config.num_hypotheses = 5
    assert config.num_hypotheses == 5

    config.temperature_override = 1.5
    assert config.temperature_override == 1.5


def test_orchestrator_config_validation_max_retries_positive():
    """Test validation checks max_retries is positive when retry enabled."""
    from mad_spark_alt.core.orchestrator_config import (
        OrchestratorConfig,
        TimeoutConfig
    )

    timeout = TimeoutConfig(enable_retry=True, max_retries=0)
    config = OrchestratorConfig(timeout_config=timeout)

    with pytest.raises(ValueError, match="max_retries must be positive when retry enabled"):
        config.validate()


def test_orchestrator_config_repr():
    """Test that config has useful string representation."""
    from mad_spark_alt.core.orchestrator_config import OrchestratorConfig

    config = OrchestratorConfig()
    repr_str = repr(config)

    # Should contain key information
    assert "OrchestratorConfig" in repr_str
    assert "strategy" in repr_str or "Strategy" in repr_str


def test_orchestrator_config_equality():
    """Test that two configs with same values are equal."""
    from mad_spark_alt.core.orchestrator_config import (
        OrchestratorConfig,
        Strategy
    )

    config1 = OrchestratorConfig(strategy=Strategy.SIMPLE, num_hypotheses=5)
    config2 = OrchestratorConfig(strategy=Strategy.SIMPLE, num_hypotheses=5)

    assert config1 == config2


def test_timeout_config_validation_in_orchestrator():
    """Test that OrchestratorConfig validates its TimeoutConfig."""
    from mad_spark_alt.core.orchestrator_config import (
        OrchestratorConfig,
        TimeoutConfig
    )

    # Valid timeout config
    config = OrchestratorConfig(
        timeout_config=TimeoutConfig(
            phase_timeout=120.0,
            total_timeout=600.0
        )
    )
    config.validate()  # Should not raise
