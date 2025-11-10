"""
Configuration system for UnifiedQADIOrchestrator.

Provides type-safe, validated configuration with factory methods for common use cases.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, TYPE_CHECKING

from .system_constants import CONSTANTS

if TYPE_CHECKING:
    from .llm_provider import ModelConfig


class ExecutionMode(Enum):
    """Execution mode for QADI orchestration."""

    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"


class Strategy(Enum):
    """Orchestration strategy selection."""

    SIMPLE = "simple"
    MULTI_PERSPECTIVE = "multi_perspective"


@dataclass
class TimeoutConfig:
    """Timeout configuration for orchestrator operations."""

    phase_timeout: float = float(CONSTANTS.TIMEOUTS.PHASE_TIMEOUT_BASE)
    total_timeout: float = float(CONSTANTS.TIMEOUTS.PHASE_TIMEOUT_TOTAL)
    enable_retry: bool = True
    max_retries: int = 3


@dataclass
class OrchestratorConfig:
    """
    Configuration for UnifiedQADIOrchestrator behavior.

    Controls strategy selection, execution mode, timeouts, and feature flags.
    """

    # Execution
    execution_mode: ExecutionMode = ExecutionMode.SEQUENTIAL
    strategy: Strategy = Strategy.SIMPLE

    # QADI Parameters
    num_hypotheses: int = 3
    enable_scoring: bool = True

    # Multi-Perspective
    perspectives: Optional[List[str]] = None
    auto_detect_perspectives: bool = False
    max_perspectives: int = 3

    # Enhancements
    enable_answer_extraction: bool = False
    enable_robust_timeout: bool = True
    timeout_config: TimeoutConfig = field(default_factory=TimeoutConfig)

    # Model Configuration
    model_config: Optional["ModelConfig"] = None
    temperature_override: Optional[float] = None

    def validate(self) -> None:
        """
        Validate configuration consistency.

        Raises:
            ValueError: If configuration is invalid.
        """
        # Multi-perspective requires perspectives or auto-detect
        if self.strategy == Strategy.MULTI_PERSPECTIVE:
            if not self.perspectives and not self.auto_detect_perspectives:
                raise ValueError(
                    "Multi-perspective requires perspectives or auto-detect"
                )

            # Validate perspective names if provided
            if self.perspectives:
                from .intent_detector import QuestionIntent
                valid_perspectives = [intent.value for intent in QuestionIntent]
                for p in self.perspectives:
                    if p.lower() not in valid_perspectives:
                        raise ValueError(
                            f"Invalid perspective: {p}. Valid perspectives: "
                            f"{valid_perspectives}"
                        )

        # Num hypotheses must be positive
        if self.num_hypotheses < 1:
            raise ValueError("num_hypotheses must be >= 1")

        # Max perspectives must be positive
        if self.max_perspectives < 1:
            raise ValueError("max_perspectives must be >= 1")

        # Temperature range validation
        if self.temperature_override is not None:
            if not (0.0 <= self.temperature_override <= 2.0):
                raise ValueError(
                    "temperature_override must be between 0.0 and 2.0"
                )

        # Timeout validation
        if self.timeout_config.phase_timeout <= 0:
            raise ValueError("phase_timeout must be positive")
        if self.timeout_config.total_timeout <= 0:
            raise ValueError("total_timeout must be positive")

        # Retry validation
        if self.timeout_config.enable_retry and self.timeout_config.max_retries <= 0:
            raise ValueError(
                "max_retries must be positive when retry enabled"
            )

    @classmethod
    def simple_config(cls) -> "OrchestratorConfig":
        """
        Factory: Simple sequential QADI.

        Returns:
            OrchestratorConfig: Configuration for simple QADI analysis.
        """
        return cls(
            execution_mode=ExecutionMode.SEQUENTIAL,
            strategy=Strategy.SIMPLE,
            num_hypotheses=3,
            enable_scoring=True
        )

    @classmethod
    def fast_config(cls) -> "OrchestratorConfig":
        """
        Factory: Parallel execution for speed.

        Returns:
            OrchestratorConfig: Configuration for parallel QADI execution.
        """
        return cls(
            execution_mode=ExecutionMode.PARALLEL,
            strategy=Strategy.SIMPLE,
            num_hypotheses=3
        )

    @classmethod
    def multi_perspective_config(
        cls,
        perspectives: List[str]
    ) -> "OrchestratorConfig":
        """
        Factory: Multi-perspective analysis.

        Args:
            perspectives: List of perspective names.

        Returns:
            OrchestratorConfig: Configuration for multi-perspective analysis.
        """
        return cls(
            execution_mode=ExecutionMode.SEQUENTIAL,
            strategy=Strategy.MULTI_PERSPECTIVE,
            perspectives=perspectives,
            num_hypotheses=3
        )

