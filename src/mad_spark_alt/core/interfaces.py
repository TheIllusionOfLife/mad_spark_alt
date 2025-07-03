"""
Core interfaces and abstract base classes for the creativity evaluation system.
"""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union


class EvaluationLayer(Enum):
    """The three layers of evaluation in the framework."""

    QUANTITATIVE = "quantitative"
    LLM_JUDGE = "llm_judge"
    HUMAN = "human"


class OutputType(Enum):
    """Supported types of AI model output."""

    TEXT = "text"
    CODE = "code"
    IMAGE = "image"
    STRUCTURED = "structured"


@dataclass
class ModelOutput:
    """Represents output from an AI model to be evaluated."""

    content: Union[str, bytes, Dict[str, Any]]
    output_type: OutputType
    model_name: str
    prompt: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[str] = None


@dataclass
class EvaluationResult:
    """Results from a single evaluation method."""

    evaluator_name: str
    layer: EvaluationLayer
    scores: Dict[str, float]
    explanations: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    execution_time: Optional[float] = None

    @property
    def overall_score(self) -> Optional[float]:
        """Calculate overall score if multiple scores exist."""
        if not self.scores:
            return None
        return sum(self.scores.values()) / len(self.scores)


@dataclass
class EvaluationRequest:
    """Request for evaluating model outputs."""

    outputs: List[ModelOutput]
    evaluation_config: Dict[str, Any] = field(default_factory=dict)
    target_layers: List[EvaluationLayer] = field(default_factory=list)
    task_context: Optional[str] = None


class EvaluatorInterface(ABC):
    """Abstract base class for all evaluators."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name for this evaluator."""
        pass

    @property
    @abstractmethod
    def layer(self) -> EvaluationLayer:
        """Which evaluation layer this evaluator belongs to."""
        pass

    @property
    @abstractmethod
    def supported_output_types(self) -> List[OutputType]:
        """Output types this evaluator can handle."""
        pass

    @abstractmethod
    async def evaluate(self, request: EvaluationRequest) -> List[EvaluationResult]:
        """
        Evaluate the given outputs.

        Args:
            request: The evaluation request containing outputs and config

        Returns:
            List of evaluation results, one per output
        """
        pass

    @abstractmethod
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate that the configuration is valid for this evaluator."""
        pass


class AsyncEvaluatorMixin:
    """Mixin for evaluators that support async batch processing."""

    @abstractmethod
    async def evaluate_batch(
        self, requests: List[EvaluationRequest]
    ) -> List[List[EvaluationResult]]:
        """
        Evaluate multiple requests in batch for efficiency.

        Args:
            requests: List of evaluation requests

        Returns:
            List of result lists, one per request
        """
        pass


class CacheableEvaluatorMixin:
    """Mixin for evaluators that support result caching."""

    @abstractmethod
    def get_cache_key(self, request: EvaluationRequest) -> str:
        """Generate a cache key for the given request."""
        pass

    @abstractmethod
    def is_cacheable(self, request: EvaluationRequest) -> bool:
        """Determine if this request's results should be cached."""
        pass


class ConfigurableEvaluatorMixin:
    """Mixin for evaluators with dynamic configuration."""

    @abstractmethod
    def get_default_config(self) -> Dict[str, Any]:
        """Get the default configuration for this evaluator."""
        pass

    @abstractmethod
    def get_config_schema(self) -> Dict[str, Any]:
        """Get JSON schema for configuration validation."""
        pass


# Keep old names for backward compatibility
AsyncEvaluatorInterface = AsyncEvaluatorMixin
CacheableEvaluatorInterface = CacheableEvaluatorMixin
ConfigurableEvaluatorInterface = ConfigurableEvaluatorMixin
