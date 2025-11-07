"""Core evaluation system components and idea generation framework."""

from .base_orchestrator import AgentCircuitBreaker, BaseOrchestrator
from .evaluator import CreativityEvaluator, EvaluationSummary
from .interfaces import (  # New idea generation interfaces
    AsyncEvaluatorInterface,
    CacheableEvaluatorInterface,
    ConfigurableEvaluatorInterface,
    EvaluationLayer,
    EvaluationRequest,
    EvaluationResult,
    EvaluatorInterface,
    GeneratedIdea,
    IdeaGenerationRequest,
    IdeaGenerationResult,
    ModelOutput,
    OutputType,
    ThinkingAgentInterface,
    ThinkingMethod,
)
from .llm_provider import (
    LLMManager,
    LLMProvider,
    LLMRequest,
    LLMResponse,
    ModelConfig,
    RateLimitConfig,
    UsageStats,
    llm_manager,
    setup_llm_providers,
)
from .orchestrator import QADICycleResult, QADIOrchestrator
from .registry import (
    EvaluatorRegistry,
    ThinkingAgentRegistry,
    agent_registry,
    register_agent,
    register_evaluator,
    registry,
    unified_registry,
)
from .smart_orchestrator import SmartQADICycleResult, SmartQADIOrchestrator
from .smart_registry import (
    SmartAgentRegistry,
    get_smart_agent,
    setup_smart_agents,
    smart_registry,
)

# Import new simplified QADI components
from .simple_qadi_orchestrator import SimpleQADIOrchestrator, SimpleQADIResult
from .unified_evaluator import UnifiedEvaluator, HypothesisEvaluation

# Import unified orchestrator components
from .orchestrator_config import (
    OrchestratorConfig,
    ExecutionMode,
    Strategy,
    TimeoutConfig,
)
from .unified_orchestrator import UnifiedQADIOrchestrator, UnifiedQADIResult

__all__ = [
    # Base orchestrator
    "BaseOrchestrator",
    "AgentCircuitBreaker",
    # Evaluation system
    "CreativityEvaluator",
    "EvaluationSummary",
    "EvaluationRequest",
    "EvaluationResult",
    "EvaluationLayer",
    "OutputType",
    "ModelOutput",
    "EvaluatorInterface",
    "AsyncEvaluatorInterface",
    "CacheableEvaluatorInterface",
    "ConfigurableEvaluatorInterface",
    "EvaluatorRegistry",
    "registry",
    "register_evaluator",
    # LLM provider system
    "LLMManager",
    "LLMProvider",
    "LLMRequest",
    "LLMResponse",
    "ModelConfig",
    "RateLimitConfig",
    "UsageStats",
    "llm_manager",
    "setup_llm_providers",
    # Idea generation system
    "ThinkingMethod",
    "ThinkingAgentInterface",
    "GeneratedIdea",
    "IdeaGenerationRequest",
    "IdeaGenerationResult",
    "QADIOrchestrator",
    "QADICycleResult",
    "SmartQADIOrchestrator",
    "SmartQADICycleResult",
    # Agent registry system
    "ThinkingAgentRegistry",
    "agent_registry",
    "register_agent",
    "unified_registry",
    # Smart registry system
    "SmartAgentRegistry",
    "smart_registry",
    "setup_smart_agents",
    "get_smart_agent",
    # Simple QADI components
    "SimpleQADIOrchestrator",
    "SimpleQADIResult",
    "UnifiedEvaluator",
    "HypothesisEvaluation",
    # Unified orchestrator components
    "UnifiedQADIOrchestrator",
    "UnifiedQADIResult",
    "OrchestratorConfig",
    "ExecutionMode",
    "Strategy",
    "TimeoutConfig",
    # Deprecated orchestrators (compatibility shims)
    "FastQADIOrchestrator",
    "RobustQADIOrchestrator",
    "RobustQADICycleResult",
    "EnhancedQADIOrchestrator",
]


# Compatibility shims for removed orchestrators (deprecated in v2.0.0, will be removed in v3.0.0)
import warnings


def _create_deprecation_shim(
    old_name: str, new_class: type, removal_version: str = "v3.0.0"
) -> type:
    """Create a deprecated class shim that warns on instantiation."""

    class DeprecatedShim(new_class):  # type: ignore[misc,valid-type]
        def __init__(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
            warnings.warn(
                f"{old_name} has been removed as of v2.0.0 and will be completely unavailable in {removal_version}. "
                f"Use {new_class.__name__} instead. "
                f"See DEPRECATED.md for migration guide.",
                DeprecationWarning,
                stacklevel=2,
            )
            super().__init__(*args, **kwargs)

    DeprecatedShim.__name__ = old_name
    DeprecatedShim.__qualname__ = old_name
    return DeprecatedShim


# Create compatibility shims pointing to SmartQADIOrchestrator
FastQADIOrchestrator = _create_deprecation_shim(
    "FastQADIOrchestrator", SmartQADIOrchestrator
)
RobustQADIOrchestrator = _create_deprecation_shim(
    "RobustQADIOrchestrator", SmartQADIOrchestrator
)
EnhancedQADIOrchestrator = _create_deprecation_shim(
    "EnhancedQADIOrchestrator", SmartQADIOrchestrator
)

# RobustQADICycleResult is just an alias
RobustQADICycleResult = SmartQADICycleResult
