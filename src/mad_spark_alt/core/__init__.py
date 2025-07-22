"""Core evaluation system components and idea generation framework."""

from .evaluator import CreativityEvaluator, EvaluationSummary
from .fast_orchestrator import FastQADIOrchestrator
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

# Import robust orchestrator if available
try:
    from .robust_orchestrator import RobustQADIOrchestrator
    from .robust_orchestrator import SmartQADICycleResult as RobustQADICycleResult
except ImportError as e:
    # Fallback to smart orchestrator if robust version not available
    import logging

    logging.debug(f"Failed to import RobustQADIOrchestrator: {e}")
    RobustQADIOrchestrator = SmartQADIOrchestrator  # type: ignore[misc,assignment]
    RobustQADICycleResult = SmartQADICycleResult  # type: ignore[misc]

__all__ = [
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
    "RobustQADIOrchestrator",
    "RobustQADICycleResult",
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
    # Fast orchestrator
    "FastQADIOrchestrator",
    # Simple QADI components
    "SimpleQADIOrchestrator",
    "SimpleQADIResult",
    "UnifiedEvaluator",
    "HypothesisEvaluation",
]
