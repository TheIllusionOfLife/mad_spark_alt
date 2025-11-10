"""Core evaluation system components and idea generation framework."""

from typing import Any, Dict

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
    old_name: str, new_class_name: str, removal_version: str = "v3.0.0"
) -> type:
    """Create a deprecated class shim that warns on instantiation."""
    import importlib

    # Lazy load the target class
    module = importlib.import_module("mad_spark_alt.core.smart_orchestrator")
    new_class = getattr(module, new_class_name)

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
    "FastQADIOrchestrator", "SmartQADIOrchestrator"
)
RobustQADIOrchestrator = _create_deprecation_shim(
    "RobustQADIOrchestrator", "SmartQADIOrchestrator"
)
EnhancedQADIOrchestrator = _create_deprecation_shim(
    "EnhancedQADIOrchestrator", "SmartQADIOrchestrator"
)

# RobustQADICycleResult is just an alias (lazy-loaded via __getattr__)

# Lazy import mechanism for deprecated modules
# This prevents warnings from firing on `import mad_spark_alt.core`
_DEPRECATED_IMPORTS = {
    "SmartQADIOrchestrator": {
        "module": "smart_orchestrator",
        "message": (
            "SmartQADIOrchestrator is deprecated and will be removed in v2.0.0. "
            "Use UnifiedQADIOrchestrator with OrchestratorConfig.simple_config() instead. "
            "The Smart strategy has been removed in favor of the simpler and more reliable Simple strategy."
        ),
    },
    "SmartQADICycleResult": {
        "module": "smart_orchestrator",
        "message": (
            "SmartQADICycleResult is deprecated and will be removed in v2.0.0. "
            "Use UnifiedQADIResult or SimpleQADIResult instead."
        ),
    },
    "answer_extractor": {
        "module": "answer_extractor",
        "message": (
            "answer_extractor module is deprecated and will be removed in v2.0.0. "
            "This module was primarily used by EnhancedQADIOrchestrator (now removed). "
            "If you need answer extraction functionality, you can continue using this "
            "module directly or implement your own extraction logic."
        ),
    },
    "robust_json_handler": {
        "module": "robust_json_handler",
        "message": (
            "robust_json_handler is deprecated and will be removed in v2.0.0. "
            "Use json_utils.extract_and_parse_json() and json_utils.parse_ideas_array() instead."
        ),
    },
    "RobustQADICycleResult": {
        "module": "smart_orchestrator",
        "attr": "SmartQADICycleResult",
        "message": (
            "RobustQADICycleResult is deprecated and will be removed in v2.0.0. "
            "Use UnifiedQADIResult or SimpleQADIResult instead."
        ),
    },
}

# Cache for already-imported deprecated items to prevent duplicate warnings
# NOTE: Despite gemini-code-assist's claim, PEP 562 does NOT automatically cache
# __getattr__ return values in __dict__. Manual caching is necessary to prevent
# __getattr__ from being called repeatedly for the same attribute, which would
# trigger duplicate deprecation warnings. Verified with PEP 562 documentation
# and empirical testing.
_deprecated_cache: Dict[str, Any] = {}


def __getattr__(name: str) -> Any:
    """
    Lazy import handler for deprecated modules and classes.

    This function is called when an attribute is not found in the module's namespace.
    It allows us to defer deprecation warnings until the deprecated item is actually used,
    rather than firing warnings on `import mad_spark_alt.core`.
    """
    if name in _DEPRECATED_IMPORTS:
        # Check cache first to avoid duplicate warnings
        if name in _deprecated_cache:
            return _deprecated_cache[name]

        import importlib

        info = _DEPRECATED_IMPORTS[name]

        # Issue deprecation warning
        warnings.warn(
            info["message"],
            DeprecationWarning,
            stacklevel=2,
        )

        # Import and return the deprecated item
        module_name = f"mad_spark_alt.core.{info['module']}"
        module = importlib.import_module(module_name)

        # Handle cases where we need a specific attribute from the module
        attr_name = info.get("attr", name)
        if attr_name == name:
            # Return the class/object directly, or the module itself if the
            # class doesn't exist. This handles both class imports
            # (SmartQADIOrchestrator) and module imports (answer_extractor).
            result = getattr(module, name) if hasattr(module, name) else module
        else:
            # Return the aliased attribute (e.g., SmartQADICycleResult → RobustQADICycleResult)
            result = getattr(module, attr_name)

        # Cache the result
        _deprecated_cache[name] = result
        return result

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
