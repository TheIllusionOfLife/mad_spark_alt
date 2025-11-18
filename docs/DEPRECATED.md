# Deprecated Components

This document tracks deprecated and removed components, providing migration paths for users.

## Recently Removed (v2.0.0)

### Prompt Classification & Adaptive Prompts (Removed: 2025-11-06)

**Removed Modules:**
- `mad_spark_alt.core.prompt_classifier` (749 lines)
- `mad_spark_alt.core.adaptive_prompts` (504 lines)

**Removed Script:**
- `qadi_simple_multi.py` (568 lines)

**Reason for Removal:**
These modules were made obsolete by SimpleQADIOrchestrator, which automatically
handles question type detection and prompt adaptation with superior accuracy.

**Migration Path:**

#### Before (Deprecated):
```python
from mad_spark_alt.core import classify_question, get_adaptive_prompt

# Manual classification
result = classify_question("How can we reduce costs?")
prompt = get_adaptive_prompt(result.question_type, "How can we reduce costs?")

# Use in custom orchestration...
```

#### After (Recommended):
```python
import asyncio
import os
from mad_spark_alt.core import SimpleQADIOrchestrator, setup_llm_providers

async def run_qadi():
    # 1. Setup LLM provider (requires API key)
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set")

    await setup_llm_providers(google_api_key=google_api_key)

    # 2. Create orchestrator (uses global llm_manager)
    orchestrator = SimpleQADIOrchestrator(num_hypotheses=3)

    # 3. Run QADI cycle
    result = await orchestrator.run_qadi_cycle("How can we reduce costs?")
    print(result.hypotheses)

# Run the async function
asyncio.run(run_qadi())
```

**Why This Is Better:**
- ✅ No manual question type detection needed
- ✅ Automatically adapts prompts based on context
- ✅ Handles scoring and synthesis in one call
- ✅ Production-tested and actively maintained
- ✅ Better accuracy than manual classification

**CLI Alternative:**
```bash
# Instead of: python qadi_simple_multi.py "question" --type=business
# Use: uv run mad-spark qadi "question"

uv run mad-spark qadi "How can we reduce costs?"
```

---

## Files Retained for Backward Compatibility

### prompt_classifier.py & adaptive_prompts.py
- **Status**: ~~Deprecated as of PR #40~~ **REMOVED** as of 2025-11-06
- **Reason**: Replaced by the universal QADI prompts in `SimpleQADIOrchestrator`
- **Migration Path**: See "Recently Removed" section above
- **Future**: ~~Will be removed in v2.0.0~~ **COMPLETED**

### Old Orchestrators
The following orchestrators are superseded by `SimpleQADIOrchestrator`:
- `QADIOrchestrator` - Original implementation with prompt classification
- `SmartQADIOrchestrator` - Enhanced version with adaptive prompts (still available, but `SimpleQADIOrchestrator` preferred)

**Migration Path**: Replace with `SimpleQADIOrchestrator` which provides:
- Cleaner implementation following true QADI methodology
- Universal prompts without classification overhead
- Temperature control for hypothesis generation
- Unified evaluation with the 5-criteria system

### Legacy Orchestrators (Recently Removed)
The following orchestrators have been removed as of v2.0.0 (2025-11-07):
- ~~`RobustQADIOrchestrator`~~ - Error-recovery focused version (use `timeout_wrapper.py` directly)
- ~~`FastQADIOrchestrator`~~ - Performance-optimized version (parallel features now in `BaseOrchestrator`)
- ~~`EnhancedQADIOrchestrator`~~ - Answer extraction wrapper (use `answer_extractor.py` directly, now deprecated)

**Migration Path**:
- For timeout handling: Use `timeout_wrapper.py` with `SmartQADIOrchestrator` or `SimpleQADIOrchestrator`
- For parallel execution: Use `SmartQADIOrchestrator` (circuit breaker in `BaseOrchestrator`)
- For answer extraction: Use `answer_extractor.py` directly (deprecated, will be removed in v3.0.0)

**Files Removed**: 738 lines across 3 orchestrator files
**Date**: 2025-11-07
**Reason**: Consolidation to reduce duplication and maintenance burden

---

## SmartQADIOrchestrator (Deprecated: 2025-11-08)

### Status
- **Deprecated as of**: 2025-11-08
- **Removal planned for**: v2.0.0
- **Replacement**: `UnifiedQADIOrchestrator` with `OrchestratorConfig.simple_config()`

### Reason for Deprecation
The Smart strategy has been removed from UnifiedQADIOrchestrator in favor of the simpler and more reliable Simple strategy. The Smart agent selection added complexity without providing clear benefits over the Simple approach.

### Migration Path

#### Before (Deprecated):
```python
from mad_spark_alt.core import SmartQADIOrchestrator, smart_registry

orchestrator = SmartQADIOrchestrator(
    registry=smart_registry,
    auto_setup=True
)

result = await orchestrator.run_qadi_cycle("How can we improve efficiency?")
```

#### After (Recommended):
```python
from mad_spark_alt.core import UnifiedQADIOrchestrator, OrchestratorConfig

# Create config with simple strategy
config = OrchestratorConfig.simple_config()
config.temperature_override = 1.2  # Optional customization
config.num_hypotheses = 5  # Optional customization

orchestrator = UnifiedQADIOrchestrator(config=config)

result = await orchestrator.run_qadi_cycle("How can we improve efficiency?")
```

#### Alternative (Direct SimpleQADI):
```python
from mad_spark_alt.core import SimpleQADIOrchestrator

orchestrator = SimpleQADIOrchestrator(
    temperature_override=1.2,
    num_hypotheses=5
)

result = await orchestrator.run_qadi_cycle("How can we improve efficiency?")
```

### Why This Is Better
- ✅ Simpler architecture - fewer moving parts
- ✅ More reliable - no agent selection complexity
- ✅ Better maintained - focus on one strategy
- ✅ Easier to test - deterministic behavior
- ✅ Same quality results - Simple strategy is proven effective

### Breaking Changes
- `Strategy.SMART` enum value removed from `OrchestratorConfig`
- `OrchestratorConfig.smart_config()` factory method removed
- `UnifiedQADIOrchestrator` no longer supports Smart strategy
- Import of `SmartQADIOrchestrator` triggers deprecation warning

## CLI Flag Changes

### Removed Flags
- `--type` - No longer needed as universal prompts work for all question types
- `--concrete` - Implementation-focused mode removed in favor of universal approach

**Migration**: Simply remove these flags from your CLI commands. The new system automatically handles all question types effectively.

## Import Changes

### Old Import Pattern (No longer works)
```python
from mad_spark_alt.core import QADIOrchestrator, SmartQADIOrchestrator
from mad_spark_alt.core.prompt_classifier import classify_question  # ❌ REMOVED
from mad_spark_alt.core.adaptive_prompts import get_adaptive_prompt  # ❌ REMOVED
```

### New Import Pattern
```python
from mad_spark_alt.core import SimpleQADIOrchestrator, setup_llm_providers
# prompt_classifier and adaptive_prompts modules have been removed
```

## Example Migration

### Old Code
```python
# Complex setup with classification
from mad_spark_alt.core import SmartQADIOrchestrator

orchestrator = SmartQADIOrchestrator()
result = await orchestrator.run_qadi_cycle(
    user_prompt="How to improve team productivity?",
    question_type="business",  # Manual type specification
    concrete_mode=True,  # Implementation focus
)
```

### New Code
```python
# Simplified universal approach
from mad_spark_alt.core import SimpleQADIOrchestrator

orchestrator = SimpleQADIOrchestrator(
    temperature_override=1.2  # Optional: control creativity
)
result = await orchestrator.run_qadi_cycle(
    user_input="How to improve team productivity?",
    context="Remote team of 20 engineers",  # Optional context
)
```

## Benefits of Migration

1. **Simpler API** - No need to classify questions or specify types
2. **Better Results** - Universal prompts tuned for hypothesis-driven analysis
3. **Consistent Evaluation** - 5-criteria system across all components
4. **Reduced Complexity** - 40% less code, easier to maintain
5. **True QADI** - Follows the original methodology more faithfully

## Timeline

- **v1.x** - Deprecated components remain for compatibility
- **v2.0** - Breaking changes, deprecated components removed
- **Migration Period** - 6 months from PR #40 merge date