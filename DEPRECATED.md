# Deprecated Components

This document tracks deprecated components that are retained for backward compatibility but should not be used in new code.

## Files Retained for Backward Compatibility

### prompt_classifier.py & adaptive_prompts.py
- **Status**: Deprecated as of PR #40
- **Reason**: Replaced by the universal QADI prompts in `qadi_prompts.py`
- **Migration Path**: Use `SimpleQADIOrchestrator` which provides universal prompts without classification
- **Retention Reason**: Kept to avoid breaking existing code that imports these modules
- **Future**: Will be removed in v2.0.0

### Old Orchestrators
The following orchestrators are superseded by `SimpleQADIOrchestrator`:
- `QADIOrchestrator` - Original implementation with prompt classification
- `SmartQADIOrchestrator` - Enhanced version with adaptive prompts
- `RobustQADIOrchestrator` - Error-recovery focused version
- `FastQADIOrchestrator` - Performance-optimized version

**Migration Path**: Replace with `SimpleQADIOrchestrator` which provides:
- Cleaner implementation following true QADI methodology
- Universal prompts without classification overhead
- Temperature control for hypothesis generation
- Unified evaluation with the 5-criteria system

## CLI Flag Changes

### Removed Flags
- `--type` - No longer needed as universal prompts work for all question types
- `--concrete` - Implementation-focused mode removed in favor of universal approach

**Migration**: Simply remove these flags from your CLI commands. The new system automatically handles all question types effectively.

## Import Changes

### Old Import Pattern
```python
from mad_spark_alt.core import QADIOrchestrator, SmartQADIOrchestrator
from mad_spark_alt.core.prompt_classifier import classify_question
from mad_spark_alt.core.adaptive_prompts import get_adaptive_prompt
```

### New Import Pattern
```python
from mad_spark_alt.core import SimpleQADIOrchestrator, SimpleQADIResult
from mad_spark_alt.core import UnifiedEvaluator, HypothesisEvaluation
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