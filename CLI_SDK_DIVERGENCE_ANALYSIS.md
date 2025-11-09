# CLI/SDK Divergence Analysis
**Date**: 2025-11-10
**Branch**: fix/cli-sdk-parity
**Status**: Phase 1 - Investigation Complete

---

## Executive Summary

**Critical Finding**: The CLI contains a custom `SimplerQADIOrchestrator` class (line 489 in `unified_cli.py`) that inherits from `SimpleQADIOrchestrator` but overrides key behavior with custom prompts (`SimplerQADIPrompts`). This creates **two different QADI implementations** with divergent behavior.

### Impact
- **Dual Maintenance**: Changes to QADI logic must be made in two places
- **Inconsistent Behavior**: CLI users get different results than SDK users
- **Testing Complexity**: Need separate test suites for CLI and SDK
- **Feature Drift Risk**: CLI and SDK implementations will continue to diverge over time

---

## Detailed Comparison

### 1. Class Hierarchy

**SDK (Core Library)**:
```
SimpleQADIOrchestrator
├── Uses: QADIPrompts (core/qadi_prompts.py)
└── Located: src/mad_spark_alt/core/simple_qadi_orchestrator.py
```

**CLI (Unified CLI)**:
```
SimplerQADIOrchestrator (extends SimpleQADIOrchestrator)
├── Uses: SimplerQADIPrompts (unified_cli.py:473)
└── Located: src/mad_spark_alt/unified_cli.py:489
```

---

## Key Difference: Phase 1 (Questioning) Prompts

### SDK Version (`QADIPrompts.get_questioning_prompt`)
**Location**: `core/qadi_prompts.py:21-35`

```python
def get_questioning_prompt(user_input: str) -> str:
    return f"""As an analytical expert, identify THE single most important question to answer based on the user's input.

User's input:
{user_input}

Think about:
- What is the core challenge or desire expressed?
- What fundamental question needs answering to make progress?
- What would provide the most helpful insight?

Output exactly ONE core question that gets to the heart of the matter.
Format: "Q: [Your core question]"
"""
```

**Characteristics**:
- ✅ Detailed thinking prompts
- ✅ Structured guidance (3 bullet points)
- ✅ Professional "analytical expert" persona
- ✅ More elaborate prompt (~150 words)

---

### CLI Version (`SimplerQADIPrompts.get_questioning_prompt`)
**Location**: `unified_cli.py:476-486`

```python
def get_questioning_prompt(user_input: str) -> str:
    """Get a much simpler prompt for Phase 1."""
    return f"""What is the user asking?

User's input:
{user_input}

State their question clearly and directly. If they made a statement, rephrase it as the implied question.
Format: "Q: [The user's question]"
"""
```

**Characteristics**:
- ❌ Minimal prompt
- ❌ No thinking guidance
- ❌ Simple directive
- ❌ Much shorter (~40 words)

---

## Behavioral Impact

### Expected Differences

| Aspect | SDK (QADIPrompts) | CLI (SimplerQADIPrompts) |
|--------|-------------------|--------------------------|
| **Prompt Length** | ~150 words | ~40 words |
| **Guidance** | Structured (3 bullet points) | Direct command only |
| **Persona** | "Analytical expert" | Neutral |
| **LLM Token Cost** | Higher (more detailed) | Lower (minimal) |
| **Question Quality** | More thoughtful/refined | More literal/direct |
| **Processing Time** | Slightly slower | Slightly faster |

### Hypothesis on User Impact

**SDK Users**:
- Get more refined, analytical core questions
- Questions are "sharpened" by LLM reasoning
- Better suited for complex, ambiguous inputs
- Example: "How can we reduce waste?" → "What systemic changes would reduce waste at production, consumption, and disposal stages?"

**CLI Users**:
- Get more literal, direct question restatements
- Faster turnaround time
- Better for clear, well-formed questions
- Example: "How can we reduce waste?" → "How can we reduce waste?"

---

## Other Shared Behavior

### Phases 2-4 (Abduction, Deduction, Induction)

✅ **NO DIFFERENCES** - Both use the same `QADIPrompts` methods:
- `get_abduction_prompt()` - Identical
- `get_deduction_prompt()` - Identical
- `get_induction_prompt()` - Identical

### Orchestration Logic

✅ **NO DIFFERENCES** - `SimplerQADIOrchestrator` inherits all orchestration from `SimpleQADIOrchestrator`:
- Phase execution order
- Error handling
- Cost tracking
- Multimodal support
- Result formatting

---

## Why This Divergence Exists

### Historical Context (Inferred)

Based on class name and docstring:
1. **Original Goal**: Make CLI faster/simpler than SDK
2. **Implementation**: Override Phase 1 prompt only
3. **Assumption**: Users calling CLI want quick, literal questions
4. **Result**: Divergent behavior that wasn't fully documented

### Current State
- No documentation explaining why CLI behavior differs
- No tests comparing CLI vs SDK output
- No user-facing documentation of this difference
- Unclear if the simplification is actually beneficial

---

## Risk Assessment

### High Risk Areas

1. **Silent Behavioral Difference**
   - Users don't know CLI and SDK behave differently
   - No warning or documentation
   - Could cause confusion when switching between interfaces

2. **Maintenance Burden**
   - Must update both `QADIPrompts` and `SimplerQADIPrompts`
   - Easy to forget to sync changes
   - Increases testing surface area

3. **Feature Drift**
   - Over time, differences will compound
   - More temptation to add CLI-specific logic
   - Eventually becomes impossible to unify

### Low Risk Areas

✅ Phases 2-4 are identical (shared code)
✅ Clear inheritance structure (easy to trace)
✅ Only affects Phase 1 prompt text

---

## Recommended Fix Strategy

### Option A: Remove SimplerQADIOrchestrator (Recommended)

**Approach**: Make CLI use `SimpleQADIOrchestrator` directly

**Steps**:
1. Update CLI to instantiate `SimpleQADIOrchestrator` instead of `SimplerQADIOrchestrator`
2. Remove `SimplerQADIPrompts` class (lines 473-486)
3. Remove `SimplerQADIOrchestrator` class (lines 489-495)
4. Run full test suite to verify behavior

**Pros**:
- ✅ Single source of truth
- ✅ Consistent behavior across CLI and SDK
- ✅ Reduces code by ~20 lines
- ✅ Easier maintenance

**Cons**:
- ⚠️ CLI Phase 1 might be slightly slower (marginal)
- ⚠️ Some CLI users might prefer simpler prompts

**Impact**: ~20 line reduction, behavioral change for CLI users in Phase 1 only

---

### Option B: Make Prompt Simplicity a Feature Flag

**Approach**: Add `--simple-prompts` flag to CLI

**Steps**:
1. Keep both prompt classes
2. Add CLI flag: `--simple-prompts` (default: False)
3. Document the difference
4. Let users choose

**Pros**:
- ✅ Preserves both behaviors
- ✅ User choice
- ✅ No breaking changes

**Cons**:
- ❌ Maintains dual codebase
- ❌ More complexity
- ❌ More testing burden
- ❌ Still need to maintain both

**Impact**: +10 lines (flag handling), no reduction

---

### Option C: Merge SimplerQADIPrompts into QADIPrompts

**Approach**: Make SDK prompt configurable

**Steps**:
1. Add `simple: bool = False` parameter to `get_questioning_prompt()`
2. If simple=True, use shorter prompt
3. CLI sets simple=True, SDK uses default

**Pros**:
- ✅ Single class for prompts
- ✅ Preserves both behaviors
- ✅ More flexible

**Cons**:
- ⚠️ Adds complexity to core library
- ⚠️ Still need dual testing
- ⚠️ Leaks CLI concern into SDK

**Impact**: Neutral lines, maintains some divergence

---

## Decision Matrix

| Criterion | Option A (Remove) | Option B (Flag) | Option C (Merge) |
|-----------|-------------------|-----------------|------------------|
| **Code Simplicity** | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ |
| **Maintenance Burden** | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ |
| **Consistency** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **User Choice** | ⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Testing Burden** | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ |
| **Breaking Changes** | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |

### **Recommendation: Option A (Remove SimplerQADIOrchestrator)**

**Rationale**:
1. **Minimal Real Benefit**: The "simpler" prompt doesn't meaningfully improve CLI UX
2. **Marginal Cost**: Phase 1 is fast either way (~2-5 seconds)
3. **High Maintenance Cost**: Dual codebases compound over time
4. **Better Alternatives**: If speed is critical, add Phase 1 caching instead

---

## Implementation Plan

### Phase 2: Unification (8 hours)

#### Step 1: Update CLI Instantiation (2 hours)
**File**: `unified_cli.py`

**Find**:
```python
orchestrator = SimplerQADIOrchestrator(
    temperature_override=temperature,
    num_hypotheses=3
)
```

**Replace with**:
```python
orchestrator = SimpleQADIOrchestrator(
    temperature_override=temperature,
    num_hypotheses=3
)
```

**Locations**: Search for all instantiations of `SimplerQADIOrchestrator`

---

#### Step 2: Remove Custom Classes (1 hour)
**File**: `unified_cli.py`

**Delete**:
- Lines 473-486: `SimplerQADIPrompts` class
- Lines 489-495: `SimplerQADIOrchestrator` class

**Update imports**: Remove any SimplerQADI* references

---

#### Step 3: Update Tests (3 hours)

**Create new test**: `tests/test_cli_sdk_parity.py`

```python
import pytest
import asyncio
from mad_spark_alt.core.simple_qadi_orchestrator import SimpleQADIOrchestrator
from mad_spark_alt.unified_cli import _run_qadi_async  # If exposed

@pytest.mark.asyncio
async def test_cli_uses_core_orchestrator():
    """Verify CLI uses core SimpleQADIOrchestrator."""
    # Test that CLI and SDK produce same Phase 1 output
    question = "How can we improve productivity?"

    # SDK path
    sdk_orchestrator = SimpleQADIOrchestrator()
    sdk_result = await sdk_orchestrator.run_qadi_cycle(question)

    # CLI path (would need to expose internal logic or test via CLI)
    # ... test equivalence ...

    assert sdk_result.core_question  # Verify structure
```

**Update existing tests**: Check if any tests depend on `SimplerQADIOrchestrator` behavior

---

#### Step 4: Documentation (1 hour)

**Update**:
- `ARCHITECTURE.md`: Document single orchestrator
- `DEPRECATED.md`: Add migration note (if external users used SimplerQADI)
- `CHANGELOG.md`: Document breaking change

---

#### Step 5: Real API Validation (1 hour)

**Test with actual Google API**:
```bash
# Test CLI
msa "How can we reduce carbon emissions?" --verbose

# Test SDK (Python script)
python -c "
import asyncio
from mad_spark_alt.core.simple_qadi_orchestrator import SimpleQADIOrchestrator
async def test():
    orch = SimpleQADIOrchestrator()
    result = await orch.run_qadi_cycle('How can we reduce carbon emissions?')
    print(result.core_question)
asyncio.run(test())
"

# Compare outputs - should be equivalent
```

---

### Phase 3: Verification (4 hours)

1. **Run full test suite**: `uv run pytest` (844 tests)
2. **Manual CLI testing**: Try 10 different questions
3. **Compare before/after**: Save outputs from current CLI, compare with unified version
4. **Performance check**: Measure Phase 1 time before/after
5. **Edge case testing**: Ambiguous inputs, statements vs questions

---

## Success Criteria

- ✅ All 844 tests pass
- ✅ No `SimplerQADI` classes in codebase
- ✅ CLI behavior matches SDK for Phase 1
- ✅ No performance degradation (< 10% slower acceptable)
- ✅ Documentation updated
- ✅ Real API validation complete

---

## Rollback Plan

If unification causes issues:
1. Revert branch: `git checkout main`
2. Cherry-pick non-CLI changes if needed
3. Re-evaluate Option B or C

**Rollback trigger**: If Phase 1 time increases >30% or user complaints

---

## Next Steps

1. ✅ **DONE**: Phase 1 investigation complete
2. **NEXT**: Get approval on Option A approach
3. **THEN**: Begin Phase 2 implementation
4. **FINALLY**: Merge to main after validation

---

## Questions for Review

1. **User Impact**: Is it acceptable for CLI users to get slightly more detailed Phase 1 questions?
2. **Performance**: Is a potential 2-3 second increase in Phase 1 acceptable for code simplicity?
3. **Alternative**: Should we implement Option C (configurable prompts) instead?

---

**End of Investigation - Ready for Phase 2**
