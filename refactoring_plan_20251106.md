# Refactoring Plan: Mad Spark Alt Codebase Consolidation

**Date:** 2025-11-06
**Status:** In Progress (Phase 1 & 2 Complete, Phase 3 In Progress)
**Last Updated:** 2025-11-08
**Estimated Timeline:** 14-20 days across 3 phases

---

## 🎯 Implementation Status (As of 2025-11-08)

### Phase 1: Quick Wins ✅ **COMPLETE**
- ✅ **Item 1**: Remove deprecated code (-1,251 lines) - [PR #105](https://github.com/TheIllusionOfLife/mad_spark_alt/pull/105)
- ✅ **Item 2**: Fix CLI --evaluators flag - [PR #106](https://github.com/TheIllusionOfLife/mad_spark_alt/pull/106)
- ✅ **Item 3**: Verify & fix Gemini structured output - [PR #107](https://github.com/TheIllusionOfLife/mad_spark_alt/pull/107)
- ✅ **Item 4**: Consolidate JSON parsing utilities - [PR #109, #110](https://github.com/TheIllusionOfLife/mad_spark_alt/pull/110)

### Phase 2: Core Refactoring ✅ **COMPLETE**
- ✅ **Item 5**: Create parsing_utils.py (843 lines) - [PR #110](https://github.com/TheIllusionOfLife/mad_spark_alt/pull/110)
- ✅ **Item 6**: Create phase_logic.py (791 lines) - [PR #111](https://github.com/TheIllusionOfLife/mad_spark_alt/pull/111)
- ✅ **Item 7**: Create base_orchestrator.py (468 lines) - [PR #112](https://github.com/TheIllusionOfLife/mad_spark_alt/pull/112)
- ✅ **Item 8**: Refactor SimpleQADI (1,296 → 221 lines, 83% reduction!) - [PR #111](https://github.com/TheIllusionOfLife/mad_spark_alt/pull/111)
- ✅ **Item 9**: Refactor MultiPerspective (507 → 312 lines, 38% reduction!) - **COMPLETED 2025-11-08**
- ✅ **Item 10**: Remove legacy orchestrators (738 lines removed) - **COMPLETED 2025-11-07**

### Phase 3: Architecture Consolidation ✅ **COMPLETE**
- ✅ **Item 11**: UnifiedOrchestrator - Smart strategy removed (Simple + MultiPerspective only) - **COMPLETED 2025-11-08**
- ✅ **Item 12**: OrchestratorConfig updated - Removed SMART enum and smart_config() - **COMPLETED 2025-11-08**
- ✅ **Item 13**: Split semantic_operators.py (1,926 → 62 lines, 97% reduction!) - **COMPLETED 2025-11-08**
- ✅ **Item 14**: SmartQADIOrchestrator deprecated (remove in v2.0.0) - **COMPLETED 2025-11-08**

### 📊 Progress Summary
- **Phase 1**: 4/4 items complete (100%) ✅
- **Phase 2**: 6/6 items complete (100%) ✅
- **Phase 3**: 4/4 items complete (100%) ✅
- **Overall**: 14/14 items complete (100%) 🎉

### 🎉 Key Achievements
- **Total lines removed**: ~4,496+ lines across PRs #105, #109-112, Step 9, Step 10, and Step 13
- **SimpleQADI reduction**: 1,296 → 221 lines (83% reduction, exceeded target!)
- **MultiPerspective reduction**: 507 → 312 lines (38% reduction!)
- **Semantic operators reduction**: 1,926 → 62 lines (97% reduction!)
- **Legacy orchestrators removed**: 738 lines (enhanced, robust, fast)
- **Unified orchestrator**: Smart strategy removed, Simple + MultiPerspective supported
- **Modular architecture**: 7 new foundational modules (parsing_utils, phase_logic, base_orchestrator, semantic_utils, operator_cache, semantic_mutation, semantic_crossover)
- **Test coverage**: Comprehensive unit and integration tests (815+ tests passing)
- **All PRs**: Merged successfully with CI passing
- **Real API testing**: All refactored code verified with live Google API
- **CLI Integration**: 10 new tests verify backward compatibility

### 🎊 REFACTORING COMPLETE!
**Phase 3 Final Status:**
- ✅ UnifiedOrchestrator simplified (removed Smart strategy)
- ✅ OrchestratorConfig streamlined (2 strategies: Simple, MultiPerspective)
- ✅ SmartQADIOrchestrator deprecated with migration guide
- ✅ CLI backward compatible - no breaking changes
- ✅ All 815+ tests passing
- ✅ Real API validation successful (basic QADI, evolution, temperature override)
- ✅ Zero regressions

---

## Executive Summary

Based on multiple code reviews and comprehensive codebase exploration, this plan addresses critical architectural issues in the Mad Spark Alt system:

- **7 orchestrator classes** with ~3,850 lines of duplicated code
- **1,251 lines of deprecated code** still in the codebase
- **Duplicate JSON parsing utilities** causing maintenance burden
- **God objects** (1,926-line semantic_operators.py, 1,296-line simple_qadi_orchestrator.py)
- **Broken CLI features** (--evaluators flag non-functional)

**Expected Impact:**
- **-29% codebase size** (~12,000 → ~8,500 lines)
- **-86% orchestrator files** (7 → 1 unified + base)
- **+20% test coverage** (comprehensive integration tests)
- **Unified architecture** with clear separation of concerns

---

## Review Analysis & Scoring

### Review 1: "Key Findings & Refactoring Plans" - 7.5/10

**Strengths:**
- ✅ Accurately identifies CLI --evaluators bug (cli.py:233-303)
- ✅ Correctly identifies JSON extraction divergence (3 codepaths)
- ✅ Accurately describes SimpleQADI god object (1,296 lines)
- ✅ Clear priority roadmap with phased approach

**Weaknesses:**
- ❌ Incorrect on session leak (close() methods exist, just not invoked)
- ⚠️ Unverified Gemini structured output claim (needs API docs verification)

### Review 2: "Orchestrator Consolidation Plan" - 6/10

**Strengths:**
- ✅ Correctly identifies core problem (7 orchestrators)
- ✅ Strategic vision is sound (unified orchestrator)

**Weaknesses:**
- ❌ Lacks specificity (no file paths, line numbers)
- ❌ No prioritization (doesn't distinguish quick wins)
- ❌ Missing context (which orchestrators are actively used)

### Review 3: "Categorized Problems Summary" - 8.5/10

**Strengths:**
- ✅ All findings confirmed accurate
- ✅ Excellent categorization (Critical vs Major)
- ✅ Quantified impact (specific line counts)
- ✅ Risk-aware (identifies low-risk starting points)

**Weaknesses:**
- ⚠️ Missing CLI --evaluators bug
- ⚠️ No file paths for verification

### Review 4: "Modular Refactoring Approach" - 9/10 ⭐ BEST

**Strengths:**
- ✅ Best implementation detail (specific modules to create)
- ✅ Clear separation of concerns
- ✅ Reusability focus (centralized logic)
- ✅ Follows SOLID principles

**Weaknesses:**
- ⚠️ Doesn't address quick wins first
- ⚠️ No prioritization guidance

---

## Current State Analysis

### Orchestrator Files (7 total, ~3,850 lines)

| File | Lines | Status | Purpose |
|------|-------|--------|---------|
| `orchestrator.py` | 279 | Legacy | Base QADI (rarely used) |
| `smart_orchestrator.py` | 953 | Legacy | LLM agent selection |
| `simple_qadi_orchestrator.py` | **1,296** | **ACTIVE** | **PRIMARY - Used in CLI** |
| `multi_perspective_orchestrator.py` | 602 | Active | Multi-perspective analysis |
| `enhanced_orchestrator.py` | 201 | Deprecated | Answer extraction wrapper |
| `robust_orchestrator.py` | 306 | Deprecated | Timeout handling wrapper |
| `fast_orchestrator.py` | 217 | Deprecated | Parallel execution wrapper |

**Finding:** Only 2 actively used, 3 deprecated wrappers (724 lines), 2 legacy (1,232 lines).

### JSON Parsing Duplication

**A. `json_utils.py` (428 lines)**
- Function: `extract_json_from_response(text: str) -> Optional[str]`
- Returns: Raw JSON string
- Used in: simple_qadi_orchestrator.py, LLM prompts

**B. `robust_json_handler.py` (194 lines)**
- Function: `extract_json_from_response(response: str, expected_keys, fallback) -> Dict`
- Returns: Parsed dictionary
- Used in: robust_orchestrator.py, multi_perspective_orchestrator.py

**C. Inline parsing in orchestrators**
- `simple_qadi_orchestrator.py` starting at line 432
- Custom regex loops for hypothesis extraction

**Impact:** 622+ lines of duplicate logic, inconsistent behavior.

### Deprecated Code

| File | Lines | Status | Scheduled Removal |
|------|-------|--------|-------------------|
| `prompt_classifier.py` | 748 | Deprecated | v2.0.0 |
| `adaptive_prompts.py` | 503 | Deprecated | v2.0.0 |
| **Total** | **1,251** | **10% of core module** | - |

Both have deprecation warnings at module import but still in codebase.

### Large Files (>800 lines)

| File | Lines | Module | Issues |
|------|-------|--------|--------|
| `semantic_operators.py` | **1,926** | evolution/ | Mutation, crossover, batch ops, caching |
| `simple_qadi_orchestrator.py` | **1,296** | core/ | All QADI phases + parsing + scoring |
| `genetic_algorithm.py` | 1,004 | evolution/ | GA implementation with caching |
| `cli.py` | 864 | Root | Multiple command groups |

### Critical Bugs

**1. CLI --evaluators Flag Non-Functional**
- Location: `cli.py:233` (flag defined), `cli.py:252-303` (parameter ignored)
- Impact: Users cannot filter evaluators despite help text
- Fix complexity: Low (2-3 hours)

**2. Gemini Structured Output Placement**
- Location: `llm_provider.py:292-294`
- Claim: responseMimeType/responseSchema in wrong location
- Status: ⚠️ Needs verification against Gemini API docs
- Impact: May force regex fallbacks instead of structured JSON

---

## Phase 1: Quick Wins (2-3 days, Low Risk, High Impact)

### 1. Remove Deprecated Code (-1,251 lines) ⏱️ 2 hours

**Actions:**
1. Delete `src/mad_spark_alt/core/prompt_classifier.py` (748 lines)
2. Delete `src/mad_spark_alt/core/adaptive_prompts.py` (503 lines)
3. Update `src/mad_spark_alt/core/__init__.py` (remove exports)
4. Search for imports and remove them
5. Add deprecation notice to CHANGELOG.md

**Risk:** Low (already deprecated, scheduled for v2.0.0 removal)

**Testing:**
```bash
# Verify no imports remain
uv run pytest tests/ -v
uv run mypy src/
```

**Success Criteria:**
- ✅ All tests pass
- ✅ No import errors
- ✅ Codebase reduced by 1,251 lines

---

### 2. Fix CLI --evaluators Flag ⏱️ 3 hours

**Current Bug:**
```python
# cli.py:233
@click.option("--evaluators", "-e", help="Comma-separated list of evaluators")

# cli.py:252
def evaluate(text: Optional[str], ..., evaluators: Optional[str], ...) -> None:
    # evaluators parameter accepted but NEVER USED

# cli.py:300-303
request = EvaluationRequest(
    outputs=[model_output],
    target_layers=target_layers,
    # ❌ evaluators not passed
)
```

**Fix:**
```python
# Parse evaluators string
selected_evaluators = None
if evaluators:
    selected_evaluators = [e.strip() for e in evaluators.split(",")]
    # Validate against registry
    available = [e.name for e in registry.get_compatible_evaluators(target_layers)]
    invalid = [e for e in selected_evaluators if e not in available]
    if invalid:
        console.print(f"[red]Unknown evaluators: {', '.join(invalid)}[/red]")
        console.print(f"Available: {', '.join(available)}")
        return

# Pass to request
request = EvaluationRequest(
    outputs=[model_output],
    target_layers=target_layers,
    evaluator_names=selected_evaluators,  # ✅ Add this
)
```

**Testing:**
```bash
# Add CLI test
uv run pytest tests/test_cli.py::test_evaluators_flag -v

# Manual verification
uv run mad-spark evaluate "test" --evaluators creativity,coherence
```

**Success Criteria:**
- ✅ CLI accepts comma-separated evaluator names
- ✅ Only specified evaluators run
- ✅ Error on unknown evaluator names
- ✅ Help text matches behavior

---

### 3. Verify & Fix Gemini Structured Output ⏱️ 4 hours

**Investigation Required:**
1. Check Gemini API documentation for correct payload structure
2. Verify if `responseMimeType`/`responseSchema` should be:
   - Inside `generationConfig` (current implementation)
   - At top level of request (reviewer's claim)

**Current Implementation:**
```python
# llm_provider.py:292-294
if request.response_schema and request.response_mime_type:
    generation_config["responseMimeType"] = request.response_mime_type
    generation_config["responseSchema"] = request.response_schema
```

**If Fix Required:**
```python
# Move to top-level of payload
payload = {
    "contents": [{"role": "user", "parts": [{"text": full_prompt}]}],
    "generationConfig": generation_config,  # temperature, maxOutputTokens, etc.
}

# Add at top level if structured output requested
if request.response_schema and request.response_mime_type:
    payload["responseMimeType"] = request.response_mime_type
    payload["responseSchema"] = request.response_schema
```

**Testing (TDD Approach):**
```python
# tests/test_llm_provider.py
@pytest.mark.asyncio
async def test_gemini_structured_output_placement():
    """Verify structured output config placed correctly in payload"""
    provider = GoogleProvider(api_key="test")

    request = LLMRequest(
        user_prompt="Generate hypothesis",
        response_schema={"type": "object", "properties": {...}},
        response_mime_type="application/json"
    )

    # Mock aiohttp to capture payload
    with patch("aiohttp.ClientSession.post") as mock_post:
        await provider.generate(request)

        # Verify payload structure
        call_args = mock_post.call_args
        payload = json.loads(call_args[1]["data"])

        # Assert correct placement (adjust based on API docs)
        assert "responseMimeType" in payload  # or in payload["generationConfig"]
        assert "responseSchema" in payload
```

**Success Criteria:**
- ✅ Payload structure matches Gemini API specification
- ✅ Structured JSON responses work without regex fallbacks
- ✅ Hypothesis extraction logs show "structured output" path
- ✅ No "insufficient hypotheses" warnings for valid responses

---

### 4. Consolidate JSON Parsing Utilities ⏱️ 1 day

**Goal:** Single source of truth for JSON extraction.

**Strategy: Keep `json_utils.py`, deprecate `robust_json_handler.py`**

**Step 1: Create Unified Extractor**
```python
# src/mad_spark_alt/core/json_utils.py
class JsonExtractor:
    """Unified JSON extraction with multiple strategies"""

    @staticmethod
    def extract_with_strategies(
        text: str,
        expected_keys: Optional[List[str]] = None,
        fallback: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Try multiple extraction strategies in order:
        1. Direct JSON parsing
        2. Markdown code blocks (```json ... ```)
        3. Brace balancing (find outermost {})
        4. Nested object patterns
        5. Return fallback if all fail
        """
        # Strategy 1: Direct parsing
        try:
            result = json.loads(text)
            if expected_keys and validate_keys(result, expected_keys):
                return result
        except json.JSONDecodeError:
            pass

        # Strategy 2: Markdown blocks
        result = extract_from_markdown(text)
        if result and (not expected_keys or validate_keys(result, expected_keys)):
            return result

        # Strategy 3: Brace balancing
        result = extract_by_braces(text)
        if result and (not expected_keys or validate_keys(result, expected_keys)):
            return result

        # Strategy 4: Nested patterns
        result = extract_nested_objects(text)
        if result and (not expected_keys or validate_keys(result, expected_keys)):
            return result

        # Fallback
        return fallback or {}
```

**Step 2: Update All Consumers**
- `simple_qadi_orchestrator.py` (line 432+)
- `multi_perspective_orchestrator.py`
- Any inline regex parsing loops

**Step 3: Deprecate `robust_json_handler.py`**
```python
# Add deprecation warning
import warnings
warnings.warn(
    "robust_json_handler is deprecated. Use json_utils.JsonExtractor instead.",
    DeprecationWarning,
    stacklevel=2
)
```

**Testing:**
```python
# tests/test_json_utils.py
def test_extraction_strategies():
    """Test all extraction strategies with realistic data"""

    # Test 1: Clean JSON
    assert JsonExtractor.extract_with_strategies('{"key": "value"}') == {"key": "value"}

    # Test 2: Markdown wrapped
    markdown = '```json\n{"key": "value"}\n```'
    assert JsonExtractor.extract_with_strategies(markdown) == {"key": "value"}

    # Test 3: ANSI codes
    ansi = '\x1b[1m{"key": "value"}\x1b[0m'
    assert JsonExtractor.extract_with_strategies(ansi) == {"key": "value"}

    # Test 4: Nested braces
    nested = 'Some text {"outer": {"inner": "value"}} more text'
    assert JsonExtractor.extract_with_strategies(nested) == {"outer": {"inner": "value"}}

    # Test 5: Expected keys validation
    result = JsonExtractor.extract_with_strategies(
        '{"key1": "a", "key2": "b"}',
        expected_keys=["key1", "key2"]
    )
    assert result == {"key1": "a", "key2": "b"}

    # Test 6: Fallback
    result = JsonExtractor.extract_with_strategies(
        "not json",
        fallback={"default": True}
    )
    assert result == {"default": True}
```

**Success Criteria:**
- ✅ Single `JsonExtractor` class handles all cases
- ✅ All orchestrators use unified API
- ✅ Comprehensive tests cover edge cases
- ✅ `robust_json_handler.py` deprecated (not yet removed)

---

## Phase 2: Core Refactoring (5-7 days, Medium Risk)

### 5. Create Parsing Utilities Module ⏱️ 1 day

**New File:** `src/mad_spark_alt/core/parsing_utils.py`

**Purpose:** Centralize all LLM response parsing logic.

**Modules:**
```python
class HypothesisParser:
    """Parse hypotheses from LLM responses"""

    @staticmethod
    def parse_from_json(response: str, num_expected: int) -> List[GeneratedIdea]:
        """Parse structured JSON response"""

    @staticmethod
    def parse_from_text(response: str, num_expected: int) -> List[GeneratedIdea]:
        """Parse unstructured text with regex"""

    @staticmethod
    def parse_with_fallback(response: str, num_expected: int) -> List[GeneratedIdea]:
        """Try JSON first, fall back to text parsing"""


class ScoreParser:
    """Parse QADI scores and fitness scores"""

    @staticmethod
    def parse_qadi_scores(response: str) -> HypothesisScore:
        """Parse Impact, Feasibility, Accessibility scores"""

    @staticmethod
    def parse_fitness_scores(response: str) -> float:
        """Parse evolutionary fitness values"""


class ActionPlanParser:
    """Parse action plans and next steps"""

    @staticmethod
    def parse_action_items(response: str) -> List[str]:
        """Extract numbered or bulleted action items"""
```

**Benefits:**
- Reusable across all orchestrators
- Single place for parsing improvements
- Easy to unit test in isolation

---

### 6. Create Phase Logic Module ⏱️ 2 days

**New File:** `src/mad_spark_alt/core/phase_logic.py`

**Purpose:** Implement standalone QADI phase functions.

**Structure:**
```python
@dataclass
class PhaseInput:
    """Common inputs for all phases"""
    question: str
    llm_manager: LLMManager
    model_config: ModelConfig
    context: Dict[str, Any]  # Accumulated results from previous phases


@dataclass
class PhaseResult:
    """Common outputs from all phases"""
    success: bool
    data: Any  # Phase-specific data
    llm_cost: float
    errors: List[str]


async def execute_questioning_phase(phase_input: PhaseInput) -> PhaseResult:
    """
    Phase 1: Clarify and reframe the question
    Returns: clarified question, assumptions, constraints
    """
    # Build prompt
    # Call LLM
    # Parse response using parsing_utils
    # Return structured result


async def execute_abduction_phase(
    phase_input: PhaseInput,
    num_hypotheses: int = 3
) -> PhaseResult:
    """
    Phase 2: Generate hypotheses
    Returns: List[GeneratedIdea]
    """
    # Build prompt with question context
    # Call LLM with structured output schema
    # Parse hypotheses using HypothesisParser
    # Return ideas with costs


async def execute_deduction_phase(
    phase_input: PhaseInput,
    hypotheses: List[GeneratedIdea]
) -> PhaseResult:
    """
    Phase 3: Score and validate hypotheses
    Returns: hypotheses with QADI scores
    """
    # Build prompt with hypotheses
    # Call LLM for scoring
    # Parse scores using ScoreParser
    # Attach scores to ideas


async def execute_induction_phase(
    phase_input: PhaseInput,
    scored_hypotheses: List[GeneratedIdea]
) -> PhaseResult:
    """
    Phase 4: Synthesize insights and action plan
    Returns: synthesis text, action items, final answer
    """
    # Build prompt with scored hypotheses
    # Call LLM for synthesis
    # Parse action plan using ActionPlanParser
    # Return comprehensive result
```

**Benefits:**
- Testable in isolation (mock LLM calls)
- Clear inputs/outputs
- Reusable across orchestrators
- No orchestration logic mixed in

---

### 7. Create Base Orchestrator ⏱️ 1 day

**New File:** `src/mad_spark_alt/core/base_orchestrator.py`

**Purpose:** Shared orchestration logic without phase implementation details.

**Structure:**
```python
class BaseOrchestrator:
    """Base class for QADI orchestration with shared logic"""

    def __init__(
        self,
        llm_manager: LLMManager,
        model_config: ModelConfig,
        num_hypotheses: int = 3
    ):
        self.llm_manager = llm_manager
        self.model_config = model_config
        self.num_hypotheses = num_hypotheses
        self.total_cost = 0.0
        self.context: Dict[str, Any] = {}

    async def run_qadi_cycle(
        self,
        question: str,
        perspectives: Optional[List[str]] = None
    ) -> QADIResult:
        """
        Execute full QADI cycle:
        Question → Abduction → Deduction → Induction
        """
        try:
            # Phase 1: Questioning
            q_result = await self._execute_questioning(question)
            self._accumulate_cost(q_result.llm_cost)
            self.context["clarified_question"] = q_result.data

            # Phase 2: Abduction
            a_result = await self._execute_abduction(q_result.data)
            self._accumulate_cost(a_result.llm_cost)
            self.context["hypotheses"] = a_result.data

            # Phase 3: Deduction
            d_result = await self._execute_deduction(a_result.data)
            self._accumulate_cost(d_result.llm_cost)
            self.context["scored_hypotheses"] = d_result.data

            # Phase 4: Induction
            i_result = await self._execute_induction(d_result.data)
            self._accumulate_cost(i_result.llm_cost)

            return self._build_final_result(i_result)

        except Exception as e:
            logger.error(f"QADI cycle failed: {e}")
            return self._build_error_result(e)

    # Abstract methods for subclasses to implement
    async def _execute_questioning(self, question: str) -> PhaseResult:
        raise NotImplementedError

    async def _execute_abduction(self, clarified_q: str) -> PhaseResult:
        raise NotImplementedError

    async def _execute_deduction(self, hypotheses: List) -> PhaseResult:
        raise NotImplementedError

    async def _execute_induction(self, scored: List) -> PhaseResult:
        raise NotImplementedError

    def _accumulate_cost(self, cost: float) -> None:
        self.total_cost += cost

    def _build_final_result(self, induction: PhaseResult) -> QADIResult:
        """Build final result with all accumulated data"""
        return QADIResult(
            question=self.context.get("clarified_question", ""),
            hypotheses=self.context.get("scored_hypotheses", []),
            synthesis=induction.data.get("synthesis", ""),
            action_plan=induction.data.get("action_items", []),
            llm_cost=self.total_cost
        )
```

**Benefits:**
- DRY: Cost tracking, context management, error handling
- Clear contract for subclasses
- State management in one place
- Easy to add hooks/callbacks

---

### 8. Refactor SimpleQADI (1,296 → ~400 lines) ⏱️ 2 days

**Goal:** Inherit from `BaseOrchestrator`, delegate to `phase_logic.py`.

**Before:** 1,296 lines mixing prompts, transport, parsing, scoring

**After:**
```python
# src/mad_spark_alt/core/simple_qadi_orchestrator.py
from .base_orchestrator import BaseOrchestrator
from .phase_logic import (
    execute_questioning_phase,
    execute_abduction_phase,
    execute_deduction_phase,
    execute_induction_phase,
    PhaseInput
)


class SimpleQADIOrchestrator(BaseOrchestrator):
    """Simplified QADI orchestrator using modular phase logic"""

    async def _execute_questioning(self, question: str) -> PhaseResult:
        phase_input = PhaseInput(
            question=question,
            llm_manager=self.llm_manager,
            model_config=self.model_config,
            context=self.context
        )
        return await execute_questioning_phase(phase_input)

    async def _execute_abduction(self, clarified_q: str) -> PhaseResult:
        phase_input = PhaseInput(
            question=clarified_q,
            llm_manager=self.llm_manager,
            model_config=self.model_config,
            context=self.context
        )
        return await execute_abduction_phase(phase_input, self.num_hypotheses)

    async def _execute_deduction(self, hypotheses: List[GeneratedIdea]) -> PhaseResult:
        phase_input = PhaseInput(
            question=self.context["clarified_question"],
            llm_manager=self.llm_manager,
            model_config=self.model_config,
            context=self.context
        )
        return await execute_deduction_phase(phase_input, hypotheses)

    async def _execute_induction(self, scored: List[GeneratedIdea]) -> PhaseResult:
        phase_input = PhaseInput(
            question=self.context["clarified_question"],
            llm_manager=self.llm_manager,
            model_config=self.model_config,
            context=self.context
        )
        return await execute_induction_phase(phase_input, scored)
```

**Result:** ~400 lines (down from 1,296)

**Risk Mitigation:**
1. **Write comprehensive integration tests FIRST** (TDD)
2. **Keep old implementation** as `simple_qadi_orchestrator_legacy.py` during transition
3. **Feature flag** to toggle between old/new
4. **Run both in parallel** on test questions, compare outputs

**Testing Strategy:**
```python
# tests/test_simple_qadi_integration.py
@pytest.mark.asyncio
async def test_full_qadi_cycle():
    """End-to-end test of refactored SimpleQADI"""
    orchestrator = SimpleQADIOrchestrator(...)

    result = await orchestrator.run_qadi_cycle(
        "How can we reduce plastic waste in oceans?"
    )

    # Verify structure
    assert result.question
    assert len(result.hypotheses) == 3
    assert all(h.qadi_score for h in result.hypotheses)
    assert result.synthesis
    assert result.action_plan
    assert result.llm_cost > 0


@pytest.mark.asyncio
async def test_phase_independence():
    """Verify each phase can be tested independently"""
    orchestrator = SimpleQADIOrchestrator(...)

    # Test questioning phase only
    q_result = await orchestrator._execute_questioning("test question")
    assert q_result.success
    assert q_result.data
```

**Success Criteria:**
- ✅ All existing tests pass
- ✅ New integration tests verify correctness
- ✅ File size reduced to ~400 lines
- ✅ No behavioral regressions
- ✅ CLI commands work identically

---

### 9. Refactor MultiPerspective (602 → ~200 lines) ⏱️ 1 day

**Goal:** Reuse refactored `SimpleQADIOrchestrator` instances.

**Before:** 602 lines re-implementing QADI cycle

**After:**
```python
# src/mad_spark_alt/core/multi_perspective_orchestrator.py
from .simple_qadi_orchestrator import SimpleQADIOrchestrator


class MultiPerspectiveOrchestrator:
    """Run QADI analysis from multiple perspectives"""

    def __init__(self, llm_manager: LLMManager, model_config: ModelConfig):
        self.llm_manager = llm_manager
        self.model_config = model_config

    async def run_multi_perspective_analysis(
        self,
        question: str,
        perspectives: List[str]
    ) -> MultiPerspectiveResult:
        """
        Run SimpleQADI for each perspective, then synthesize
        """
        perspective_results = []

        for idx, perspective in enumerate(perspectives):
            # Create SimpleQADI instance for this perspective
            orchestrator = SimpleQADIOrchestrator(
                llm_manager=self.llm_manager,
                model_config=self.model_config,
                num_hypotheses=3
            )

            # Augment question with perspective
            perspective_question = f"From a {perspective} perspective: {question}"

            # Run QADI cycle
            result = await orchestrator.run_qadi_cycle(perspective_question)

            # Calculate relevance score
            relevance = 1.0 if idx == 0 else max(0.1, 0.8 - idx * 0.1)

            perspective_results.append(PerspectiveResult(
                perspective=perspective,
                qadi_result=result,
                relevance_score=relevance
            ))

        # Synthesize across perspectives
        synthesis = await self._synthesize_perspectives(perspective_results)

        return MultiPerspectiveResult(
            question=question,
            perspectives=perspective_results,
            unified_synthesis=synthesis,
            total_cost=sum(r.qadi_result.llm_cost for r in perspective_results)
        )
```

**Result:** ~200 lines (down from 602)

**Benefits:**
- Eliminates duplication
- Leverages SimpleQADI improvements automatically
- Clear separation: SimpleQADI = single analysis, Multi = coordination

---

### 10. Remove Legacy Orchestrators (-724 lines) ⏱️ 2 hours

**Files to Delete:**
1. `enhanced_orchestrator.py` (201 lines) - Answer extraction wrapper
2. `robust_orchestrator.py` (306 lines) - Timeout handling wrapper
3. `fast_orchestrator.py` (217 lines) - Parallel execution wrapper

**Verification:**
```bash
# Check for usage
uv run grep -r "EnhancedQADIOrchestrator" src/
uv run grep -r "RobustQADIOrchestrator" src/
uv run grep -r "FastQADIOrchestrator" src/

# If none found, safe to delete
```

**Migration Notes:**
- Enhanced → Use `answer_extractor.py` directly if needed
- Robust → Timeout handling now in `base_orchestrator.py`
- Fast → Parallel execution built into phase_logic

**Success Criteria:**
- ✅ Files deleted
- ✅ No import errors
- ✅ All tests pass
- ✅ Documentation updated

---

## Phase 3: Architecture Consolidation (7-10 days, Higher Risk)

### 11. Create Unified Orchestrator ⏱️ 3 days

**New File:** `src/mad_spark_alt/core/unified_orchestrator.py`

**Goal:** Single orchestrator class with configuration-based behavior.

**Structure:**
```python
from .orchestrator_config import OrchestratorConfig, ExecutionMode, Strategy


class UnifiedQADIOrchestrator:
    """
    Single orchestrator supporting all execution modes and strategies
    via configuration
    """

    def __init__(
        self,
        llm_manager: LLMManager,
        config: OrchestratorConfig
    ):
        self.llm_manager = llm_manager
        self.config = config
        self.total_cost = 0.0

    async def run_analysis(self, question: str) -> QADIResult:
        """
        Main entry point - behavior determined by config
        """
        # Select strategy
        if self.config.strategy == Strategy.MULTI_PERSPECTIVE:
            return await self._run_multi_perspective(question)
        elif self.config.strategy == Strategy.SMART:
            return await self._run_smart(question)
        else:  # Strategy.SIMPLE
            return await self._run_simple(question)

    async def _run_simple(self, question: str) -> QADIResult:
        """Simple QADI cycle (default)"""
        if self.config.execution_mode == ExecutionMode.PARALLEL:
            return await self._run_simple_parallel(question)
        else:
            return await self._run_simple_sequential(question)

    async def _run_multi_perspective(self, question: str) -> QADIResult:
        """Multi-perspective analysis"""
        perspectives = self.config.perspectives or ["technical", "business"]
        # Delegate to multi-perspective logic
        ...

    async def _run_smart(self, question: str) -> QADIResult:
        """Smart agent selection"""
        # Detect question type
        # Select appropriate agents
        # Run cycle
        ...
```

**Benefits:**
- Single entry point for all QADI operations
- Configuration makes behavior explicit
- Easy to add new strategies
- No more proliferation of orchestrator classes

---

### 12. Create Orchestrator Configuration ⏱️ 1 day

**New File:** `src/mad_spark_alt/core/orchestrator_config.py`

**Structure:**
```python
from dataclasses import dataclass
from enum import Enum
from typing import Optional, List


class ExecutionMode(Enum):
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"


class Strategy(Enum):
    SIMPLE = "simple"
    SMART = "smart"
    MULTI_PERSPECTIVE = "multi_perspective"


@dataclass
class TimeoutConfig:
    phase_timeout: float = 90.0
    total_timeout: float = 900.0
    enable_retry: bool = True
    max_retries: int = 3


@dataclass
class OrchestratorConfig:
    """Configuration for UnifiedQADIOrchestrator behavior"""

    # Execution
    execution_mode: ExecutionMode = ExecutionMode.SEQUENTIAL
    strategy: Strategy = Strategy.SIMPLE

    # QADI Parameters
    num_hypotheses: int = 3
    enable_scoring: bool = True

    # Multi-Perspective
    perspectives: Optional[List[str]] = None
    auto_detect_perspectives: bool = False

    # Enhancements
    enable_answer_extraction: bool = False
    enable_robust_timeout: bool = True
    timeout_config: TimeoutConfig = TimeoutConfig()

    # Model
    model_config: Optional[ModelConfig] = None

    def validate(self) -> None:
        """Validate configuration consistency"""
        if self.strategy == Strategy.MULTI_PERSPECTIVE:
            if not self.perspectives and not self.auto_detect_perspectives:
                raise ValueError("Multi-perspective requires perspectives or auto-detect")

        if self.num_hypotheses < 1:
            raise ValueError("num_hypotheses must be >= 1")

    @classmethod
    def simple_config(cls) -> "OrchestratorConfig":
        """Factory: Simple sequential QADI"""
        return cls(
            execution_mode=ExecutionMode.SEQUENTIAL,
            strategy=Strategy.SIMPLE,
            num_hypotheses=3
        )

    @classmethod
    def fast_config(cls) -> "OrchestratorConfig":
        """Factory: Parallel execution for speed"""
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
        """Factory: Multi-perspective analysis"""
        return cls(
            execution_mode=ExecutionMode.SEQUENTIAL,
            strategy=Strategy.MULTI_PERSPECTIVE,
            perspectives=perspectives,
            num_hypotheses=3
        )
```

**Usage Example:**
```python
# Simple usage
config = OrchestratorConfig.simple_config()
orchestrator = UnifiedQADIOrchestrator(llm_manager, config)

# Advanced usage
config = OrchestratorConfig(
    execution_mode=ExecutionMode.PARALLEL,
    strategy=Strategy.MULTI_PERSPECTIVE,
    perspectives=["technical", "business", "environmental"],
    num_hypotheses=5,
    enable_answer_extraction=True
)
orchestrator = UnifiedQADIOrchestrator(llm_manager, config)
```

---

### 13. Split semantic_operators.py (1,926 → 4 files) ⏱️ 3 days

**Current:** 1,926 lines in single file

**Target Structure:**
```
src/mad_spark_alt/evolution/
├── mutation_operators.py (~600 lines)
│   ├── SemanticMutationOperator
│   ├── BreakthroughMutationOperator
│   ├── _generate_mutation_prompt()
│   └── _parse_mutation_response()
│
├── crossover_operators.py (~400 lines)
│   ├── SemanticCrossoverOperator
│   ├── _generate_crossover_prompt()
│   └── _parse_crossover_response()
│
├── batch_operations.py (~500 lines)
│   ├── BatchSemanticMutationOperator
│   ├── BatchSemanticCrossoverOperator
│   └── Batch processing logic
│
└── operator_cache.py (~400 lines)
    ├── OperatorCache
    ├── Cache management
    └── Cache statistics
```

**Benefits:**
- Easier to navigate and understand
- Clearer responsibility boundaries
- Faster test runs (can test modules independently)
- Reduces merge conflicts

**Risk:** High (complex evolution logic with caching)

**Mitigation:**
1. Comprehensive tests BEFORE splitting
2. Split incrementally (one module at a time)
3. Keep original file until all tests pass
4. Run extensive evolution tests after split

---

### 14. Deprecate Old Orchestrators ⏱️ 1 day

**Files to Deprecate:**
1. `orchestrator.py` (279 lines) - Base QADI
2. `smart_orchestrator.py` (953 lines) - Smart agent selection

**Process:**
```python
# Add to top of each file
import warnings

warnings.warn(
    "orchestrator.py is deprecated and will be removed in v3.0.0. "
    "Use UnifiedQADIOrchestrator with OrchestratorConfig instead.",
    DeprecationWarning,
    stacklevel=2
)
```

**Update Imports:**
```python
# Before
from mad_spark_alt.core import QADIOrchestrator

# After
from mad_spark_alt.core import UnifiedQADIOrchestrator, OrchestratorConfig
orchestrator = UnifiedQADIOrchestrator(llm_manager, OrchestratorConfig.simple_config())
```

**Success Criteria:**
- ✅ Deprecation warnings added
- ✅ Migration guide in docs
- ✅ All CLI commands use UnifiedQADIOrchestrator
- ✅ Scheduled removal: v3.0.0

---

## Implementation Order & Dependencies

```
Phase 1 (Parallel Execution ✅):
├─ 1. Remove deprecated code          [No deps]
├─ 2. Fix CLI evaluators              [No deps]
├─ 3. Verify Gemini API               [No deps]
└─ 4. Consolidate JSON                [No deps]

Phase 2 (Sequential ⚠️):
├─ 5. Create parsing_utils.py         ⬅ Depends on: #4
├─ 6. Create phase_logic.py           ⬅ Depends on: #5
├─ 7. Create base_orchestrator.py     ⬅ Depends on: #6
├─ 8. Refactor SimpleQADI              ⬅ Depends on: #7 (HIGH RISK)
├─ 9. Refactor MultiPerspective       ⬅ Depends on: #8
└─ 10. Remove legacy orchestrators    ⬅ Depends on: #9

Phase 3 (Mixed ⚠️):
├─ 11. Create unified_orchestrator.py ⬅ Depends on: Phase 2 complete
├─ 12. Create orchestrator_config.py  [Can be parallel with #11]
├─ 13. Split semantic_operators.py    [Independent, parallel OK]
└─ 14. Deprecate old orchestrators    ⬅ Depends on: #11
```

---

## Risk Mitigation Strategies

### 1. Write Integration Tests First (TDD)

**Critical for Item #8 (SimpleQADI Refactoring)**

Before any refactoring:
```python
# tests/test_simple_qadi_integration.py
@pytest.mark.asyncio
async def test_full_qadi_cycle_baseline():
    """Baseline behavior of current SimpleQADI"""
    orchestrator = SimpleQADIOrchestrator(...)
    result = await orchestrator.run_qadi_cycle("Test question")

    # Save as golden output
    save_golden_output("qadi_baseline.json", result)


@pytest.mark.asyncio
async def test_refactored_matches_baseline():
    """Verify refactored version matches baseline"""
    orchestrator = SimpleQADIOrchestrator(...)  # New implementation
    result = await orchestrator.run_qadi_cycle("Test question")

    baseline = load_golden_output("qadi_baseline.json")
    assert_results_equivalent(result, baseline)
```

### 2. Feature Flags for New Orchestrator

```python
# Environment variable toggle
USE_UNIFIED_ORCHESTRATOR = os.getenv("USE_UNIFIED_ORCHESTRATOR", "false").lower() == "true"

if USE_UNIFIED_ORCHESTRATOR:
    orchestrator = UnifiedQADIOrchestrator(llm_manager, config)
else:
    orchestrator = SimpleQADIOrchestrator(llm_manager, model_config)
```

**Benefits:**
- Gradual rollout
- Easy rollback if issues found
- A/B testing in production

### 3. Parallel Development Branches

```bash
# Phase 1
git checkout -b refactor/phase1-cleanup

# Phase 2
git checkout -b refactor/phase2-modular

# Phase 3
git checkout -b refactor/phase3-unified
```

Merge incrementally:
- Merge Phase 1 items as they complete (don't wait for all 4)
- Merge Phase 2 after extensive testing
- Merge Phase 3 with feature flags enabled

### 4. Comprehensive Test Coverage Goals

| Component | Current | Target | Strategy |
|-----------|---------|--------|----------|
| JSON Utils | ~60% | 95% | Edge cases (ANSI, nested, markdown) |
| Parsing Utils | New | 90% | Both structured and regex paths |
| Phase Logic | New | 85% | Mock LLM calls |
| Base Orchestrator | New | 80% | Integration tests |
| SimpleQADI | ~50% | 90% | End-to-end + golden outputs |
| Evolution Ops | ~70% | 85% | More cache tests |

---

## Expected Outcomes

### Quantitative Improvements

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| **Total Lines** | ~12,000 | ~8,500 | **-29%** |
| **Orchestrator Files** | 7 | 1 unified + 1 base | **-71%** |
| **JSON Utilities** | 2 modules | 1 module | **-50%** |
| **Deprecated Code** | 1,251 lines | 0 lines | **-100%** |
| **Largest File** | 1,926 lines | ~600 lines | **-69%** |
| **God Objects** | 2 (1,926 + 1,296) | 0 | **-100%** |
| **Test Coverage** | ~65% | ~90% | **+38%** |
| **CLI Bugs** | 1 (evaluators) | 0 | **-100%** |

### Qualitative Improvements

**Maintainability:**
- ✅ Single source of truth for JSON parsing
- ✅ Clear separation of concerns (parsing, logic, orchestration)
- ✅ No duplicate code across orchestrators
- ✅ Easy to add new QADI strategies

**Testability:**
- ✅ Phase logic testable in isolation
- ✅ Parsing utilities have comprehensive edge case tests
- ✅ Orchestration separated from implementation
- ✅ Mock-friendly interfaces

**Developer Experience:**
- ✅ Clear entry point (UnifiedQADIOrchestrator)
- ✅ Configuration makes behavior explicit
- ✅ Smaller files easier to navigate
- ✅ Consistent patterns across codebase

**User Experience:**
- ✅ CLI --evaluators flag works as documented
- ✅ More reliable JSON parsing (fewer fallbacks)
- ✅ Better structured output from Gemini (if fix needed)
- ✅ No breaking changes during transition

---

## Timeline & Resource Allocation

### Phase 1: Quick Wins (2-3 days)
- **Day 1:** Items 1-2 (remove deprecated, fix CLI)
- **Day 2:** Item 3 (verify/fix Gemini)
- **Day 3:** Item 4 (consolidate JSON)

### Phase 2: Core Refactoring (5-7 days)
- **Days 1-2:** Items 5-6 (parsing_utils, phase_logic)
- **Days 3-4:** Item 7 (base_orchestrator)
- **Days 5-6:** Item 8 (refactor SimpleQADI) ⚠️ HIGH RISK
- **Day 7:** Items 9-10 (refactor multi-perspective, remove legacy)

### Phase 3: Architecture Consolidation (7-10 days)
- **Days 1-3:** Item 11 (unified orchestrator)
- **Day 4:** Item 12 (orchestrator config)
- **Days 5-7:** Item 13 (split semantic_operators) ⚠️ HIGH RISK
- **Day 8:** Item 14 (deprecate old orchestrators)
- **Days 9-10:** Buffer for testing, documentation, issues

**Total Timeline:** 14-20 days

---

## Success Criteria

### Phase 1 Complete When:
- ✅ All deprecated code removed (1,251 lines)
- ✅ CLI --evaluators flag functional with tests
- ✅ Gemini structured output verified/fixed
- ✅ Single JSON utility with comprehensive tests
- ✅ All existing tests pass
- ✅ No regressions in CLI behavior

### Phase 2 Complete When:
- ✅ parsing_utils.py has 90%+ test coverage
- ✅ phase_logic.py has 85%+ test coverage
- ✅ base_orchestrator.py has 80%+ test coverage
- ✅ SimpleQADI reduced to ~400 lines
- ✅ MultiPerspective reduced to ~200 lines
- ✅ Legacy orchestrators removed (724 lines)
- ✅ All integration tests pass
- ✅ CLI produces identical outputs

### Phase 3 Complete When:
- ✅ UnifiedQADIOrchestrator handles all strategies
- ✅ OrchestratorConfig with factory methods
- ✅ semantic_operators split into 4 focused modules
- ✅ Old orchestrators deprecated with warnings
- ✅ Migration guide written
- ✅ All tests pass with 90%+ coverage
- ✅ Documentation updated
- ✅ Feature flags in place for gradual rollout

---

## Rollback Plan

### If Phase 1 Issues:
- **Item 1 (Deprecated code):** Restore from git
- **Item 2 (CLI evaluators):** Revert PR, keep flag non-functional
- **Item 3 (Gemini):** Revert to original payload structure
- **Item 4 (JSON):** Keep both utilities, remove deprecation

### If Phase 2 Issues:
- **Item 8 (SimpleQADI):** Feature flag to old implementation
- **Other items:** Revert individual PRs (independent changes)

### If Phase 3 Issues:
- **UnifiedQADI:** Feature flag to SimpleQADI
- **Split semantic_operators:** Keep original file until stable

---

## Next Steps

1. **Get stakeholder approval** on this plan
2. **Set up feature flag infrastructure** (environment variables)
3. **Create baseline integration tests** for SimpleQADI
4. **Start Phase 1, Item 1** (remove deprecated code)
5. **Track progress** with daily updates

---

## References

### Related Documents
- [ARCHITECTURE.md](ARCHITECTURE.md) - Current system architecture
- [DEVELOPMENT.md](DEVELOPMENT.md) - Development guidelines
- [CLAUDE.md](CLAUDE.md) - AI assistant instructions

### Key Files for Review
- `src/mad_spark_alt/core/simple_qadi_orchestrator.py` (1,296 lines)
- `src/mad_spark_alt/evolution/semantic_operators.py` (1,926 lines)
- `src/mad_spark_alt/core/json_utils.py` (428 lines)
- `src/mad_spark_alt/core/robust_json_handler.py` (194 lines)
- `src/mad_spark_alt/cli.py` (864 lines)

### Review Scores
- Review 1: 7.5/10 (accurate but one incorrect finding)
- Review 2: 6/10 (good vision, lacks detail)
- Review 3: 8.5/10 (excellent categorization)
- Review 4: 9/10 (best implementation approach) ⭐

---

**End of Refactoring Plan**
