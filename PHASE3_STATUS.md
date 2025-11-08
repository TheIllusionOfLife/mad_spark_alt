# Phase 3: QADI Orchestrator Integration - Implementation Status

**Branch**: `feature/phase3-orchestrator-multimodal`
**Last Updated**: 2025-11-08
**Status**: ~40% Complete (Core foundations done, orchestrators and CLI pending)

---

## ✅ Completed Components

### 1. Core Data Structures (100% Complete)
**Commit**: `840641a` - test: add tests for PhaseInput multimodal support and validation

**Changes**:
- ✅ Extended `PhaseInput` with multimodal fields (lines 159-161):
  - `multimodal_inputs: Optional[List["MultimodalInput"]] = None`
  - `urls: Optional[List[str]] = None`
  - `tools: Optional[List[Dict[str, Any]]] = None`

- ✅ Added `multimodal_metadata` to all phase result dataclasses:
  - `QuestioningResult` (line 191)
  - `AbductionResult` (line 203)
  - `DeductionResult` (line 216)
  - `InductionResult` (line 227)

- ✅ Validation function implemented (lines 235-266):
  - `_validate_multimodal_inputs()` validates inputs and URLs
  - Checks file sizes, MIME types, page counts, URL formats
  - Enforces Gemini limits (20 URLs max, 1000 pages max, etc.)

- ✅ TYPE_CHECKING import pattern for circular import avoidance

**Tests**: 24 tests passing
- `tests/core/test_phase_input_multimodal.py` (12 tests)
- `tests/core/test_multimodal_validation.py` (12 tests)

---

### 2. Phase Logic Functions (100% Complete)
**Commit**: `38b0794` - feat: add multimodal support to all QADI phase execution functions

**Changes**:
All 4 phase functions updated with multimodal support:

#### `execute_questioning_phase()` (lines 274-375)
- ✅ Validation call at start (lines 290-292)
- ✅ Multimodal context added to prompts (lines 300-304)
- ✅ LLMRequest includes multimodal fields (lines 313-321)
- ✅ Metadata extraction from response (lines 327-333)
- ✅ All return statements include `multimodal_metadata`

#### `execute_abduction_phase()` (lines 384-519)
- ✅ Validation call (lines 408-410)
- ✅ Multimodal context in prompts (lines 417-421)
- ✅ LLMRequest with multimodal fields (lines 436-446)
- ✅ Metadata extraction (lines 452-458)
- ✅ All returns (success + failure cases) include metadata

#### `execute_deduction_phase()` (lines 527-748)
- ✅ Validation call (lines 547-549)
- ✅ Multimodal context in prompts (lines 564-568)
- ✅ LLMRequest with multimodal fields (lines 580-590)
- ✅ Metadata extraction (lines 597-603)
- ✅ Both structured + fallback returns include metadata

#### `execute_induction_phase()` (lines 751-950)
- ✅ Validation call (lines 775-777)
- ✅ Multimodal context in prompts (lines 784-788)
- ✅ LLMRequest with multimodal fields (lines 798-806)
- ✅ Metadata extraction (lines 813-819)
- ✅ Return includes metadata (line 931)

**Tests**: All 892 existing tests passing (zero regressions)

---

## 🚧 Remaining Work

### 3. Orchestrator Updates (0% Complete) - NEXT PRIORITY

#### Files to Modify:
1. `src/mad_spark_alt/core/base_orchestrator.py`
2. `src/mad_spark_alt/core/simple_qadi_orchestrator.py`
3. `src/mad_spark_alt/core/multi_perspective_orchestrator.py`
4. `src/mad_spark_alt/core/unified_orchestrator.py`

#### Required Changes:

**BaseOrchestrator** (`base_orchestrator.py:80-469`):
```python
# Line ~137-157: Update abstract method signature
async def run_qadi_cycle(
    self,
    problem_statement: str,
    context: Optional[str] = None,
    cycle_config: Optional[Dict[str, Any]] = None,
    multimodal_inputs: Optional[List["MultimodalInput"]] = None,  # NEW
    urls: Optional[List[str]] = None,  # NEW
    tools: Optional[List[Dict[str, Any]]] = None,  # NEW
) -> Any:

# Add helper method after line 350
def _build_multimodal_phase_input(
    self,
    base_phase_input: PhaseInput,
    multimodal_inputs: Optional[List["MultimodalInput"]] = None,
    urls: Optional[List[str]] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
) -> PhaseInput:
    """Build PhaseInput with multimodal support."""
    return PhaseInput(
        user_input=base_phase_input.user_input,
        llm_manager=base_phase_input.llm_manager,
        model_config=base_phase_input.model_config,
        context=base_phase_input.context,
        max_retries=base_phase_input.max_retries,
        multimodal_inputs=multimodal_inputs or base_phase_input.multimodal_inputs,
        urls=urls or base_phase_input.urls,
        tools=tools or base_phase_input.tools,
    )
```

**SimpleQADIOrchestrator** (`simple_qadi_orchestrator.py:56-221`):
```python
# Line 35-53: Update SimpleQADIResult dataclass
@dataclass
class SimpleQADIResult:
    # ... existing fields ...
    multimodal_metadata: Dict[str, Any] = field(default_factory=dict)  # NEW
    total_images_processed: int = 0  # NEW
    total_pages_processed: int = 0  # NEW
    total_urls_processed: int = 0  # NEW

# Line 81-86: Update run_qadi_cycle signature
async def run_qadi_cycle(
    self,
    user_input: str,
    context: Optional[str] = None,
    max_retries: int = 2,
    multimodal_inputs: Optional[List["MultimodalInput"]] = None,  # NEW
    urls: Optional[List[str]] = None,  # NEW
    tools: Optional[List[Dict[str, Any]]] = None,  # NEW
) -> SimpleQADIResult:

# Lines 113-193: Pass multimodal data to PhaseInput
phase_input = PhaseInput(
    user_input=full_input,
    llm_manager=llm_manager,
    context={},
    max_retries=max_retries,
    multimodal_inputs=multimodal_inputs,  # NEW
    urls=urls,  # NEW
    tools=tools,  # NEW
)

# Aggregate multimodal metadata from all phases
result.multimodal_metadata = {
    "questioning": questioning_result.multimodal_metadata,
    "abduction": abduction_result.multimodal_metadata,
    "deduction": deduction_result.multimodal_metadata,
    "induction": induction_result.multimodal_metadata,
}
result.total_images_processed = sum(...)
result.total_pages_processed = sum(...)
result.total_urls_processed = urls count
```

**MultiPerspectiveQADIOrchestrator** (`multi_perspective_orchestrator.py:62-317`):
```python
# Line 38-59: Update MultiPerspectiveQADIResult
@dataclass
class MultiPerspectiveQADIResult:
    # ... existing fields ...
    multimodal_metadata: Dict[str, Any] = field(default_factory=dict)  # NEW

# Line 83-99: Update run_multi_perspective_analysis signature
async def run_multi_perspective_analysis(
    self,
    user_input: str,
    max_perspectives: int = 3,
    force_perspectives: Optional[List[QuestionIntent]] = None,
    multimodal_inputs: Optional[List["MultimodalInput"]] = None,  # NEW
    urls: Optional[List[str]] = None,  # NEW
    tools: Optional[List[Dict[str, Any]]] = None,  # NEW
) -> MultiPerspectiveQADIResult:

# Line 155-183: Update _run_perspective_analysis
async def _run_perspective_analysis(
    self,
    user_input: str,
    perspective: QuestionIntent,
    multimodal_inputs: Optional[List["MultimodalInput"]] = None,  # NEW
    urls: Optional[List[str]] = None,  # NEW
    tools: Optional[List[Dict[str, Any]]] = None,  # NEW
) -> Optional[SimpleQADIResult]:
    # Pass to SimpleQADI
    result = await orchestrator.run_qadi_cycle(
        perspective_question,
        multimodal_inputs=multimodal_inputs,
        urls=urls,
        tools=tools,
    )

# Line 118-121: Update perspective_tasks loop
for perspective in perspectives:
    task = self._run_perspective_analysis(
        user_input,
        perspective,
        multimodal_inputs=multimodal_inputs,
        urls=urls,
        tools=tools,
    )
```

**UnifiedQADIOrchestrator** (`unified_orchestrator.py:61-200+`):
```python
# Line 30-59: Update UnifiedQADIResult
@dataclass
class UnifiedQADIResult:
    # ... existing fields ...
    multimodal_metadata: Optional[Dict[str, Any]] = None  # NEW

# Line 99-104: Update run_qadi_cycle signature
async def run_qadi_cycle(
    self,
    problem_statement: str,
    context: Optional[str] = None,
    cycle_config: Optional[Dict[str, Any]] = None,
    multimodal_inputs: Optional[List["MultimodalInput"]] = None,  # NEW
    urls: Optional[List[str]] = None,  # NEW
    tools: Optional[List[Dict[str, Any]]] = None,  # NEW
) -> UnifiedQADIResult:

# Propagate through _run_simple_strategy and _run_multi_perspective_strategy
```

---

### 4. CLI Integration (0% Complete)

#### Files to Modify:
- Main CLI entry point (needs investigation - could be `cli.py` or `qadi_simple.py`)

#### Required CLI Options:
```python
@click.option('--image', '-i', multiple=True, type=click.Path(exists=True),
              help='Image file(s) for analysis (PNG, JPEG, WebP, HEIC)')
@click.option('--document', '-d', multiple=True, type=click.Path(exists=True),
              help='Document file(s) (PDF recommended, up to 1000 pages)')
@click.option('--url', '-u', multiple=True,
              help='URL(s) for context retrieval (max 20)')
@click.option('--use-url-context', is_flag=True, default=False,
              help='Enable Gemini URL context tool')
```

#### Implementation Pattern:
```python
def main(..., image, document, url, use_url_context):
    # Build multimodal inputs from CLI options
    multimodal_inputs = []

    for img_path in image:
        # Resolve path, detect MIME, get file size
        multimodal_inputs.append(MultimodalInput(...))

    for doc_path in document:
        # Resolve path, detect MIME, get page count
        multimodal_inputs.append(MultimodalInput(...))

    # Setup tools if URL context enabled
    tools = [{"url_context": {}}] if use_url_context and url else None

    # Pass to orchestrator
    result = await orchestrator.run_qadi_cycle(
        problem_statement,
        multimodal_inputs=multimodal_inputs or None,
        urls=list(url) if url else None,
        tools=tools,
    )

    # Display multimodal processing stats
    if result.multimodal_metadata:
        console.print("\n[bold cyan]📊 Processing Stats[/bold cyan]")
        console.print(f"  Images: {result.total_images_processed}")
        console.print(f"  Pages: {result.total_pages_processed}")
        console.print(f"  URLs: {result.total_urls_processed}")
```

---

### 5. Testing Strategy

#### Integration Tests Needed:
```python
# tests/core/test_phase_logic_multimodal_integration.py
@pytest.mark.asyncio
async def test_questioning_phase_with_multimodal(mock_llm_manager):
    """Test questioning phase passes multimodal data correctly."""
    phase_input = PhaseInput(
        user_input="Analyze this",
        llm_manager=mock_llm_manager,
        multimodal_inputs=[...],
        urls=["https://example.com"],
    )
    result = await execute_questioning_phase(phase_input)
    assert result.multimodal_metadata["images_processed"] == 1

# Similar tests for abduction, deduction, induction

# tests/core/test_orchestrator_multimodal_integration.py
@pytest.mark.asyncio
async def test_simple_qadi_with_image(mock_llm_manager):
    """Test SimpleQADI end-to-end with image."""
    orchestrator = SimpleQADIOrchestrator(...)
    result = await orchestrator.run_qadi_cycle(
        "Describe this",
        multimodal_inputs=[MultimodalInput(...)],
    )
    assert result.hypotheses
    assert result.total_images_processed > 0

# tests/test_cli_multimodal.py
def test_cli_image_option():
    """Test CLI accepts --image option."""
    result = runner.invoke(main, ["Test", "--image", "test.png"])
    assert result.exit_code == 0

# Integration tests with real API
@pytest.mark.integration
@pytest.mark.asyncio
async def test_real_api_with_image():
    """Test with real Gemini API and actual image."""
    # Uses GOOGLE_API_KEY from environment
    # Verifies no timeouts, truncation, or errors
```

---

### 6. Documentation Updates

#### Files to Update:
1. **README.md** - Add multimodal usage examples
2. **CLAUDE.md** - Document Phase 3 patterns
3. **CLI help text** - Include multimodal examples
4. **MULTIMODAL_ORCHESTRATION.md** (new) - Detailed guide

#### Example Documentation:
```markdown
## Multimodal Analysis

Mad Spark Alt now supports multimodal inputs through the Gemini API:

### Basic Usage

```bash
# Analyze an image
uv run mad_spark_alt "What improvements can be made?" --image diagram.png

# Process a PDF document
uv run mad_spark_alt "Summarize key findings" --document paper.pdf

# Fetch URL context
uv run mad_spark_alt "Compare approaches" --url https://example.com/article

# Combine multiple inputs
uv run mad_spark_alt "Analyze competitive landscape" \
  --image product_a.jpg \
  --image product_b.jpg \
  --document market_report.pdf \
  --url https://competitor.com/pricing
```

### API Usage

```python
from mad_spark_alt.core import UnifiedQADIOrchestrator
from mad_spark_alt.core.multimodal import MultimodalInput, MultimodalInputType, MultimodalSourceType

image = MultimodalInput(
    input_type=MultimodalInputType.IMAGE,
    source_type=MultimodalSourceType.FILE_PATH,
    data="/path/to/image.png",
    mime_type="image/png"
)

orchestrator = UnifiedQADIOrchestrator()
result = await orchestrator.run_qadi_cycle(
    "Analyze this system",
    multimodal_inputs=[image]
)
```
```

---

## 🎯 Completion Roadmap

### Remaining Effort Estimate: 2-3 days

**Day 3 (Orchestrators)**:
1. Morning: Update BaseOrchestrator + SimpleQADIOrchestrator
2. Afternoon: Update MultiPerspective + Unified orchestrators
3. Evening: Write orchestrator integration tests

**Day 4 (CLI & Testing)**:
1. Morning: Add CLI options and input processing
2. Afternoon: Write CLI integration tests
3. Evening: Real API testing with images/documents/URLs

**Day 5 (Verification & Documentation)**:
1. Morning: User testing scenarios (follow README examples literally)
2. Afternoon: Update all documentation
3. Evening: Final verification, PR creation

---

## 📝 Testing Checklist

### Unit Tests
- [x] PhaseInput accepts multimodal fields
- [x] Phase result dataclasses include metadata
- [x] Validation function catches invalid inputs
- [ ] Orchestrators accept multimodal parameters
- [ ] Result aggregation includes metadata

### Integration Tests
- [ ] Each phase function with mock LLM
- [ ] SimpleQADI end-to-end with multimodal
- [ ] MultiPerspective propagates multimodal data
- [ ] CLI options work correctly

### Real API Tests
- [ ] Image analysis (no timeout, no truncation)
- [ ] Document processing (PDF up to 100 pages)
- [ ] URL context retrieval (3-5 URLs)
- [ ] Combined multimodal inputs

### User Verification
- [ ] Follow README examples exactly
- [ ] Test various file path types (relative, absolute, ~/)
- [ ] Verify output quality (not placeholder content)
- [ ] Check metadata display accuracy

---

## 🚨 Critical Patterns to Follow

1. **Backward Compatibility**: All multimodal parameters are optional (default to None)
2. **Validation First**: Call `_validate_multimodal_inputs()` before processing
3. **Metadata Tracking**: Extract and aggregate metadata at each level
4. **TYPE_CHECKING Imports**: Avoid circular imports with MultimodalInput
5. **Prompt Context**: Add multimodal context hints to LLM prompts
6. **TDD Approach**: Tests first, implementation second
7. **Zero Regressions**: Run full test suite after each change

---

## 📊 Progress Summary

**Overall**: ~40% Complete

| Component | Status | Tests |
|-----------|--------|-------|
| Data Structures | ✅ 100% | 24/24 passing |
| Phase Logic | ✅ 100% | 892/892 passing |
| Orchestrators | ⏳ 0% | Not started |
| CLI Integration | ⏳ 0% | Not started |
| Documentation | ⏳ 0% | Not started |
| Real API Testing | ⏳ 0% | Not started |

---

## 🔄 How to Continue

```bash
# Ensure you're on the feature branch
git checkout feature/phase3-orchestrator-multimodal

# Check current status
git log --oneline -n 5
# Should show:
# 38b0794 feat: add multimodal support to all QADI phase execution functions
# 840641a test: add tests for PhaseInput multimodal support and validation
# 08bd2b5 Phase 2: Gemini Provider Multimodal Support (#124)

# Continue with orchestrator updates
# 1. Start with BaseOrchestrator
# 2. Then SimpleQADIOrchestrator
# 3. Then MultiPerspective + Unified
# 4. Write tests at each step
# 5. Commit frequently

# Run tests after each change
uv run pytest tests/ -m "not integration" -v

# When complete, test with real API
export GOOGLE_API_KEY="your-key"
uv run pytest tests/ -m "integration" -v -k "multimodal"
```

---

## 📞 Questions or Issues?

If you encounter any issues continuing this work:

1. **Check existing tests**: The test files show expected behavior
2. **Review Phase 1 & 2**: Multimodal foundations are solid
3. **Follow TDD**: Write tests before implementation
4. **Run frequently**: `uv run pytest` after every change

The foundation is solid - orchestrators and CLI are straightforward extensions of the phase logic patterns already established.
