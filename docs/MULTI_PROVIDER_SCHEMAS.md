# Multi-Provider Pydantic Schemas

## Overview

This document describes the Pydantic-based schema architecture that enables Mad Spark Alt to work seamlessly across multiple LLM providers (Google Gemini, OpenAI, Anthropic, and local LLMs).

## Background

### Gemini API Update (January 2025)

Google announced enhanced Structured Outputs support with expanded JSON Schema features:

- Standard JSON Schema format (lowercase types: "object", "string", "number", "array")
- Property ordering preservation for Gemini 2.5+
- Support for `anyOf`, `$ref`, `minimum/maximum`, `additionalProperties`
- Support for `type: 'null'`, `prefixItems`, and nested schemas

### Migration Rationale

**Before**: Manual dict schemas with Gemini-specific OpenAPI 3.0 format (uppercase "OBJECT", "STRING")
```python
schema = {
    "type": "OBJECT",
    "properties": {
        "hypotheses": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "id": {"type": "STRING"},
                    "content": {"type": "STRING"}
                }
            }
        }
    }
}
```

**After**: Pydantic models generating standard JSON Schema (universal compatibility)
```python
class HypothesisListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hypotheses: List[Hypothesis]

# Automatically generates standard JSON Schema:
# {"type": "object", "properties": {"hypotheses": {"type": "array", ...}}}
```

## Schema Architecture

### Core Pydantic Models

Located in `src/mad_spark_alt/core/schemas.py`:

1. **QADI Phase Schemas**
   - `HypothesisScores`: Score validation (0.0-1.0 range)
   - `Hypothesis`: Single hypothesis with ID and content
   - `HypothesisEvaluation`: Hypothesis with scores
   - `DeductionResponse`: Complete deduction phase output
   - `HypothesisListResponse`: Abduction phase output

2. **Evolution Operator Schemas**
   - `MutationResponse`: Single mutation result
   - `BatchMutationResponse`: Batch mutation results
   - `CrossoverResponse`: Single crossover result
   - `BatchCrossoverResponse`: Batch crossover results

### Key Features

#### 1. Automatic Score Validation

```python
class HypothesisScores(BaseModel):
    model_config = ConfigDict(extra="forbid")

    impact: float = Field(ge=0.0, le=1.0, description="Impact score (0.0-1.0)")
    feasibility: float = Field(ge=0.0, le=1.0, description="Feasibility score (0.0-1.0)")
    accessibility: float = Field(ge=0.0, le=1.0, description="Accessibility score (0.0-1.0)")
    sustainability: float = Field(ge=0.0, le=1.0, description="Sustainability score (0.0-1.0)")
    scalability: float = Field(ge=0.0, le=1.0, description="Scalability score (0.0-1.0)")
```

**Benefits**:
- Gemini API enforces constraints server-side
- Invalid scores rejected at API level
- Clear error messages from Pydantic validation

#### 2. Strict Validation

```python
model_config = ConfigDict(extra="forbid")
```

**Benefits**:
- Catches LLM hallucinations (extra fields rejected)
- Prevents schema drift over time
- Clear error messages when LLM deviates from schema

#### 3. Property Ordering Preservation

Pydantic v2 preserves field order by default, which is important for Gemini 2.5+ consistency.

```python
class DeductionResponse(BaseModel):
    evaluations: List[HypothesisEvaluation]  # First in schema
    answer: str                               # Second
    action_plan: List[str]                    # Third
```

#### 4. Schema Reusability via Nested Models

```python
class HypothesisEvaluation(BaseModel):
    hypothesis_id: str
    scores: HypothesisScores  # Reusable nested model

class DeductionResponse(BaseModel):
    evaluations: List[HypothesisEvaluation]  # Uses $ref in JSON Schema
    answer: str
    action_plan: List[str]
```

## Multi-Provider Support

### Schema Conversion Utilities

Located in `src/mad_spark_alt/core/schema_utils.py`:

```python
def to_gemini_schema(pydantic_model: Type[BaseModel]) -> Dict[str, Any]:
    """Convert Pydantic model to Gemini-compatible JSON Schema."""
    return pydantic_model.model_json_schema()

def to_openai_schema(pydantic_model: Type[BaseModel]) -> Dict[str, Any]:
    """Convert Pydantic model to OpenAI-compatible JSON Schema."""
    return pydantic_model.model_json_schema()

def to_anthropic_schema(pydantic_model: Type[BaseModel]) -> Dict[str, Any]:
    """Convert Pydantic model to Anthropic-compatible JSON Schema."""
    return pydantic_model.model_json_schema()

def to_standard_json_schema(pydantic_model: Type[BaseModel]) -> Dict[str, Any]:
    """Convert Pydantic model to standard JSON Schema."""
    return pydantic_model.model_json_schema()
```

### LLM Provider Integration

Updated `LLMRequest` to accept both Pydantic models and dict schemas:

```python
@dataclass
class LLMRequest:
    user_prompt: str
    response_schema: Optional[Union[Dict[str, Any], type]] = None
    response_mime_type: Optional[str] = None

    def get_json_schema(self) -> Optional[Dict[str, Any]]:
        """Convert response_schema to JSON Schema format."""
        if self.response_schema is None:
            return None
        if isinstance(self.response_schema, type) and issubclass(self.response_schema, BaseModel):
            return self.response_schema.model_json_schema()
        return self.response_schema
```

**Backward Compatibility**: Existing code using dict schemas continues to work without changes.

## Usage Examples

### 1. Using Pydantic Models in LLM Requests

```python
from mad_spark_alt.core.schemas import DeductionResponse
from mad_spark_alt.core.llm_provider import LLMRequest

request = LLMRequest(
    user_prompt="Evaluate these hypotheses...",
    response_schema=DeductionResponse,  # Pydantic model class
    response_mime_type="application/json"
)

# LLMRequest automatically converts to JSON Schema
response = await llm_provider.generate(request)
```

### 2. Parsing Validated Responses

```python
# Parse response with automatic validation
result = DeductionResponse.model_validate_json(response.content)

# Access validated data with type safety
for eval in result.evaluations:
    print(f"Hypothesis {eval.hypothesis_id}:")
    print(f"  Impact: {eval.scores.impact}")
    print(f"  Feasibility: {eval.scores.feasibility}")
```

### 3. Handling Validation Errors

```python
from pydantic import ValidationError

try:
    result = DeductionResponse.model_validate_json(response_text)
except ValidationError as e:
    # e.errors() provides detailed error information
    logger.error(f"Validation failed: {e.errors()}")
    # Fall back to text parsing or default values
```

## Adding New Providers

To add support for a new LLM provider:

1. **Check if standard JSON Schema works** (most providers accept it):
   ```python
   schema = MyPydanticModel.model_json_schema()
   ```

2. **If provider needs custom format**, add conversion function:
   ```python
   def to_newprovider_schema(pydantic_model: Type[BaseModel]) -> Dict[str, Any]:
       base_schema = pydantic_model.model_json_schema()
       # Apply provider-specific transformations
       return transformed_schema
   ```

3. **Update provider class** to use `get_json_schema()`:
   ```python
   class NewProvider(LLMProviderInterface):
       async def generate(self, request: LLMRequest) -> LLMResponse:
           schema = request.get_json_schema()
           if schema:
               # Use schema in API call
               ...
   ```

## Testing Strategy

### 1. Schema Model Tests (`test_schemas.py`)

- JSON Schema generation correctness
- Score range validation (0.0-1.0)
- Strict validation (extra fields rejected)
- Nested schema reusability
- Property ordering preservation

### 2. LLM Provider Integration Tests (`test_llm_provider_pydantic.py`)

- Pydantic model acceptance in LLMRequest
- Automatic conversion to JSON Schema
- Backward compatibility with dict schemas
- Multi-provider schema format generation

### 3. QADI Phase Tests (`test_phase_logic_pydantic.py`)

- Schema generation for each phase
- Validation with valid responses
- Fallback behavior on validation errors
- Backward compatibility with existing parsing

### 4. Integration Tests with Real API

```bash
# Test with real Gemini API
GOOGLE_API_KEY=xxx pytest tests/test_multi_provider_schemas.py::test_real_api -v
```

## Troubleshooting

### Issue: "Extra inputs not permitted"

**Cause**: LLM returned fields not in schema
**Solution**: Check if LLM prompt and schema match exactly

### Issue: "Input should be less than or equal to 1"

**Cause**: LLM returned score > 1.0
**Solution**: Update prompt to emphasize 0.0-1.0 range, or adjust Field constraints

### Issue: "Field required"

**Cause**: LLM didn't return all required fields
**Solution**: Make field optional or improve prompt clarity

### Issue: Schema not enforced by API

**Cause**: Using old Gemini format or wrong API configuration
**Solution**: Verify using `responseMimeType: "application/json"` and `responseJsonSchema` (not `responseSchema`)

## Design Decisions

### 1. Union Type for Backward Compatibility

**Decision**: `response_schema: Optional[Union[Dict[str, Any], type]]`

**Rationale**:
- Gradual migration (not breaking change)
- Existing dict schemas continue working
- New code can use Pydantic models

### 2. Explicit Conversion Method

**Decision**: `get_json_schema()` method vs automatic conversion

**Rationale**:
- Clear conversion point for debugging
- Allows future caching optimization
- Explicit > implicit

### 3. Standard JSON Schema Output

**Decision**: Output lowercase "object", "string", etc.

**Rationale**:
- Gemini API update accepts standard format
- OpenAI, Anthropic use standard format
- Future-proof for new providers

### 4. Strict Validation Default

**Decision**: `ConfigDict(extra="forbid")` on all models

**Rationale**:
- Catch LLM hallucinations early
- Prevent schema drift
- Clear error messages

## Benefits Summary

1. **Type Safety**: IDE autocomplete and type checking for LLM responses
2. **Automatic Validation**: Gemini API enforces constraints server-side
3. **Multi-Provider Compatibility**: Standard JSON Schema works everywhere
4. **Clear Error Messages**: Pydantic provides detailed validation errors
5. **Schema Reusability**: Define once, use across multiple operations
6. **Future-Proof**: Ready for new JSON Schema features (anyOf, $ref, etc.)
7. **Backward Compatible**: Existing dict schemas continue working

## Migration Checklist

For migrating other parts of the codebase:

- [ ] Identify manual dict schemas
- [ ] Create equivalent Pydantic models in `schemas.py`
- [ ] Update function signatures to use Pydantic models
- [ ] Add Pydantic validation tests
- [ ] Update parsing logic to use `model_validate_json()`
- [ ] Test with real API to verify compatibility
- [ ] Update documentation

## References

- [Pydantic v2 Documentation](https://docs.pydantic.dev/latest/)
- [JSON Schema Specification](https://json-schema.org/)
- [Gemini API Structured Outputs](https://ai.google.dev/gemini-api/docs/structured-output)
- [OpenAI Function Calling](https://platform.openai.com/docs/guides/function-calling)
- [Anthropic Tool Use](https://docs.anthropic.com/claude/docs/tool-use)
