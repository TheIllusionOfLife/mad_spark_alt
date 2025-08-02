# Structured Output Implementation

This document describes how the Mad Spark Alt system uses Gemini's structured output feature to improve reliability and eliminate parsing failures.

## Overview

The system uses Gemini's `responseSchema` and `responseMimeType` parameters to ensure LLM responses follow a specific JSON structure. This eliminates the need for complex regex parsing and provides consistent, reliable data extraction.

## Implementation Details

### Core Infrastructure

The LLM provider infrastructure already supports structured output:

```python
# In LLMRequest (llm_provider.py)
class LLMRequest(BaseModel):
    response_schema: Optional[Dict[str, Any]] = None
    response_mime_type: Optional[str] = None
```

The GoogleProvider automatically includes these in the generation config when provided:

```python
if request.response_schema and request.response_mime_type:
    generation_config["responseMimeType"] = request.response_mime_type
    generation_config["responseSchema"] = request.response_schema
```

### Schema Definitions

#### Hypothesis Generation Schema

Used in the abduction phase to generate structured hypotheses:

```python
def get_hypothesis_generation_schema() -> Dict[str, Any]:
    return {
        "type": "OBJECT",
        "properties": {
            "hypotheses": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "id": {"type": "STRING"},
                        "content": {"type": "STRING"},
                    },
                    "required": ["id", "content"]
                }
            }
        },
        "required": ["hypotheses"]
    }
```

#### Deduction (Score Evaluation) Schema

Used to evaluate hypotheses with consistent scoring:

```python
def get_deduction_schema() -> Dict[str, Any]:
    return {
        "type": "OBJECT",
        "properties": {
            "evaluations": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "hypothesis_id": {"type": "STRING"},
                        "scores": {
                            "type": "OBJECT",
                            "properties": {
                                "impact": {"type": "NUMBER"},
                                "feasibility": {"type": "NUMBER"},
                                "accessibility": {"type": "NUMBER"},
                                "sustainability": {"type": "NUMBER"},
                                "scalability": {"type": "NUMBER"}
                            },
                            "required": ["impact", "feasibility", "accessibility", 
                                       "sustainability", "scalability"]
                        }
                    },
                    "required": ["hypothesis_id", "scores"]
                }
            }
        },
        "required": ["evaluations"]
    }
```

#### Semantic Operator Schemas

For evolution operations:

```python
# Mutation Schema
{
    "type": "OBJECT",
    "properties": {
        "mutations": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "idea_id": {"type": "INTEGER"},
                    "mutated_content": {"type": "STRING"}
                },
                "required": ["idea_id", "mutated_content"]
            }
        }
    },
    "required": ["mutations"]
}

# Crossover Schema
{
    "type": "OBJECT",
    "properties": {
        "offspring_1": {"type": "STRING"},
        "offspring_2": {"type": "STRING"}
    },
    "required": ["offspring_1", "offspring_2"]
}
```

## Benefits

1. **Reliability**: Structured output guarantees consistent JSON format
2. **Simplicity**: No complex regex patterns to maintain
3. **Performance**: Single parsing attempt instead of multiple fallbacks
4. **Clarity**: Schema serves as documentation for expected format
5. **Error Reduction**: Eliminates parsing failures that default scores to 0.5

## Fallback Mechanisms

While structured output is highly reliable, the system maintains fallback parsing for robustness:

1. **Primary**: Parse structured JSON response
2. **Fallback**: Use regex patterns for common text formats
3. **Last Resort**: Extract meaningful content from unstructured text

This layered approach ensures the system continues to function even if structured output fails.

## Testing

Comprehensive integration tests verify structured output functionality:

```bash
# Run structured output tests
uv run pytest tests/test_structured_output_integration.py -v

# Run with real API (requires GOOGLE_API_KEY)
GOOGLE_API_KEY=your_key uv run pytest tests/test_structured_output_integration.py::TestRealLLMStructuredOutput -v
```

## Troubleshooting

### Common Issues

1. **Schema Too Complex**: Gemini may return errors for overly complex schemas
   - Solution: Simplify schema, use shorter property names
   
2. **Missing Required Fields**: LLM may omit required fields
   - Solution: Fallback parsing handles this gracefully
   
3. **Type Mismatches**: LLM returns string instead of number
   - Solution: Type coercion in parsing logic

### Debugging

Enable debug logging to see structured output details:

```python
import logging
logging.getLogger('mad_spark_alt.core.simple_qadi_orchestrator').setLevel(logging.DEBUG)
```

## Future Improvements

1. **Schema Validation**: Add pydantic models for schema validation
2. **Error Metrics**: Track structured output success rates
3. **Schema Evolution**: Version schemas for backward compatibility
4. **Provider Support**: Extend to other LLM providers that support structured output