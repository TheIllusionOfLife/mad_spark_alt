"""
Schema Conversion Utilities for Multi-Provider LLM Support

This module provides utilities for converting Pydantic models to provider-specific
schema formats. Currently, most providers (Google Gemini, OpenAI, Anthropic) support
standard JSON Schema, so conversion is straightforward.

Future providers with non-standard formats can add specialized converters here.
"""

from typing import Any, Dict, Type

from pydantic import BaseModel


def to_standard_json_schema(pydantic_model: Type[BaseModel]) -> Dict[str, Any]:
    """
    Convert Pydantic model to standard JSON Schema.

    This is the universal converter that works for any provider
    supporting JSON Schema (which is most modern LLM APIs).

    Args:
        pydantic_model: Pydantic model class to convert

    Returns:
        Standard JSON Schema dict

    Example:
        >>> from mad_spark_alt.core.schemas import HypothesisScores
        >>> schema = to_standard_json_schema(HypothesisScores)
        >>> print(schema["type"])  # "object"
        >>> print(schema["properties"].keys())  # impact, feasibility, ...
    """
    return pydantic_model.model_json_schema()


def to_gemini_schema(pydantic_model: Type[BaseModel]) -> Dict[str, Any]:
    """
    Convert Pydantic model to Gemini API's response_json_schema format.

    As of Gemini API update (2025), Gemini accepts standard JSON Schema,
    so this delegates to to_standard_json_schema().

    Args:
        pydantic_model: Pydantic model class to convert

    Returns:
        Standard JSON Schema dict compatible with Gemini API

    Example:
        >>> from mad_spark_alt.core.schemas import DeductionResponse
        >>> schema = to_gemini_schema(DeductionResponse)
        >>> # Use in LLM API call
        >>> response = client.models.generate_content(
        ...     model="gemini-2.5-flash",
        ...     contents=prompt,
        ...     config={
        ...         "response_mime_type": "application/json",
        ...         "response_json_schema": schema,
        ...     }
        ... )
    """
    return to_standard_json_schema(pydantic_model)


def to_openai_schema(pydantic_model: Type[BaseModel]) -> Dict[str, Any]:
    """
    Convert Pydantic model to OpenAI API's response_format schema.

    OpenAI's Structured Outputs feature also uses standard JSON Schema,
    so this delegates to to_standard_json_schema().

    Args:
        pydantic_model: Pydantic model class to convert

    Returns:
        Standard JSON Schema dict compatible with OpenAI API

    Example:
        >>> from mad_spark_alt.core.schemas import DeductionResponse
        >>> schema = to_openai_schema(DeductionResponse)
        >>> # Use in OpenAI API call
        >>> response = openai.chat.completions.create(
        ...     messages=[...],
        ...     response_format={
        ...         "type": "json_schema",
        ...         "json_schema": {
        ...             "name": "deduction_response",
        ...             "schema": schema,
        ...         }
        ...     }
        ... )
    """
    return to_standard_json_schema(pydantic_model)


def to_anthropic_schema(pydantic_model: Type[BaseModel]) -> Dict[str, Any]:
    """
    Convert Pydantic model to Anthropic Claude's tool schema format.

    Anthropic uses JSON Schema in their function calling feature,
    which can be adapted for structured outputs.

    Args:
        pydantic_model: Pydantic model class to convert

    Returns:
        Standard JSON Schema dict compatible with Anthropic API

    Example:
        >>> from mad_spark_alt.core.schemas import DeductionResponse
        >>> schema = to_anthropic_schema(DeductionResponse)
        >>> # Use in Anthropic API call with tool use pattern
        >>> response = anthropic.messages.create(
        ...     model="claude-3-5-sonnet-20241022",
        ...     messages=[...],
        ...     tools=[{
        ...         "name": "return_structured_output",
        ...         "description": "Return a structured deduction response",
        ...         "input_schema": schema
        ...     }]
        ... )
    """
    return to_standard_json_schema(pydantic_model)


def is_pydantic_model(obj: Any) -> bool:
    """
    Check if an object is a Pydantic model class.

    Args:
        obj: Object to check

    Returns:
        True if obj is a Pydantic BaseModel subclass

    Example:
        >>> from mad_spark_alt.core.schemas import DeductionResponse
        >>> is_pydantic_model(DeductionResponse)  # True
        >>> is_pydantic_model({"type": "object"})  # False
    """
    # isinstance(obj, type) check prevents TypeError in issubclass
    return isinstance(obj, type) and issubclass(obj, BaseModel)
