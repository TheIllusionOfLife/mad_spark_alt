"""
Universal Pydantic Schema Models for LLM Structured Outputs

This module provides type-safe schema definitions that generate standard JSON Schema
compatible with multiple LLM providers (Google Gemini, OpenAI, Anthropic, local LLMs).

Key Features:
- Standard JSON Schema generation (not provider-specific formats)
- Automatic validation (score ranges, required fields, no extra properties)
- Type safety with IDE autocomplete
- Schema reusability via nested models
- Property ordering preservation (Gemini 2.5+)

Usage:
    # Generate JSON Schema for API calls
    schema = DeductionResponse.model_json_schema()

    # Parse and validate LLM responses
    response = DeductionResponse.model_validate_json(llm_response_text)

    # Access validated data with type safety
    for evaluation in response.evaluations:
        print(f"Hypothesis {evaluation.hypothesis_id}: {evaluation.scores.impact}")
"""

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class HypothesisScores(BaseModel):
    """
    QADI hypothesis evaluation scores.

    All scores must be in the range [0.0, 1.0] and are automatically validated.
    This schema works with all LLM providers supporting JSON Schema.
    """

    model_config = ConfigDict(extra="forbid")  # Strict validation: reject extra fields

    impact: float = Field(
        ge=0.0,
        le=1.0,
        description="Impact score: What level of positive change will this create?",
    )
    feasibility: float = Field(
        ge=0.0,
        le=1.0,
        description="Feasibility score: How practical is implementation?",
    )
    accessibility: float = Field(
        ge=0.0,
        le=1.0,
        description="Accessibility score: How easily can people adopt this?",
    )
    sustainability: float = Field(
        ge=0.0,
        le=1.0,
        description="Sustainability score: Can this be maintained long-term?",
    )
    scalability: float = Field(
        ge=0.0,
        le=1.0,
        description="Scalability score: Can this grow to serve more users/use cases?",
    )


class Hypothesis(BaseModel):
    """
    A single hypothesis generated during the Abduction phase.

    Combines a unique identifier with the hypothesis content.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(description="Unique hypothesis identifier (e.g., 'H1', 'H2')")
    content: str = Field(description="The hypothesis text describing the approach or solution")


class HypothesisEvaluation(BaseModel):
    """
    Evaluation result linking a hypothesis ID to its QADI scores.

    Used in the Deduction phase to assess each hypothesis.
    """

    model_config = ConfigDict(extra="forbid")

    hypothesis_id: str = Field(description="ID of the hypothesis being evaluated")
    scores: HypothesisScores = Field(description="QADI evaluation scores for this hypothesis")


class DeductionResponse(BaseModel):
    """
    Complete response from the Deduction phase (Phase 2).

    Contains evaluations of all hypotheses, a synthesized answer,
    and an actionable plan. Property order is preserved for Gemini 2.5+.
    """

    model_config = ConfigDict(extra="forbid")

    evaluations: List[HypothesisEvaluation] = Field(
        description="Scored evaluations for each hypothesis"
    )
    answer: str = Field(description="Synthesized answer based on the highest-scoring hypothesis")
    action_plan: List[str] = Field(
        min_length=3,
        max_length=3,
        description="Exactly 3 action items: (1) immediate action for today, (2) short-term weekly/monthly goal, (3) long-term strategy. Each item is a clean sentence WITHOUT numbering or bullet points - the list structure handles ordering.",
    )


class HypothesisListResponse(BaseModel):
    """
    Response from the Abduction phase (Phase 1) containing generated hypotheses.

    Used for hypothesis generation with configurable quantities.
    """

    model_config = ConfigDict(extra="forbid")

    hypotheses: List[Hypothesis] = Field(
        description="List of generated hypotheses with IDs and content"
    )


class MutationResponse(BaseModel):
    """
    Response from a single mutation operation in evolution.

    Contains the mutated version of an idea.
    """

    model_config = ConfigDict(extra="forbid")

    mutated_idea: str = Field(description="The mutated version of the original idea")


class MutationResult(BaseModel):
    """
    Single mutation result in a batch operation.

    Includes an ID for ordering and optional mutation type for breakthrough mutations.
    """

    model_config = ConfigDict(extra="forbid")

    id: int = Field(description="Mutation ID for maintaining order (1-based indexing)")
    mutated_idea: str = Field(description="The mutated idea content")
    mutation_type: Optional[str] = Field(
        default=None,
        description="Type of mutation applied (e.g., 'paradigm_shift', 'scale_amplification')",
    )


class BatchMutationResponse(BaseModel):
    """
    Response from batch mutation operations in evolution.

    Processes multiple ideas in a single LLM call for efficiency.
    Supports both regular and breakthrough mutations with type annotations.
    """

    model_config = ConfigDict(extra="forbid")

    mutations: List[MutationResult] = Field(
        description="List of mutation results with IDs for ordering"
    )


class CrossoverResponse(BaseModel):
    """
    Response from a single crossover operation.

    Produces two offspring ideas from combining parent ideas.
    """

    model_config = ConfigDict(extra="forbid")

    offspring1: str = Field(description="First offspring idea combining parent characteristics")
    offspring2: str = Field(description="Second offspring idea combining parent characteristics")


class CrossoverResult(BaseModel):
    """
    Single crossover result in a batch operation.

    Includes pair_id for maintaining correspondence between parent pairs and offspring.
    """

    model_config = ConfigDict(extra="forbid")

    pair_id: int = Field(
        description="Parent pair ID for maintaining order (1-based indexing)"
    )
    offspring1: str = Field(description="First offspring from this parent pair")
    offspring2: str = Field(description="Second offspring from this parent pair")


class BatchCrossoverResponse(BaseModel):
    """
    Response from batch crossover operations in evolution.

    Processes multiple parent pairs in a single LLM call for efficiency.
    """

    model_config = ConfigDict(extra="forbid")

    crossovers: List[CrossoverResult] = Field(
        description="List of crossover results with pair IDs for ordering"
    )
