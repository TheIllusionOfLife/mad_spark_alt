"""
Universal QADI Prompts for Hypothesis-Driven Analysis

This module provides the true QADI methodology prompts based on
hypothesis-driven consulting approach from "Shin Logical Thinking".

The QADI process:
1. Q (Question): Identify THE core question to answer
2. A (Abduction): Generate hypotheses that answer the question
3. D (Deduction): Evaluate hypotheses and conclude the answer
4. I (Induction): Verify the answer with examples
"""

from typing import Dict

from .language_utils import get_strategy_1_instruction
from .system_constants import CONSTANTS


class QADIPrompts:
    """Universal prompts for the QADI methodology."""

    @staticmethod
    def get_questioning_prompt(user_input: str) -> str:
        """Get the prompt for extracting the core question."""
        return f"""{get_strategy_1_instruction()}As an analytical expert, identify THE single most important question to answer based on the user's input.

User's input:
{user_input}

Think about:
- What is the core challenge or desire expressed?
- What fundamental question needs answering to make progress?
- What would provide the most helpful insight?

Output exactly ONE core question that gets to the heart of the matter.
Format: "Q: [Your core question]"
"""

    @staticmethod
    def get_abduction_prompt(user_input: str, core_question: str, num_hypotheses: int = 3) -> str:
        """Get the prompt for generating hypotheses.

        Note: Format instructions (e.g., "1. [Title]\\n[Explanation]") were removed.
        Output structure is now controlled by Pydantic schema field descriptions
        (see Hypothesis.content in schemas.py). This follows the "Structured Output
        Over Prompt Engineering" pattern - prompts describe WHAT content to provide,
        schemas define HOW to structure it.
        """
        return f"""{get_strategy_1_instruction()}As a creative problem-solver, generate {num_hypotheses} distinct approaches that could answer this core question.

Core Question: {core_question}

User's original input:
{user_input}

Generate hypotheses that cover different scales or perspectives:
- Consider individual/personal approaches
- Think about community/collective solutions
- Explore systemic/structural changes

Each hypothesis should:
- Directly address the core question
- Be concrete and actionable with specific implementation details
- Provide comprehensive explanation (minimum 100 words per hypothesis)
- Offer a meaningfully different path forward

Generate exactly {num_hypotheses} hypotheses, each with a unique ID (H1, H2, H3, etc.) and comprehensive content."""

    @staticmethod
    def get_deduction_prompt(
        user_input: str, core_question: str, hypotheses: str
    ) -> str:
        """Get the prompt for evaluating hypotheses and determining the answer."""
        return f"""{get_strategy_1_instruction()}As an analytical expert, evaluate each approach and determine the best answer to our core question.

Core Question: {core_question}

Approaches to evaluate:
{hypotheses}

User's original context:
{user_input}

Score each approach from 0.0 to 1.0 on these universal criteria:
- Impact: How much positive change will this create? (0=minimal, 1=transformative)
- Feasibility: How easy is it to implement? (0=very difficult, 1=very easy)
- Accessibility: Can most people do this? (0=requires special resources, 1=anyone can do it)
- Sustainability: Will this solution last? (0=temporary fix, 1=permanent solution)
- Scalability: Can this grow from small to large? (0=limited scope, 1=unlimited potential)

IMPORTANT: You MUST use the exact format below with "Approach 1:", "Approach 2:", "Approach 3:" headers and numerical scores.

Approach 1:
* Impact: [decimal score like 0.8] - [brief explanation]
* Feasibility: [decimal score] - [brief explanation]
* Accessibility: [decimal score] - [brief explanation]
* Sustainability: [decimal score] - [brief explanation]
* Scalability: [decimal score] - [brief explanation]
* Overall: [calculated weighted score]

Approach 2:
* Impact: [decimal score] - [brief explanation]
* Feasibility: [decimal score] - [brief explanation]
* Accessibility: [decimal score] - [brief explanation]
* Sustainability: [decimal score] - [brief explanation]
* Scalability: [decimal score] - [brief explanation]
* Overall: [calculated weighted score]

Approach 3:
* Impact: [decimal score] - [brief explanation]
* Feasibility: [decimal score] - [brief explanation]
* Accessibility: [decimal score] - [brief explanation]
* Sustainability: [decimal score] - [brief explanation]
* Scalability: [decimal score] - [brief explanation]
* Overall: [calculated weighted score]

ANSWER: [Clear, comprehensive answer explaining which approach works best and why, written for a general audience]

Action Plan - MUST be specific to the best approach above. Provide exactly 3 items as plain sentences without any formatting (no numbers, bullets, or dashes). The schema handles the list structure.

First item: An immediate first step to START implementing the recommended approach today
Second item: A concrete milestone to achieve this week/month using the recommended approach
Third item: A long-term goal that fully realizes the recommended approach
"""

    @staticmethod
    def get_induction_prompt(
        user_input: str,
        core_question: str,
        answer: str,
        hypotheses_with_scores: str = "",
        action_plan: str = "",
    ) -> str:
        """Get the prompt for synthesizing all QADI findings into a final conclusion.

        Args:
            user_input: Original user question/context
            core_question: Extracted core question from Phase 1
            answer: Analysis from deduction phase explaining the recommended approach
            hypotheses_with_scores: Formatted string of all hypotheses with their scores
            action_plan: Formatted string of the 3 action items

        Note: Format instructions removed - relying on Pydantic schema field descriptions
        for output structure. This follows the "Structured Output Over Prompt Engineering"
        pattern - prompts describe WHAT content to provide, schemas define HOW to structure it.
        """
        # Build context section based on available data
        context_parts = [f"Original question: {user_input}"]

        if hypotheses_with_scores:
            context_parts.append(f"\nHYPOTHESES EVALUATED:\n{hypotheses_with_scores}")

        if action_plan:
            context_parts.append(f"\nACTION PLAN:\n{action_plan}")

        context = "\n".join(context_parts)

        return f"""{get_strategy_1_instruction()}You are synthesizing a QADI (Question-Abduction-Deduction-Induction) analysis into a final conclusion.

CORE QUESTION: {core_question}

{context}

ANALYSIS SUMMARY:
{answer}

Write a comprehensive synthesis (3-4 paragraphs) that:
1. Directly answers the core question with the recommended approach
2. Explains WHY this approach was chosen based on the evaluation criteria (reference specific scores if available)
3. Acknowledges trade-offs compared to alternatives (e.g., "While Approach 2 scored higher on feasibility...")
4. Provides practical guidance for implementation, contextualizing the action plan within the broader conclusion

Do NOT simply repeat the action plan or analysis. Instead, synthesize the findings into a cohesive narrative that helps the reader understand both the conclusion and the reasoning behind it.
"""


# Phase-specific hyperparameters
PHASE_HYPERPARAMETERS = {
    "questioning": {
        "temperature": 0.3,  # Low - need focused, precise question
        "max_tokens": 150,  # Short - just one question
        "top_p": 0.9,
    },
    "abduction": {
        "temperature": CONSTANTS.LLM.DEFAULT_QADI_TEMPERATURE,  # Creative hypotheses
        "max_tokens": 2500,  # Increased for detailed hypotheses with 150+ words each
        "top_p": CONSTANTS.LLM.DEFAULT_QADI_TOP_P,
        "user_adjustable": True,  # Allow --temperature override
    },
    "deduction": {
        "temperature": 0.2,  # Very low - need analytical precision
        "max_tokens": 3000,  # Increased for complete analysis with scores, answer, and action plan
        "top_p": 0.9,
    },
    "induction": {
        "temperature": 0.3,  # Low - synthesis requires analytical precision
        "max_tokens": 1500,  # Sufficient for 3-4 paragraph synthesis
        "top_p": 0.9,
    },
}


# Unified evaluation criteria weights
EVALUATION_CRITERIA = {
    "impact": 0.3,  # How much positive change
    "feasibility": 0.2,  # How easy to implement
    "accessibility": 0.2,  # Can most people do this
    "sustainability": 0.2,  # Will solution last
    "scalability": 0.1,  # Can grow from small to large
}


def calculate_hypothesis_score(scores: Dict[str, float]) -> float:
    """
    Calculate the overall score for a hypothesis using unified criteria.

    Args:
        scores: Dictionary with keys: impact, feasibility, accessibility, sustainability, scalability

    Returns:
        Weighted overall score between 0.0 and 1.0
    """
    return (
        scores.get("impact", 0.0) * EVALUATION_CRITERIA["impact"]
        + scores.get("feasibility", 0.0) * EVALUATION_CRITERIA["feasibility"]
        + scores.get("accessibility", 0.0) * EVALUATION_CRITERIA["accessibility"]
        + scores.get("sustainability", 0.0) * EVALUATION_CRITERIA["sustainability"]
        + scores.get("scalability", 0.0) * EVALUATION_CRITERIA["scalability"]
    )
