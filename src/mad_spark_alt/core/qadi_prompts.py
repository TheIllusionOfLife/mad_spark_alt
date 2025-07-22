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

from typing import Any, Dict


class QADIPrompts:
    """Universal prompts for the QADI methodology."""

    @staticmethod
    def get_questioning_prompt(user_input: str) -> str:
        """Get the prompt for extracting the core question."""
        return f"""As a strategic consultant, identify THE single most important question to answer for the user's input.

User's input:
{user_input}

Think about:
- What is the real problem or goal behind this input?
- What question, if answered well, would provide maximum value?
- What decision or action depends on this answer?

Output exactly ONE core question.
Format: "Q: [Your core question]"
"""

    @staticmethod
    def get_abduction_prompt(user_input: str, core_question: str) -> str:
        """Get the prompt for generating hypotheses."""
        return f"""As a hypothesis-driven consultant, generate 3 specific hypotheses that could answer this core question.

Core Question: {core_question}

User's original input:
{user_input}

Each hypothesis should:
- Directly answer the core question
- Be specific and actionable
- Represent a distinct approach

Format:
H1: [First hypothesis]
H2: [Second hypothesis]
H3: [Third hypothesis]
"""

    @staticmethod
    def get_deduction_prompt(
        user_input: str, core_question: str, hypotheses: str
    ) -> str:
        """Get the prompt for evaluating hypotheses and determining the answer."""
        return f"""As an analytical consultant, evaluate each hypothesis and determine the best answer to our core question.

Core Question: {core_question}

Hypotheses to evaluate:
{hypotheses}

User's original context:
{user_input}

Score each hypothesis from 0.0 to 1.0 on:
- Novelty: How innovative/unique is this approach?
- Impact: What level of positive change will this create?
- Cost: What resources required? (0=expensive, 1=cheap)
- Feasibility: How practical is implementation?
- Risks: What could go wrong? (0=high risk, 1=low risk)

Format:
Analysis:
- H1: 
  * Novelty: [score] - [brief explanation]
  * Impact: [score] - [brief explanation]
  * Cost: [score] - [brief explanation]
  * Feasibility: [score] - [brief explanation]
  * Risks: [score] - [brief explanation]
  * Overall: [calculated weighted score]

- H2: 
  * Novelty: [score] - [brief explanation]
  * Impact: [score] - [brief explanation]
  * Cost: [score] - [brief explanation]
  * Feasibility: [score] - [brief explanation]
  * Risks: [score] - [brief explanation]
  * Overall: [calculated weighted score]

- H3: 
  * Novelty: [score] - [brief explanation]
  * Impact: [score] - [brief explanation]
  * Cost: [score] - [brief explanation]
  * Feasibility: [score] - [brief explanation]
  * Risks: [score] - [brief explanation]
  * Overall: [calculated weighted score]

ANSWER: [Your definitive answer to the core question based on the highest scoring hypothesis]

Action Plan:
1. [Specific first step to implement the answer]
2. [Second concrete action]
3. [Third actionable step]
"""

    @staticmethod
    def get_induction_prompt(user_input: str, core_question: str, answer: str) -> str:
        """Get the prompt for verifying the answer with examples."""
        return f"""As a consultant validating the solution, verify that our answer is robust by testing it across different scenarios.

Core Question: {core_question}
Our Answer: {answer}

Original context:
{user_input}

Provide 3 examples where this answer/principle successfully applies:

Format:
Verification:
1. [Real-world example where this approach worked]
2. [Different industry/context where this principle applies]
3. [Future scenario where this would be effective]

Conclusion: [Is our answer universally applicable, context-specific, or needs refinement?]
"""


# Phase-specific hyperparameters
PHASE_HYPERPARAMETERS = {
    "questioning": {
        "temperature": 0.3,  # Low - need focused, precise question
        "max_tokens": 150,  # Short - just one question
        "top_p": 0.9,
    },
    "abduction": {
        "temperature": 0.8,  # High - need creative hypotheses (default)
        "max_tokens": 400,  # Medium - 3 hypotheses
        "top_p": 0.95,
        "user_adjustable": True,  # Allow --temperature override
    },
    "deduction": {
        "temperature": 0.2,  # Very low - need analytical precision
        "max_tokens": 800,  # Long - detailed analysis
        "top_p": 0.9,
    },
    "induction": {
        "temperature": 0.5,  # Medium - balanced examples
        "max_tokens": 600,  # Medium - 3 examples
        "top_p": 0.9,
    },
}


# Unified evaluation criteria weights
EVALUATION_CRITERIA = {
    "novelty": 0.2,  # How innovative/unique
    "impact": 0.3,  # Level of positive change
    "cost": 0.2,  # Resource efficiency (inverse - lower cost = higher score)
    "feasibility": 0.2,  # Implementation practicality
    "risks": 0.1,  # Risk mitigation (inverse - lower risk = higher score)
}


def calculate_hypothesis_score(scores: Dict[str, float]) -> float:
    """
    Calculate the overall score for a hypothesis using unified criteria.

    Args:
        scores: Dictionary with keys: novelty, impact, cost, feasibility, risks

    Returns:
        Weighted overall score between 0.0 and 1.0
    """
    return (
        scores.get("novelty", 0.0) * EVALUATION_CRITERIA["novelty"]
        + scores.get("impact", 0.0) * EVALUATION_CRITERIA["impact"]
        + scores.get("cost", 0.0)
        * EVALUATION_CRITERIA["cost"]  # Already inverted in prompt
        + scores.get("feasibility", 0.0) * EVALUATION_CRITERIA["feasibility"]
        + scores.get("risks", 0.0)
        * EVALUATION_CRITERIA["risks"]  # Already inverted in prompt
    )
