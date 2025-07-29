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


class QADIPrompts:
    """Universal prompts for the QADI methodology."""

    @staticmethod
    def get_questioning_prompt(user_input: str) -> str:
        """Get the prompt for extracting the core question."""
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

    @staticmethod
    def get_abduction_prompt(user_input: str, core_question: str, num_hypotheses: int = 3) -> str:
        """Get the prompt for generating hypotheses."""
        return f"""As a creative problem-solver, generate {num_hypotheses} distinct approaches that could answer this core question.

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

IMPORTANT OUTPUT FORMAT - You MUST follow this exact format without any modifications:

H1: [First approach title]
[Detailed explanation of the first approach with specific steps, technologies, methodologies, and implementation details. This should be a comprehensive paragraph explaining how this approach works, what resources it requires, and why it addresses the core question effectively.]

H2: [Second approach title]
[Detailed explanation of the second approach with specific steps, technologies, methodologies, and implementation details. This should be a comprehensive paragraph explaining how this approach works, what resources it requires, and why it addresses the core question effectively.]

H3: [Third approach title]
[Detailed explanation of the third approach with specific steps, technologies, methodologies, and implementation details. This should be a comprehensive paragraph explaining how this approach works, what resources it requires, and why it addresses the core question effectively.]""" + (f"""

H4: [Fourth approach title]
[Detailed explanation of the fourth approach with specific steps, technologies, methodologies, and implementation details. This should be a comprehensive paragraph explaining how this approach works, what resources it requires, and why it addresses the core question effectively.]

H5: [Fifth approach title]
[Detailed explanation of the fifth approach with specific steps, technologies, methodologies, and implementation details. This should be a comprehensive paragraph explaining how this approach works, what resources it requires, and why it addresses the core question effectively.]""" if num_hypotheses > 3 else "") + (f"""

H6: [Sixth approach title]
[Detailed explanation of the sixth approach with specific steps, technologies, methodologies, and implementation details. This should be a comprehensive paragraph explaining how this approach works, what resources it requires, and why it addresses the core question effectively.]

H7: [Seventh approach title]
[Detailed explanation of the seventh approach with specific steps, technologies, methodologies, and implementation details. This should be a comprehensive paragraph explaining how this approach works, what resources it requires, and why it addresses the core question effectively.]""" if num_hypotheses > 5 else "")

    @staticmethod
    def get_deduction_prompt(
        user_input: str, core_question: str, hypotheses: str
    ) -> str:
        """Get the prompt for evaluating hypotheses and determining the answer."""
        return f"""As an analytical expert, evaluate each approach and determine the best answer to our core question.

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

IMPORTANT: You MUST use the exact format below with "H1:", "H2:", "H3:" headers and numerical scores.

H1:
* Impact: [decimal score like 0.8] - [brief explanation]
* Feasibility: [decimal score] - [brief explanation]
* Accessibility: [decimal score] - [brief explanation]
* Sustainability: [decimal score] - [brief explanation]
* Scalability: [decimal score] - [brief explanation]
* Overall: [calculated weighted score]

H2:
* Impact: [decimal score] - [brief explanation]
* Feasibility: [decimal score] - [brief explanation]
* Accessibility: [decimal score] - [brief explanation]
* Sustainability: [decimal score] - [brief explanation]
* Scalability: [decimal score] - [brief explanation]
* Overall: [calculated weighted score]

H3:
* Impact: [decimal score] - [brief explanation]
* Feasibility: [decimal score] - [brief explanation]
* Accessibility: [decimal score] - [brief explanation]
* Sustainability: [decimal score] - [brief explanation]
* Scalability: [decimal score] - [brief explanation]
* Overall: [calculated weighted score]

ANSWER: [Clear, comprehensive answer explaining which approach works best and why, written for a general audience]

Action Plan:
1. [Immediate action anyone can take today]
2. [Short-term goal to work toward this week/month]
3. [Long-term strategy for lasting change]
"""

    @staticmethod
    def get_induction_prompt(user_input: str, core_question: str, answer: str) -> str:
        """Get the prompt for verifying the answer with examples."""
        return f"""As a thoughtful analyst, verify the recommended approach by examining real-world applications.

Core Question: {core_question}
Recommended Approach: {answer}

Original context:
{user_input}

Provide 3 diverse examples that demonstrate this approach in action:

Format each example clearly:

Example 1: [Individual/Personal Level]
- Context: [Brief situation description]
- Application: [How the approach was used]
- Result: [What positive outcome occurred]

Example 2: [Community/Group Level]
- Context: [Brief situation description]
- Application: [How the approach was used]
- Result: [What positive outcome occurred]

Example 3: [Larger Scale/Future Application]
- Context: [Brief situation description]
- Application: [How the approach could be used]
- Result: [Expected positive outcome]

Conclusion: In 2-3 sentences, explain whether the recommended approach is broadly applicable or needs adaptation for specific contexts. Focus on one key practical insight.
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
        "max_tokens": 2500,  # Increased for detailed hypotheses with 150+ words each
        "top_p": 0.95,
        "user_adjustable": True,  # Allow --temperature override
    },
    "deduction": {
        "temperature": 0.2,  # Very low - need analytical precision
        "max_tokens": 3000,  # Increased for complete analysis with scores, answer, and action plan
        "top_p": 0.9,
    },
    "induction": {
        "temperature": 0.5,  # Medium - balanced examples
        "max_tokens": 1200,  # Increased for complete examples without truncation
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
