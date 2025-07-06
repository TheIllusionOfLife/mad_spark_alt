"""Thinking method agents for idea generation."""

from .abduction.agent import AbductionAgent
from .abduction.llm_agent import LLMAbductiveAgent
from .deduction.agent import DeductionAgent
from .deduction.llm_agent import LLMDeductiveAgent
from .induction.agent import InductionAgent
from .induction.llm_agent import LLMInductiveAgent
from .questioning.agent import QuestioningAgent
from .questioning.llm_agent import LLMQuestioningAgent

__all__ = [
    "QuestioningAgent",
    "LLMQuestioningAgent",
    "AbductionAgent",
    "LLMAbductiveAgent",
    "DeductionAgent",
    "LLMDeductiveAgent",
    "InductionAgent",
    "LLMInductiveAgent",
]
