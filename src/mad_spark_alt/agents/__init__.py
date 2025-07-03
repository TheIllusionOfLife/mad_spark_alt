"""Thinking method agents for idea generation."""

from .abduction.agent import AbductionAgent
from .deduction.agent import DeductionAgent
from .induction.agent import InductionAgent
from .questioning.agent import QuestioningAgent

__all__ = [
    "QuestioningAgent",
    "AbductionAgent",
    "DeductionAgent",
    "InductionAgent",
]
