"""Thinking method agents for idea generation."""

from .questioning.agent import QuestioningAgent
from .abduction.agent import AbductionAgent
from .deduction.agent import DeductionAgent
from .induction.agent import InductionAgent

__all__ = [
    "QuestioningAgent",
    "AbductionAgent", 
    "DeductionAgent",
    "InductionAgent",
]