"""
Mad Spark Alt - AI Creativity Evaluation System

A multi-layer framework for evaluating AI model creativity across different dimensions.
"""

__version__ = "0.1.0"
__author__ = "TheIllusionOfLife"

from .core.evaluator import CreativityEvaluator
from .core.interfaces import EvaluationResult, EvaluatorInterface

__all__ = ["CreativityEvaluator", "EvaluationResult", "EvaluatorInterface"]