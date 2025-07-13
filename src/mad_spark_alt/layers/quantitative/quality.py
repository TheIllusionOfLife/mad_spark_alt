"""
Quality and consistency evaluation metrics.
"""

import asyncio
import logging
import re
from typing import Any, Dict, List, Optional, cast

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from ...core.evaluation_utils import (
    CodeAnalyzer,
    CodeStructureMetricsDict,
    TextAnalyzer,
)
from ...core.interfaces import (
    AsyncEvaluatorMixin,
    EvaluationLayer,
    EvaluationRequest,
    EvaluationResult,
    EvaluatorInterface,
    ModelOutput,
    OutputType,
)

logger = logging.getLogger(__name__)


class QualityEvaluator(EvaluatorInterface, AsyncEvaluatorMixin):
    """Evaluates basic quality metrics of AI outputs."""

    def __init__(self, model_name: str = "gpt2"):
        self.model_name = model_name
        self._model: Optional[GPT2LMHeadModel] = None
        self._tokenizer: Optional[GPT2TokenizerFast] = None

    @property
    def name(self) -> str:
        return "quality_evaluator"

    @property
    def layer(self) -> EvaluationLayer:
        return EvaluationLayer.QUANTITATIVE

    @property
    def supported_output_types(self) -> List[OutputType]:
        return [OutputType.TEXT, OutputType.CODE]

    @property
    def model(self) -> GPT2LMHeadModel:
        """Lazy load the language model for perplexity calculation."""
        if self._model is None:
            self._model = GPT2LMHeadModel.from_pretrained(self.model_name)
            self._model.eval()
        return cast(GPT2LMHeadModel, self._model)

    @property
    def tokenizer(self) -> GPT2TokenizerFast:
        """Lazy load the tokenizer."""
        if self._tokenizer is None:
            self._tokenizer = GPT2TokenizerFast.from_pretrained(self.model_name)
            self._tokenizer.pad_token = self._tokenizer.eos_token
        return cast(GPT2TokenizerFast, self._tokenizer)

    async def evaluate(self, request: EvaluationRequest) -> List[EvaluationResult]:
        """Evaluate quality metrics for the given outputs."""
        tasks = [self._evaluate_single_output(output) for output in request.outputs]
        return await asyncio.gather(*tasks)

    async def evaluate_batch(
        self, requests: List[EvaluationRequest]
    ) -> List[List[EvaluationResult]]:
        """Evaluate multiple requests in batch."""
        tasks = [self.evaluate(request) for request in requests]
        return await asyncio.gather(*tasks)

    async def _evaluate_single_output(self, output: ModelOutput) -> EvaluationResult:
        """Evaluate quality metrics for a single output."""
        content = str(output.content)

        # Calculate various quality metrics
        scores: Dict[str, float] = {}
        explanations = {}

        # Basic text quality metrics
        scores["length"] = float(len(content))
        scores["word_count"] = float(len(content.split()))
        scores["sentence_count"] = float(len(TextAnalyzer.split_sentences(content)))

        # Grammar and structure metrics
        scores["grammar_score"] = self._calculate_grammar_score(content)
        scores["readability_score"] = self._calculate_readability_score(content)

        # Fluency metric (perplexity)
        try:
            scores["fluency_score"] = await self._calculate_fluency_score(content)
        except Exception as e:
            logger.warning(f"Failed to calculate fluency score: {e}")
            scores["fluency_score"] = 0.5  # Default middle score

        # Content structure metrics
        scores["coherence_score"] = self._calculate_coherence_score(content)

        # Output type specific metrics
        if output.output_type == OutputType.CODE:
            code_metrics = self._evaluate_code_quality(content)
            scores.update(code_metrics)

        # Calculate overall quality score
        quality_components = [
            scores.get("grammar_score", 0.5),
            scores.get("readability_score", 0.5),
            scores.get("fluency_score", 0.5),
            scores.get("coherence_score", 0.5),
        ]
        scores["overall_quality"] = sum(quality_components) / len(quality_components)

        # Add explanations
        explanations.update(
            {
                "length": "Total character count of the output",
                "word_count": "Total number of words",
                "sentence_count": "Total number of sentences",
                "grammar_score": "Basic grammar correctness (0-1)",
                "readability_score": "Text readability assessment (0-1)",
                "fluency_score": "Language fluency based on perplexity (0-1)",
                "coherence_score": "Structural coherence assessment (0-1)",
                "overall_quality": "Combined quality score across all metrics",
            }
        )

        metadata = {
            "output_type": output.output_type.value,
            "model_name": output.model_name,
            "perplexity_model": self.model_name,
        }

        return EvaluationResult(
            evaluator_name=self.name,
            layer=self.layer,
            scores=scores,
            explanations=explanations,
            metadata=metadata,
        )

    def _calculate_grammar_score(self, text: str) -> float:
        """Basic grammar score based on simple heuristics."""
        # Use TextAnalyzer for basic grammar checking
        grammar_scores = TextAnalyzer.check_basic_grammar(text)

        score = 1.0

        # Apply grammar scores
        score *= grammar_scores["has_punctuation"]
        score *= grammar_scores["capitalization_score"]

        # Check for excessive repetition
        word_freq = TextAnalyzer.calculate_word_frequency(text)
        if word_freq:
            total_words = sum(word_freq.values())
            max_freq = max(word_freq.values())
            if max_freq / total_words > 0.2:
                score -= 0.2

        return max(0.0, min(1.0, score))

    def _calculate_readability_score(self, text: str) -> float:
        """Simple readability score based on sentence and word length."""
        # Use TextAnalyzer for readability analysis
        readability_metrics = TextAnalyzer.analyze_readability(text)

        if readability_metrics["avg_sentence_length"] == 0:
            return 0.0

        # Combine the two normalized scores
        sentence_score = readability_metrics["sentence_length_score"]
        word_score = readability_metrics["word_length_score"]

        return (sentence_score + word_score) / 2

    async def _calculate_fluency_score(self, text: str) -> float:
        """Calculate fluency score based on perplexity."""
        if not text.strip():
            return 0.0

        try:
            # Tokenize text
            inputs = self.tokenizer(
                text, return_tensors="pt", truncation=True, max_length=512
            )

            with torch.no_grad():
                outputs = self.model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss
                perplexity = torch.exp(loss).item()

            # Convert perplexity to 0-1 score (lower perplexity = higher fluency)
            # Typical perplexity ranges: 10-100 for good text, 100+ for poor text
            fluency_score = max(0.0, min(1.0, 1.0 - (perplexity - 10) / 90))
            return fluency_score

        except Exception as e:
            logger.warning(f"Perplexity calculation failed: {e}")
            return 0.5

    def _calculate_coherence_score(self, text: str) -> float:
        """Basic coherence score based on text structure."""
        if not text.strip():
            return 0.0

        score = 1.0

        # Check for abrupt topic changes (simplified)
        sentences = TextAnalyzer.split_sentences(text)
        if len(sentences) > 1:
            # Very basic coherence check - similar to readability
            consistent_length = True
            sentence_lengths = [len(s.split()) for s in sentences]
            if sentence_lengths:
                avg_len = sum(sentence_lengths) / len(sentence_lengths)
                # Check if sentence lengths vary too much
                variance = sum((l - avg_len) ** 2 for l in sentence_lengths) / len(
                    sentence_lengths
                )
                if variance > avg_len**2:  # High variance
                    score -= 0.2

        # Check for logical connectors
        connectors = [
            "however",
            "therefore",
            "moreover",
            "furthermore",
            "additionally",
            "consequently",
            "meanwhile",
            "similarly",
            "in contrast",
            "for example",
        ]
        text_lower = text.lower()
        connector_count = sum(1 for conn in connectors if conn in text_lower)
        if len(sentences) > 2 and connector_count == 0:
            score -= 0.1

        return max(0.0, min(1.0, score))

    def _evaluate_code_quality(self, code: str) -> Dict[str, float]:
        """Evaluate code-specific quality metrics."""
        # Use CodeAnalyzer for code structure analysis
        metrics = CodeAnalyzer.analyze_code_structure(code)

        # Calculate structure score based on analyzed metrics
        structure_components = [
            min(metrics["indentation_ratio"] * 2, 1.0),  # Good indentation
            min(metrics["comment_ratio"] * 5, 1.0),  # Some comments
        ]
        metrics["code_structure_score"] = float(
            sum(structure_components) / len(structure_components)
        )

        return cast(Dict[str, float], dict(metrics))

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate configuration."""
        return True  # This evaluator has no required config
