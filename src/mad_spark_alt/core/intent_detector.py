"""
Intent Detection for Multi-Perspective QADI Analysis

This module detects the primary intent behind user questions to enable
appropriate perspective selection for QADI analysis.
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Tuple


class QuestionIntent(Enum):
    """Primary intents for questions."""

    ENVIRONMENTAL = "environmental"
    PERSONAL = "personal"
    TECHNICAL = "technical"
    BUSINESS = "business"
    SCIENTIFIC = "scientific"
    PHILOSOPHICAL = "philosophical"
    GENERAL = "general"


@dataclass
class IntentResult:
    """Result of intent detection."""

    primary_intent: QuestionIntent
    confidence: float  # 0.0 to 1.0
    secondary_intents: List[QuestionIntent]
    keywords_matched: List[str]


class IntentDetector:
    """Detects the primary intent behind user questions."""

    def __init__(self) -> None:
        """Initialize the detector with intent patterns."""

        # Intent patterns with keywords and phrases
        self.intent_patterns: Dict[QuestionIntent, Dict[str, List[str]]] = {
            QuestionIntent.ENVIRONMENTAL: {
                "keywords": [
                    "environment",
                    "environmental",
                    "sustainable",
                    "sustainability",
                    "plastic",
                    "waste",
                    "pollution",
                    "climate",
                    "carbon",
                    "emissions",
                    "recycle",
                    "recycling",
                    "green",
                    "eco",
                    "ecological",
                    "nature",
                    "conservation",
                    "biodegradable",
                    "renewable",
                    "energy",
                    "solar",
                    "wind",
                    "ocean",
                    "forest",
                    "ecosystem",
                    "habitat",
                    "species",
                ],
                "phrases": [
                    "reduce waste",
                    "save the planet",
                    "environmental impact",
                    "climate change",
                    "global warming",
                    "carbon footprint",
                    "go green",
                    "eco-friendly",
                    "renewable energy",
                    "protect nature",
                ],
            },
            QuestionIntent.PERSONAL: {
                "keywords": [
                    "myself",
                    "personal",
                    "personally",
                    "individual",
                    "lifestyle",
                    "habit",
                    "daily",
                    "routine",
                    "health",
                    "wellness",
                    "self",
                    "mindset",
                    "behavior",
                    "practice",
                    "motivation",
                ],
                "phrases": [
                    "how can I",
                    "what should I",
                    "personal development",
                    "self improvement",
                    "my life",
                    "individual action",
                    "lifestyle change",
                    "personal growth",
                    "daily routine",
                ],
            },
            QuestionIntent.TECHNICAL: {
                "keywords": [
                    "build",
                    "code",
                    "program",
                    "software",
                    "hardware",
                    "system",
                    "algorithm",
                    "database",
                    "API",
                    "framework",
                    "technology",
                    "implement",
                    "develop",
                    "architecture",
                    "infrastructure",
                    "debug",
                    "optimize",
                    "deploy",
                    "scale",
                    "security",
                    "network",
                ],
                "phrases": [
                    "how to build",
                    "technical implementation",
                    "system design",
                    "software development",
                    "code architecture",
                    "tech stack",
                    "programming solution",
                    "technical approach",
                ],
            },
            QuestionIntent.BUSINESS: {
                "keywords": [
                    "business",
                    "company",
                    "organization",
                    "profit",
                    "revenue",
                    "market",
                    "customer",
                    "strategy",
                    "growth",
                    "sales",
                    "marketing",
                    "ROI",
                    "investment",
                    "competitive",
                    "brand",
                    "enterprise",
                    "startup",
                    "economy",
                    "financial",
                    "commercial",
                    "corporate",
                ],
                "phrases": [
                    "business strategy",
                    "market analysis",
                    "revenue growth",
                    "customer acquisition",
                    "competitive advantage",
                    "ROI analysis",
                    "business model",
                    "market share",
                    "profit margin",
                ],
            },
            QuestionIntent.SCIENTIFIC: {
                "keywords": [
                    "science",
                    "scientific",
                    "research",
                    "study",
                    "experiment",
                    "data",
                    "analysis",
                    "hypothesis",
                    "theory",
                    "evidence",
                    "biology",
                    "chemistry",
                    "physics",
                    "mathematics",
                    "statistics",
                    "methodology",
                    "peer-review",
                    "publication",
                    "discovery",
                ],
                "phrases": [
                    "scientific method",
                    "research study",
                    "data analysis",
                    "experimental design",
                    "hypothesis testing",
                    "evidence-based",
                    "scientific research",
                    "peer-reviewed",
                    "empirical data",
                ],
            },
            QuestionIntent.PHILOSOPHICAL: {
                "keywords": [
                    "philosophy",
                    "philosophical",
                    "ethics",
                    "ethical",
                    "moral",
                    "meaning",
                    "purpose",
                    "value",
                    "belief",
                    "principle",
                    "wisdom",
                    "truth",
                    "existence",
                    "consciousness",
                    "reality",
                    "society",
                    "justice",
                    "freedom",
                    "humanity",
                    "virtue",
                    "good",
                    "evil",
                ],
                "phrases": [
                    "meaning of",
                    "purpose of",
                    "ethical implications",
                    "moral responsibility",
                    "philosophical question",
                    "value system",
                    "greater good",
                    "human nature",
                    "societal impact",
                ],
            },
        }

    def detect_intent(self, question: str) -> IntentResult:
        """
        Detect the primary intent of a question.

        Args:
            question: The user's question or input

        Returns:
            IntentResult with primary and secondary intents
        """
        question_lower = question.lower()
        intent_scores: Dict[QuestionIntent, Tuple[float, List[str]]] = {}

        # Calculate scores for each intent
        for intent, patterns in self.intent_patterns.items():
            score = 0.0
            matched_keywords = []

            # Check keywords (with word boundaries)
            for keyword in patterns["keywords"]:
                if re.search(rf"\b{re.escape(keyword)}\b", question_lower):
                    score += 1.0
                    matched_keywords.append(keyword)

            # Check phrases (higher weight)
            for phrase in patterns["phrases"]:
                if phrase in question_lower:
                    score += 2.0
                    matched_keywords.append(phrase)

            # Normalize score - use a more reasonable normalization
            # Don't expect all keywords to match, just a few good ones
            if score > 0:
                # Normalize based on actual matches, not total possible
                normalized_score = min(score / 5.0, 1.0)  # 5 matches = 100% confidence
            else:
                normalized_score = 0.0

            intent_scores[intent] = (normalized_score, matched_keywords)

        # Sort intents by score
        sorted_intents = sorted(
            intent_scores.items(), key=lambda x: x[1][0], reverse=True
        )

        # Determine primary intent
        if sorted_intents[0][1][0] > 0.0:
            primary_intent = sorted_intents[0][0]
            confidence = sorted_intents[0][1][0]  # Already normalized
            keywords_matched = sorted_intents[0][1][1]
        else:
            # No clear intent detected
            primary_intent = QuestionIntent.GENERAL
            confidence = 0.5
            keywords_matched = []

        # Collect secondary intents (score > 0)
        secondary_intents = [
            intent for intent, (score, _) in sorted_intents[1:] if score > 0.0
        ][
            :2
        ]  # Maximum 2 secondary intents

        return IntentResult(
            primary_intent=primary_intent,
            confidence=confidence,
            secondary_intents=secondary_intents,
            keywords_matched=keywords_matched,
        )

    def get_recommended_perspectives(
        self, intent_result: IntentResult, max_perspectives: int = 3
    ) -> List[QuestionIntent]:
        """
        Get recommended perspectives based on intent detection.

        Args:
            intent_result: Result from detect_intent
            max_perspectives: Maximum number of perspectives to recommend

        Returns:
            List of recommended intents/perspectives
        """
        perspectives = [intent_result.primary_intent]

        # Add secondary intents if confidence is not too high
        if intent_result.confidence < 0.8 and intent_result.secondary_intents:
            perspectives.extend(intent_result.secondary_intents[: max_perspectives - 1])

        # Always include business perspective for general questions
        if (
            intent_result.primary_intent == QuestionIntent.GENERAL
            and QuestionIntent.BUSINESS not in perspectives
        ):
            perspectives.append(QuestionIntent.BUSINESS)

        return perspectives[:max_perspectives]
