"""
Prompt Classification System for Dynamic QADI Analysis

This module automatically detects question types and complexity to enable
adaptive prompt selection for optimal QADI analysis results.
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, cast


class QuestionType(Enum):
    """Types of questions for adaptive prompt selection."""

    TECHNICAL = "technical"
    BUSINESS = "business"
    CREATIVE = "creative"
    RESEARCH = "research"
    PLANNING = "planning"
    PERSONAL = "personal"
    UNKNOWN = "unknown"


class ComplexityLevel(Enum):
    """Complexity levels for questions."""

    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"


@dataclass
class ClassificationResult:
    """Result of question classification."""

    question_type: QuestionType
    complexity: ComplexityLevel
    confidence: float  # 0.0 to 1.0
    detected_patterns: List[str]
    domain_hints: List[str]


class PromptClassifier:
    """Classifies questions to enable adaptive prompt selection."""

    def __init__(self) -> None:
        """Initialize the classifier with pattern dictionaries."""

        # Question type patterns - keywords that indicate specific types
        self.type_patterns = {
            QuestionType.TECHNICAL: {
                "keywords": [
                    "build",
                    "implement",
                    "code",
                    "develop",
                    "create",
                    "design",
                    "architecture",
                    "API",
                    "database",
                    "algorithm",
                    "system",
                    "software",
                    "app",
                    "website",
                    "platform",
                    "framework",
                    "programming",
                    "technical",
                    "engineering",
                    "deployment",
                    "infrastructure",
                    "scalability",
                    "performance",
                    "security",
                    # New additions
                    "REST",
                    "microservices",
                    "docker",
                    "kubernetes",
                    "cloud",
                    "backend",
                    "frontend",
                    "fullstack",
                    "DevOps",
                    "CI/CD",
                    "testing",
                    "debugging",
                    "optimization",
                    "refactor",
                    "migrate",
                    "integration",
                    "authentication",
                    "authorization",
                    "encryption",
                    "server",
                    "client",
                    "mobile",
                    "web",
                    "desktop",
                    "embedded",
                    "git",
                    "version",
                    "deploy",
                    "monitor",
                    "scale",
                    "cache",
                ],
                "phrases": [
                    "how to build",
                    "how to implement",
                    "technical approach",
                    "software solution",
                    "system design",
                    "tech stack",
                    "development process",
                    "coding approach",
                    # New additions
                    "REST API",
                    "microservices architecture",
                    "cloud migration",
                    "continuous integration",
                    "continuous deployment",
                    "test automation",
                    "performance optimization",
                    "security implementation",
                    "scale up",
                ],
            },
            QuestionType.BUSINESS: {
                "keywords": [
                    "strategy",
                    "revenue",
                    "profit",
                    "market",
                    "customers",
                    "sales",
                    "growth",
                    "business",
                    "company",
                    "startup",
                    "entrepreneur",
                    "investment",
                    "funding",
                    "monetize",
                    "competition",
                    "ROI",
                    "stakeholders",
                    "partnership",
                    "expansion",
                    "operations",
                    "productivity",
                    "efficiency",
                    "team",
                    "management",
                    # New additions
                    "quarterly",
                    "annual",
                    "KPI",
                    "metrics",
                    "B2B",
                    "B2C",
                    "SaaS",
                    "acquisition",
                    "retention",
                    "churn",
                    "CAC",
                    "LTV",
                    "margin",
                    "pricing",
                    "positioning",
                    "branding",
                    "marketing",
                    "enterprise",
                    "scale",
                    "pivot",
                    "disruption",
                    "innovation",
                    "valuation",
                    "competitive",
                    "differentiation",
                    "moat",
                    "TAM",
                    "SAM",
                    "SOM",
                ],
                "phrases": [
                    "business strategy",
                    "market opportunity",
                    "revenue model",
                    "growth strategy",
                    "competitive advantage",
                    "business plan",
                    "team productivity",
                    "operational efficiency",
                    # New additions
                    "increase revenue",
                    "market share",
                    "customer acquisition",
                    "business model",
                    "go to market",
                    "product market fit",
                    "competitive analysis",
                    "market penetration",
                    "value proposition",
                ],
            },
            QuestionType.CREATIVE: {
                "keywords": [
                    "creative",
                    "artistic",
                    "design",
                    "innovative",
                    "novel",
                    "brainstorm",
                    "imagination",
                    "original",
                    "unique",
                    "inspiring",
                    "aesthetic",
                    "visual",
                    "artistic",
                    "conceptual",
                    "experimental",
                    "unconventional",
                    "breakthrough",
                    "revolutionary",
                    "visionary",
                    # New additions
                    "logo",
                    "brand",
                    "UI",
                    "UX",
                    "user experience",
                    "interface",
                    "graphics",
                    "illustration",
                    "animation",
                    "color",
                    "typography",
                    "layout",
                    "composition",
                    "style",
                    "theme",
                    "mood",
                    "atmosphere",
                    "storytelling",
                    "narrative",
                    "concept",
                    "ideation",
                    "prototype",
                    "sketch",
                    "mockup",
                    "wireframe",
                    "art",
                    "craft",
                    "media",
                ],
                "phrases": [
                    "creative solution",
                    "artistic approach",
                    "innovative idea",
                    "design thinking",
                    "creative process",
                    "artistic expression",
                    "novel approach",
                    "thinking outside the box",
                    # New additions
                    "design a logo",
                    "create a brand",
                    "user interface design",
                    "visual identity",
                    "creative concept",
                    "artistic vision",
                    "design language",
                    "creative direction",
                    "brand experience",
                ],
            },
            QuestionType.RESEARCH: {
                "keywords": [
                    "analyze",
                    "study",
                    "investigate",
                    "research",
                    "understand",
                    "explore",
                    "examine",
                    "evaluate",
                    "assess",
                    "compare",
                    "data",
                    "evidence",
                    "findings",
                    "methodology",
                    "hypothesis",
                    "theory",
                    "academic",
                    "scientific",
                    "empirical",
                    "statistical",
                    # New additions
                    "effectiveness",
                    "impact",
                    "correlation",
                    "causation",
                    "trend",
                    "pattern",
                    "insight",
                    "observation",
                    "experiment",
                    "survey",
                    "qualitative",
                    "quantitative",
                    "metrics",
                    "measurement",
                    "benchmark",
                    "literature",
                    "review",
                    "meta-analysis",
                    "case study",
                    "sample",
                    "variable",
                    "control",
                    "significance",
                    "validation",
                    "peer-review",
                ],
                "phrases": [
                    "research question",
                    "analysis of",
                    "study of",
                    "investigation into",
                    "understanding of",
                    "exploration of",
                    "what causes",
                    "why does",
                    "relationship between",
                    "impact of",
                    "effect of",
                    # New additions
                    "analyze the effectiveness",
                    "evaluate the impact",
                    "assess the correlation",
                    "investigate the relationship",
                    "examine the influence",
                    "study the effects",
                    "research the connection",
                    "measure the outcome",
                    "test the hypothesis",
                ],
            },
            QuestionType.PLANNING: {
                "keywords": [
                    "plan",
                    "steps",
                    "timeline",
                    "schedule",
                    "organize",
                    "structure",
                    "process",
                    "workflow",
                    "procedure",
                    "roadmap",
                    "milestone",
                    "phases",
                    "stages",
                    "sequence",
                    "order",
                    "prioritize",
                    "coordinate",
                    "manage",
                    "execute",
                    "deliver",
                ],
                "phrases": [
                    "step by step",
                    "action plan",
                    "project plan",
                    "timeline for",
                    "how to organize",
                    "planning process",
                    "implementation plan",
                    "roadmap for",
                    "phases of",
                    "sequence of steps",
                ],
            },
            QuestionType.PERSONAL: {
                "keywords": [
                    "personal",
                    "myself",
                    "my life",
                    "career",
                    "health",
                    "habits",
                    "skills",
                    "learning",
                    "growth",
                    "development",
                    "improvement",
                    "self",
                    "individual",
                    "lifestyle",
                    "wellbeing",
                    "goals",
                    "motivation",
                    "confidence",
                    "relationships",
                    "work-life",
                    # New additions
                    "daily",
                    "routine",
                    "balance",
                    "stress",
                    "mindfulness",
                    "meditation",
                    "fitness",
                    "nutrition",
                    "sleep",
                    "energy",
                    "focus",
                    "discipline",
                    "happiness",
                    "fulfillment",
                    "purpose",
                    "passion",
                    "values",
                    "beliefs",
                    "mindset",
                    "attitude",
                    "resilience",
                    "emotional",
                    "mental",
                    "physical",
                ],
                "phrases": [
                    "improve my",
                    "develop my",
                    "how can I",
                    "personal growth",
                    "self improvement",
                    "life goals",
                    "career development",
                    "personal skills",
                    "individual growth",
                    # New additions
                    "work life balance",
                    "daily productivity",
                    "personal effectiveness",
                    "career advancement",
                    "life satisfaction",
                    "stress management",
                    "time management",
                    "personal development",
                    "self care",
                ],
            },
        }

        # Complexity indicators
        self.complexity_patterns = {
            ComplexityLevel.SIMPLE: {
                "indicators": [
                    "basic",
                    "simple",
                    "easy",
                    "quick",
                    "fast",
                    "straightforward",
                    "beginner",
                    "introduction",
                    "overview",
                    "summary",
                ],
                "question_words": ["what", "when", "where", "who"],
                "length_threshold": 50,  # characters
            },
            ComplexityLevel.COMPLEX: {
                "indicators": [
                    "complex",
                    "comprehensive",
                    "detailed",
                    "advanced",
                    "sophisticated",
                    "enterprise",
                    "large-scale",
                    "multi-faceted",
                    "interdisciplinary",
                    "strategic",
                    "long-term",
                    "systematic",
                    "integrated",
                ],
                "question_words": ["how might", "what if", "why does", "how could"],
                "length_threshold": 150,  # characters
            },
        }

        # Domain-specific hints
        self.domain_patterns = {
            "software": ["code", "programming", "development", "API", "framework"],
            "education": ["learn", "teach", "student", "education", "training"],
            "healthcare": ["health", "medical", "patient", "treatment", "diagnosis"],
            "finance": ["money", "investment", "financial", "budget", "cost"],
            "marketing": ["marketing", "advertising", "brand", "promotion", "campaign"],
            "gaming": ["game", "player", "gaming", "entertainment", "fun"],
            "science": ["research", "experiment", "data", "hypothesis", "theory"],
        }

    def classify(self, question: str) -> ClassificationResult:
        """
        Classify a question and return detailed results.

        Args:
            question: The question text to classify

        Returns:
            ClassificationResult with type, complexity, and confidence
        """
        question_lower = question.lower()

        # Detect question type
        type_scores = self._calculate_type_scores(question_lower)
        question_type, type_confidence = self._get_best_type(type_scores)

        # Detect complexity
        complexity = self._detect_complexity(question_lower, len(question))

        # Find detected patterns
        detected_patterns = self._get_detected_patterns(question_lower, question_type)

        # Detect domain hints
        domain_hints = self._detect_domains(question_lower)

        return ClassificationResult(
            question_type=question_type,
            complexity=complexity,
            confidence=type_confidence,
            detected_patterns=detected_patterns,
            domain_hints=domain_hints,
        )

    def _calculate_type_scores(self, question_lower: str) -> Dict[QuestionType, float]:
        """Calculate scores for each question type using word boundary matching."""
        scores = {}

        for question_type, patterns in self.type_patterns.items():
            score = 0.0
            matched_keywords = 0

            # Score keywords with word boundary matching
            for keyword in patterns["keywords"]:
                # Create regex pattern with word boundaries
                # Always escape keywords to prevent regex metacharacter issues
                pattern = r"\b" + re.escape(keyword.lower()) + r"\b"

                if re.search(pattern, question_lower, re.IGNORECASE):
                    score += 1.0
                    matched_keywords += 1

            # Score phrases (higher weight) - exact phrase matching
            for phrase in patterns["phrases"]:
                if phrase.lower() in question_lower:
                    score += 3.0  # Increased weight for phrases
                    matched_keywords += 1

            # Bonus for multiple matches (indicates stronger signal)
            if matched_keywords > 2:
                score *= 1.2  # 20% bonus for multiple matches

            # Store raw score (not normalized yet)
            scores[question_type] = score

        # Normalize scores relative to each other rather than max possible
        max_score = max(scores.values()) if scores else 1.0
        if max_score > 0:
            for qtype in scores:
                scores[qtype] = scores[qtype] / max_score

        return scores

    def _get_best_type(
        self, scores: Dict[QuestionType, float]
    ) -> Tuple[QuestionType, float]:
        """Get the best question type and confidence score."""
        if not scores:
            return QuestionType.UNKNOWN, 0.0

        # Filter out zero scores
        non_zero_scores = {k: v for k, v in scores.items() if v > 0}
        if not non_zero_scores:
            return QuestionType.UNKNOWN, 0.0

        # Find the type with highest score
        best_type = max(non_zero_scores.keys(), key=lambda k: non_zero_scores[k])
        best_score = non_zero_scores[best_type]

        # Calculate confidence based on:
        # 1. Absolute score (how many matches)
        # 2. Separation from second best (how clear the winner is)
        sorted_scores = sorted(non_zero_scores.values(), reverse=True)

        if len(sorted_scores) > 1:
            second_best = sorted_scores[1]
            # Separation factor: how much better is best than second
            separation_factor = (
                (best_score - second_best) / best_score if best_score > 0 else 0
            )

            # Confidence combines normalized score and separation
            # More weight on separation for clearer classification
            confidence = (best_score * 0.4) + (separation_factor * 0.6)
        else:
            # Only one type matched - confidence based on score alone
            confidence = best_score * 0.7  # Slightly lower confidence for single match

        # Adjust confidence based on raw match count from original scores
        # This rewards questions that match multiple keywords
        raw_matches = sum(1 for v in scores.values() if v > 0)
        if raw_matches > 1:
            confidence *= 1.1  # Boost for matching multiple types (indicates richness)

        # Ensure confidence is between 0 and 1
        confidence = min(max(confidence, 0.0), 1.0)

        # Lower minimum confidence threshold to 15%
        if confidence < 0.15:
            return QuestionType.UNKNOWN, confidence

        return best_type, confidence

    def _detect_complexity(
        self, question_lower: str, question_length: int
    ) -> ComplexityLevel:
        """Detect the complexity level of the question."""

        # Check for simple indicators
        simple_patterns = cast(
            Dict[str, Any], self.complexity_patterns[ComplexityLevel.SIMPLE]
        )
        simple_score = 0

        for indicator in cast(List[str], simple_patterns["indicators"]):
            if indicator in question_lower:
                simple_score += 1

        for qword in cast(List[str], simple_patterns["question_words"]):
            if question_lower.startswith(qword):
                simple_score += 1

        if question_length < cast(int, simple_patterns["length_threshold"]):
            simple_score += 1

        # Check for complex indicators
        complex_patterns = cast(
            Dict[str, Any], self.complexity_patterns[ComplexityLevel.COMPLEX]
        )
        complex_score = 0

        for indicator in cast(List[str], complex_patterns["indicators"]):
            if indicator in question_lower:
                complex_score += 1

        for qword in cast(List[str], complex_patterns["question_words"]):
            if qword in question_lower:
                complex_score += 1

        if question_length > cast(int, complex_patterns["length_threshold"]):
            complex_score += 1

        # Determine complexity
        if complex_score > simple_score and complex_score >= 2:
            return ComplexityLevel.COMPLEX
        elif simple_score > complex_score and simple_score >= 2:
            return ComplexityLevel.SIMPLE
        else:
            return ComplexityLevel.MEDIUM

    def _get_detected_patterns(
        self, question_lower: str, question_type: QuestionType
    ) -> List[str]:
        """Get the specific patterns that were detected."""
        patterns = []

        if question_type in self.type_patterns:
            type_patterns = self.type_patterns[question_type]

            # Find matching keywords
            for keyword in type_patterns["keywords"]:
                if keyword in question_lower:
                    patterns.append(f"keyword: {keyword}")

            # Find matching phrases
            for phrase in type_patterns["phrases"]:
                if phrase in question_lower:
                    patterns.append(f"phrase: {phrase}")

        return patterns[:5]  # Limit to top 5 for readability

    def _detect_domains(self, question_lower: str) -> List[str]:
        """Detect domain-specific hints in the question."""
        domains = []

        for domain, keywords in self.domain_patterns.items():
            domain_score = sum(1 for keyword in keywords if keyword in question_lower)
            if domain_score > 0:
                domains.append((domain, domain_score))

        # Sort by domain score (descending) and format
        domains.sort(key=lambda x: x[1], reverse=True)
        formatted_domains = [
            f"{domain} ({score} indicators)" for domain, score in domains
        ]

        return formatted_domains[:3]  # Top 3 domains


# Global classifier instance
classifier = PromptClassifier()


def classify_question(question: str) -> ClassificationResult:
    """
    Convenience function to classify a question.

    Args:
        question: The question text to classify

    Returns:
        ClassificationResult with type, complexity, and confidence
    """
    return classifier.classify(question)
