"""
Prompt Classification System for Dynamic QADI Analysis

This module automatically detects question types and complexity to enable
adaptive prompt selection for optimal QADI analysis results.
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple


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
    
    def __init__(self):
        """Initialize the classifier with pattern dictionaries."""
        
        # Question type patterns - keywords that indicate specific types
        self.type_patterns = {
            QuestionType.TECHNICAL: {
                "keywords": [
                    "build", "implement", "code", "develop", "create", "design",
                    "architecture", "API", "database", "algorithm", "system",
                    "software", "app", "website", "platform", "framework",
                    "programming", "technical", "engineering", "deployment",
                    "infrastructure", "scalability", "performance", "security"
                ],
                "phrases": [
                    "how to build", "how to implement", "technical approach",
                    "software solution", "system design", "tech stack",
                    "development process", "coding approach"
                ]
            },
            
            QuestionType.BUSINESS: {
                "keywords": [
                    "strategy", "revenue", "profit", "market", "customers", "sales",
                    "growth", "business", "company", "startup", "entrepreneur",
                    "investment", "funding", "monetize", "competition", "ROI",
                    "stakeholders", "partnership", "expansion", "operations",
                    "productivity", "efficiency", "team", "management"
                ],
                "phrases": [
                    "business strategy", "market opportunity", "revenue model",
                    "growth strategy", "competitive advantage", "business plan",
                    "team productivity", "operational efficiency"
                ]
            },
            
            QuestionType.CREATIVE: {
                "keywords": [
                    "creative", "artistic", "design", "innovative", "novel",
                    "brainstorm", "imagination", "original", "unique", "inspiring",
                    "aesthetic", "visual", "artistic", "conceptual", "experimental",
                    "unconventional", "breakthrough", "revolutionary", "visionary"
                ],
                "phrases": [
                    "creative solution", "artistic approach", "innovative idea",
                    "design thinking", "creative process", "artistic expression",
                    "novel approach", "thinking outside the box"
                ]
            },
            
            QuestionType.RESEARCH: {
                "keywords": [
                    "analyze", "study", "investigate", "research", "understand",
                    "explore", "examine", "evaluate", "assess", "compare",
                    "data", "evidence", "findings", "methodology", "hypothesis",
                    "theory", "academic", "scientific", "empirical", "statistical"
                ],
                "phrases": [
                    "research question", "analysis of", "study of", "investigation into",
                    "understanding of", "exploration of", "what causes", "why does",
                    "relationship between", "impact of", "effect of"
                ]
            },
            
            QuestionType.PLANNING: {
                "keywords": [
                    "plan", "steps", "timeline", "schedule", "organize", "structure",
                    "process", "workflow", "procedure", "roadmap", "milestone",
                    "phases", "stages", "sequence", "order", "prioritize",
                    "coordinate", "manage", "execute", "deliver"
                ],
                "phrases": [
                    "step by step", "action plan", "project plan", "timeline for",
                    "how to organize", "planning process", "implementation plan",
                    "roadmap for", "phases of", "sequence of steps"
                ]
            },
            
            QuestionType.PERSONAL: {
                "keywords": [
                    "personal", "myself", "my life", "career", "health", "habits",
                    "skills", "learning", "growth", "development", "improvement",
                    "self", "individual", "lifestyle", "wellbeing", "goals",
                    "motivation", "confidence", "relationships", "work-life"
                ],
                "phrases": [
                    "improve my", "develop my", "how can I", "personal growth",
                    "self improvement", "life goals", "career development",
                    "personal skills", "individual growth"
                ]
            }
        }
        
        # Complexity indicators
        self.complexity_patterns = {
            ComplexityLevel.SIMPLE: {
                "indicators": [
                    "basic", "simple", "easy", "quick", "fast", "straightforward",
                    "beginner", "introduction", "overview", "summary"
                ],
                "question_words": ["what", "when", "where", "who"],
                "length_threshold": 50  # characters
            },
            
            ComplexityLevel.COMPLEX: {
                "indicators": [
                    "complex", "comprehensive", "detailed", "advanced", "sophisticated",
                    "enterprise", "large-scale", "multi-faceted", "interdisciplinary",
                    "strategic", "long-term", "systematic", "integrated"
                ],
                "question_words": ["how might", "what if", "why does", "how could"],
                "length_threshold": 150  # characters
            }
        }
        
        # Domain-specific hints
        self.domain_patterns = {
            "software": ["code", "programming", "development", "API", "framework"],
            "education": ["learn", "teach", "student", "education", "training"],
            "healthcare": ["health", "medical", "patient", "treatment", "diagnosis"],
            "finance": ["money", "investment", "financial", "budget", "cost"],
            "marketing": ["marketing", "advertising", "brand", "promotion", "campaign"],
            "gaming": ["game", "player", "gaming", "entertainment", "fun"],
            "science": ["research", "experiment", "data", "hypothesis", "theory"]
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
            domain_hints=domain_hints
        )
    
    def _calculate_type_scores(self, question_lower: str) -> Dict[QuestionType, float]:
        """Calculate scores for each question type."""
        scores = {}
        
        for question_type, patterns in self.type_patterns.items():
            score = 0.0
            
            # Score keywords
            for keyword in patterns["keywords"]:
                if keyword in question_lower:
                    score += 1.0
            
            # Score phrases (higher weight)
            for phrase in patterns["phrases"]:
                if phrase in question_lower:
                    score += 2.0
            
            # Normalize by total possible score
            max_possible = len(patterns["keywords"]) + (len(patterns["phrases"]) * 2)
            scores[question_type] = score / max_possible if max_possible > 0 else 0.0
        
        return scores
    
    def _get_best_type(self, scores: Dict[QuestionType, float]) -> Tuple[QuestionType, float]:
        """Get the best question type and confidence score."""
        if not scores:
            return QuestionType.UNKNOWN, 0.0
        
        # Find the type with highest score
        best_type = max(scores, key=scores.get)
        best_score = scores[best_type]
        
        # Calculate confidence based on score and separation from second best
        sorted_scores = sorted(scores.values(), reverse=True)
        
        if best_score == 0.0:
            return QuestionType.UNKNOWN, 0.0
        
        # Confidence is higher if there's clear separation from second best
        if len(sorted_scores) > 1 and sorted_scores[1] > 0:
            separation = (best_score - sorted_scores[1]) / best_score
            confidence = min(best_score + (separation * 0.3), 1.0)
        else:
            confidence = best_score
        
        # Minimum confidence threshold
        if confidence < 0.3:
            return QuestionType.UNKNOWN, confidence
        
        return best_type, confidence
    
    def _detect_complexity(self, question_lower: str, question_length: int) -> ComplexityLevel:
        """Detect the complexity level of the question."""
        
        # Check for simple indicators
        simple_patterns = self.complexity_patterns[ComplexityLevel.SIMPLE]
        simple_score = 0
        
        for indicator in simple_patterns["indicators"]:
            if indicator in question_lower:
                simple_score += 1
        
        for qword in simple_patterns["question_words"]:
            if question_lower.startswith(qword):
                simple_score += 1
        
        if question_length < simple_patterns["length_threshold"]:
            simple_score += 1
        
        # Check for complex indicators
        complex_patterns = self.complexity_patterns[ComplexityLevel.COMPLEX]
        complex_score = 0
        
        for indicator in complex_patterns["indicators"]:
            if indicator in question_lower:
                complex_score += 1
        
        for qword in complex_patterns["question_words"]:
            if qword in question_lower:
                complex_score += 1
        
        if question_length > complex_patterns["length_threshold"]:
            complex_score += 1
        
        # Determine complexity
        if complex_score > simple_score and complex_score >= 2:
            return ComplexityLevel.COMPLEX
        elif simple_score > complex_score and simple_score >= 2:
            return ComplexityLevel.SIMPLE
        else:
            return ComplexityLevel.MEDIUM
    
    def _get_detected_patterns(self, question_lower: str, question_type: QuestionType) -> List[str]:
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
                domains.append(f"{domain} ({domain_score} indicators)")
        
        # Sort by number of indicators (descending)
        domains.sort(key=lambda x: int(x.split('(')[1].split(' ')[0]), reverse=True)
        
        return domains[:3]  # Top 3 domains


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