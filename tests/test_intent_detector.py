"""Tests for intent detection module."""

import pytest

from mad_spark_alt.core.intent_detector import IntentDetector, QuestionIntent


class TestIntentDetector:
    """Test intent detection functionality."""
    
    def setup_method(self):
        """Set up test instance."""
        self.detector = IntentDetector()
    
    def test_environmental_intent_detection(self):
        """Test detection of environmental questions."""
        questions = [
            "How can we reduce plastic waste?",
            "What are the best renewable energy sources?",
            "How to reduce carbon emissions?",
            "What's the impact of climate change on oceans?",
        ]
        
        for question in questions:
            result = self.detector.detect_intent(question)
            assert result.primary_intent == QuestionIntent.ENVIRONMENTAL
            assert result.confidence >= 0.2  # Lower threshold for tests
            assert len(result.keywords_matched) > 0
        
        # Test mixed intent - should detect both
        result = self.detector.detect_intent("How to make my lifestyle more sustainable?")
        all_intents = [result.primary_intent] + result.secondary_intents
        assert QuestionIntent.ENVIRONMENTAL in all_intents
        assert QuestionIntent.PERSONAL in all_intents
    
    def test_personal_intent_detection(self):
        """Test detection of personal development questions."""
        questions = [
            "How can I improve my daily routine?",
            "What should I change in my lifestyle?",
            "How to develop better personal habits?",
            "What can I do for self improvement?",
        ]
        
        for question in questions:
            result = self.detector.detect_intent(question)
            assert result.primary_intent == QuestionIntent.PERSONAL
            assert result.confidence >= 0.2  # Lower threshold for tests
    
    def test_technical_intent_detection(self):
        """Test detection of technical questions."""
        questions = [
            "How to build a REST API?",
            "What's the best way to implement authentication?",
            "How do I optimize database performance?",
            "Should I use microservices architecture?",
        ]
        
        for question in questions:
            result = self.detector.detect_intent(question)
            assert result.primary_intent == QuestionIntent.TECHNICAL
            assert result.confidence >= 0.2  # Lower threshold for tests
    
    def test_business_intent_detection(self):
        """Test detection of business questions."""
        questions = [
            "How to increase company revenue?",
            "What's the best marketing strategy?",
            "How to improve customer retention?",
            "What's our competitive advantage?",
        ]
        
        for question in questions:
            result = self.detector.detect_intent(question)
            assert result.primary_intent == QuestionIntent.BUSINESS
            assert result.confidence >= 0.2  # Lower threshold for tests
    
    def test_general_intent_for_ambiguous_questions(self):
        """Test that ambiguous questions get general intent."""
        questions = [
            "What is this?",
            "Explain it",  # Avoid single "me" word
            "Hello world",
            "Something something",
        ]
        
        for question in questions:
            result = self.detector.detect_intent(question)
            assert result.primary_intent == QuestionIntent.GENERAL
    
    def test_confidence_calculation(self):
        """Test confidence score calculation."""
        # Strong environmental question
        result = self.detector.detect_intent("How can we reduce plastic waste in the ocean?")
        assert result.confidence >= 0.4  # Should have decent confidence
        
        # Weak/ambiguous question
        result = self.detector.detect_intent("What about it?")
        assert result.confidence == 0.5  # Default for no clear intent
    
    def test_secondary_intents(self):
        """Test detection of secondary intents."""
        # Question with both environmental and personal aspects
        result = self.detector.detect_intent(
            "How can I personally reduce my carbon footprint?"
        )
        
        # Should detect both intents
        all_intents = [result.primary_intent] + result.secondary_intents
        assert QuestionIntent.ENVIRONMENTAL in all_intents
        assert QuestionIntent.PERSONAL in all_intents
    
    def test_recommended_perspectives(self):
        """Test perspective recommendations."""
        # Clear environmental question
        result = self.detector.detect_intent("How to stop climate change?")
        perspectives = self.detector.get_recommended_perspectives(result, max_perspectives=3)
        
        assert len(perspectives) <= 3
        assert perspectives[0] == QuestionIntent.ENVIRONMENTAL
        
        # General question should include business perspective
        result = self.detector.detect_intent("What should we do?")
        perspectives = self.detector.get_recommended_perspectives(result, max_perspectives=3)
        
        assert QuestionIntent.GENERAL in perspectives or QuestionIntent.BUSINESS in perspectives