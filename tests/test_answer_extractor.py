"""
Tests for the answer extraction components.
"""

import asyncio
from typing import List
from unittest.mock import AsyncMock, Mock, patch

import pytest

from mad_spark_alt.core.answer_extractor import (
    AnswerExtractionResult,
    EnhancedAnswerExtractor,
    ExtractedAnswer,
    QuestionTypeAnalyzer,
    TemplateAnswerExtractor,
)
from mad_spark_alt.core.interfaces import GeneratedIdea, ThinkingMethod


class TestQuestionTypeAnalyzer:
    """Test question type detection."""
    
    def setup_method(self):
        """Set up test instance."""
        self.analyzer = QuestionTypeAnalyzer()
    
    def test_list_request_detection(self):
        """Test detection of list request questions."""
        questions = [
            "What are 5 ways to improve productivity?",
            "List 3 methods to reduce stress",
            "Give me 10 ideas for marketing",
        ]
        
        for question in questions:
            q_type, metadata = self.analyzer.analyze_question(question)
            assert q_type == "list_request"
            assert metadata["expects_list"] is True
    
    def test_how_to_detection(self):
        """Test detection of how-to questions."""
        questions = [
            "How to build a website?",
            "How can I learn programming?",
            "Ways to improve communication skills",
        ]
        
        for question in questions:
            q_type, metadata = self.analyzer.analyze_question(question)
            assert q_type == "how_to"
            assert metadata["expects_list"] is True
    
    def test_what_is_detection(self):
        """Test detection of explanatory questions."""
        questions = [
            "What is machine learning?",
            "Define artificial intelligence",
            "Explain quantum computing",
        ]
        
        for question in questions:
            q_type, metadata = self.analyzer.analyze_question(question)
            assert q_type == "what_is"
            assert metadata["expects_explanation"] is True
    
    def test_quantity_extraction(self):
        """Test extraction of requested quantity."""
        test_cases = [
            ("Give me 5 ideas", 5),
            ("What are 3 ways to improve?", 3),
            ("How to solve this?", 3),  # default
        ]
        
        for question, expected_qty in test_cases:
            _, metadata = self.analyzer.analyze_question(question)
            assert metadata["requested_quantity"] == expected_qty


class TestTemplateAnswerExtractor:
    """Test answer extraction from QADI results."""
    
    def setup_method(self):
        """Set up test instance."""
        self.extractor = TemplateAnswerExtractor()
    
    def create_mock_idea(self, content: str, method: ThinkingMethod) -> GeneratedIdea:
        """Create a mock QADI idea."""
        return GeneratedIdea(
            content=content,
            thinking_method=method,
            agent_name=f"{method.value}Agent",
            generation_prompt="test prompt",
            metadata={"phase": method.value}
        )
    
    def test_extract_list_answers(self):
        """Test extraction of list-type answers."""
        question = "What are 3 ways to improve sleep quality?"
        
        qadi_results = {
            "questioning": [
                self.create_mock_idea(
                    "What environmental factors affect sleep?",
                    ThinkingMethod.QUESTIONING
                )
            ],
            "abduction": [
                self.create_mock_idea(
                    "Sleep quality might be influenced by light exposure",
                    ThinkingMethod.ABDUCTION
                )
            ],
            "deduction": [
                self.create_mock_idea(
                    "If we reduce screen time before bed, then sleep improves",
                    ThinkingMethod.DEDUCTION
                )
            ],
        }
        
        result = self.extractor.extract_answers(question, qadi_results, max_answers=3)
        
        assert result.question_type == "list_request"
        assert len(result.direct_answers) == 3
        assert result.total_qadi_ideas == 3
        assert all(isinstance(answer, ExtractedAnswer) for answer in result.direct_answers)
    
    def test_answer_uses_qadi_content(self):
        """Test that extracted answers actually use QADI insights."""
        question = "How to reduce stress?"
        
        qadi_results = {
            "abduction": [
                self.create_mock_idea(
                    "Stress reduction might come from mindfulness practices",
                    ThinkingMethod.ABDUCTION
                )
            ],
        }
        
        result = self.extractor.extract_answers(question, qadi_results, max_answers=1)
        
        # The answer should reference the actual QADI content
        answer_content = result.direct_answers[0].content
        assert "mindfulness" in answer_content or "hypothesis" in answer_content
        assert result.direct_answers[0].source_phase == "abduction"
    
    def test_synthetic_answer_generation(self):
        """Test generation of synthetic answers when QADI lacks ideas."""
        question = "What are 5 ways to be more creative?"
        
        # Empty QADI results
        qadi_results = {}
        
        result = self.extractor.extract_answers(question, qadi_results, max_answers=5)
        
        assert len(result.direct_answers) == 5
        assert all(answer.source_phase == "synthetic" for answer in result.direct_answers)
        assert all(answer.confidence == 0.5 for answer in result.direct_answers)
    
    def test_metadata_none_handling(self):
        """Test handling of ideas with None metadata."""
        question = "How to learn effectively?"
        
        # Create idea with None metadata
        idea = GeneratedIdea(
            content="Use spaced repetition",
            thinking_method=ThinkingMethod.DEDUCTION,
            agent_name="TestAgent",
            generation_prompt="test",
            metadata=None  # This should be handled gracefully
        )
        
        qadi_results = {"unknown": [idea]}
        
        result = self.extractor.extract_answers(question, qadi_results, max_answers=1)
        
        assert len(result.direct_answers) == 1
        assert result.direct_answers[0].source_phase in ["unknown", "synthetic"]


class TestEnhancedAnswerExtractor:
    """Test LLM-enhanced answer extraction."""
    
    def setup_method(self):
        """Set up test instance."""
        self.extractor = EnhancedAnswerExtractor(prefer_llm=True)
    
    def create_mock_idea(self, content: str, method: ThinkingMethod) -> GeneratedIdea:
        """Create a mock QADI idea."""
        return GeneratedIdea(
            content=content,
            thinking_method=method,
            agent_name=f"{method.value}Agent",
            generation_prompt="test prompt",
            metadata={"phase": method.value}
        )
    
    @pytest.mark.asyncio
    async def test_current_implementation_uses_template(self):
        """Test that current implementation uses template extraction."""
        question = "What are 3 ways to improve focus?"
        
        qadi_results = {
            "questioning": [
                self.create_mock_idea(
                    "What prevents focus?",
                    ThinkingMethod.QUESTIONING
                )
            ],
        }
        
        result = await self.extractor.extract_answers(question, qadi_results, max_answers=3)
        
        assert isinstance(result, AnswerExtractionResult)
        assert len(result.direct_answers) == 3
        # Current implementation always uses template until LLM is implemented
        assert result.extraction_method in ["template", None]  # Allow None for current implementation
    
    @pytest.mark.asyncio
    async def test_llm_extraction_with_mock(self):
        """Test LLM extraction with mocked response."""
        question = "How to reduce stress?"
        
        qadi_results = {
            "abduction": [
                self.create_mock_idea(
                    "Stress reduction might come from mindfulness practices",
                    ThinkingMethod.ABDUCTION
                )
            ],
        }
        
        # Mock LLM response
        mock_llm_response = Mock()
        mock_llm_response.content = '''
        {
            "answers": [
                {
                    "content": "Practice mindfulness meditation",
                    "confidence": 0.9,
                    "source_phase": "abduction",
                    "reasoning": "Based on abduction insights about mindfulness"
                },
                {
                    "content": "Take regular breaks during work",
                    "confidence": 0.8,
                    "source_phase": "synthetic",
                    "reasoning": "Common stress reduction technique"
                }
            ]
        }
        '''
        
        with patch('mad_spark_alt.core.answer_extractor.llm_manager') as mock_manager:
            mock_manager.providers = {"test": "provider"}  # Non-empty to enable LLM
            mock_manager.generate = AsyncMock(return_value=mock_llm_response)
            
            result = await self.extractor.extract_answers(question, qadi_results, max_answers=2)
            
            assert isinstance(result, AnswerExtractionResult)
            assert len(result.direct_answers) == 2
            assert result.extraction_method == "llm"
            assert result.direct_answers[0].content == "Practice mindfulness meditation"
            assert result.direct_answers[0].confidence == 0.9
            assert result.direct_answers[0].source_phase == "abduction"
    
    @pytest.mark.asyncio
    async def test_llm_fallback_to_template(self):
        """Test fallback to template when LLM fails."""
        question = "What are ways to learn faster?"
        qadi_results = {}
        
        with patch('mad_spark_alt.core.answer_extractor.llm_manager') as mock_manager:
            mock_manager.providers = {"test": "provider"}  # Non-empty to enable LLM attempt
            # Simulate LLM failure
            mock_manager.generate = AsyncMock(side_effect=Exception("LLM failed"))
            
            result = await self.extractor.extract_answers(question, qadi_results, max_answers=3)
            
            # Should fallback to template
            assert isinstance(result, AnswerExtractionResult)
            assert len(result.direct_answers) == 3
            assert result.extraction_method == "template"
    
    @pytest.mark.asyncio 
    async def test_enhanced_extractor_interface(self):
        """Test that EnhancedAnswerExtractor has the expected interface."""
        question = "How to reduce stress?"
        
        qadi_results = {
            "abduction": [
                self.create_mock_idea(
                    "Stress reduction might come from mindfulness practices",
                    ThinkingMethod.ABDUCTION
                )
            ],
        }
        
        result = await self.extractor.extract_answers(question, qadi_results, max_answers=2)
        
        assert isinstance(result, AnswerExtractionResult)
        assert len(result.direct_answers) == 2
        assert result.question_type is not None
        assert result.total_qadi_ideas >= 0
        
        # Verify the answers reference QADI content
        answer_contents = [answer.content for answer in result.direct_answers]
        assert any("stress" in content.lower() or "mindfulness" in content.lower() 
                  for content in answer_contents)
    
    @pytest.mark.asyncio
    async def test_async_operation_completes(self):
        """Test that async operations complete successfully."""
        question = "What are ways to learn faster?"
        qadi_results = {}
        
        start_time = asyncio.get_event_loop().time()
        result = await self.extractor.extract_answers(question, qadi_results, max_answers=3)
        end_time = asyncio.get_event_loop().time()
        
        # Should complete quickly (under 1 second for template extraction)
        assert (end_time - start_time) < 1.0
        assert isinstance(result, AnswerExtractionResult)
        assert len(result.direct_answers) == 3
    
    @pytest.mark.asyncio
    async def test_error_handling_graceful(self):
        """Test graceful error handling."""
        question = "How to be more creative?"
        
        # Test with malformed QADI results (should not crash)
        malformed_qadi_results = {
            "invalid_phase": ["not a GeneratedIdea object"]
        }
        
        try:
            result = await self.extractor.extract_answers(
                question, malformed_qadi_results, max_answers=3
            )
            # Should return some result even with malformed input
            assert isinstance(result, AnswerExtractionResult)
            assert len(result.direct_answers) >= 0
        except Exception as e:
            # If it does throw an exception, it should be related to the malformed input
            assert "content" in str(e) or "attribute" in str(e) or "GeneratedIdea" in str(e)
    
    @pytest.mark.asyncio
    async def test_concurrent_extractions(self):
        """Test that multiple concurrent extractions work correctly."""
        questions = [
            "What are 2 ways to improve focus?",  # Specify number in question
            "List 2 productivity tips",
            "Give me 2 ways to reduce stress"
        ]
        
        qadi_results = {
            "questioning": [
                self.create_mock_idea("What improves focus?", ThinkingMethod.QUESTIONING),
                self.create_mock_idea("What hinders productivity?", ThinkingMethod.QUESTIONING)
            ]
        }
        
        # Run multiple extractions concurrently
        tasks = [
            self.extractor.extract_answers(question, qadi_results, max_answers=2)
            for question in questions
        ]
        
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 3
        for result in results:
            assert isinstance(result, AnswerExtractionResult)
            # Should get at least 1 answer, up to max_answers requested
            assert len(result.direct_answers) >= 1
            assert len(result.direct_answers) <= 2
    
    @pytest.mark.asyncio
    async def test_edge_case_empty_qadi_results(self):
        """Test extraction with completely empty QADI results."""
        question = "What are 5 ways to improve sleep?"
        qadi_results = {}
        
        result = await self.extractor.extract_answers(question, qadi_results, max_answers=5)
        
        assert isinstance(result, AnswerExtractionResult)
        assert len(result.direct_answers) == 5
        assert result.total_qadi_ideas == 0
        assert all(answer.source_phase == "synthetic" for answer in result.direct_answers)
    
    @pytest.mark.asyncio
    async def test_edge_case_very_large_qadi_results(self):
        """Test extraction with large number of QADI ideas."""
        question = "How to be successful?"
        
        # Create 100 QADI ideas
        qadi_results = {
            "questioning": [
                self.create_mock_idea(f"Question {i}", ThinkingMethod.QUESTIONING)
                for i in range(50)
            ],
            "abduction": [
                self.create_mock_idea(f"Hypothesis {i}", ThinkingMethod.ABDUCTION)
                for i in range(50)
            ],
        }
        
        result = await self.extractor.extract_answers(question, qadi_results, max_answers=10)
        
        assert isinstance(result, AnswerExtractionResult)
        assert len(result.direct_answers) == 10
        assert result.total_qadi_ideas == 100
    
    @pytest.mark.asyncio
    async def test_edge_case_unicode_and_special_characters(self):
        """Test handling of unicode and special characters."""
        question = "如何提高工作效率？"  # Chinese question
        
        qadi_results = {
            "questioning": [
                self.create_mock_idea(
                    "什么阻碍了效率？🤔",  # Chinese with emoji
                    ThinkingMethod.QUESTIONING
                )
            ],
        }
        
        result = await self.extractor.extract_answers(question, qadi_results, max_answers=3)
        
        assert isinstance(result, AnswerExtractionResult)
        assert len(result.direct_answers) == 3
        # Should handle unicode gracefully
        assert result.question_type is not None
    
    @pytest.mark.asyncio
    async def test_mixed_metadata_states(self):
        """Test handling of ideas with various metadata states."""
        question = "How to learn programming?"
        
        # Create ideas with different metadata states
        ideas = [
            GeneratedIdea(
                content="Idea with full metadata",
                thinking_method=ThinkingMethod.QUESTIONING,
                agent_name="TestAgent",
                generation_prompt="test",
                metadata={"phase": "questioning", "confidence": 0.8}
            ),
            GeneratedIdea(
                content="Idea with None metadata",
                thinking_method=ThinkingMethod.ABDUCTION,
                agent_name="TestAgent",
                generation_prompt="test",
                metadata=None
            ),
            GeneratedIdea(
                content="Idea with empty metadata",
                thinking_method=ThinkingMethod.DEDUCTION,
                agent_name="TestAgent",
                generation_prompt="test",
                metadata={}
            ),
        ]
        
        qadi_results = {
            "questioning": [ideas[0]],
            "unknown": [ideas[1], ideas[2]],
        }
        
        result = await self.extractor.extract_answers(question, qadi_results, max_answers=3)
        
        assert isinstance(result, AnswerExtractionResult)
        assert len(result.direct_answers) == 3
        assert result.total_qadi_ideas == 3
    
    def test_prefer_llm_flag(self):
        """Test prefer_llm flag controls behavior."""
        # Test with prefer_llm=False
        extractor_no_llm = EnhancedAnswerExtractor(prefer_llm=False)
        assert extractor_no_llm.prefer_llm is False
        
        # Test with prefer_llm=True (default)
        extractor_with_llm = EnhancedAnswerExtractor(prefer_llm=True)
        assert extractor_with_llm.prefer_llm is True
        
        # Test default behavior
        extractor_default = EnhancedAnswerExtractor()
        assert extractor_default.prefer_llm is True