"""
Tests for the enhanced QADI orchestrator.
"""

import asyncio
from typing import Dict, List
from unittest.mock import Mock, patch

import pytest

from mad_spark_alt.core.enhanced_orchestrator import (
    EnhancedQADICycleResult,
    EnhancedQADIOrchestrator,
)
from mad_spark_alt.core.interfaces import (
    GeneratedIdea,
    IdeaGenerationResult,
    ThinkingMethod,
)
from mad_spark_alt.core.smart_registry import SmartAgentRegistry


class TestEnhancedQADIOrchestrator:
    """Test the enhanced orchestrator with answer extraction."""
    
    @pytest.fixture
    def orchestrator(self):
        """Create test orchestrator instance."""
        return EnhancedQADIOrchestrator(auto_setup=False)
    
    def create_mock_idea(self, content: str, phase: str) -> GeneratedIdea:
        """Create a mock idea with metadata."""
        return GeneratedIdea(
            content=content,
            thinking_method=ThinkingMethod.QUESTIONING,
            agent_name="MockAgent",
            generation_prompt="test",
            metadata={"phase": phase}
        )
    
    @pytest.mark.asyncio
    async def test_run_qadi_cycle_with_answers(self, orchestrator):
        """Test running QADI cycle with answer extraction."""
        # Mock the parent run_qadi_cycle method
        mock_base_result = Mock()
        mock_base_result.problem_statement = "How to improve productivity?"
        mock_base_result.cycle_id = "test-123"
        mock_base_result.phases = {}
        mock_base_result.synthesized_ideas = [
            self.create_mock_idea("What blocks productivity?", "questioning"),
            self.create_mock_idea("Productivity comes from focus", "abduction"),
        ]
        mock_base_result.execution_time = 1.0
        mock_base_result.metadata = {}
        mock_base_result.timestamp = "2025-01-01T00:00:00Z"
        mock_base_result.agent_types = {"questioning": "template"}
        mock_base_result.llm_cost = 0.0
        mock_base_result.setup_time = 0.1
        mock_base_result.conclusion = None
        
        # Patch the parent class method directly
        with patch('mad_spark_alt.core.enhanced_orchestrator.SmartQADIOrchestrator.run_qadi_cycle', return_value=mock_base_result):
            result = await orchestrator.run_qadi_cycle_with_answers(
                problem_statement="How to improve productivity?",
                max_answers=3
            )
        
        assert isinstance(result, EnhancedQADICycleResult)
        assert result.extracted_answers is not None
        assert len(result.extracted_answers.direct_answers) == 3
        assert result.answer_extraction_time > 0
    
    def test_group_ideas_by_phase(self, orchestrator):
        """Test grouping ideas by phase with various metadata states."""
        ideas = [
            self.create_mock_idea("Question 1", "questioning"),
            self.create_mock_idea("Hypothesis 1", "abduction"),
            GeneratedIdea(
                content="No metadata idea",
                thinking_method=ThinkingMethod.DEDUCTION,
                agent_name="Test",
                generation_prompt="test",
                metadata=None  # Test None metadata
            ),
            GeneratedIdea(
                content="Empty metadata",
                thinking_method=ThinkingMethod.INDUCTION,
                agent_name="Test",
                generation_prompt="test",
                metadata={}  # Test empty metadata
            ),
        ]
        
        grouped = orchestrator._group_ideas_by_phase(ideas)
        
        assert "questioning" in grouped
        assert "abduction" in grouped
        assert "unknown" in grouped
        assert len(grouped["unknown"]) == 2  # Both None and empty metadata
    
    @pytest.mark.asyncio
    async def test_get_direct_answers(self, orchestrator):
        """Test the convenience method for getting direct answers."""
        # Mock the full cycle method
        mock_result = Mock()
        mock_result.extracted_answers = Mock()
        mock_result.extracted_answers.direct_answers = [
            Mock(content="Answer 1"),
            Mock(content="Answer 2"),
        ]
        
        with patch.object(
            orchestrator,
            'run_qadi_cycle_with_answers',
            return_value=mock_result
        ):
            answers = await orchestrator.get_direct_answers(
                "What are ways to learn faster?",
                max_answers=2
            )
        
        assert len(answers) == 2
        assert answers[0] == "Answer 1"
        assert answers[1] == "Answer 2"
    
    @pytest.mark.asyncio
    async def test_extract_answers_disabled(self, orchestrator):
        """Test disabling answer extraction."""
        mock_base_result = Mock()
        mock_base_result.problem_statement = "Test question"
        mock_base_result.cycle_id = "test-123"
        mock_base_result.phases = {}
        mock_base_result.synthesized_ideas = []
        mock_base_result.execution_time = 1.0
        mock_base_result.metadata = {}
        mock_base_result.timestamp = "2025-01-01T00:00:00Z"
        mock_base_result.agent_types = {}
        mock_base_result.llm_cost = 0.0
        mock_base_result.setup_time = 0.1
        mock_base_result.conclusion = None
        
        with patch.object(orchestrator, 'run_qadi_cycle', return_value=mock_base_result):
            result = await orchestrator.run_qadi_cycle_with_answers(
                problem_statement="Test question",
                extract_answers=False  # Disable extraction
            )
        
        assert result.extracted_answers is None
        assert result.answer_extraction_time == 0.0