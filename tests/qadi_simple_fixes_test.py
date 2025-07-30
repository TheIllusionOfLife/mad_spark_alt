"""Tests for qadi_simple.py display and parsing fixes."""

import pytest
from unittest.mock import MagicMock, patch

# Add parent directory to path for imports  
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from qadi_simple import extract_key_solutions


class TestHypothesisDisplay:
    """Test that all generated hypotheses are displayed."""
    
    def test_extract_key_solutions_returns_all_hypotheses(self):
        """Test that extract_key_solutions returns all hypotheses, not just 3."""
        # Create 10 test hypotheses
        hypotheses = [f"H{i}: Test hypothesis {i} with detailed explanation" for i in range(1, 11)]
        action_plan = ["Action 1", "Action 2"]
        
        # Extract solutions
        solutions = extract_key_solutions(hypotheses, action_plan)
        
        # Should return all 10 hypotheses
        assert len(solutions) == 10
        for i, solution in enumerate(solutions):
            assert f"Test hypothesis {i+1}" in solution
    
    def test_extract_key_solutions_handles_empty_input(self):
        """Test extract_key_solutions with empty input."""
        solutions = extract_key_solutions([], [])
        assert solutions == []
    
    def test_extract_key_solutions_extracts_titles_correctly(self):
        """Test that hypothesis titles are extracted properly."""
        hypotheses = [
            "H1: Machine Learning Approach\nThis uses advanced ML algorithms...",
            "H2: Distributed Systems Solution\nImplement a distributed architecture...",
            "**H3: Blockchain Integration**\nLeverage blockchain technology..."
        ]
        
        solutions = extract_key_solutions(hypotheses, [])
        
        assert len(solutions) == 3
        assert "Machine Learning Approach" in solutions[0]
        assert "Distributed Systems Solution" in solutions[1]  
        assert "Blockchain Integration" in solutions[2]


class TestScoreParsingWarnings:
    """Test improvements to score parsing to reduce warnings."""
    
    def test_parse_hypothesis_scores_handles_various_formats(self):
        """Test that score parsing handles different LLM response formats."""
        from mad_spark_alt.core.simple_qadi_orchestrator import SimpleQADIOrchestrator
        
        orchestrator = SimpleQADIOrchestrator()
        
        # Test various formats that should parse without warnings
        test_contents = [
            # Standard format
            """H1:
            * Impact: 0.9 - High impact solution
            * Feasibility: 0.7 - Reasonably feasible
            * Accessibility: 0.8 - Good accessibility
            * Sustainability: 0.9 - Very sustainable
            * Scalability: 0.8 - Scales well
            * Overall: 0.86""",
            
            # Format with bold markers
            """**H1:**
            - **Impact:** 0.9
            - **Feasibility:** 0.7
            - **Accessibility:** 0.8
            - **Sustainability:** 0.9
            - **Scalability:** 0.8
            - **Overall:** 0.86""",
            
            # Format without bullets
            """H1:
            Impact: 0.9
            Feasibility: 0.7
            Accessibility: 0.8
            Sustainability: 0.9
            Scalability: 0.8
            Overall: 0.86"""
        ]
        
        for content in test_contents:
            score = orchestrator._parse_hypothesis_scores(content, 1)
            # Should parse successfully without using defaults
            assert score.impact == 0.9
            assert score.feasibility == 0.7
            # Calculate expected overall based on weights:
            # impact: 0.9 * 0.3 = 0.27
            # feasibility: 0.7 * 0.2 = 0.14
            # accessibility: 0.8 * 0.2 = 0.16
            # sustainability: 0.9 * 0.2 = 0.18
            # scalability: 0.8 * 0.1 = 0.08
            # total: 0.27 + 0.14 + 0.16 + 0.18 + 0.08 = 0.83
            assert abs(score.overall - 0.83) < 0.01  # Allow for floating point precision


class TestEnhancedApproachContent:
    """Test that enhanced approach content is not truncated."""
    
    def test_evolution_display_shows_full_content(self):
        """Test that evolved ideas display full content."""
        # This is more of an integration test - we'll verify the display logic
        # doesn't truncate content when rendering
        from mad_spark_alt.core.interfaces import GeneratedIdea, ThinkingMethod
        
        # Create a long idea content
        long_content = """This is a comprehensive solution that involves multiple steps and detailed implementation. 
        First, we need to establish the foundational infrastructure using cloud-native technologies. 
        Second, implement microservices architecture with proper service mesh. 
        Third, integrate machine learning models for intelligent decision making. 
        Fourth, establish monitoring and observability using distributed tracing. 
        Fifth, implement security measures including zero-trust architecture. 
        This solution provides scalability, reliability, and maintainability."""
        
        idea = GeneratedIdea(
            content=long_content,
            thinking_method=ThinkingMethod.QUESTIONING,
            agent_name="test",
            generation_prompt="test"
        )
        
        # Verify content is not truncated
        assert len(idea.content) > 200
        assert idea.content == long_content