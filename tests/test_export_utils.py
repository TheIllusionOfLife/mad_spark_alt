"""
Tests for export utilities.

Following TDD approach - tests written before implementation.
"""

import json
import pytest
from pathlib import Path
from datetime import datetime
from mad_spark_alt.core.simple_qadi_orchestrator import SimpleQADIResult
from mad_spark_alt.core.unified_orchestrator import UnifiedQADIResult, Strategy, ExecutionMode
from mad_spark_alt.core.phase_logic import HypothesisScore
from mad_spark_alt.core.interfaces import GeneratedIdea, ThinkingMethod
from mad_spark_alt.evolution.interfaces import EvolutionResult, IndividualFitness, PopulationSnapshot
from mad_spark_alt.utils.export_utils import (
    export_to_json,
    export_to_markdown,
    generate_export_filename,
)


class TestJSONExport:
    """Test JSON export functionality."""

    def test_export_simple_qadi_to_json(self, tmp_path):
        """Test exporting SimpleQADIResult to JSON file."""
        # Create a simple result
        result = SimpleQADIResult(
            core_question="What is AI?",
            hypotheses=["H1: ML", "H2: DL"],
            hypothesis_scores=[
                HypothesisScore(0.8, 0.7, 0.6, 0.9, 0.5, 0.7),
                HypothesisScore(0.7, 0.8, 0.7, 0.8, 0.6, 0.72),
            ],
            final_answer="AI is machine learning and deep learning.",
            action_plan=["Learn ML", "Learn DL"],
            verification_examples=["Example 1"],
            verification_conclusion="Verified",
            total_llm_cost=0.005,
        )

        # Export to JSON
        filepath = tmp_path / "test_export.json"
        export_to_json(result, filepath)

        # Verify file exists
        assert filepath.exists()

        # Verify JSON content
        with open(filepath) as f:
            data = json.load(f)

        assert data["core_question"] == "What is AI?"
        assert len(data["hypotheses"]) == 2
        assert data["metadata"]["total_llm_cost"] == 0.005

    def test_export_unified_qadi_to_json(self, tmp_path):
        """Test exporting UnifiedQADIResult to JSON file."""
        result = UnifiedQADIResult(
            strategy_used=Strategy.SIMPLE,
            execution_mode=ExecutionMode.SEQUENTIAL,
            core_question="Test Q",
            hypotheses=["H1"],
            final_answer="Answer",
            action_plan=["Action"],
            total_llm_cost=0.003,
            synthesized_ideas=[],
        )

        filepath = tmp_path / "unified_export.json"
        export_to_json(result, filepath)

        assert filepath.exists()

        with open(filepath) as f:
            data = json.load(f)

        assert data["strategy_used"] == "simple"
        assert data["execution_mode"] == "sequential"

    def test_export_with_evolution_result(self, tmp_path):
        """Test exporting QADI result with evolution data."""
        qadi_result = SimpleQADIResult(
            core_question="Q",
            hypotheses=["H"],
            hypothesis_scores=[HypothesisScore(0.5, 0.5, 0.5, 0.5, 0.5, 0.5)],
            final_answer="A",
            action_plan=[],
            verification_examples=[],
            verification_conclusion="C",
        )

        idea = GeneratedIdea(
            content="Evolved idea",
            thinking_method=ThinkingMethod.ABDUCTION,
            agent_name="Agent",
            generation_prompt="Prompt",
        )

        evolution_result = EvolutionResult(
            final_population=[],
            best_ideas=[idea],
            generation_snapshots=[],
            total_generations=5,
            execution_time=120.0,
        )

        filepath = tmp_path / "with_evolution.json"
        export_to_json(qadi_result, filepath, evolution_result=evolution_result)

        assert filepath.exists()

        with open(filepath) as f:
            data = json.load(f)

        assert "qadi_analysis" in data
        assert "evolution_results" in data
        assert data["evolution_results"]["total_generations"] == 5

    def test_export_with_timestamp(self, tmp_path):
        """Test that timestamp is added to export when requested."""
        result = SimpleQADIResult(
            core_question="Q",
            hypotheses=[],
            hypothesis_scores=[],
            final_answer="A",
            action_plan=[],
            verification_examples=[],
            verification_conclusion="C",
        )

        filepath = tmp_path / "timestamped.json"
        export_to_json(result, filepath, timestamp=True)

        with open(filepath) as f:
            data = json.load(f)

        assert "exported_at" in data
        # Verify it's a valid ISO format timestamp
        datetime.fromisoformat(data["exported_at"])

    def test_export_without_timestamp(self, tmp_path):
        """Test that timestamp can be omitted."""
        result = SimpleQADIResult(
            core_question="Q",
            hypotheses=[],
            hypothesis_scores=[],
            final_answer="A",
            action_plan=[],
            verification_examples=[],
            verification_conclusion="C",
        )

        filepath = tmp_path / "no_timestamp.json"
        export_to_json(result, filepath, timestamp=False)

        with open(filepath) as f:
            data = json.load(f)

        assert "exported_at" not in data

    def test_export_creates_parent_directories(self, tmp_path):
        """Test that export creates parent directories if they don't exist."""
        result = SimpleQADIResult(
            core_question="Q",
            hypotheses=[],
            hypothesis_scores=[],
            final_answer="A",
            action_plan=[],
            verification_examples=[],
            verification_conclusion="C",
        )

        filepath = tmp_path / "subdir1" / "subdir2" / "export.json"
        export_to_json(result, filepath)

        assert filepath.exists()
        assert filepath.parent.exists()


class TestMarkdownExport:
    """Test Markdown export functionality."""

    def test_export_simple_qadi_to_markdown(self, tmp_path):
        """Test exporting SimpleQADIResult to Markdown file."""
        result = SimpleQADIResult(
            core_question="What is the best programming language?",
            hypotheses=["Python is versatile", "Rust is safe"],
            hypothesis_scores=[
                HypothesisScore(0.9, 0.8, 0.9, 0.7, 0.8, 0.82),
                HypothesisScore(0.8, 0.7, 0.6, 0.8, 0.7, 0.72),
            ],
            final_answer="Python for versatility, Rust for safety.",
            action_plan=["Learn Python basics", "Explore Rust ownership"],
            verification_examples=["Python used in AI", "Rust in systems programming"],
            verification_conclusion="Both languages excel in different domains.",
            total_llm_cost=0.008,
        )

        filepath = tmp_path / "test_export.md"
        export_to_markdown(result, filepath)

        assert filepath.exists()

        content = filepath.read_text()

        # Verify key sections are present
        assert "# QADI Analysis Results" in content
        assert "## Core Question" in content
        assert "What is the best programming language?" in content
        assert "## Hypotheses" in content
        assert "Python is versatile" in content
        assert "## Final Answer" in content
        assert "## Action Plan" in content
        assert "## Verification" in content

    def test_export_unified_qadi_to_markdown(self, tmp_path):
        """Test exporting UnifiedQADIResult to Markdown."""
        result = UnifiedQADIResult(
            strategy_used=Strategy.MULTI_PERSPECTIVE,
            execution_mode=ExecutionMode.PARALLEL,
            core_question="How to improve productivity?",
            hypotheses=["Time management", "Automation"],
            final_answer="Combine time management with automation.",
            action_plan=["Use calendar", "Automate tasks"],
            total_llm_cost=0.012,
            synthesized_ideas=[],
            perspectives_used=["technical", "business"],
        )

        filepath = tmp_path / "unified.md"
        export_to_markdown(result, filepath)

        content = filepath.read_text()

        assert "**Strategy:** multi_perspective" in content
        assert "**Execution Mode:** parallel" in content
        assert "**Perspectives:** technical, business" in content

    def test_export_with_evolution_results(self, tmp_path):
        """Test exporting with evolution data in Markdown."""
        qadi_result = SimpleQADIResult(
            core_question="Q",
            hypotheses=["H1", "H2"],
            hypothesis_scores=[
                HypothesisScore(0.7, 0.7, 0.7, 0.7, 0.7, 0.7),
                HypothesisScore(0.8, 0.8, 0.8, 0.8, 0.8, 0.8),
            ],
            final_answer="A",
            action_plan=["Step 1"],
            verification_examples=["Ex 1"],
            verification_conclusion="Verified",
        )

        ideas = [
            GeneratedIdea(
                content="Best evolved idea",
                thinking_method=ThinkingMethod.ABDUCTION,
                agent_name="Agent",
                generation_prompt="Prompt",
            )
        ]

        snapshots = [
            PopulationSnapshot(
                generation=0,
                population=[],
                best_fitness=0.6,
                average_fitness=0.5,
                worst_fitness=0.4,
                diversity_score=0.8,
            ),
            PopulationSnapshot(
                generation=1,
                population=[],
                best_fitness=0.8,
                average_fitness=0.7,
                worst_fitness=0.6,
                diversity_score=0.75,
            ),
        ]

        evolution_result = EvolutionResult(
            final_population=[],
            best_ideas=ideas,
            generation_snapshots=snapshots,
            total_generations=2,
            execution_time=60.0,
            evolution_metrics={"diversity_avg": 0.775},
        )

        filepath = tmp_path / "with_evolution.md"
        export_to_markdown(qadi_result, filepath, evolution_result=evolution_result)

        content = filepath.read_text()

        assert "# QADI Analysis Results" in content
        assert "## Evolution Results" in content
        assert "Best evolved idea" in content
        assert "**Total Generations:** 2" in content
        assert "**Execution Time:** 60.0s" in content
        assert "### Fitness Progression" in content

    def test_markdown_hypothesis_scores_formatting(self, tmp_path):
        """Test that hypothesis scores are properly formatted in Markdown."""
        result = SimpleQADIResult(
            core_question="Q",
            hypotheses=["H1"],
            hypothesis_scores=[
                HypothesisScore(
                    impact=0.85,
                    feasibility=0.75,
                    accessibility=0.65,
                    sustainability=0.95,
                    scalability=0.55,
                    overall=0.75,
                )
            ],
            final_answer="A",
            action_plan=[],
            verification_examples=[],
            verification_conclusion="C",
        )

        filepath = tmp_path / "scores.md"
        export_to_markdown(result, filepath)

        content = filepath.read_text()

        # Verify score formatting
        assert "Impact: 0.85" in content or "impact: 0.85" in content
        assert "Feasibility: 0.75" in content or "feasibility: 0.75" in content
        assert "Overall: 0.75" in content or "overall: 0.75" in content


class TestFilenameGeneration:
    """Test automatic filename generation."""

    def test_generate_filename_with_timestamp(self):
        """Test generating filename with timestamp."""
        filename = generate_export_filename("json", include_timestamp=True)

        assert filename.endswith(".json")
        assert "qadi_analysis_" in filename
        # Should contain date-like pattern
        assert any(char.isdigit() for char in filename)

    def test_generate_filename_without_timestamp(self):
        """Test generating filename without timestamp."""
        filename = generate_export_filename("md", include_timestamp=False)

        assert filename == "qadi_analysis.md"

    def test_generate_filename_with_evolution(self):
        """Test generating filename indicating evolution was used."""
        filename = generate_export_filename("json", include_evolution=True)

        assert "evolution" in filename
        assert filename.endswith(".json")

    def test_generate_filename_format_validation(self):
        """Test that only valid formats are accepted."""
        # Valid formats should work
        assert generate_export_filename("json").endswith(".json")
        assert generate_export_filename("md").endswith(".md")

        # Invalid format should raise error
        with pytest.raises(ValueError, match="Unsupported format"):
            generate_export_filename("txt")
