"""
Tests for result serialization methods.

Following TDD approach - tests written before implementation.
"""

import pytest
from mad_spark_alt.core.simple_qadi_orchestrator import SimpleQADIResult
from mad_spark_alt.core.unified_orchestrator import UnifiedQADIResult, Strategy, ExecutionMode
from mad_spark_alt.core.phase_logic import HypothesisScore
from mad_spark_alt.core.interfaces import GeneratedIdea, ThinkingMethod
from mad_spark_alt.evolution.interfaces import EvolutionResult, IndividualFitness, PopulationSnapshot


class TestSimpleQADIResultSerialization:
    """Test SimpleQADIResult.to_dict() method."""

    def test_to_dict_basic_structure(self):
        """Test that to_dict() returns correct structure with all required fields."""
        # Create a simple result
        result = SimpleQADIResult(
            core_question="What is the meaning of life?",
            hypotheses=["H1: Purpose", "H2: Happiness"],
            hypothesis_scores=[
                HypothesisScore(
                    impact=0.8,
                    feasibility=0.7,
                    accessibility=0.6,
                    sustainability=0.9,
                    scalability=0.5,
                    overall=0.7,
                ),
                HypothesisScore(
                    impact=0.7,
                    feasibility=0.8,
                    accessibility=0.7,
                    sustainability=0.8,
                    scalability=0.6,
                    overall=0.72,
                ),
            ],
            final_answer="The meaning of life is to find purpose and happiness.",
            action_plan=["Step 1", "Step 2"],
            verification_examples=["Example 1", "Example 2"],
            verification_conclusion="Verified through examples.",
            total_llm_cost=0.0042,
        )

        # Convert to dict
        result_dict = result.to_dict()

        # Verify structure
        assert isinstance(result_dict, dict)
        assert "core_question" in result_dict
        assert "hypotheses" in result_dict
        assert "hypothesis_scores" in result_dict
        assert "final_answer" in result_dict
        assert "action_plan" in result_dict
        assert "verification_examples" in result_dict
        assert "verification_conclusion" in result_dict
        assert "metadata" in result_dict

    def test_to_dict_core_fields_values(self):
        """Test that core fields are correctly serialized."""
        result = SimpleQADIResult(
            core_question="Test question",
            hypotheses=["H1", "H2", "H3"],
            hypothesis_scores=[
                HypothesisScore(0.8, 0.7, 0.6, 0.9, 0.5, 0.7),
                HypothesisScore(0.7, 0.8, 0.7, 0.8, 0.6, 0.72),
                HypothesisScore(0.9, 0.6, 0.8, 0.7, 0.7, 0.74),
            ],
            final_answer="Test answer",
            action_plan=["Action 1", "Action 2"],
            verification_examples=["Ex 1", "Ex 2"],
            verification_conclusion="Conclusion",
        )

        result_dict = result.to_dict()

        assert result_dict["core_question"] == "Test question"
        assert result_dict["hypotheses"] == ["H1", "H2", "H3"]
        assert result_dict["final_answer"] == "Test answer"
        assert result_dict["action_plan"] == ["Action 1", "Action 2"]
        assert result_dict["verification_examples"] == ["Ex 1", "Ex 2"]
        assert result_dict["verification_conclusion"] == "Conclusion"

    def test_to_dict_hypothesis_scores_structure(self):
        """Test that hypothesis scores are correctly converted to dicts."""
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

        result_dict = result.to_dict()

        assert isinstance(result_dict["hypothesis_scores"], list)
        assert len(result_dict["hypothesis_scores"]) == 1

        score_dict = result_dict["hypothesis_scores"][0]
        assert score_dict["impact"] == 0.85
        assert score_dict["feasibility"] == 0.75
        assert score_dict["accessibility"] == 0.65
        assert score_dict["sustainability"] == 0.95
        assert score_dict["scalability"] == 0.55
        assert score_dict["overall"] == 0.75

    def test_to_dict_metadata_section(self):
        """Test that metadata section is correctly populated."""
        result = SimpleQADIResult(
            core_question="Q",
            hypotheses=["H"],
            hypothesis_scores=[HypothesisScore(0.5, 0.5, 0.5, 0.5, 0.5, 0.5)],
            final_answer="A",
            action_plan=[],
            verification_examples=[],
            verification_conclusion="C",
            total_llm_cost=0.0123,
            total_images_processed=3,
            total_pages_processed=10,
            total_urls_processed=2,
        )

        result_dict = result.to_dict()

        metadata = result_dict["metadata"]
        assert metadata["total_llm_cost"] == 0.0123
        assert metadata["total_images_processed"] == 3
        assert metadata["total_pages_processed"] == 10
        assert metadata["total_urls_processed"] == 2

    def test_to_dict_with_synthesized_ideas(self):
        """Test that synthesized ideas are correctly serialized."""
        ideas = [
            GeneratedIdea(
                content="Idea 1 content",
                thinking_method=ThinkingMethod.ABDUCTION,
                agent_name="AbductionAgent",
                generation_prompt="Generate ideas",
                confidence_score=0.8,
                reasoning="Good idea",
            ),
            GeneratedIdea(
                content="Idea 2 content",
                thinking_method=ThinkingMethod.DEDUCTION,
                agent_name="DeductionAgent",
                generation_prompt="Evaluate ideas",
                confidence_score=0.9,
                reasoning="Strong logic",
            ),
        ]

        result = SimpleQADIResult(
            core_question="Q",
            hypotheses=["H"],
            hypothesis_scores=[HypothesisScore(0.5, 0.5, 0.5, 0.5, 0.5, 0.5)],
            final_answer="A",
            action_plan=[],
            verification_examples=[],
            verification_conclusion="C",
            synthesized_ideas=ideas,
        )

        result_dict = result.to_dict()

        assert "synthesized_ideas" in result_dict
        assert len(result_dict["synthesized_ideas"]) == 2

        idea1 = result_dict["synthesized_ideas"][0]
        assert idea1["content"] == "Idea 1 content"
        assert idea1["thinking_method"] == "abduction"
        assert idea1["confidence_score"] == 0.8

        idea2 = result_dict["synthesized_ideas"][1]
        assert idea2["content"] == "Idea 2 content"
        assert idea2["thinking_method"] == "deduction"
        assert idea2["confidence_score"] == 0.9

    def test_to_dict_empty_lists(self):
        """Test that empty lists are handled correctly."""
        result = SimpleQADIResult(
            core_question="Q",
            hypotheses=[],
            hypothesis_scores=[],
            final_answer="A",
            action_plan=[],
            verification_examples=[],
            verification_conclusion="C",
        )

        result_dict = result.to_dict()

        assert result_dict["hypotheses"] == []
        assert result_dict["hypothesis_scores"] == []
        assert result_dict["action_plan"] == []
        assert result_dict["verification_examples"] == []
        assert result_dict["synthesized_ideas"] == []


class TestUnifiedQADIResultSerialization:
    """Test UnifiedQADIResult.to_dict() method."""

    def test_to_dict_basic_structure(self):
        """Test basic structure of unified result serialization."""
        result = UnifiedQADIResult(
            strategy_used=Strategy.SIMPLE,
            execution_mode=ExecutionMode.SEQUENTIAL,
            core_question="What is AI?",
            hypotheses=["H1", "H2"],
            final_answer="AI is...",
            action_plan=["Learn", "Practice"],
            total_llm_cost=0.005,
            synthesized_ideas=[],
        )

        result_dict = result.to_dict()

        assert isinstance(result_dict, dict)
        assert "strategy_used" in result_dict
        assert "execution_mode" in result_dict
        assert "core_question" in result_dict
        assert "hypotheses" in result_dict
        assert "final_answer" in result_dict
        assert "action_plan" in result_dict
        assert "metadata" in result_dict

    def test_to_dict_strategy_and_mode_serialization(self):
        """Test that Strategy and ExecutionMode enums are converted to strings."""
        result = UnifiedQADIResult(
            strategy_used=Strategy.MULTI_PERSPECTIVE,
            execution_mode=ExecutionMode.PARALLEL,
            core_question="Q",
            hypotheses=["H"],
            final_answer="A",
            action_plan=[],
            total_llm_cost=0.0,
            synthesized_ideas=[],
        )

        result_dict = result.to_dict()

        assert result_dict["strategy_used"] == "multi_perspective"
        assert result_dict["execution_mode"] == "parallel"

    def test_to_dict_optional_fields_present(self):
        """Test that optional fields are included when present."""
        result = UnifiedQADIResult(
            strategy_used=Strategy.SIMPLE,
            execution_mode=ExecutionMode.SEQUENTIAL,
            core_question="Q",
            hypotheses=["H"],
            final_answer="A",
            action_plan=[],
            total_llm_cost=0.0,
            synthesized_ideas=[],
            hypothesis_scores=[HypothesisScore(0.5, 0.5, 0.5, 0.5, 0.5, 0.5)],
            verification_examples=["Ex 1"],
            verification_conclusion="Verified",
            perspectives_used=["technical", "business"],
            synthesized_answer="Synthesized",
        )

        result_dict = result.to_dict()

        assert "hypothesis_scores" in result_dict
        assert "verification_examples" in result_dict
        assert "verification_conclusion" in result_dict
        assert "perspectives_used" in result_dict
        assert "synthesized_answer" in result_dict

    def test_to_dict_optional_fields_none(self):
        """Test that None optional fields are handled correctly."""
        result = UnifiedQADIResult(
            strategy_used=Strategy.SIMPLE,
            execution_mode=ExecutionMode.SEQUENTIAL,
            core_question="Q",
            hypotheses=["H"],
            final_answer="A",
            action_plan=[],
            total_llm_cost=0.0,
            synthesized_ideas=[],
            hypothesis_scores=None,
            verification_examples=None,
            verification_conclusion=None,
        )

        result_dict = result.to_dict()

        # None fields should either be excluded or explicitly None
        assert result_dict.get("hypothesis_scores") is None or "hypothesis_scores" not in result_dict


class TestEvolutionResultSerialization:
    """Test EvolutionResult.to_export_dict() method."""

    def test_to_export_dict_basic_structure(self):
        """Test basic structure of evolution result export."""
        # Create mock evolution result
        idea1 = GeneratedIdea(
            content="Evolved idea 1",
            thinking_method=ThinkingMethod.ABDUCTION,
            agent_name="Agent",
            generation_prompt="Prompt",
            confidence_score=0.85,
        )
        idea2 = GeneratedIdea(
            content="Evolved idea 2",
            thinking_method=ThinkingMethod.DEDUCTION,
            agent_name="Agent",
            generation_prompt="Prompt",
            confidence_score=0.90,
        )

        result = EvolutionResult(
            final_population=[],
            best_ideas=[idea1, idea2],
            generation_snapshots=[],
            total_generations=5,
            execution_time=120.5,
            evolution_metrics={"diversity": 0.7, "convergence": 0.8},
        )

        export_dict = result.to_export_dict()

        assert isinstance(export_dict, dict)
        assert "best_ideas" in export_dict
        assert "total_generations" in export_dict
        assert "execution_time" in export_dict
        assert "evolution_metrics" in export_dict
        assert "fitness_progression" in export_dict

    def test_to_export_dict_best_ideas_content(self):
        """Test that best ideas are exported as content strings."""
        ideas = [
            GeneratedIdea(
                content="Great idea 1",
                thinking_method=ThinkingMethod.ABDUCTION,
                agent_name="Agent",
                generation_prompt="Prompt",
            ),
            GeneratedIdea(
                content="Great idea 2",
                thinking_method=ThinkingMethod.INDUCTION,
                agent_name="Agent",
                generation_prompt="Prompt",
            ),
        ]

        result = EvolutionResult(
            final_population=[],
            best_ideas=ideas,
            generation_snapshots=[],
            total_generations=3,
            execution_time=60.0,
        )

        export_dict = result.to_export_dict()

        assert export_dict["best_ideas"] == ["Great idea 1", "Great idea 2"]

    def test_to_export_dict_fitness_progression(self):
        """Test that fitness progression is correctly formatted."""
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
                best_fitness=0.7,
                average_fitness=0.6,
                worst_fitness=0.5,
                diversity_score=0.75,
            ),
            PopulationSnapshot(
                generation=2,
                population=[],
                best_fitness=0.85,
                average_fitness=0.72,
                worst_fitness=0.6,
                diversity_score=0.7,
            ),
        ]

        result = EvolutionResult(
            final_population=[],
            best_ideas=[],
            generation_snapshots=snapshots,
            total_generations=3,
            execution_time=90.0,
        )

        export_dict = result.to_export_dict()

        progression = export_dict["fitness_progression"]
        assert len(progression) == 3
        assert progression[0]["generation"] == 0
        assert progression[0]["best_fitness"] == 0.6
        assert progression[0]["avg_fitness"] == 0.5
        assert progression[2]["generation"] == 2
        assert progression[2]["best_fitness"] == 0.85

    def test_to_export_dict_metrics(self):
        """Test that evolution metrics are included."""
        result = EvolutionResult(
            final_population=[],
            best_ideas=[],
            generation_snapshots=[],
            total_generations=5,
            execution_time=150.0,
            evolution_metrics={
                "diversity_avg": 0.75,
                "fitness_improvement": 0.3,
                "semantic_operators_used": 15,
            },
        )

        export_dict = result.to_export_dict()

        metrics = export_dict["evolution_metrics"]
        assert metrics["diversity_avg"] == 0.75
        assert metrics["fitness_improvement"] == 0.3
        assert metrics["semantic_operators_used"] == 15
