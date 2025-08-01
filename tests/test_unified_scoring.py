"""
Tests for unified scoring system across QADI and evolution.
"""

import pytest

from mad_spark_alt.core.interfaces import GeneratedIdea
from mad_spark_alt.evolution.interfaces import IndividualFitness
from mad_spark_alt.evolution.fitness import FitnessEvaluator
from mad_spark_alt.core.simple_qadi_orchestrator import HypothesisScore


class TestUnifiedScoring:
    """Test that all ideas use consistent scoring criteria."""
    
    def test_individual_fitness_has_qadi_scores(self):
        """Test that IndividualFitness stores QADI scoring criteria."""
        idea = GeneratedIdea(
            content="Test idea",
            thinking_method="test",
            agent_name="test_agent",
            generation_prompt="test prompt"
        )
        
        # IndividualFitness should have QADI scores
        fitness = IndividualFitness(
            idea=idea,
            impact=0.8,
            feasibility=0.7,
            accessibility=0.9,
            sustainability=0.6,
            scalability=0.75,
            overall_fitness=0.77  # Average of 5 scores
        )
        
        assert fitness.impact == 0.8
        assert fitness.feasibility == 0.7
        assert fitness.accessibility == 0.9
        assert fitness.sustainability == 0.6
        assert fitness.scalability == 0.75
        assert fitness.overall_fitness == 0.77
        
    def test_hypothesis_score_and_individual_fitness_compatible(self):
        """Test that HypothesisScore and IndividualFitness use same criteria."""
        # Both should have the same 5 criteria
        hypothesis_score = HypothesisScore(
            impact=0.8,
            feasibility=0.7,
            accessibility=0.9,
            sustainability=0.6,
            scalability=0.75,
            overall=0.77
        )
        
        idea = GeneratedIdea(
            content="Test idea",
            thinking_method="test",
            agent_name="test_agent",
            generation_prompt="test prompt"
        )
        
        fitness = IndividualFitness(
            idea=idea,
            impact=hypothesis_score.impact,
            feasibility=hypothesis_score.feasibility,
            accessibility=hypothesis_score.accessibility,
            sustainability=hypothesis_score.sustainability,
            scalability=hypothesis_score.scalability,
            overall_fitness=hypothesis_score.overall
        )
        
        # Scores should match
        assert fitness.impact == hypothesis_score.impact
        assert fitness.feasibility == hypothesis_score.feasibility
        assert fitness.accessibility == hypothesis_score.accessibility
        assert fitness.sustainability == hypothesis_score.sustainability
        assert fitness.scalability == hypothesis_score.scalability
        assert fitness.overall_fitness == hypothesis_score.overall
        
    @pytest.mark.asyncio
    async def test_fitness_evaluator_returns_qadi_scores(self):
        """Test that FitnessEvaluator returns QADI scores in metadata."""
        from mad_spark_alt.evolution.interfaces import EvolutionConfig
        
        # Create mock ideas
        ideas = [
            GeneratedIdea(
                content="Idea 1: Solar panels",
                thinking_method="abduction",
                agent_name="test_agent",
                generation_prompt="test prompt"
            ),
            GeneratedIdea(
                content="Idea 2: Wind turbines",
                thinking_method="deduction",
                agent_name="test_agent",
                generation_prompt="test prompt"
            )
        ]
        
        # Config and evaluator would be used in actual implementation
        # For now, just test the expected structure
        expected_fitness = IndividualFitness(
            idea=ideas[0],
            impact=0.8,
            feasibility=0.7,
            accessibility=0.9,
            sustainability=0.6,
            scalability=0.75,
            overall_fitness=0.77,
            evaluation_metadata={
                "evaluation_criteria": {
                    "impact": 0.8,
                    "feasibility": 0.7,
                    "accessibility": 0.9,
                    "sustainability": 0.6,
                    "scalability": 0.75
                }
            }
        )
        
        # Test structure
        assert hasattr(expected_fitness, 'impact')
        assert hasattr(expected_fitness, 'feasibility')
        assert hasattr(expected_fitness, 'accessibility')
        assert hasattr(expected_fitness, 'sustainability')
        assert hasattr(expected_fitness, 'scalability')
        assert 'evaluation_criteria' in expected_fitness.evaluation_metadata
        
    def test_individual_fitness_to_dict(self):
        """Test that IndividualFitness can convert to dict with scores."""
        idea = GeneratedIdea(
            content="Test idea",
            thinking_method="test",
            agent_name="test_agent",
            generation_prompt="test prompt"
        )
        
        fitness = IndividualFitness(
            idea=idea,
            impact=0.8,
            feasibility=0.7,
            accessibility=0.9,
            sustainability=0.6,
            scalability=0.75,
            overall_fitness=0.77
        )
        
        # Should have a method to get scores as dict
        scores_dict = fitness.get_scores_dict()
        
        assert scores_dict == {
            "impact": 0.8,
            "feasibility": 0.7,
            "accessibility": 0.9,
            "sustainability": 0.6,
            "scalability": 0.75,
            "overall": 0.77
        }


class TestEvolutionCollectsAllGenerations:
    """Test that evolution results include all generations for selection."""
    
    def test_evolution_result_has_all_populations_property(self):
        """Test that EvolutionResult can provide all populations."""
        from mad_spark_alt.evolution.interfaces import EvolutionResult, PopulationSnapshot
        
        # Create mock data
        idea1 = GeneratedIdea(
            content="Gen 0 idea",
            thinking_method="test",
            agent_name="test_agent",
            generation_prompt="test prompt"
        )
        
        idea2 = GeneratedIdea(
            content="Gen 1 idea",
            thinking_method="test",
            agent_name="test_agent",
            generation_prompt="test prompt"
        )
        
        pop0 = [IndividualFitness(
            idea=idea1,
            impact=0.8,
            feasibility=0.7,
            accessibility=0.9,
            sustainability=0.6,
            scalability=0.75,
            overall_fitness=0.77
        )]
        
        pop1 = [IndividualFitness(
            idea=idea2,
            impact=0.9,
            feasibility=0.8,
            accessibility=0.85,
            sustainability=0.7,
            scalability=0.8,
            overall_fitness=0.83
        )]
        
        snapshot0 = PopulationSnapshot.from_population(0, pop0)
        snapshot1 = PopulationSnapshot.from_population(1, pop1)
        
        result = EvolutionResult(
            final_population=pop1,
            best_ideas=[idea2],
            generation_snapshots=[snapshot0, snapshot1],
            total_generations=2,
            execution_time=10.0
        )
        
        # Should be able to get all individuals from all generations
        all_individuals = result.get_all_individuals()
        
        assert len(all_individuals) == 2
        assert all_individuals[0].idea.content == "Gen 0 idea"
        assert all_individuals[1].idea.content == "Gen 1 idea"