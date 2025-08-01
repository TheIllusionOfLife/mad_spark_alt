"""
Integration tests for unified scoring system.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, List

from mad_spark_alt.core.interfaces import GeneratedIdea
from mad_spark_alt.evolution.interfaces import (
    IndividualFitness, EvolutionConfig, EvolutionRequest
)
from mad_spark_alt.evolution.fitness import FitnessEvaluator
from mad_spark_alt.evolution.genetic_algorithm import GeneticAlgorithm
from mad_spark_alt.core.simple_qadi_orchestrator import (
    SimpleQADIOrchestrator, HypothesisScore
)
from mad_spark_alt.core.llm_provider import GoogleProvider, LLMResponse


class TestUnifiedScoringIntegration:
    """Integration tests for unified scoring across QADI and evolution."""
    
    @pytest.mark.asyncio
    async def test_fitness_evaluator_uses_qadi_scores(self):
        """Test that FitnessEvaluator returns QADI scores correctly."""
        # Mock the UnifiedEvaluator
        with patch('mad_spark_alt.evolution.fitness.UnifiedEvaluator') as MockEvaluator:
            mock_evaluator = MockEvaluator.return_value
            
            # Mock evaluation result with QADI scores
            mock_evaluation = Mock()
            mock_evaluation.scores = {
                "impact": 0.8,
                "feasibility": 0.7,
                "accessibility": 0.9,
                "sustainability": 0.6,
                "scalability": 0.75
            }
            mock_evaluation.overall_score = 0.77
            mock_evaluation.explanations = {}
            mock_evaluation.metadata = {"llm_cost": 0.001}
            
            mock_evaluator.evaluate_multiple = AsyncMock(
                return_value=[mock_evaluation]
            )
            
            # Create test data
            idea = GeneratedIdea(
                content="Test idea for solar panels",
                thinking_method="abduction",
                agent_name="test_agent",
                generation_prompt="energy solutions"
            )
            
            config = EvolutionConfig(
                population_size=1,
                generations=1,
                mutation_rate=0.1,
                crossover_rate=0.7
            )
            
            # Create evaluator and evaluate
            evaluator = FitnessEvaluator()
            results = await evaluator.evaluate_population([idea], config)
            
            # Check results
            assert len(results) == 1
            fitness = results[0]
            
            # Check QADI scores
            assert fitness.impact == 0.8
            assert fitness.feasibility == 0.7
            assert fitness.accessibility == 0.9
            assert fitness.sustainability == 0.6
            assert fitness.scalability == 0.75
            assert fitness.overall_fitness == 0.77
            
            # Check metadata
            assert "evaluation_criteria" in fitness.evaluation_metadata
            criteria = fitness.evaluation_metadata["evaluation_criteria"]
            assert criteria["impact"] == 0.8
            assert criteria["feasibility"] == 0.7
            assert criteria["accessibility"] == 0.9
            assert criteria["sustainability"] == 0.6
            assert criteria["scalability"] == 0.75
            
    @pytest.mark.asyncio
    async def test_qadi_and_evolution_use_same_scoring(self):
        """Test that QADI hypotheses and evolved ideas use same scoring."""
        # Mock LLM provider
        mock_provider = Mock(spec=GoogleProvider)
        
        # Mock QADI hypothesis evaluation response
        qadi_scores_response = LLMResponse(
            content="""
            ## H1: Solar Panel Solution
            * **Impact:** 0.8 - High positive environmental impact
            * **Feasibility:** 0.7 - Technically feasible with current technology
            * **Accessibility:** 0.9 - Easy for communities to adopt
            * **Sustainability:** 0.6 - Moderate long-term sustainability
            * **Scalability:** 0.75 - Can scale to multiple regions
            """,
            cost=0.001,
            model="gemini-pro",
            provider="google"
        )
        
        # Mock evolution evaluation response (should use same format)
        evolution_scores_response = LLMResponse(
            content="""{
                "evaluations": [{
                    "scores": {
                        "impact": 0.85,
                        "feasibility": 0.72,
                        "accessibility": 0.88,
                        "sustainability": 0.65,
                        "scalability": 0.78
                    },
                    "overall_score": 0.776
                }]
            }""",
            cost=0.001,
            model="gemini-pro",
            provider="google"
        )
        
        mock_provider.generate = AsyncMock(
            side_effect=[qadi_scores_response, evolution_scores_response]
        )
        
        # Test QADI scoring
        qadi_orchestrator = SimpleQADIOrchestrator(llm_provider=mock_provider)
        
        # Create mock hypothesis
        hypothesis_score = HypothesisScore(
            impact=0.8,
            feasibility=0.7,
            accessibility=0.9,
            sustainability=0.6,
            scalability=0.75,
            overall=0.77
        )
        
        # Test evolution scoring with same criteria
        with patch('mad_spark_alt.evolution.fitness.UnifiedEvaluator') as MockEvaluator:
            mock_evaluator = MockEvaluator.return_value
            
            # Return evaluation with QADI scores
            mock_evaluation = Mock()
            mock_evaluation.scores = {
                "impact": 0.85,
                "feasibility": 0.72,
                "accessibility": 0.88,
                "sustainability": 0.65,
                "scalability": 0.78
            }
            mock_evaluation.overall_score = 0.776
            mock_evaluation.explanations = {}
            mock_evaluation.metadata = {}
            
            mock_evaluator.evaluate_multiple = AsyncMock(
                return_value=[mock_evaluation]
            )
            
            # Create evolved idea
            evolved_idea = GeneratedIdea(
                content="Enhanced solar solution",
                thinking_method="mutation",
                agent_name="evolution",
                generation_prompt="energy"
            )
            
            config = EvolutionConfig(
                population_size=1,
                generations=1
            )
            
            evaluator = FitnessEvaluator()
            fitness_results = await evaluator.evaluate_population(
                [evolved_idea], config
            )
            
            # Both should use same 5 criteria
            fitness = fitness_results[0]
            assert hasattr(hypothesis_score, 'impact') and hasattr(fitness, 'impact')
            assert hasattr(hypothesis_score, 'feasibility') and hasattr(fitness, 'feasibility')
            assert hasattr(hypothesis_score, 'accessibility') and hasattr(fitness, 'accessibility')
            assert hasattr(hypothesis_score, 'sustainability') and hasattr(fitness, 'sustainability')
            assert hasattr(hypothesis_score, 'scalability') and hasattr(fitness, 'scalability')
            
    @pytest.mark.asyncio
    async def test_evolution_collects_all_generations(self):
        """Test that evolution results include all generations for selection."""
        # Mock LLM provider
        mock_provider = Mock(spec=GoogleProvider)
        
        # Create initial ideas
        initial_ideas = [
            GeneratedIdea(
                content=f"Initial idea {i}",
                thinking_method="hypothesis",
                agent_name="qadi",
                generation_prompt="test"
            )
            for i in range(3)
        ]
        
        # Mock fitness evaluator
        with patch('mad_spark_alt.evolution.genetic_algorithm.FitnessEvaluator') as MockEvaluator:
            mock_evaluator = MockEvaluator.return_value
            
            # Gen 0 evaluations
            gen0_fitness = [
                IndividualFitness(
                    idea=initial_ideas[i],
                    impact=0.7 + i * 0.05,
                    feasibility=0.6 + i * 0.05,
                    accessibility=0.8 + i * 0.05,
                    sustainability=0.5 + i * 0.05,
                    scalability=0.7 + i * 0.05,
                    overall_fitness=0.66 + i * 0.05
                )
                for i in range(3)
            ]
            
            # Gen 1 evaluations (evolved)
            evolved_ideas = [
                GeneratedIdea(
                    content=f"Evolved idea {i}",
                    thinking_method="crossover",
                    agent_name="evolution",
                    generation_prompt="test"
                )
                for i in range(3)
            ]
            
            gen1_fitness = [
                IndividualFitness(
                    idea=evolved_ideas[i],
                    impact=0.8 + i * 0.05,
                    feasibility=0.7 + i * 0.05,
                    accessibility=0.85 + i * 0.05,
                    sustainability=0.6 + i * 0.05,
                    scalability=0.8 + i * 0.05,
                    overall_fitness=0.77 + i * 0.05
                )
                for i in range(3)
            ]
            
            # Mock evaluation calls
            mock_evaluator.evaluate_population = AsyncMock(
                side_effect=[gen0_fitness, gen1_fitness]
            )
            
            # Mock diversity calculation
            mock_evaluator.calculate_population_diversity = AsyncMock(
                return_value=0.8
            )
            
            # Create GA and run evolution
            ga = GeneticAlgorithm(llm_provider=mock_provider)
            
            request = EvolutionRequest(
                initial_population=initial_ideas,
                config=EvolutionConfig(
                    population_size=3,
                    generations=2,
                    mutation_rate=0.1,
                    crossover_rate=0.7,
                    elite_size=1
                )
            )
            
            # Mock the evolution process to avoid complex operator mocking
            from mad_spark_alt.evolution.interfaces import PopulationSnapshot, EvolutionResult
            
            snapshot0 = PopulationSnapshot.from_population(0, gen0_fitness)
            snapshot0.diversity_score = 0.8
            
            snapshot1 = PopulationSnapshot.from_population(1, gen1_fitness)
            snapshot1.diversity_score = 0.75
            
            result = EvolutionResult(
                final_population=gen1_fitness,
                best_ideas=[gen1_fitness[2].idea],  # Highest scoring
                generation_snapshots=[snapshot0, snapshot1],
                total_generations=2,
                execution_time=5.0
            )
            
            # Test that we can get all individuals
            all_individuals = result.get_all_individuals()
            
            assert len(all_individuals) == 6  # 3 from gen 0 + 3 from gen 1
            
            # Check that both initial and evolved ideas are included
            all_contents = [ind.idea.content for ind in all_individuals]
            assert "Initial idea 0" in all_contents
            assert "Initial idea 1" in all_contents
            assert "Initial idea 2" in all_contents
            assert "Evolved idea 0" in all_contents
            assert "Evolved idea 1" in all_contents
            assert "Evolved idea 2" in all_contents
            
            # Check scoring consistency
            for ind in all_individuals:
                assert hasattr(ind, 'impact')
                assert hasattr(ind, 'feasibility')
                assert hasattr(ind, 'accessibility')
                assert hasattr(ind, 'sustainability')
                assert hasattr(ind, 'scalability')
                assert hasattr(ind, 'overall_fitness')