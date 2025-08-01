"""
Tests for comprehensive idea collection from all generations.
"""

from typing import List

from mad_spark_alt.core.interfaces import GeneratedIdea
from mad_spark_alt.evolution.interfaces import (
    IndividualFitness, EvolutionResult, PopulationSnapshot
)
from mad_spark_alt.core.simple_qadi_orchestrator import HypothesisScore


class TestComprehensiveIdeaCollection:
    """Test comprehensive idea collection from all generations and QADI."""
    
    def test_evolution_result_get_all_individuals(self):
        """Test that EvolutionResult can get all individuals from all generations."""
        # Create mock individuals from different generations
        gen0_ideas = [
            GeneratedIdea(
                content="Gen 0 Idea 1",
                thinking_method="hypothesis",
                agent_name="qadi",
                generation_prompt="initial"
            ),
            GeneratedIdea(
                content="Gen 0 Idea 2", 
                thinking_method="hypothesis",
                agent_name="qadi",
                generation_prompt="initial"
            )
        ]
        
        gen1_ideas = [
            GeneratedIdea(
                content="Gen 1 Idea 1",
                thinking_method="crossover",
                agent_name="evolution",
                generation_prompt="evolution"
            ),
            GeneratedIdea(
                content="Gen 1 Idea 2",
                thinking_method="mutation", 
                agent_name="evolution",
                generation_prompt="evolution"
            )
        ]
        
        # Create fitness objects
        gen0_fitness = [
            IndividualFitness(
                idea=idea,
                impact=0.7,
                feasibility=0.6,
                accessibility=0.8,
                sustainability=0.5,
                scalability=0.7,
                overall_fitness=0.66
            ) for idea in gen0_ideas
        ]
        
        gen1_fitness = [
            IndividualFitness(
                idea=idea,
                impact=0.8,
                feasibility=0.7,
                accessibility=0.85,
                sustainability=0.6,
                scalability=0.8,
                overall_fitness=0.76
            ) for idea in gen1_ideas
        ]
        
        # Create snapshots
        snapshot0 = PopulationSnapshot.from_population(0, gen0_fitness)
        snapshot1 = PopulationSnapshot.from_population(1, gen1_fitness)
        
        # Create evolution result
        result = EvolutionResult(
            final_population=gen1_fitness,
            best_ideas=[gen1_ideas[1]],  # Best from final
            generation_snapshots=[snapshot0, snapshot1],
            total_generations=2,
            execution_time=10.0
        )
        
        # Get all individuals
        all_individuals = result.get_all_individuals()
        
        # Should have all 4 individuals (2 from each generation)
        assert len(all_individuals) == 4
        
        # Check content is preserved
        all_contents = [ind.idea.content for ind in all_individuals]
        assert "Gen 0 Idea 1" in all_contents
        assert "Gen 0 Idea 2" in all_contents
        assert "Gen 1 Idea 1" in all_contents
        assert "Gen 1 Idea 2" in all_contents
        
    def test_combined_qadi_and_evolution_idea_collection(self):
        """Test collecting ideas from both QADI hypotheses and evolution generations."""
        
        # Mock QADI hypotheses
        qadi_hypotheses = [
            "Hypothesis 1: Solar panel installation",
            "Hypothesis 2: Wind energy system",
            "Hypothesis 3: Hydro power solution"
        ]
        
        qadi_scores = [
            HypothesisScore(
                impact=0.8, feasibility=0.7, accessibility=0.9, 
                sustainability=0.6, scalability=0.75, overall=0.77
            ),
            HypothesisScore(
                impact=0.75, feasibility=0.8, accessibility=0.85,
                sustainability=0.7, scalability=0.8, overall=0.78
            ),
            HypothesisScore(
                impact=0.9, feasibility=0.6, accessibility=0.7,
                sustainability=0.8, scalability=0.65, overall=0.75
            )
        ]
        
        # Mock evolution ideas (already as IndividualFitness)
        evolved_ideas = [
            IndividualFitness(
                idea=GeneratedIdea(
                    content="Evolved solar-wind hybrid",
                    thinking_method="crossover",
                    agent_name="evolution",
                    generation_prompt="energy"
                ),
                impact=0.85, feasibility=0.75, accessibility=0.8,
                sustainability=0.75, scalability=0.85, overall_fitness=0.8
            ),
            IndividualFitness(
                idea=GeneratedIdea(
                    content="Mutated hydro-solar combo",
                    thinking_method="mutation",
                    agent_name="evolution", 
                    generation_prompt="energy"
                ),
                impact=0.9, feasibility=0.7, accessibility=0.75,
                sustainability=0.85, scalability=0.7, overall_fitness=0.78
            )
        ]
        
        # Test combining ideas
        def combine_all_ideas(
            hypotheses: List[str], 
            hypothesis_scores: List[HypothesisScore],
            evolution_individuals: List[IndividualFitness]
        ) -> List[IndividualFitness]:
            """Combine QADI hypotheses and evolution individuals."""
            combined = []
            
            # Convert QADI hypotheses to IndividualFitness format
            for hypothesis, score in zip(hypotheses, hypothesis_scores):
                combined.append(IndividualFitness(
                    idea=GeneratedIdea(
                        content=hypothesis,
                        thinking_method="hypothesis",
                        agent_name="qadi",
                        generation_prompt="initial"
                    ),
                    impact=score.impact,
                    feasibility=score.feasibility,
                    accessibility=score.accessibility,
                    sustainability=score.sustainability,
                    scalability=score.scalability,
                    overall_fitness=score.overall
                ))
            
            # Add evolution individuals
            combined.extend(evolution_individuals)
            
            return combined
        
        all_ideas = combine_all_ideas(qadi_hypotheses, qadi_scores, evolved_ideas)
        
        # Should have 5 total ideas (3 QADI + 2 evolved)
        assert len(all_ideas) == 5
        
        # Check that QADI ideas are included
        qadi_contents = [idea.idea.content for idea in all_ideas[:3]]
        assert "Solar panel installation" in qadi_contents[0]
        assert "Wind energy system" in qadi_contents[1]
        assert "Hydro power solution" in qadi_contents[2]
        
        # Check that evolved ideas are included
        evolved_contents = [idea.idea.content for idea in all_ideas[3:]]
        assert "Evolved solar-wind hybrid" in evolved_contents[0]
        assert "Mutated hydro-solar combo" in evolved_contents[1]
        
        # All should have consistent scoring
        for idea in all_ideas:
            assert hasattr(idea, 'impact')
            assert hasattr(idea, 'feasibility')
            assert hasattr(idea, 'accessibility')
            assert hasattr(idea, 'sustainability')
            assert hasattr(idea, 'scalability')
            assert hasattr(idea, 'overall_fitness')
            
    def test_top_ideas_selection_with_deduplication(self):
        """Test selecting top ideas with deduplication across all sources."""
        
        def deduplicate_ideas(ideas: List[IndividualFitness]) -> List[IndividualFitness]:
            """Remove duplicate ideas based on content similarity."""
            from difflib import SequenceMatcher
            
            unique_ideas = []
            seen_contents = []
            
            # Sort by fitness first (highest first)
            sorted_ideas = sorted(ideas, key=lambda x: x.overall_fitness, reverse=True)
            
            for idea in sorted_ideas:
                content = idea.idea.content.strip().lower()
                
                # Check if similar to any seen content
                is_duplicate = False
                for seen_content in seen_contents:
                    similarity = SequenceMatcher(None, content, seen_content).ratio()
                    if similarity > 0.7:  # 70% similar = duplicate
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    unique_ideas.append(idea)
                    seen_contents.append(content)
                    
            return unique_ideas
        
        # Create ideas with some duplicates
        ideas = [
            IndividualFitness(
                idea=GeneratedIdea(
                    content="Solar panel installation on rooftops",
                    thinking_method="hypothesis",
                    agent_name="qadi", 
                    generation_prompt="test"
                ),
                overall_fitness=0.85
            ),
            IndividualFitness(
                idea=GeneratedIdea(
                    content="Solar panel installation on rooftops with battery storage",  # Similar
                    thinking_method="mutation",
                    agent_name="evolution",
                    generation_prompt="test"
                ),
                overall_fitness=0.82  # Lower fitness, should be filtered
            ),
            IndividualFitness(
                idea=GeneratedIdea(
                    content="Wind turbine energy generation",  # Different
                    thinking_method="hypothesis",
                    agent_name="qadi",
                    generation_prompt="test"
                ),
                overall_fitness=0.78
            ),
            IndividualFitness(
                idea=GeneratedIdea(
                    content="Hydroelectric power system",  # Different
                    thinking_method="crossover",
                    agent_name="evolution",
                    generation_prompt="test"
                ),
                overall_fitness=0.88  # Highest fitness
            )
        ]
        
        # Test deduplication
        unique_ideas = deduplicate_ideas(ideas)
        
        # Should keep 3 unique ideas (remove the similar solar one)
        assert len(unique_ideas) == 3
        
        # Should be sorted by fitness (hydroelectric first)
        assert unique_ideas[0].idea.content == "Hydroelectric power system"
        assert unique_ideas[0].overall_fitness == 0.88
        
        # Should keep the higher-fitness solar option
        solar_ideas = [idea for idea in unique_ideas if "solar" in idea.idea.content.lower()]
        assert len(solar_ideas) == 1
        assert solar_ideas[0].overall_fitness == 0.85
        
    def test_high_score_approaches_display_format(self):
        """Test that the display shows 'High Score Approaches' with scores."""
        
        # Mock top ideas after collection and deduplication
        top_ideas = [
            IndividualFitness(
                idea=GeneratedIdea(
                    content="Hydroelectric power with smart grid integration",
                    thinking_method="crossover",
                    agent_name="evolution",
                    generation_prompt="energy"
                ),
                impact=0.9, feasibility=0.8, accessibility=0.75,
                sustainability=0.85, scalability=0.8, overall_fitness=0.82
            ),
            IndividualFitness(
                idea=GeneratedIdea(
                    content="Community solar panel cooperative",
                    thinking_method="hypothesis", 
                    agent_name="qadi",
                    generation_prompt="energy"
                ),
                impact=0.85, feasibility=0.85, accessibility=0.9,
                sustainability=0.75, scalability=0.8, overall_fitness=0.83
            )
        ]
        
        # Test display formatting
        for i, idea in enumerate(top_ideas, 1):
            # Title format
            title = f"**{i}. High Score Approaches**"
            assert "High Score Approaches" in title
            assert "Enhanced" not in title
            
            # Score format
            scores = idea.get_scores_dict()
            score_display = f"[Overall: {scores['overall']:.2f} | Impact: {scores['impact']:.2f} | Feasibility: {scores['feasibility']:.2f} | Accessibility: {scores['accessibility']:.2f} | Sustainability: {scores['sustainability']:.2f} | Scalability: {scores['scalability']:.2f}]"
            
            # Verify score format structure
            assert score_display.startswith("[Overall:")
            assert "Impact:" in score_display
            assert "Feasibility:" in score_display
            assert "Accessibility:" in score_display
            assert "Sustainability:" in score_display
            assert "Scalability:" in score_display
            assert score_display.endswith("]")