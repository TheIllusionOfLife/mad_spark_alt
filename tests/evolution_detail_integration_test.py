"""
Integration tests for evolution producing detailed outputs.
These tests verify the complete flow from hypothesis generation to evolution.
"""

import pytest
from mad_spark_alt.core.simple_qadi_orchestrator import SimpleQADIOrchestrator


class TestEvolutionDetailIntegration:
    """Integration tests for detailed evolution outputs."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_hypothesis_generation_produces_detailed_content(self):
        """Test that hypotheses generated contain detailed explanations."""
        orchestrator = SimpleQADIOrchestrator(num_hypotheses=3)
        
        # Run QADI cycle
        result = await orchestrator.run_qadi_cycle(
            "How can we implement AGI?",
            "Focus on practical approaches"
        )
        
        # Verify hypotheses are detailed
        assert len(result.hypotheses) >= 3
        for hypothesis in result.hypotheses:
            # Each hypothesis should be detailed (at least 100 words ~ 500 chars)
            assert len(hypothesis) >= 500, f"Hypothesis too short: {len(hypothesis)} chars"
        
        # Verify synthesized ideas preserve full content
        assert len(result.synthesized_ideas) >= 3
        for idea in result.synthesized_ideas:
            assert len(idea.content) >= 500, f"Synthesized idea too short: {len(idea.content)} chars"
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_evolution_maintains_detail_level(self):
        """Test that evolution maintains or enhances detail level."""
        from mad_spark_alt.evolution.genetic_algorithm import GeneticAlgorithm
        from mad_spark_alt.evolution.interfaces import EvolutionRequest, EvolutionConfig
        from mad_spark_alt.core.interfaces import GeneratedIdea, ThinkingMethod
        from mad_spark_alt.core.llm_provider import get_google_provider
        
        # Create detailed initial ideas
        initial_ideas = [
            GeneratedIdea(
                content="A" * 600,  # 600 character detailed idea
                thinking_method=ThinkingMethod.ABDUCTION,
                agent_name="test",
                generation_prompt="test",
                confidence_score=0.8,
                reasoning="test"
            ),
            GeneratedIdea(
                content="B" * 700,  # 700 character detailed idea
                thinking_method=ThinkingMethod.DEDUCTION,
                agent_name="test",
                generation_prompt="test",
                confidence_score=0.7,
                reasoning="test"
            ),
            GeneratedIdea(
                content="C" * 800,  # 800 character detailed idea
                thinking_method=ThinkingMethod.INDUCTION,
                agent_name="test",
                generation_prompt="test",
                confidence_score=0.6,
                reasoning="test"
            )
        ]
        
        # Create GA with semantic operators
        llm_provider = get_google_provider()
        ga = GeneticAlgorithm(llm_provider=llm_provider)
        
        # Configure evolution
        config = EvolutionConfig(
            population_size=3,
            generations=1,
            mutation_rate=0.5,
            crossover_rate=0.5
        )
        
        request = EvolutionRequest(
            initial_population=initial_ideas,
            config=config,
            context="Test evolution maintaining detail"
        )
        
        # Run evolution
        result = await ga.evolve(request)
        
        # Verify evolved ideas maintain detail
        assert result.success
        for individual in result.final_population:
            # Evolved ideas should maintain or increase detail
            assert len(individual.idea.content) >= 600, f"Evolved idea too short: {len(individual.idea.content)} chars"
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_end_to_end_qadi_evolution_detail(self):
        """Test complete QADI + evolution flow produces detailed results."""
        from mad_spark_alt.evolution.genetic_algorithm import GeneticAlgorithm
        from mad_spark_alt.evolution.interfaces import EvolutionRequest, EvolutionConfig
        from mad_spark_alt.core.llm_provider import get_google_provider
        
        # Step 1: Run QADI
        orchestrator = SimpleQADIOrchestrator(num_hypotheses=3)
        qadi_result = await orchestrator.run_qadi_cycle(
            "How can we solve climate change?",
            "Focus on technological solutions"
        )
        
        # Verify QADI produced detailed hypotheses
        assert len(qadi_result.synthesized_ideas) >= 3
        min_initial_length = min(len(idea.content) for idea in qadi_result.synthesized_ideas)
        assert min_initial_length >= 500, f"Initial ideas too short: {min_initial_length} chars"
        
        # Step 2: Evolve the ideas
        llm_provider = get_google_provider()
        ga = GeneticAlgorithm(llm_provider=llm_provider)
        
        config = EvolutionConfig(
            population_size=3,
            generations=2,
            mutation_rate=0.3,
            crossover_rate=0.7
        )
        
        request = EvolutionRequest(
            initial_population=qadi_result.synthesized_ideas[:3],
            config=config,
            context="Evolve climate change solutions"
        )
        
        evolution_result = await ga.evolve(request)
        
        # Verify evolution maintained/enhanced detail
        assert evolution_result.success
        for individual in evolution_result.final_population:
            assert len(individual.idea.content) >= min_initial_length, \
                f"Evolution reduced detail: {len(individual.idea.content)} < {min_initial_length}"
        
        # Verify fitness improved
        assert evolution_result.evolution_metrics.get('fitness_improvement_percent', 0) > 0