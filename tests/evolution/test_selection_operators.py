"""
Tests for selection operators.
"""

import pytest
from mad_spark_alt.evolution.operators import (
    RouletteWheelSelection,
    RankSelection,
    TournamentSelection,
    EliteSelection,
    RandomSelection
)
from mad_spark_alt.evolution.interfaces import IndividualFitness, EvolutionConfig
from mad_spark_alt.core.interfaces import GeneratedIdea, ThinkingMethod

class TestSelectionOperators:

    @pytest.fixture
    def population(self):
        """Create a sample population with known fitness values."""
        pop = []
        # Create 10 individuals with fitness 0.1, 0.2, ... 1.0
        for i in range(1, 11):
            idea = GeneratedIdea(
                content=f"Idea {i}",
                thinking_method=ThinkingMethod.QUESTIONING,
                agent_name="test",
                generation_prompt="test",
                confidence_score=0.5
            )
            ind = IndividualFitness(
                idea=idea,
                overall_fitness=i / 10.0,
                impact=0.5,
                feasibility=0.5,
                accessibility=0.5,
                sustainability=0.5,
                scalability=0.5
            )
            pop.append(ind)
        return pop

    @pytest.fixture
    def config(self):
        return EvolutionConfig()

    @pytest.mark.asyncio
    async def test_roulette_wheel_selection(self, population, config):
        selector = RouletteWheelSelection()
        selected = await selector.select(population, 5, config)

        assert len(selected) == 5
        assert all(isinstance(ind, IndividualFitness) for ind in selected)
        # Higher fitness should be more likely, but it's probabilistic.
        # Just check it runs and returns valid individuals.

    @pytest.mark.asyncio
    async def test_rank_selection(self, population, config):
        selector = RankSelection()
        selected = await selector.select(population, 5, config)

        assert len(selected) == 5
        assert all(isinstance(ind, IndividualFitness) for ind in selected)

    @pytest.mark.asyncio
    async def test_tournament_selection(self, population, config):
        selector = TournamentSelection()
        selected = await selector.select(population, 5, config)

        assert len(selected) == 5
        assert all(isinstance(ind, IndividualFitness) for ind in selected)

    @pytest.mark.asyncio
    async def test_elite_selection(self, population, config):
        selector = EliteSelection()
        selected = await selector.select(population, 3, config)

        assert len(selected) == 3
        # Should select top 3 (0.8, 0.9, 1.0)
        fitnesses = sorted([ind.overall_fitness for ind in selected])
        assert fitnesses == [0.8, 0.9, 1.0]

    @pytest.mark.asyncio
    async def test_random_selection(self, population, config):
        selector = RandomSelection()
        selected = await selector.select(population, 5, config)

        assert len(selected) == 5
        assert all(isinstance(ind, IndividualFitness) for ind in selected)

    @pytest.mark.asyncio
    async def test_roulette_negative_fitness(self, config):
        """Test roulette selection handles negative fitness values."""
        pop = []
        for i in range(5):
            idea = GeneratedIdea(content=f"Idea {i}", thinking_method=ThinkingMethod.QUESTIONING, agent_name="test", generation_prompt="test", confidence_score=0.5)
            # Create negative fitness
            ind = IndividualFitness(
                idea=idea,
                overall_fitness = -10 + i, # -10, -9, -8, -7, -6
                impact=0.5, feasibility=0.5, accessibility=0.5, sustainability=0.5, scalability=0.5
            )
            pop.append(ind)

        selector = RouletteWheelSelection()
        selected = await selector.select(pop, 3, config)
        assert len(selected) == 3
