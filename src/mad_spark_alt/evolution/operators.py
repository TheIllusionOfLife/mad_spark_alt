"""
Genetic operators for idea evolution.

This module implements crossover, mutation, and selection operators
for evolving ideas through genetic algorithms.
"""

import random
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from mad_spark_alt.core.interfaces import GeneratedIdea, ThinkingMethod
from mad_spark_alt.evolution.interfaces import (
    CrossoverInterface,
    EvolutionConfig,
    IndividualFitness,
    MutationInterface,
    SelectionInterface,
)


class CrossoverOperator(CrossoverInterface):
    """
    Implements semantic crossover for ideas.

    This operator combines elements from two parent ideas to create offspring
    that inherit characteristics from both parents.
    """

    @property
    def name(self) -> str:
        return "semantic_crossover"

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate crossover configuration."""
        return True  # No specific config needed for basic crossover

    async def crossover(
        self,
        parent1: GeneratedIdea,
        parent2: GeneratedIdea,
        context: Optional[str] = None,
    ) -> Tuple[GeneratedIdea, GeneratedIdea]:
        """
        Perform semantic crossover between two parent ideas.

        Args:
            parent1: First parent idea
            parent2: Second parent idea
            context: Optional context for crossover

        Returns:
            Tuple of two offspring ideas
        """
        # Extract key components from parent ideas
        p1_components = self._extract_components(parent1.content)
        p2_components = self._extract_components(parent2.content)

        # Create crossover points
        max_components = min(len(p1_components), len(p2_components))
        if max_components <= 1:
            # If content is too simple, just swap
            return parent2, parent1

        crossover_point = random.randint(1, max_components - 1)

        # Create offspring by combining components
        offspring1_content = self._combine_components(
            p1_components[:crossover_point] + p2_components[crossover_point:]
        )
        offspring2_content = self._combine_components(
            p2_components[:crossover_point] + p1_components[crossover_point:]
        )

        # Create offspring ideas
        offspring1 = GeneratedIdea(
            content=offspring1_content,
            thinking_method=parent1.thinking_method,  # Inherit from parent1
            agent_name="CrossoverOperator",
            generation_prompt=f"Crossover of ideas: '{parent1.content[:50]}...' and '{parent2.content[:50]}...'",
            confidence_score=(
                (parent1.confidence_score or 0.5) + (parent2.confidence_score or 0.5)
            )
            / 2,
            reasoning=f"Combined elements from both parent ideas at crossover point {crossover_point}",
            parent_ideas=[parent1.content, parent2.content],
            metadata={
                "operator": "crossover",
                "crossover_point": crossover_point,
                "generation": max(
                    parent1.metadata.get("generation", 0),
                    parent2.metadata.get("generation", 0),
                )
                + 1,
            },
            timestamp=datetime.now().isoformat(),
        )

        offspring2 = GeneratedIdea(
            content=offspring2_content,
            thinking_method=parent2.thinking_method,  # Inherit from parent2
            agent_name="CrossoverOperator",
            generation_prompt=f"Crossover of ideas: '{parent2.content[:50]}...' and '{parent1.content[:50]}...'",
            confidence_score=(
                (parent1.confidence_score or 0.5) + (parent2.confidence_score or 0.5)
            )
            / 2,
            reasoning=f"Combined elements from both parent ideas at crossover point {crossover_point}",
            parent_ideas=[parent2.content, parent1.content],
            metadata={
                "operator": "crossover",
                "crossover_point": crossover_point,
                "generation": max(
                    parent1.metadata.get("generation", 0),
                    parent2.metadata.get("generation", 0),
                )
                + 1,
            },
            timestamp=datetime.now().isoformat(),
        )

        return offspring1, offspring2

    def _extract_components(self, content: str) -> List[str]:
        """Extract semantic components from idea content."""
        # Simple implementation: split by sentences or key phrases
        # In a more sophisticated version, this could use NLP
        sentences = content.split(". ")
        return [s.strip() for s in sentences if s.strip()]

    def _combine_components(self, components: List[str]) -> str:
        """Combine components back into coherent content."""
        # Ensure proper sentence structure
        combined = ". ".join(components)
        if not combined.endswith("."):
            combined += "."
        return combined


class MutationOperator(MutationInterface):
    """
    Implements semantic mutation for ideas.

    This operator introduces random variations to ideas to maintain
    genetic diversity and explore new solution spaces.
    """

    @property
    def name(self) -> str:
        return "semantic_mutation"

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate mutation configuration."""
        return True  # No specific config needed

    async def mutate(
        self, idea: GeneratedIdea, mutation_rate: float, context: Optional[str] = None
    ) -> GeneratedIdea:
        """
        Mutate an idea with given mutation rate.

        Args:
            idea: Idea to mutate
            mutation_rate: Probability of mutation (0-1)
            context: Optional context for mutation

        Returns:
            Mutated idea (or original if no mutation occurs)
        """
        # Check if mutation should occur
        if random.random() < mutation_rate:
            # Choose mutation type
            mutation_type = random.choice(
                [
                    "word_substitution",
                    "phrase_reordering",
                    "concept_addition",
                    "concept_removal",
                    "emphasis_change",
                ]
            )

            # Apply mutation
            mutated_content = self._apply_mutation(idea.content, mutation_type)

            # Create mutated idea
            mutated_idea = GeneratedIdea(
                content=mutated_content,
                thinking_method=idea.thinking_method,
                agent_name="MutationOperator",
                generation_prompt=f"Mutation ({mutation_type}) of: '{idea.content[:50]}...'",
                confidence_score=(idea.confidence_score or 0.5)
                * 0.95,  # Slightly reduce confidence
                reasoning=f"Applied {mutation_type} mutation to introduce variation",
                parent_ideas=[idea.content],
                metadata={
                    "operator": "mutation",
                    "mutation_type": mutation_type,
                    "mutation_rate": mutation_rate,
                    "generation": idea.metadata.get("generation", 0) + 1,
                },
                timestamp=datetime.now().isoformat(),
            )

            return mutated_idea
        else:
            return idea  # No mutation

    def _apply_mutation(self, content: str, mutation_type: str) -> str:
        """Apply specific mutation type to content."""
        if mutation_type == "word_substitution":
            # Replace random words with synonyms or related concepts
            words = content.split()
            if len(words) > 3:
                idx = random.randint(0, len(words) - 1)
                # Simple substitution - in production, use word embeddings
                substitutions = {
                    # Action verbs
                    "improve": "enhance",
                    "improves": "enhances",
                    "create": "develop",
                    "creates": "develops",
                    "use": "utilize",
                    "uses": "utilizes",
                    "make": "construct",
                    "makes": "constructs",
                    "reduce": "minimize",
                    "reduces": "minimizes",
                    "increase": "amplify",
                    "increases": "amplifies",
                    "build": "construct",
                    "builds": "constructs",
                    "design": "architect",
                    "designs": "architects",
                    "implement": "deploy",
                    "implements": "deploys",
                    # Descriptive adjectives
                    "innovative": "creative",
                    "effective": "efficient",
                    "simple": "streamlined",
                    "complex": "sophisticated",
                    "advanced": "cutting-edge",
                    "modern": "contemporary",
                    "traditional": "conventional",
                    "unique": "distinctive",
                    "powerful": "robust",
                    "flexible": "adaptable",
                    # Nouns and concepts
                    "efficiency": "effectiveness",
                    "idea": "concept",
                    "ideas": "concepts",
                    "solution": "approach",
                    "solutions": "approaches",
                    "method": "technique",
                    "methods": "techniques",
                    "system": "framework",
                    "systems": "frameworks",
                    "process": "procedure",
                    "processes": "procedures",
                    "technology": "innovation",
                    "problem": "challenge",
                    "problems": "challenges",
                    "opportunity": "possibility",
                    "opportunities": "possibilities",
                }
                word = words[idx].lower().strip(".,!?")
                if word in substitutions:
                    words[idx] = substitutions[word]
                else:
                    # If no substitution found, add a modifier
                    words[idx] = f"enhanced_{words[idx]}"
            return " ".join(words)

        elif mutation_type == "phrase_reordering":
            # Reorder phrases or sentences
            # Handle both ". " and "." as sentence separators
            if ". " in content:
                sentences = content.split(". ")
                if len(sentences) > 1:
                    random.shuffle(sentences)
                    return ". ".join(sentences)
            elif "." in content:
                # Split by "." and preserve the dots
                sentences = [s.strip() for s in content.split(".") if s.strip()]
                if len(sentences) > 1:
                    random.shuffle(sentences)
                    return ". ".join(sentences) + "."
            return content

        elif mutation_type == "concept_addition":
            # Add a related concept
            additions = [
                " Additionally, consider sustainability aspects.",
                " This could also incorporate user feedback mechanisms.",
                " Integration with existing systems would enhance adoption.",
                " Scalability should be a key consideration.",
            ]
            return content + random.choice(additions)

        elif mutation_type == "concept_removal":
            # Remove a sentence (if multiple exist)
            sentences = content.split(". ")
            if len(sentences) > 2:
                sentences.pop(random.randint(0, len(sentences) - 1))
            return ". ".join(sentences)

        elif mutation_type == "emphasis_change":
            # Change emphasis or priority
            emphasis_words = ["critically", "importantly", "primarily", "especially"]
            words = content.split()
            insert_pos = random.randint(0, len(words))
            words.insert(insert_pos, random.choice(emphasis_words))
            return " ".join(words)

        return content  # Fallback


class TournamentSelection(SelectionInterface):
    """
    Tournament selection operator.

    Selects individuals by running tournaments between random subsets
    of the population.
    """

    @property
    def name(self) -> str:
        return "tournament_selection"

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate selection configuration."""
        return True

    async def select(
        self,
        population: List[IndividualFitness],
        num_selected: int,
        config: EvolutionConfig,
    ) -> List[IndividualFitness]:
        """
        Select individuals using tournament selection.

        Args:
            population: Population to select from
            num_selected: Number of individuals to select
            config: Evolution configuration with tournament size

        Returns:
            Selected individuals
        """
        selected = []

        for _ in range(num_selected):
            # Run a tournament
            tournament_size = min(config.tournament_size, len(population))
            tournament = random.sample(population, tournament_size)

            # Select winner (highest fitness)
            winner = max(tournament, key=lambda x: x.overall_fitness)
            selected.append(winner)

        return selected


class EliteSelection(SelectionInterface):
    """
    Elite selection operator.

    Always preserves the best individuals from the population.
    """

    @property
    def name(self) -> str:
        return "elite_selection"

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate selection configuration."""
        return True

    async def select(
        self,
        population: List[IndividualFitness],
        num_selected: int,
        config: EvolutionConfig,
    ) -> List[IndividualFitness]:
        """
        Select top individuals by fitness.

        Args:
            population: Population to select from
            num_selected: Number of individuals to select
            config: Evolution configuration

        Returns:
            Top individuals by fitness
        """
        # Sort by fitness (descending)
        sorted_pop = sorted(population, key=lambda x: x.overall_fitness, reverse=True)

        # Return top individuals
        return sorted_pop[:num_selected]


class RouletteWheelSelection(SelectionInterface):
    """
    Roulette wheel selection operator.

    Selection probability is proportional to fitness scores.
    """

    @property
    def name(self) -> str:
        return "roulette_wheel_selection"

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate selection configuration."""
        return True

    async def select(
        self,
        population: List[IndividualFitness],
        num_selected: int,
        config: EvolutionConfig,
    ) -> List[IndividualFitness]:
        """
        Select individuals using roulette wheel (fitness-proportionate) selection.

        Args:
            population: Population to select from
            num_selected: Number of individuals to select
            config: Evolution configuration

        Returns:
            Selected individuals
        """
        selected = []

        # Calculate total fitness (ensure all positive)
        min_fitness = min(ind.overall_fitness for ind in population)
        if min_fitness < 0:
            # Shift all fitnesses to be positive
            adjusted_fitnesses = [
                ind.overall_fitness - min_fitness + 0.001 for ind in population
            ]
        else:
            adjusted_fitnesses = [
                ind.overall_fitness + 0.001 for ind in population
            ]  # Avoid zero

        total_fitness = sum(adjusted_fitnesses)

        for _ in range(num_selected):
            # Spin the roulette wheel
            spin = random.uniform(0, total_fitness)
            cumulative = 0.0

            for i, fitness in enumerate(adjusted_fitnesses):
                cumulative += fitness
                if cumulative >= spin:
                    selected.append(population[i])
                    break

        return selected


class RankSelection(SelectionInterface):
    """
    Rank-based selection operator.

    Selection is based on rank rather than raw fitness values.
    """

    @property
    def name(self) -> str:
        return "rank_selection"

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate selection configuration."""
        return True

    async def select(
        self,
        population: List[IndividualFitness],
        num_selected: int,
        config: EvolutionConfig,
    ) -> List[IndividualFitness]:
        """
        Select individuals using rank-based selection.

        Args:
            population: Population to select from
            num_selected: Number of individuals to select
            config: Evolution configuration

        Returns:
            Selected individuals
        """
        selected = []

        # Sort population by fitness and assign ranks
        sorted_pop = sorted(population, key=lambda x: x.overall_fitness, reverse=True)
        n = len(sorted_pop)

        # Create rank-based weights (linear ranking)
        ranks = list(range(n, 0, -1))  # Best gets rank n, worst gets rank 1
        total_rank = sum(ranks)

        for _ in range(num_selected):
            # Select based on rank probability
            spin = random.uniform(0, total_rank)
            cumulative = 0.0

            for i, rank in enumerate(ranks):
                cumulative += rank
                if cumulative >= spin:
                    selected.append(sorted_pop[i])
                    break

        return selected


class RandomSelection(SelectionInterface):
    """
    Random selection operator.

    Selects individuals uniformly at random (no fitness bias).
    """

    @property
    def name(self) -> str:
        return "random_selection"

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate selection configuration."""
        return True

    async def select(
        self,
        population: List[IndividualFitness],
        num_selected: int,
        config: EvolutionConfig,
    ) -> List[IndividualFitness]:
        """
        Select individuals randomly (uniform distribution).

        Args:
            population: Population to select from
            num_selected: Number of individuals to select
            config: Evolution configuration

        Returns:
            Randomly selected individuals
        """
        return random.choices(population, k=num_selected)
