"""
Mutation strategies for genetic algorithm.

This module implements the Strategy pattern for different types of mutations,
replacing the large, complex _apply_mutation method.
"""

import random
from abc import ABC, abstractmethod
from typing import Dict, List


class MutationStrategy(ABC):
    """Abstract base class for mutation strategies."""

    @abstractmethod
    def apply(self, content: str) -> str:
        """Apply the mutation to the given content."""
        pass


class WordSubstitutions:
    """Centralized word substitution mappings."""

    SUBSTITUTIONS: Dict[str, str] = {
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


class WordSubstitutionStrategy(MutationStrategy):
    """Replace random words with synonyms or related concepts."""

    def __init__(self) -> None:
        self.substitutions = WordSubstitutions.SUBSTITUTIONS

    def apply(self, content: str) -> str:
        """Apply word substitution mutation."""
        words = content.split()
        if len(words) <= 3:
            return content

        idx = random.randint(0, len(words) - 1)
        word = words[idx].lower().strip(".,!?")

        if word in self.substitutions:
            words[idx] = self.substitutions[word]
        else:
            # If no substitution found, add a modifier
            words[idx] = f"enhanced_{words[idx]}"

        return " ".join(words)


class PhraseReorderingStrategy(MutationStrategy):
    """Reorder phrases or sentences."""

    def apply(self, content: str) -> str:
        """Apply phrase reordering mutation."""
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


class ConceptAdditionStrategy(MutationStrategy):
    """Add a related concept to the content."""

    ADDITIONS: List[str] = [
        " Additionally, consider sustainability aspects.",
        " This could also incorporate user feedback mechanisms.",
        " Integration with existing systems would enhance adoption.",
        " Scalability should be a key consideration.",
    ]

    def apply(self, content: str) -> str:
        """Apply concept addition mutation."""
        return content + random.choice(self.ADDITIONS)


class ConceptRemovalStrategy(MutationStrategy):
    """Remove a sentence (if multiple exist)."""

    def apply(self, content: str) -> str:
        """Apply concept removal mutation."""
        sentences = content.split(". ")
        if len(sentences) > 2:
            sentences.pop(random.randint(0, len(sentences) - 1))
        return ". ".join(sentences)


class EmphasisChangeStrategy(MutationStrategy):
    """Change emphasis or priority in the content."""

    EMPHASIS_WORDS: List[str] = ["critically", "importantly", "primarily", "especially"]

    def apply(self, content: str) -> str:
        """Apply emphasis change mutation."""
        words = content.split()
        insert_pos = random.randint(0, len(words))
        words.insert(insert_pos, random.choice(self.EMPHASIS_WORDS))
        return " ".join(words)


class MutationStrategyFactory:
    """Factory for creating mutation strategies."""

    _strategies: Dict[str, MutationStrategy] = {
        "word_substitution": WordSubstitutionStrategy(),
        "phrase_reordering": PhraseReorderingStrategy(),
        "concept_addition": ConceptAdditionStrategy(),
        "concept_removal": ConceptRemovalStrategy(),
        "emphasis_change": EmphasisChangeStrategy(),
    }

    @classmethod
    def get_strategy(cls, mutation_type: str) -> MutationStrategy:
        """Get a mutation strategy by type."""
        strategy = cls._strategies.get(mutation_type)
        if strategy is None:
            raise ValueError(f"Unknown mutation type: {mutation_type}")
        return strategy

    @classmethod
    def get_available_types(cls) -> List[str]:
        """Get list of available mutation types."""
        return list(cls._strategies.keys())
