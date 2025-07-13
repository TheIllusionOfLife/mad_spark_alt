"""
Tests for mutation strategy implementations.
"""

import pytest

from mad_spark_alt.evolution.mutation_strategies import (
    ConceptAdditionStrategy,
    ConceptRemovalStrategy,
    EmphasisChangeStrategy,
    MutationStrategyFactory,
    PhraseReorderingStrategy,
    WordSubstitutionStrategy,
)


class TestWordSubstitutionStrategy:
    """Test word substitution mutation strategy."""

    def test_applies_substitution(self):
        """Test that word substitution modifies content."""
        strategy = WordSubstitutionStrategy()
        content = "improve improve improve improve"  # Increase chances of hitting "improve"
        result = strategy.apply(content)
        
        # Should modify the content in some way
        words = result.split()
        assert len(words) == 4  # Same number of words
        
        # At least one word should be different or have enhanced_ prefix
        original_words = content.split()
        modified = False
        for i, word in enumerate(words):
            if word != original_words[i] or "enhanced_" in word:
                modified = True
                break
        assert modified, f"Expected modification but got: {result}"

    def test_handles_short_content(self):
        """Test that short content is returned unchanged."""
        strategy = WordSubstitutionStrategy()
        content = "too short"
        result = strategy.apply(content)
        
        # Should return unchanged for <= 3 words
        assert result == content

    def test_adds_enhanced_prefix_for_unknown_words(self):
        """Test that unknown words get enhanced_ prefix."""
        strategy = WordSubstitutionStrategy()
        content = "randomword that should be modified"
        result = strategy.apply(content)
        
        # Should either substitute a known word or add enhanced_ prefix
        assert result != content


class TestPhraseReorderingStrategy:
    """Test phrase reordering mutation strategy."""

    def test_reorders_sentences_with_period_space(self):
        """Test reordering sentences separated by '. '."""
        strategy = PhraseReorderingStrategy()
        content = "First sentence. Second sentence. Third sentence"
        result = strategy.apply(content)
        
        # Should have same sentences but potentially different order
        sentences = result.split(". ")
        assert len(sentences) == 3
        assert "First sentence" in result
        assert "Second sentence" in result
        assert "Third sentence" in result

    def test_reorders_sentences_with_period_only(self):
        """Test reordering sentences separated by '.'."""
        strategy = PhraseReorderingStrategy()
        content = "First.Second.Third"
        result = strategy.apply(content)
        
        # Should have same content but potentially reordered
        assert "First" in result
        assert "Second" in result
        assert "Third" in result
        assert result.endswith(".")

    def test_handles_single_sentence(self):
        """Test that single sentence is returned unchanged."""
        strategy = PhraseReorderingStrategy()
        content = "Single sentence without periods"
        result = strategy.apply(content)
        
        assert result == content


class TestConceptAdditionStrategy:
    """Test concept addition mutation strategy."""

    def test_adds_concept(self):
        """Test that a concept is added to the content."""
        strategy = ConceptAdditionStrategy()
        content = "Original content"
        result = strategy.apply(content)
        
        # Should be longer than original
        assert len(result) > len(content)
        assert content in result
        
        # Should contain one of the predefined additions
        additions_found = any(
            addition.strip() in result 
            for addition in ConceptAdditionStrategy.ADDITIONS
        )
        assert additions_found


class TestConceptRemovalStrategy:
    """Test concept removal mutation strategy."""

    def test_removes_sentence(self):
        """Test that a sentence is removed when multiple exist."""
        strategy = ConceptRemovalStrategy()
        content = "First. Second. Third. Fourth"
        result = strategy.apply(content)
        
        # Should have fewer sentences
        original_sentences = content.split(". ")
        result_sentences = result.split(". ")
        assert len(result_sentences) < len(original_sentences)

    def test_handles_few_sentences(self):
        """Test that content with <= 2 sentences is returned unchanged."""
        strategy = ConceptRemovalStrategy()
        content = "First. Second"
        result = strategy.apply(content)
        
        # Should return unchanged for <= 2 sentences
        assert result == content


class TestEmphasisChangeStrategy:
    """Test emphasis change mutation strategy."""

    def test_adds_emphasis_word(self):
        """Test that an emphasis word is added."""
        strategy = EmphasisChangeStrategy()
        content = "Original content here"
        result = strategy.apply(content)
        
        # Should be longer than original
        assert len(result.split()) > len(content.split())
        
        # Should contain one of the emphasis words
        emphasis_found = any(
            word in result 
            for word in EmphasisChangeStrategy.EMPHASIS_WORDS
        )
        assert emphasis_found


class TestMutationStrategyFactory:
    """Test mutation strategy factory."""

    def test_get_strategy_returns_correct_type(self):
        """Test that factory returns correct strategy types."""
        word_strategy = MutationStrategyFactory.get_strategy("word_substitution")
        assert isinstance(word_strategy, WordSubstitutionStrategy)
        
        phrase_strategy = MutationStrategyFactory.get_strategy("phrase_reordering")
        assert isinstance(phrase_strategy, PhraseReorderingStrategy)
        
        addition_strategy = MutationStrategyFactory.get_strategy("concept_addition")
        assert isinstance(addition_strategy, ConceptAdditionStrategy)
        
        removal_strategy = MutationStrategyFactory.get_strategy("concept_removal")
        assert isinstance(removal_strategy, ConceptRemovalStrategy)
        
        emphasis_strategy = MutationStrategyFactory.get_strategy("emphasis_change")
        assert isinstance(emphasis_strategy, EmphasisChangeStrategy)

    def test_get_strategy_raises_for_unknown_type(self):
        """Test that factory raises ValueError for unknown types."""
        with pytest.raises(ValueError, match="Unknown mutation type"):
            MutationStrategyFactory.get_strategy("unknown_type")

    def test_get_available_types(self):
        """Test that factory returns all available strategy types."""
        types = MutationStrategyFactory.get_available_types()
        
        expected_types = [
            "word_substitution",
            "phrase_reordering", 
            "concept_addition",
            "concept_removal",
            "emphasis_change"
        ]
        
        for expected_type in expected_types:
            assert expected_type in types
        
        assert len(types) == len(expected_types)

    def test_strategies_are_singletons(self):
        """Test that factory returns the same instance for the same type."""
        strategy1 = MutationStrategyFactory.get_strategy("word_substitution")
        strategy2 = MutationStrategyFactory.get_strategy("word_substitution")
        
        # Should be the same instance (singleton pattern)
        assert strategy1 is strategy2