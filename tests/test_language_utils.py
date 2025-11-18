"""Tests for language mirroring utilities."""

import pytest

from mad_spark_alt.core.language_utils import (
    detect_language,
    get_combined_instruction,
    get_strategy_1_instruction,
    get_strategy_2_instruction,
    prepend_language_instruction,
)


class TestLanguageDetection:
    """Unit tests for language detection."""

    def test_detect_english(self):
        """Test detection of English text."""
        assert detect_language("Hello world") == "en"
        assert detect_language("How can we reduce food waste?") == "en"
        assert detect_language("The quick brown fox jumps over the lazy dog") == "en"

    def test_detect_japanese(self):
        """Test detection of Japanese text."""
        assert detect_language("こんにちは世界") == "ja"
        assert detect_language("都市部で食品廃棄物を減らすには？") == "ja"
        assert detect_language("日本語のテキスト") == "ja"

    def test_detect_spanish(self):
        """Test detection of Spanish text."""
        assert detect_language("Hola mundo") == "es"
        assert detect_language("¿Cómo podemos reducir el desperdicio de alimentos?") == "es"
        assert detect_language("Buenos días") == "es"

    def test_detect_mixed_primarily_english(self):
        """Test mixed text that is primarily English."""
        text = "The QADI methodology uses LLM technology"
        assert detect_language(text) == "en"

    def test_detect_mixed_primarily_japanese(self):
        """Test mixed text that is primarily Japanese."""
        text = "QADI手法はLLM技術を使用します"
        assert detect_language(text) == "ja"

    def test_detect_empty_string(self):
        """Test empty string returns unknown."""
        assert detect_language("") == "unknown"

    def test_detect_whitespace_only(self):
        """Test whitespace-only returns unknown."""
        assert detect_language("   \n\t  ") == "unknown"


class TestStrategyInstructions:
    """Unit tests for strategy instruction generation."""

    def test_strategy_1_contains_key_phrases(self):
        """Test Strategy 1 contains essential directive phrases."""
        instruction = get_strategy_1_instruction()

        # Should be explicit about language matching
        assert "same language" in instruction.lower()
        assert "japanese" in instruction.lower()
        assert "english" in instruction.lower()
        assert "spanish" in instruction.lower()

        # Should mention output requirements
        assert "output" in instruction.lower() or "respond" in instruction.lower()

    def test_strategy_2_contains_examples(self):
        """Test Strategy 2 contains few-shot examples."""
        instruction = get_strategy_2_instruction()

        # Should have examples in multiple languages
        assert "環境" in instruction  # Japanese example
        assert "eco" in instruction.lower()  # English example
        assert "métodos" in instruction.lower() or "ecológicos" in instruction.lower()  # Spanish

    def test_combined_strategy_has_both(self):
        """Test combined strategy includes both Strategy 1 and 2."""
        combined = get_combined_instruction()
        strategy_1 = get_strategy_1_instruction()
        strategy_2 = get_strategy_2_instruction()

        # Should contain elements from both strategies
        assert len(combined) > len(strategy_1)
        assert len(combined) > len(strategy_2)

    def test_strategies_return_non_empty_strings(self):
        """Test all strategies return non-empty strings."""
        assert len(get_strategy_1_instruction()) > 0
        assert len(get_strategy_2_instruction()) > 0
        assert len(get_combined_instruction()) > 0


class TestPrependLanguageInstruction:
    """Unit tests for prepending language instructions."""

    def test_prepend_strategy_1(self):
        """Test prepending Strategy 1 instruction."""
        original = "You are a helpful assistant."
        result = prepend_language_instruction(original, strategy="strategy1")

        # Should start with language instruction
        assert result.startswith(get_strategy_1_instruction())
        # Should contain original prompt
        assert original in result
        # Should be longer than original
        assert len(result) > len(original)

    def test_prepend_strategy_2(self):
        """Test prepending Strategy 2 instruction."""
        original = "You are an expert analyst."
        result = prepend_language_instruction(original, strategy="strategy2")

        assert result.startswith(get_strategy_2_instruction())
        assert original in result

    def test_prepend_combined(self):
        """Test prepending combined strategy (default)."""
        original = "Analyze the following problem."
        result = prepend_language_instruction(original)  # Default is combined

        assert get_combined_instruction() in result
        assert original in result

    def test_prepend_invalid_strategy_raises_error(self):
        """Test invalid strategy raises ValueError."""
        with pytest.raises(ValueError, match="Unknown strategy"):
            prepend_language_instruction("Test prompt", strategy="invalid")

    def test_prepend_preserves_original_formatting(self):
        """Test prepending preserves original prompt formatting."""
        original = "System instructions:\n1. Be helpful\n2. Be concise"
        result = prepend_language_instruction(original, strategy="strategy1")

        # Original formatting should be preserved
        assert "System instructions:" in result
        assert "1. Be helpful" in result
        assert "2. Be concise" in result

    def test_prepend_adds_separation(self):
        """Test that instruction and original prompt are properly separated."""
        original = "You are a helpful assistant."
        result = prepend_language_instruction(original, strategy="strategy1")

        # Should have clear separation (newlines)
        assert "\n\n" in result or "\n" in result
