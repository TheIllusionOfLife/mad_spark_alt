"""Language mirroring utilities for LLM prompts.

This module provides utilities for encouraging LLMs to respond in the same
language as the user's input. Since LLMs are non-deterministic, these utilities
increase the probability of language matching but don't guarantee it.
"""

# Language detection thresholds (for test validation only)
JAPANESE_CHAR_RATIO_THRESHOLD = 0.20  # Min ratio of Japanese chars to consider text Japanese


def get_strategy_1_instruction() -> str:
    """Get Strategy 1: Direct instruction for language mirroring.

    Returns:
        Language instruction text with direct, clear directive
    """
    return """Language Instruction: Respond in the same language as the user's input.
- Japanese input (日本語) → Japanese output
- English input → English output
- Spanish input (Español) → Spanish output
- Preserve technical terms, code, and proper nouns in their original form

"""


def get_strategy_2_instruction() -> str:
    """Get Strategy 2: Few-shot examples for language mirroring.

    Returns:
        Language instruction text with concrete examples
    """
    return """Match the user's language in your response.

Examples:
User: "環境にやさしい方法は？" → You respond in Japanese
User: "What are eco-friendly methods?" → You respond in English
User: "¿Métodos ecológicos?" → You respond in Spanish

Now respond to the user's actual question in their language:

"""


def get_combined_instruction() -> str:
    """Get combined Strategy 1 + Strategy 2 instruction.

    Returns:
        Both strategies concatenated
    """
    return get_strategy_1_instruction() + get_strategy_2_instruction()


def detect_language(text: str) -> str:
    """Detect the primary language in text FOR TEST VALIDATION ONLY.

    ⚠️ WARNING: This is a simple heuristic for validating experimental results.
    DO NOT use in production code. The LLM handles language detection itself.

    Limitations:
    - Only supports EN/JA/ES
    - Character-based heuristic, not ML-based
    - Spanish detection may have false positives with Romance languages

    Args:
        text: Text to analyze

    Returns:
        Language code: "en" (English), "ja" (Japanese), "es" (Spanish),
        or "unknown" if unable to determine

    Examples:
        >>> detect_language("Hello world")
        'en'
        >>> detect_language("こんにちは世界")
        'ja'
        >>> detect_language("Hola mundo")
        'es'
    """
    if not text or not text.strip():
        return "unknown"

    text_lower = text.lower()

    # Count characters by script
    japanese_chars = 0
    english_chars = 0
    spanish_indicator_chars = 0
    total_alpha_chars = 0

    for char in text:
        code = ord(char)

        # Japanese character ranges
        # Hiragana: 3040-309F
        # Katakana: 30A0-30FF
        # Kanji: 4E00-9FFF
        if (0x3040 <= code <= 0x309F or  # Hiragana
            0x30A0 <= code <= 0x30FF or  # Katakana
            0x4E00 <= code <= 0x9FFF):   # Kanji
            japanese_chars += 1
            total_alpha_chars += 1

        # ASCII letters (could be English or Spanish)
        elif 'a' <= char.lower() <= 'z':
            english_chars += 1
            total_alpha_chars += 1

        # Spanish-specific characters (accents, ñ)
        if char in 'ñáéíóúüÑÁÉÍÓÚÜ':
            spanish_indicator_chars += 1

    # Check for Spanish-specific punctuation
    has_spanish_punctuation = '¿' in text or '¡' in text

    # Common Spanish words (for basic heuristic)
    spanish_words = {
        'hola', 'mundo', 'cómo', 'qué', 'dónde', 'cuándo',
        'por', 'para', 'de', 'el', 'la', 'los', 'las',
        'buenos', 'días', 'noches', 'gracias', 'español',
        'podemos', 'métodos', 'ecológicos', 'desperdicio',
        'alimentos', 'ciudades', 'reducir'
    }

    words = text_lower.split()
    spanish_word_matches = sum(1 for word in words if word.strip('.,!?;:') in spanish_words)

    if total_alpha_chars == 0:
        return "unknown"

    # Calculate ratios
    japanese_ratio = japanese_chars / total_alpha_chars

    # Decision logic
    # If >20% Japanese characters, it's Japanese
    if japanese_ratio > JAPANESE_CHAR_RATIO_THRESHOLD:
        return "ja"

    # If has Spanish-specific characters or punctuation, it's Spanish
    if spanish_indicator_chars > 0 or has_spanish_punctuation:
        return "es"

    # If has Spanish word matches, likely Spanish
    if spanish_word_matches >= 1:
        return "es"

    # Default to English for ASCII text
    if english_chars > 0:
        return "en"

    return "unknown"


def prepend_language_instruction(system_prompt: str, strategy: str = "combined") -> str:
    """Prepend language instruction to existing system prompt.

    Args:
        system_prompt: Existing system prompt
        strategy: Which strategy to use ("strategy1", "strategy2", "combined")

    Returns:
        System prompt with language instruction prepended

    Raises:
        ValueError: If strategy is not recognized
    """
    strategy_map = {
        "strategy1": get_strategy_1_instruction,
        "strategy2": get_strategy_2_instruction,
        "combined": get_combined_instruction,
    }

    if strategy not in strategy_map:
        raise ValueError(
            f"Unknown strategy: '{strategy}'. "
            f"Valid options: {', '.join(strategy_map.keys())}"
        )

    instruction = strategy_map[strategy]()
    return instruction + system_prompt
