#!/usr/bin/env python3
"""
Tests for QADI hypothesis title extraction functionality.
"""

import pytest
import sys
from pathlib import Path

# Add parent directory to path to import from qadi_simple
sys.path.insert(0, str(Path(__file__).parent.parent))

from qadi_simple import extract_hypothesis_title
from mad_spark_alt.utils.text_cleaning import clean_ansi_codes


class TestHypothesisTitleExtraction:
    """Test title extraction from various hypothesis formats."""
    
    def test_sentence_extraction_with_keywords_japanese(self):
        """Test extraction of actual sentence content for Japanese text with keywords."""
        # Should extract the sentence, not return category label
        hypothesis = "このアプローチは、個人の能力を最大限に活用することを目指します。"
        assert extract_hypothesis_title(hypothesis, 1) == "このアプローチは、個人の能力を最大限に活用することを目指します。"
        
        hypothesis = "チームワークと協力を通じて、革新的な解決策を生み出します。"
        assert extract_hypothesis_title(hypothesis, 2) == "チームワークと協力を通じて、革新的な解決策を生み出します。"
        
        hypothesis = "組織全体のシステムを再構築し、効率を向上させます。"
        assert extract_hypothesis_title(hypothesis, 3) == "組織全体のシステムを再構築し、効率を向上させます。"
        
    def test_sentence_extraction_with_keywords_english(self):
        """Test extraction of actual sentence content for English text with keywords."""
        hypothesis = "This approach focuses on personal growth and individual achievement."
        assert extract_hypothesis_title(hypothesis, 1) == "This approach focuses on personal growth and individual achievement."
        
        hypothesis = "Through collaborative efforts and team synergy, we can achieve more."
        assert extract_hypothesis_title(hypothesis, 2) == "Through collaborative efforts and team synergy, we can achieve more."
        
    def test_sentence_extraction_various_content(self):
        """Test extraction of sentences with various content types."""
        # Technical/Technology content
        hypothesis = "最新の技術とアルゴリズムを活用して問題を解決します。"
        assert extract_hypothesis_title(hypothesis, 4) == "最新の技術とアルゴリズムを活用して問題を解決します。"
        
        # Scale/Expansion content
        hypothesis = "大規模なスケールアップによって、システムの能力を拡大します。"
        assert extract_hypothesis_title(hypothesis, 5) == "大規模なスケールアップによって、システムの能力を拡大します。"
        
        # Evolution/Development content
        hypothesis = "継続的な進化と発達を通じて、より高度な知能を実現します。"
        assert extract_hypothesis_title(hypothesis, 6) == "継続的な進化と発達を通じて、より高度な知能を実現します。"
        
        # Integration/Hybrid content
        hypothesis = "複数のアプローチを統合し、ハイブリッドな解決策を提供します。"
        assert extract_hypothesis_title(hypothesis, 7) == "複数のアプローチを統合し、ハイブリッドな解決策を提供します。"
        
    def test_sentence_extraction_japanese(self):
        """Test extraction of first sentence for Japanese text."""
        hypothesis = "量子コンピューティングを活用した新しいアプローチです。これにより、従来の計算限界を超えることができます。"
        title = extract_hypothesis_title(hypothesis, 8)
        assert title == "量子コンピューティングを活用した新しいアプローチです。"
        
        # With exclamation mark - the function extracts up to a delimiter, not sentence boundary
        hypothesis = "革命的な発想の転換！私たちは全く新しい視点から問題に取り組みます。"
        title = extract_hypothesis_title(hypothesis, 9)
        # Should extract up to delimiter 'を'
        assert "革命的な発想" in title
        
    def test_sentence_extraction_english(self):
        """Test extraction of first sentence for English text."""
        hypothesis = "This is a groundbreaking approach to artificial intelligence. It combines multiple paradigms."
        title = extract_hypothesis_title(hypothesis, 10)
        assert title == "This is a groundbreaking approach to artificial intelligence."
        
    def test_no_clear_sentence_boundary(self):
        """Test extraction when there's no clear sentence boundary."""
        # Japanese without punctuation - should extract using delimiter
        hypothesis = "継続的学習と適応的アルゴリズムを組み合わせた革新的手法により高度な問題解決能力を実現"
        title = extract_hypothesis_title(hypothesis, 11)
        # Should extract part before delimiter 'を'
        assert "継続的学習" in title or title == hypothesis[:80]
        
    def test_long_hypothesis_truncation(self):
        """Test proper truncation of long hypotheses."""
        hypothesis = "これは非常に長い仮説の説明で、" + "詳細な技術的説明が含まれています。" * 10
        title = extract_hypothesis_title(hypothesis, 12)
        # For sentences with period, it should extract the first sentence
        assert title == "これは非常に長い仮説の説明で、詳細な技術的説明が含まれています。"
        
    def test_mixed_language_content(self):
        """Test extraction from mixed Japanese/English content."""
        hypothesis = "AIとMLを活用したDeep Learningアプローチで、次世代の知能システムを構築します。"
        title = extract_hypothesis_title(hypothesis, 13)
        # Should extract the full sentence with period
        assert title == "AIとMLを活用したDeep Learningアプローチで、次世代の知能システムを構築します。"
        
    def test_numbered_list_in_hypothesis(self):
        """Test extraction when hypothesis contains numbered lists."""
        hypothesis = "このアプローチには以下の特徴があります：(1) 高速処理 (2) 省メモリ (3) 高精度"
        title = extract_hypothesis_title(hypothesis, 14)
        # Should extract the intro sentence
        assert "このアプローチには以下の特徴があります" in title
        
    def test_empty_or_short_hypothesis(self):
        """Test handling of empty or very short hypotheses."""
        # Empty
        assert extract_hypothesis_title("", 15) == "Approach 15"
        
        # Too short
        assert extract_hypothesis_title("短い", 16) == "Approach 16"
        
    def test_ansi_code_handling(self):
        """Test that ANSI codes are properly handled."""
        # This would be handled by clean_ansi_codes before calling extract_hypothesis_title
        hypothesis = "[1mこれは太字のアプローチです。[0m革新的な解決策を提供します。"
        cleaned = clean_ansi_codes(hypothesis)
        title = extract_hypothesis_title(cleaned, 17)
        assert "[1m" not in title and "[0m" not in title
        # Should extract the first part (splits on 。)
        assert title == "これは太字のアプローチです。"




if __name__ == "__main__":
    pytest.main([__file__, "-v"])