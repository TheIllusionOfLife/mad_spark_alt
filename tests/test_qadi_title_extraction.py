#!/usr/bin/env python3
"""
Tests for QADI hypothesis title extraction functionality.
"""

import pytest
from mad_spark_alt.utils.text_cleaning import clean_ansi_codes


# Import the function we're about to create
def extract_hypothesis_title(cleaned_hypothesis: str, index: int) -> str:
    """Extract a meaningful title from hypothesis content."""
    # This function will be implemented in qadi_simple.py
    # For now, we'll define it here to make tests fail
    raise NotImplementedError("extract_hypothesis_title not implemented yet")


class TestHypothesisTitleExtraction:
    """Test title extraction from various hypothesis formats."""
    
    def test_category_based_extraction_japanese(self):
        """Test extraction of category-based titles for Japanese keywords."""
        # Individual/Personal category
        hypothesis = "このアプローチは、個人の能力を最大限に活用することを目指します。"
        assert extract_hypothesis_title(hypothesis, 1) == "Individual/Personal Approach"
        
        # Team/Collaborative category
        hypothesis = "チームワークと協力を通じて、革新的な解決策を生み出します。"
        assert extract_hypothesis_title(hypothesis, 2) == "Team/Collaborative Approach"
        
        # System/Organizational category
        hypothesis = "組織全体のシステムを再構築し、効率を向上させます。"
        assert extract_hypothesis_title(hypothesis, 3) == "System/Organizational Approach"
        
    def test_category_based_extraction_english(self):
        """Test extraction of category-based titles for English keywords."""
        hypothesis = "This approach focuses on personal growth and individual achievement."
        assert extract_hypothesis_title(hypothesis, 1) == "Individual/Personal Approach"
        
        hypothesis = "Through collaborative efforts and team synergy, we can achieve more."
        assert extract_hypothesis_title(hypothesis, 2) == "Team/Collaborative Approach"
        
    def test_new_category_extraction(self):
        """Test extraction of additional categories."""
        # Technical/Technology
        hypothesis = "最新の技術とアルゴリズムを活用して問題を解決します。"
        assert extract_hypothesis_title(hypothesis, 4) == "Technical/Technology Approach"
        
        # Scale/Expansion
        hypothesis = "大規模なスケールアップによって、システムの能力を拡大します。"
        assert extract_hypothesis_title(hypothesis, 5) == "Scale/Expansion Approach"
        
        # Evolution/Development
        hypothesis = "継続的な進化と発達を通じて、より高度な知能を実現します。"
        assert extract_hypothesis_title(hypothesis, 6) == "Evolution/Development Approach"
        
        # Integration/Hybrid
        hypothesis = "複数のアプローチを統合し、ハイブリッドな解決策を提供します。"
        assert extract_hypothesis_title(hypothesis, 7) == "Integration/Hybrid Approach"
        
    def test_sentence_extraction_japanese(self):
        """Test extraction of first sentence for Japanese text."""
        hypothesis = "量子コンピューティングを活用した新しいアプローチです。これにより、従来の計算限界を超えることができます。"
        title = extract_hypothesis_title(hypothesis, 8)
        assert title == "量子コンピューティングを活用した新しいアプローチです。"
        
        # With exclamation mark
        hypothesis = "革命的な発想の転換！私たちは全く新しい視点から問題に取り組みます。"
        title = extract_hypothesis_title(hypothesis, 9)
        assert title == "革命的な発想の転換！"
        
    def test_sentence_extraction_english(self):
        """Test extraction of first sentence for English text."""
        hypothesis = "This is a groundbreaking approach to artificial intelligence. It combines multiple paradigms."
        title = extract_hypothesis_title(hypothesis, 10)
        assert title == "This is a groundbreaking approach to artificial intelligence."
        
    def test_no_clear_sentence_boundary(self):
        """Test extraction when there's no clear sentence boundary."""
        # Japanese without punctuation
        hypothesis = "継続的学習と適応的アルゴリズムを組み合わせた革新的手法により高度な問題解決能力を実現"
        title = extract_hypothesis_title(hypothesis, 11)
        # Should extract up to a particle or truncate at 80 chars
        assert len(title) <= 83  # 80 + "..."
        assert "継続的学習" in title
        
    def test_long_hypothesis_truncation(self):
        """Test proper truncation of long hypotheses."""
        hypothesis = "これは非常に長い仮説の説明で、" + "詳細な技術的説明が含まれています。" * 10
        title = extract_hypothesis_title(hypothesis, 12)
        assert len(title) <= 83  # 80 chars + "..."
        assert title.endswith("...")
        
    def test_mixed_language_content(self):
        """Test extraction from mixed Japanese/English content."""
        hypothesis = "AIとMLを活用したDeep Learningアプローチで、次世代の知能システムを構築します。"
        title = extract_hypothesis_title(hypothesis, 13)
        assert "AI" in title and "ML" in title
        
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
        assert "これは太字のアプローチです。" in title


class TestHypothesisFormatting:
    """Test hypothesis formatting for the analysis section."""
    
    def format_hypothesis_for_answer(hypothesis: str, approach_number: int) -> str:
        """Format hypothesis content for clean display in answer."""
        # This function will be implemented in simple_qadi_orchestrator.py
        raise NotImplementedError("format_hypothesis_for_answer not implemented yet")
    
    def test_numbered_list_formatting(self):
        """Test that inline numbered lists get proper line breaks."""
        hypothesis = "このアプローチでは (1) データ収集 (2) 分析 (3) 実装を行います。"
        formatted = format_hypothesis_for_answer(hypothesis, 1)
        assert "\n(1) データ収集" in formatted
        assert "\n(2) 分析" in formatted
        assert "\n(3) 実装" in formatted
        
    def test_whitespace_normalization(self):
        """Test that excessive whitespace is normalized."""
        hypothesis = "これは    複数の     空白が     含まれる     テキストです。"
        formatted = format_hypothesis_for_answer(hypothesis, 2)
        assert "これは 複数の 空白が 含まれる テキストです。" in formatted
        
    def test_punctuation_spacing(self):
        """Test proper spacing after punctuation."""
        hypothesis = "第一に、計画を立てます。第二に、実行します。第三に、評価します。"
        formatted = format_hypothesis_for_answer(hypothesis, 3)
        # Should maintain proper spacing
        assert "。 " in formatted
        
    def test_ansi_code_removal(self):
        """Test that ANSI codes are removed."""
        hypothesis = "\x1b[1mBold approach\x1b[0m with (1) step one (2) step two"
        formatted = format_hypothesis_for_answer(hypothesis, 4)
        assert "\x1b" not in formatted
        assert "Bold approach" in formatted
        assert "\n(1) step one" in formatted


if __name__ == "__main__":
    pytest.main([__file__, "-v"])