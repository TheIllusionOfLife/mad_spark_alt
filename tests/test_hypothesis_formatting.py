#!/usr/bin/env python3
"""
Tests for hypothesis formatting in the analysis section.
"""

import pytest
from mad_spark_alt.utils.text_cleaning import clean_ansi_codes


def format_hypothesis_for_answer(hypothesis: str, approach_number: int) -> str:
    """Format hypothesis content for clean display in answer."""
    # This function will be implemented in simple_qadi_orchestrator.py
    # For now, we'll define it here to make tests fail
    raise NotImplementedError("format_hypothesis_for_answer not implemented yet")


class TestHypothesisFormatting:
    """Test hypothesis formatting for the analysis section."""
    
    def test_numbered_list_formatting(self):
        """Test that inline numbered lists get proper line breaks."""
        hypothesis = "このアプローチでは (1) データ収集 (2) 分析 (3) 実装を行います。"
        formatted = format_hypothesis_for_answer(hypothesis, 1)
        assert "\n(1) データ収集" in formatted
        assert "\n(2) 分析" in formatted
        assert "\n(3) 実装" in formatted
        # Should not have leading space before the numbered items
        assert " \n(1)" not in formatted
        
    def test_numbered_list_english(self):
        """Test English numbered lists."""
        hypothesis = "This approach involves (1) planning (2) execution (3) evaluation phases."
        formatted = format_hypothesis_for_answer(hypothesis, 2)
        assert "\n(1) planning" in formatted
        assert "\n(2) execution" in formatted
        assert "\n(3) evaluation phases" in formatted
        
    def test_whitespace_normalization(self):
        """Test that excessive whitespace is normalized."""
        hypothesis = "これは    複数の     空白が     含まれる     テキストです。"
        formatted = format_hypothesis_for_answer(hypothesis, 3)
        assert "これは 複数の 空白が 含まれる テキストです。" in formatted
        # Should not have multiple spaces
        assert "  " not in formatted
        
    def test_punctuation_spacing_japanese(self):
        """Test proper spacing after Japanese punctuation."""
        hypothesis = "第一に、計画を立てます。第二に、実行します。第三に、評価します。"
        formatted = format_hypothesis_for_answer(hypothesis, 4)
        # Should maintain single space after periods
        assert "。 " in formatted
        assert "。  " not in formatted  # No double spaces
        
    def test_punctuation_spacing_english(self):
        """Test proper spacing after English punctuation."""
        hypothesis = "First,we plan.Second,we execute.Third,we evaluate."
        formatted = format_hypothesis_for_answer(hypothesis, 5)
        # Should add space after punctuation
        assert ". " in formatted
        assert ", " in formatted
        
    def test_ansi_code_removal(self):
        """Test that ANSI codes are removed."""
        hypothesis = "\x1b[1mBold approach\x1b[0m with (1) step one (2) step two"
        formatted = format_hypothesis_for_answer(hypothesis, 6)
        assert "\x1b" not in formatted
        assert "[1m" not in formatted
        assert "[0m" not in formatted
        assert "Bold approach" in formatted
        assert "\n(1) step one" in formatted
        
    def test_mixed_content(self):
        """Test mixed Japanese/English with numbered lists."""
        hypothesis = "このHybrid Approachでは(1)AIとMLの統合(2)Real-time処理(3)最適化を実現します。"
        formatted = format_hypothesis_for_answer(hypothesis, 7)
        assert "\n(1)" in formatted or "\n(1)" in formatted
        assert "Hybrid Approach" in formatted
        
    def test_no_numbered_lists(self):
        """Test content without numbered lists."""
        hypothesis = "これは番号付きリストを含まない通常のテキストです。シンプルな説明文です。"
        formatted = format_hypothesis_for_answer(hypothesis, 8)
        # Should not add any line breaks
        assert "\n(1)" not in formatted
        assert "\n" not in formatted or formatted.strip() == formatted[:-1]  # Allow trailing newline
        
    def test_already_formatted_lists(self):
        """Test that already formatted lists are not double-formatted."""
        hypothesis = "このアプローチには以下の特徴があります：\n(1) 高速処理\n(2) 省メモリ\n(3) 高精度"
        formatted = format_hypothesis_for_answer(hypothesis, 9)
        # Should not add extra line breaks
        assert "\n\n(1)" not in formatted
        
    def test_edge_cases(self):
        """Test edge cases."""
        # Empty string
        assert format_hypothesis_for_answer("", 10) == ""
        
        # Very short string
        assert format_hypothesis_for_answer("短い", 11) == "短い"
        
        # Only whitespace
        assert format_hypothesis_for_answer("   ", 12).strip() == ""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])