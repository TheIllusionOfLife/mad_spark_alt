"""Tests for truncation detection in semantic operators."""

from mad_spark_alt.evolution.semantic_operators import is_likely_truncated


class TestTruncationDetection:
    """Test truncation detection functionality."""
    
    def test_complete_sentences_not_truncated(self):
        """Test that complete sentences are not flagged as truncated."""
        text = "This is a complete sentence. It has proper punctuation."
        assert not is_likely_truncated(text)
        
        text = "Multiple sentences here! What do you think? I think it's great."
        assert not is_likely_truncated(text)
        
        text = 'This ends with a quote."'
        assert not is_likely_truncated(text)
    
    def test_mid_sentence_truncation_detected(self):
        """Test that text ending mid-sentence is detected as truncated."""
        text = "This sentence is incomplete and just ends without"
        assert is_likely_truncated(text)
        
        text = "For instance, a character previous"
        assert is_likely_truncated(text)
        
        text = "The implementation involves setting up collaboration platforms and"
        assert is_likely_truncated(text)
    
    def test_ellipsis_truncation_detected(self):
        """Test that ellipsis indicates truncation."""
        text = "This continues..."
        assert is_likely_truncated(text)
        
        text = "And then there was more data but..."
        assert is_likely_truncated(text)
    
    def test_short_ending_words_detected(self):
        """Test that very short ending words suggest truncation."""
        text = "This is a longer sentence that ends with a"
        assert is_likely_truncated(text)
        
        text = "Implementation details include the"
        assert is_likely_truncated(text)
    
    def test_incomplete_json_detected(self):
        """Test that incomplete JSON is detected as truncated."""
        text = '{"ideas": [{"content": "First idea"}, {"content": "Second'
        assert is_likely_truncated(text)
        
        text = '[{"name": "test", "value":'
        assert is_likely_truncated(text)
        
        text = '{"complete": "json", "with": "proper", "closing": "braces"}'
        assert not is_likely_truncated(text)
    
    def test_empty_text_not_truncated(self):
        """Test that empty text is not considered truncated."""
        assert not is_likely_truncated("")
        assert not is_likely_truncated(None)
        assert not is_likely_truncated("   ")
    
    def test_special_punctuation_endings(self):
        """Test various punctuation endings."""
        # These should NOT be truncated
        assert not is_likely_truncated("Is this truncated?")
        assert not is_likely_truncated("No, it's not!")
        assert not is_likely_truncated("It ends with a period.")
        assert not is_likely_truncated('It ends with a quote."')
        assert not is_likely_truncated("It ends with a quote.'")
        
        # These SHOULD be truncated
        assert is_likely_truncated("But this one is")
        assert is_likely_truncated("And this too,")
        assert is_likely_truncated("Also this:")