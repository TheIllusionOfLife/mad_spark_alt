"""
Text cleaning utilities.

This module provides functions for cleaning and normalizing text output,
particularly for removing ANSI escape codes and other formatting artifacts.
"""

import re
from typing import Optional


def clean_ansi_codes(text: Optional[str]) -> str:
    """
    Remove all ANSI escape codes from text.
    
    This function handles various ANSI code patterns including:
    - Standard escape sequences with \x1b
    - Orphaned codes without escape character
    - Nested and complex formatting codes
    - Partial or malformed codes
    
    Args:
        text: Text that may contain ANSI codes
        
    Returns:
        Clean text with all ANSI codes removed
    """
    if not text:
        return ""
    
    # Remove standard ANSI escape sequences with \x1b
    text = re.sub(r'\x1b\[[0-9;]*m', '', text)
    
    # Remove orphaned ANSI codes (lost escape character)
    # Pattern: [Nm or [N;Nm where N is 1-3 digits
    text = re.sub(r'\[([0-9]{1,3}(?:;[0-9]{1,3})*)?m', '', text)
    
    # Handle specific patterns from LLM output
    # Use non-greedy matching and handle newlines
    # [1mApproach 1:[0m -> Approach 1:
    text = re.sub(r'\[1m(.*?)\[0m', r'\1', text, flags=re.DOTALL)
    
    # [3mText[0m -> Text (italic)
    text = re.sub(r'\[3m(.*?)\[0m', r'\1', text, flags=re.DOTALL)
    
    # [33mText[0m -> Text (color codes)
    text = re.sub(r'\[([0-9]{1,2})m(.*?)\[0m', r'\2', text, flags=re.DOTALL)
    
    # Handle compound codes like [1;33m
    text = re.sub(r'\[([0-9]{1,2});([0-9]{1,2})m(.*?)\[0m', r'\3', text, flags=re.DOTALL)
    
    # Clean up any remaining orphaned [0m reset codes
    text = re.sub(r'\[0m', '', text)
    
    # Clean up any remaining orphaned start codes at end of lines
    text = re.sub(r'\[[0-9;]*m$', '', text, flags=re.MULTILINE)
    
    # Clean up any remaining orphaned start codes at beginning of lines
    text = re.sub(r'^\[[0-9;]*m', '', text, flags=re.MULTILINE)
    
    return text


def clean_markdown_formatting(text: str) -> str:
    """
    Remove markdown formatting from text.
    
    Args:
        text: Text with markdown formatting
        
    Returns:
        Plain text with formatting removed
    """
    if not text:
        return ""
    
    # Remove **bold** markers
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    
    # Remove *italic* markers
    text = re.sub(r'\*([^*]+)\*', r'\1', text)
    
    # Remove markdown links [text](url)
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    
    # Remove any remaining asterisks
    text = text.replace('*', '')
    
    # Clean up extra whitespace
    text = ' '.join(text.split())
    
    return text.strip()


def clean_all_formatting(text: Optional[str]) -> str:
    """
    Remove all formatting (ANSI codes and markdown) from text.
    
    Args:
        text: Text with various formatting
        
    Returns:
        Plain text with all formatting removed
    """
    if not text:
        return ""
    
    # First remove ANSI codes
    text = clean_ansi_codes(text)
    
    # Then remove markdown
    text = clean_markdown_formatting(text)
    
    return text