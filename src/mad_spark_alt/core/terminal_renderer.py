"""
Terminal rendering utilities using Rich library.

This module provides a clean interface for rendering markdown and styled content
in the terminal, with graceful fallbacks for different terminal capabilities.
"""

import os
import sys
from typing import Optional, Union
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text
from rich.style import Style


class TerminalRenderer:
    """Enhanced terminal renderer using Rich library."""
    
    def __init__(self, force_color: Optional[bool] = None):
        """
        Initialize the terminal renderer.
        
        Args:
            force_color: Force color output (True/False), None for auto-detection
        """
        self.console = Console(
            force_terminal=force_color,
            color_system="auto" if force_color is None else ("standard" if force_color else None)
        )
        self._fallback_mode = not self.console.color_system
    
    def render_markdown(self, content: str) -> None:
        """
        Render markdown content to the terminal.
        
        Args:
            content: Markdown content to render
        """
        if self._fallback_mode:
            # Simple fallback for terminals without color support
            self.console.print(content)
        else:
            try:
                markdown = Markdown(content)
                self.console.print(markdown)
            except Exception:
                # Fallback to plain text if markdown rendering fails
                self.console.print(content)
    
    def render_panel(self, content: str, title: str = "", 
                    border_style: str = "blue", 
                    title_style: str = "bold") -> None:
        """
        Render content in a styled panel.
        
        Args:
            content: Content to display in panel
            title: Panel title
            border_style: Border color/style
            title_style: Title text style
        """
        if self._fallback_mode:
            # Simple fallback without panels
            if title:
                self.console.print(f"\n{title}")
                self.console.print("-" * len(title))
            self.console.print(content)
        else:
            try:
                panel = Panel(
                    content, 
                    title=title, 
                    border_style=border_style,
                    title_align="left"
                )
                self.console.print(panel)
            except Exception:
                # Fallback to simple output
                if title:
                    self.console.print(f"\n{title}")
                    self.console.print("-" * len(title))
                self.console.print(content)
    
    def render_styled_text(self, text: str, style: str = "") -> None:
        """
        Render styled text.
        
        Args:
            text: Text to render
            style: Rich style string (e.g., "bold green", "red")
        """
        if self._fallback_mode or not style:
            self.console.print(text)
        else:
            try:
                styled_text = Text(text, style=style)
                self.console.print(styled_text)
            except Exception:
                self.console.print(text)
    
    def render_section_header(self, title: str, emoji: str = "") -> None:
        """
        Render a section header with emoji and styling.
        
        Args:
            title: Section title
            emoji: Optional emoji prefix
        """
        header_text = f"{emoji} {title}" if emoji else title
        
        if self._fallback_mode:
            self.console.print(f"\n{header_text}")
            self.console.print("=" * 70)
        else:
            try:
                self.console.print(f"\n[bold cyan]{header_text}[/bold cyan]")
                self.console.print("=" * 70)
            except Exception:
                self.console.print(f"\n{header_text}")
                self.console.print("=" * 70)
    
    def render_phase_indicator(self, phase: str, emoji: str = "", 
                             status: str = "✓", duration: Optional[float] = None) -> None:
        """
        Render a phase completion indicator.
        
        Args:
            phase: Phase name
            emoji: Optional emoji prefix
            status: Status indicator (✓, ✗, etc.)
            duration: Optional duration in seconds
        """
        phase_text = f"{emoji} {phase}" if emoji else phase
        duration_text = f" ({duration:.1f}s)" if duration is not None else ""
        
        if self._fallback_mode:
            self.console.print(f"{phase_text}... {status}{duration_text}")
        else:
            try:
                status_style = "bold green" if status == "✓" else "bold red"
                self.console.print(f"{phase_text}... [bold green]{status}[/bold green]{duration_text}")
            except Exception:
                self.console.print(f"{phase_text}... {status}{duration_text}")
    
    def render_summary_section(self, items: list, title: str = "Summary") -> None:
        """
        Render a summary section with bullet points.
        
        Args:
            items: List of summary items
            title: Section title
        """
        if self._fallback_mode:
            self.console.print(f"\n📊 {title}:")
            for item in items:
                self.console.print(f"  {item}")
        else:
            try:
                self.console.print(f"\n[bold]📊 {title}:[/bold]")
                for item in items:
                    self.console.print(f"  {item}")
            except Exception:
                self.console.print(f"\n📊 {title}:")
                for item in items:
                    self.console.print(f"  {item}")
    
    def print(self, *args, **kwargs) -> None:
        """
        Enhanced print function with Rich support.
        
        Args:
            *args: Arguments to print
            **kwargs: Keyword arguments for console.print
        """
        self.console.print(*args, **kwargs)


# Global renderer instance
_renderer = None

def get_renderer() -> TerminalRenderer:
    """Get the global terminal renderer instance."""
    global _renderer
    if _renderer is None:
        _renderer = TerminalRenderer()
    return _renderer

def render_markdown(content: str) -> None:
    """Convenience function to render markdown."""
    get_renderer().render_markdown(content)

def render_panel(content: str, title: str = "", **kwargs) -> None:
    """Convenience function to render a panel."""
    get_renderer().render_panel(content, title, **kwargs)

def render_styled_text(text: str, style: str = "") -> None:
    """Convenience function to render styled text."""
    get_renderer().render_styled_text(text, style)

def render_section_header(title: str, emoji: str = "") -> None:
    """Convenience function to render section header."""
    get_renderer().render_section_header(title, emoji)

def render_phase_indicator(phase: str, emoji: str = "", 
                         status: str = "✓", duration: Optional[float] = None) -> None:
    """Convenience function to render phase indicator."""
    get_renderer().render_phase_indicator(phase, emoji, status, duration)

def render_summary_section(items: list, title: str = "Summary") -> None:
    """Convenience function to render summary section."""
    get_renderer().render_summary_section(items, title)