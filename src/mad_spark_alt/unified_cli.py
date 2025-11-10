#!/usr/bin/env python3
"""
Unified Command-line interface for Mad Spark Alt.

This unified CLI combines QADI analysis and all evaluation features into a single
command-line interface with Click framework.
"""

import asyncio
import json
import logging
import mimetypes
import os
import re
import sys
import time
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import click
from rich.console import Console
from rich.json import JSON
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from .core import (
    CreativityEvaluator,
    EvaluationLayer,
    EvaluationRequest,
    EvaluationSummary,
    GeneratedIdea,
    ModelOutput,
    OutputType,
    ThinkingMethod,
    registry,
    setup_llm_providers,
)
from .core.json_utils import format_llm_cost
from .core.llm_provider import LLMProvider, get_google_provider, llm_manager
from .core.multimodal import MultimodalInput, MultimodalInputType, MultimodalSourceType
from .core.simple_qadi_orchestrator import SimpleQADIOrchestrator, SimpleQADIResult
from .core.system_constants import CONSTANTS
from .core.terminal_renderer import render_markdown
from .evolution import (
    DiversityMethod,
    EvolutionConfig,
    EvolutionRequest,
    GeneticAlgorithm,
    SelectionStrategy,
)
from .evolution.interfaces import EvolutionResult, IndividualFitness
from .layers.quantitative import DiversityEvaluator, QualityEvaluator
from .utils.export_utils import export_to_json, export_to_markdown
from .utils.text_cleaning import clean_ansi_codes

console = Console()
logger = logging.getLogger(__name__)


# ===== UTILITY FUNCTIONS FROM BOTH FILES =====

def _get_semantic_operator_status() -> str:
    """Get status of semantic operators (ENABLED/DISABLED)."""
    if LLMProvider.GOOGLE in llm_manager.providers:
        return "Semantic operators: ENABLED"
    else:
        return "Semantic operators: DISABLED (traditional operators only)"


def _format_idea_for_display(
    content: str, max_length: Optional[int] = None
) -> str:
    """Format idea content for display with smart truncation.

    Args:
        content: The idea content to format
        max_length: Maximum length before truncation (default: CONSTANTS.TEXT.MAX_IDEA_DISPLAY_LENGTH)

    Returns:
        Formatted content string
    """
    if max_length is None:
        max_length = CONSTANTS.TEXT.MAX_IDEA_DISPLAY_LENGTH

    if len(content) <= max_length:
        return content

    # Find a good truncation point at word boundary
    truncated = content[:max_length]

    # Look for last complete word
    last_space = truncated.rfind(' ')
    if last_space > max_length * CONSTANTS.TEXT.WORD_BOUNDARY_RATIO:
        truncated = truncated[:last_space]

    # Also check for punctuation as good breaking points
    for punct in ['.', ',', ';', ')', ']']:
        punct_pos = truncated.rfind(punct)
        if punct_pos > max_length * CONSTANTS.TEXT.WORD_BOUNDARY_RATIO:
            truncated = truncated[:punct_pos + 1]
            break

    return truncated.strip() + "..."


def _create_evolution_results_table() -> Table:
    """Create a table for evolution results with proper column configuration."""
    table = Table(title="🏆 Top Evolved Ideas")
    table.add_column("Rank", style="cyan", width=4)
    table.add_column("Idea", style="white", width=None)  # No width limit
    table.add_column("Fitness", style="green", width=8)
    table.add_column("Gen", style="yellow", width=5)
    return table


def calculate_evolution_timeout(generations: int, population: int) -> float:
    """
    Calculate adaptive timeout based on evolution complexity.

    Args:
        generations: Number of generations
        population: Population size

    Returns:
        Timeout in seconds (min 120s, max 900s)
    """
    # Use configured timeout per evaluation (25s)
    estimated_time = generations * population * CONSTANTS.TIMEOUTS.CLI_EVOLUTION_TIMEOUT_PER_EVAL
    return min(
        max(CONSTANTS.TIMEOUTS.CLI_BASE_TIMEOUT_SECONDS, estimated_time),
        CONSTANTS.TIMEOUTS.CLI_MAX_TIMEOUT_SECONDS
    )


def load_env_file() -> None:
    """Load environment variables from .env file if it exists."""
    env_path = Path(__file__).parent.parent.parent / ".env"
    if env_path.exists():
        try:
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, value = line.split("=", 1)
                        # Only set if not already in environment
                        if key not in os.environ:
                            os.environ[key] = value.strip('"').strip("'")
        except Exception as e:
            # Log error but don't fail - env vars might be set elsewhere
            logging.warning(f"Failed to load .env file: {e}")


def setup_logging(verbose: bool = False) -> None:
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


def register_default_evaluators() -> None:
    """Register default evaluators."""
    registry.register(DiversityEvaluator)
    registry.register(QualityEvaluator)


# ===== TEXT PROCESSING UTILITIES FROM QADI_SIMPLE =====

def clean_markdown_text(text: str) -> str:
    """Remove markdown formatting while preserving structure."""
    if not text:
        return ""

    # Remove bold and italic markers but keep content
    cleaned = text.replace('**', '').replace('__', '')
    # Be careful with single asterisks - only remove if they're formatting
    cleaned = re.sub(r'(?<!\w)\*([^*]+)\*(?!\w)', r'\1', cleaned)

    # Remove headers but keep content
    cleaned = re.sub(r'^#{1,6}\s+', '', cleaned, flags=re.MULTILINE)

    # Keep link text, remove URL
    cleaned = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', cleaned)

    # Remove inline code markers but keep content
    cleaned = re.sub(r'`([^`]+)`', r'\1', cleaned)

    # Remove code blocks entirely (they're usually not part of main content)
    cleaned = re.sub(r'```[^`]*```', '', cleaned, flags=re.DOTALL)

    # Preserve numbered list structure but remove markers
    cleaned = re.sub(r'^(\d+)\.\s+', r'\1. ', cleaned, flags=re.MULTILINE)

    # Remove bullet markers but keep content
    cleaned = re.sub(r'^[-*+]\s+', '', cleaned, flags=re.MULTILINE)

    # Remove blockquote markers
    cleaned = re.sub(r'^>\s+', '', cleaned, flags=re.MULTILINE)

    # Remove table formatting
    cleaned = re.sub(r'^\|.*\|$', '', cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r'^[-\s]+$', '', cleaned, flags=re.MULTILINE)

    # Preserve single line breaks for readability
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)

    # Don't over-clean - keep some structure
    return cleaned.strip()


def _truncate_title(title: str) -> str:
    """Truncate title to MAX_TITLE_LENGTH if needed."""
    if len(title) <= CONSTANTS.TEXT.MAX_TITLE_LENGTH:
        return title
    return title[:CONSTANTS.TEXT.MAX_TITLE_LENGTH] + "..."


def extract_hypothesis_title(cleaned_hypothesis: str, index: int) -> str:
    """Extract a meaningful title from hypothesis content."""

    # Emergency fallback for empty or very short hypotheses
    if not cleaned_hypothesis or len(cleaned_hypothesis.strip()) < CONSTANTS.TEXT.MIN_HYPOTHESIS_LENGTH:
        return f"Approach {index}"

    # Strategy: Extract the first meaningful sentence or phrase
    # Don't use category-based extraction as it returns generic labels

    # Try Japanese sentence ending patterns first
    jp_sentence_match = re.match(r'^([^。！？]+[。！？])', cleaned_hypothesis)
    if jp_sentence_match:
        title = jp_sentence_match.group(1).strip()
        if len(title) > CONSTANTS.TEXT.MIN_MEANINGFUL_TITLE_LENGTH:
            return _truncate_title(title)

    # Try English sentence patterns
    en_sentence_match = re.match(r'^([^.!?]+[.!?])', cleaned_hypothesis)
    if en_sentence_match:
        title = en_sentence_match.group(1).strip()
        if len(title) > CONSTANTS.TEXT.MIN_MEANINGFUL_TITLE_LENGTH:
            return _truncate_title(title)

    # For numbered lists, try to extract the intro part
    if "：" in cleaned_hypothesis or ":" in cleaned_hypothesis:
        # Split on colon and take the first part
        parts = re.split(r'[:：]', cleaned_hypothesis)
        if parts[0] and len(parts[0].strip()) > CONSTANTS.TEXT.MIN_MEANINGFUL_TITLE_LENGTH:
            title = parts[0].strip()
            return _truncate_title(title)

    # Try splitting by common delimiters
    delimiters = ['。', '.', '、', ',', 'を', 'は', 'が', 'の']
    for delimiter in delimiters:
        if delimiter in cleaned_hypothesis[:100]:
            parts = cleaned_hypothesis.split(delimiter, 1)
            if parts[0] and len(parts[0].strip()) > CONSTANTS.TEXT.MIN_MEANINGFUL_TITLE_LENGTH:
                title = parts[0].strip()
                return _truncate_title(title)

    # Final fallback: First MAX_TITLE_LENGTH chars with word boundary
    if len(cleaned_hypothesis) > CONSTANTS.TEXT.MAX_TITLE_LENGTH:
        # Try to find a word boundary
        truncated = cleaned_hypothesis[:CONSTANTS.TEXT.MAX_TITLE_LENGTH]
        last_space = truncated.rfind(' ')
        last_jp_particle = max(
            truncated.rfind('を') if 'を' in truncated else -1,
            truncated.rfind('は') if 'は' in truncated else -1,
            truncated.rfind('が') if 'が' in truncated else -1,
            truncated.rfind('の') if 'の' in truncated else -1
        )
        boundary = max(last_space, last_jp_particle)

        if boundary > CONSTANTS.TEXT.WORD_BOUNDARY_THRESHOLD:
            return truncated[:boundary].strip() + "..."
        else:
            return truncated[:CONSTANTS.TEXT.MAX_TITLE_LENGTH-3] + "..."

    return cleaned_hypothesis[:CONSTANTS.TEXT.MAX_TITLE_LENGTH].strip()


def truncate_at_sentence_boundary(text: str, max_length: int) -> str:
    """Truncate text at sentence boundary to preserve readability."""
    if len(text) <= max_length:
        return text

    # We'll work with the text up to the max_length to find a boundary
    boundary_search_text = text[:max_length]

    # Find all potential sentence endings in the truncated text.
    matches = list(re.finditer(r'[.!?]["\']?(?=\s|$)', boundary_search_text))

    if matches:
        # The end position of the last match is the best place to cut
        last_break_pos = matches[-1].end()
        return text[:last_break_pos].strip()

    # Fallback: no sentence boundary found, truncate at word boundary
    last_space = boundary_search_text.rfind(' ')
    if last_space > max_length * CONSTANTS.TEXT.WORD_BOUNDARY_RATIO:  # If we found a space reasonably close
        return text[:last_space].strip() + "..."

    return boundary_search_text.strip() + "..."


def format_example_output(example: str, example_num: int) -> str:
    """Format example output with smart truncation."""
    lines = example.split('\n')

    # Extract structured parts - look for various markdown patterns
    context_content = None
    application_content = None
    result_content = None
    level_indicator = None

    for line in lines:
        line_clean = line.strip()

        # Look for level indicators like [Individual/Personal Level]
        if re.match(r'\[.*?Level\]', line_clean):
            level_indicator = line_clean.replace('[', '').replace(']', '')
        # Look for context patterns and extract just the content
        elif re.search(r'(^|[-*]\s*)(Context:|コンテキスト:|背景:)', line_clean):
            # Remove all label variations and markdown
            context_content = re.sub(r'^[-*]+\s*', '', line_clean)
            context_content = re.sub(r'(Context:|コンテキスト:|背景:)\s*', '', context_content)
            context_content = re.sub(r'\*{2,}', '', context_content).strip()
        # Look for application patterns and extract just the content
        elif re.search(r'(^|[-*]\s*)(Application:|アプリケーション:|応用:|適用:)', line_clean):
            # Remove all label variations and markdown
            application_content = re.sub(r'^[-*]+\s*', '', line_clean)
            application_content = re.sub(r'(Application:|アプリケーション:|応用:|適用:)\s*', '', application_content)
            application_content = re.sub(r'\*{2,}', '', application_content).strip()
        # Look for result patterns and extract just the content
        elif re.search(r'(^|[-*]\s*)(Result:|結果:|成果:)', line_clean):
            # Remove all label variations and markdown
            result_content = re.sub(r'^[-*]+\s*', '', line_clean)
            result_content = re.sub(r'(Result:|結果:|成果:)\s*', '', result_content)
            result_content = re.sub(r'\*{2,}', '', result_content).strip()

    output = f"**Example {example_num}**\n"

    # Add level indicator if found
    if level_indicator:
        output += f"*{level_indicator}*\n\n"

    if context_content and application_content:
        # Present clean content without redundant labels
        context_truncated = truncate_at_sentence_boundary(context_content, CONSTANTS.TEXT.MAX_CONTEXT_TRUNCATION_LENGTH)
        application_truncated = truncate_at_sentence_boundary(application_content, CONSTANTS.TEXT.MAX_CONTEXT_TRUNCATION_LENGTH)

        output += f"{context_truncated}\n\n"
        output += f"→ {application_truncated}\n"

        if result_content and len(result_content) < CONSTANTS.TEXT.MAX_RESULT_LENGTH:
            output += f"\n**Result:** {result_content}\n"
    else:
        # Fallback for unstructured examples - clean up more aggressively
        cleaned_example = example
        # Remove all instances of label text in various forms
        label_patterns = [
            r'[-*]*\s*\*?\*?(Context|コンテキスト|背景):?\*?\*?\s*',
            r'[-*]*\s*\*?\*?(Application|アプリケーション|応用|適用):?\*?\*?\s*',
            r'[-*]*\s*\*?\*?(Result|結果|成果):?\*?\*?\s*',
        ]
        for pattern in label_patterns:
            cleaned_example = re.sub(pattern, '', cleaned_example, flags=re.IGNORECASE)

        # Clean up formatting artifacts
        cleaned_example = re.sub(r'\*{2,}', '', cleaned_example)  # Remove multiple asterisks
        cleaned_example = re.sub(r'\[.*?Level\]\*{2,}', '', cleaned_example)  # Remove level indicators with trailing asterisks
        cleaned_example = re.sub(r'\n\s*\*\s*\*\s*', '\n', cleaned_example)  # Fix bullet points
        cleaned_example = re.sub(r'\n{3,}', '\n\n', cleaned_example)  # Fix excessive newlines

        truncated = truncate_at_sentence_boundary(cleaned_example.strip(), CONSTANTS.TEXT.MAX_EXAMPLE_LENGTH)
        output += truncated + "\n"

    return output


def format_evaluation_scores(hypotheses: List[str], scores: List) -> str:
    """Format evaluation scores with approach titles and consistent ordering."""
    output = ""

    for i, (hypothesis, score) in enumerate(zip(hypotheses, scores)):
        # Clean ANSI codes before extracting title
        cleaned_hypothesis = clean_ansi_codes(hypothesis)

        # Remove "Approach X:" prefix to avoid duplication in title extraction
        approach_prefix_pattern = r'^Approach\s+\d+:\s*'
        cleaned_hypothesis = re.sub(approach_prefix_pattern, '', cleaned_hypothesis, flags=re.IGNORECASE)

        # Extract title using the new function
        title = extract_hypothesis_title(cleaned_hypothesis, i + 1)

        # Ensure we always have a meaningful title
        if not title or len(title) < 5:
            # Emergency fallback
            title = f"Approach {i+1}"
            logger.warning(f"Failed to extract meaningful title for approach {i+1}")

        # Format with clean title
        output += f"**Approach {i+1} Scores: {title}**\n"

        # Consistent ordering: Overall first, then other metrics
        output += f"  - **Overall: {score.overall:.2f}**\n"
        output += f"  - Impact: {score.impact:.2f}\n"
        output += f"  - Feasibility: {score.feasibility:.2f}\n"
        output += f"  - Accessibility: {score.accessibility:.2f}\n"
        output += f"  - Sustainability: {score.sustainability:.2f}\n"
        output += f"  - Scalability: {score.scalability:.2f}\n"
        output += "\n"

    return output


def clean_evolution_output(text: str) -> str:
    """Remove parent references from evolution output."""
    result = text

    # Replace possessive forms first to avoid double replacement
    result = result.replace("Parent 1's", "the first approach's")
    result = result.replace("Parent 2's", "the second approach's")

    # Then replace non-possessive forms
    result = result.replace("Parent 1", "the first approach")
    result = result.replace("Parent 2", "the second approach")

    return result


def extract_key_solutions(hypotheses: List[str], action_plan: List[str]) -> List[str]:
    """Extract key solutions from QADI results."""
    solutions = []

    # Extract from hypotheses first
    for h in hypotheses:
        if h and h.strip():
            # Clean ANSI codes first
            h_clean = clean_ansi_codes(h)
            title = clean_markdown_text(h_clean)
            if title and len(title) > CONSTANTS.TEXT.MIN_MEANINGFUL_TITLE_LENGTH:  # Must be meaningful
                solutions.append(title[:150])  # Limit length

    # If we don't have enough solutions, add from action plan
    if len(solutions) < 3:
        for action in action_plan[:3]:
            if len(solutions) < 3 and action and action.strip():
                action_clean = clean_markdown_text(action)
                # Remove numbering from start
                action_clean = re.sub(r'^\d+\.\s*', '', action_clean)

                # Take first sentence if it's meaningful
                first_sentence = action_clean.split('.')[0].strip()
                if len(first_sentence) > CONSTANTS.TEXT.MIN_MEANINGFUL_LENGTH:
                    solutions.append(first_sentence)
                elif len(action_clean) > CONSTANTS.TEXT.MIN_MEANINGFUL_LENGTH:
                    solutions.append(action_clean[:CONSTANTS.TEXT.MAX_ACTION_LENGTH].strip())

    # Return all solutions, not just 3
    return solutions


# ===== MAIN CLI GROUP WITH INVOKE_WITHOUT_COMMAND =====

@click.group(invoke_without_command=True)
@click.pass_context
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging and detailed output')
@click.option('--temperature', '-t', type=click.FloatRange(0.0, 2.0), help='Temperature for hypothesis generation (0.0-2.0, default: 0.8)')
@click.option('--evolve', '-e', is_flag=True, help='Evolve ideas using genetic algorithm after QADI analysis')
@click.option('--generations', '-g', type=int, default=2, help='Number of evolution generations (default: 2, with --evolve)')
@click.option('--population', '-p', type=int, default=5, help='Population size for evolution (default: 5, with --evolve)')
@click.option('--traditional', is_flag=True, help='Use traditional operators instead of semantic operators (with --evolve)')
@click.option('--diversity-method', type=click.Choice(['jaccard', 'semantic'], case_sensitive=False), default='jaccard',
              help='Diversity calculation method: jaccard (fast, word-based) or semantic (slower, embedding-based)')
@click.option('--image', '-i', multiple=True, type=click.Path(exists=True),
              help='Path to image file(s) to include in analysis (PNG, JPEG, GIF, WEBP supported)')
@click.option('--document', '-d', multiple=True, type=click.Path(exists=True),
              help='Path to document file(s) to include in analysis (PDF supported)')
@click.option('--url', '-u', multiple=True, help='URL(s) for context retrieval (max 20)')
@click.option('--output', '-o', type=click.Path(), help='Export results to file (JSON or Markdown)')
@click.option('--format', 'export_format', type=click.Choice(['json', 'md'], case_sensitive=False),
              default='json', help='Export format: json or md (default: json)')
@click.argument('input', required=False)
def main(
    ctx: click.Context,
    verbose: bool,
    temperature: Optional[float],
    evolve: bool,
    generations: int,
    population: int,
    traditional: bool,
    diversity_method: str,
    image: tuple,
    document: tuple,
    url: tuple,
    output: Optional[str],
    export_format: str,
    input: Optional[str],
) -> None:
    """Mad Spark Alt - QADI Analysis & AI Creativity Evaluation System

    Run QADI analysis on any question or problem. Optionally evolve ideas with genetic algorithm.

    Examples:

      # Basic QADI analysis
      msa "How can we reduce food waste?"

      # QADI with evolution
      msa "Improve remote work" --evolve --generations 3

      # With multimodal inputs
      msa "Analyze this design" --image design.png

      # With temperature control
      msa "New product ideas" --temperature 1.5

    Use subcommands for evaluation features:

      msa evaluate "text to evaluate"
      msa list-evaluators
    """
    # Load environment variables
    load_env_file()
    setup_logging(verbose)
    register_default_evaluators()

    # Initialize Google LLM provider if API key is available
    google_key = os.getenv("GOOGLE_API_KEY")

    # CRITICAL FIX: Check if input is actually a subcommand name that Click failed to recognize
    # This happens because Click processes arguments before recognizing subcommands
    if input is not None and ctx.invoked_subcommand is None:
        # Check if command is a Group (has subcommands)
        if isinstance(ctx.command, click.Group):
            # Get all registered subcommand names
            subcommand_names = list(ctx.command.commands.keys())
            # Normalize: check both underscore and hyphenated forms
            input_normalized = input.replace('-', '_')

            # If input matches a subcommand, manually invoke it with proper argument parsing
            for cmd_name in subcommand_names:
                if input == cmd_name or input_normalized == cmd_name or input == cmd_name.replace('_', '-'):
                    # Initialize LLM provider BEFORE invoking subcommand
                    # (some subcommands like evaluate may need LLM providers)
                    if google_key:
                        async def init_llm() -> None:
                            await setup_llm_providers(google_api_key=google_key)

                        try:
                            asyncio.run(init_llm())
                        except RuntimeError:
                            # Already in event loop - skip initialization
                            if verbose:
                                console.print("[yellow]Warning: Cannot initialize LLM providers in running event loop[/yellow]")
                        except Exception as e:
                            if verbose:
                                console.print(f"[yellow]Warning: Failed to initialize LLM providers: {e}[/yellow]")

                    # Found matching subcommand - create new context to parse remaining args
                    subcommand = ctx.command.commands[cmd_name]
                    # Use make_context to parse remaining CLI tokens (ctx.args contains unparsed args)
                    sub_ctx = subcommand.make_context(cmd_name, list(ctx.args), parent=ctx)
                    # Invoke subcommand with parsed parameters
                    with sub_ctx:
                        ctx.invoke(subcommand, **sub_ctx.params)
                    return

    if ctx.invoked_subcommand is None:
        # Default QADI analysis command
        if not input:
            click.echo(ctx.get_help())
            ctx.exit(0)

        # Validate evolution parameters
        if evolve:
            if population < CONSTANTS.EVOLUTION.MIN_POPULATION_SIZE or population > CONSTANTS.EVOLUTION.MAX_POPULATION_SIZE:
                console.print(f"[red]Error: Population size must be between {CONSTANTS.EVOLUTION.MIN_POPULATION_SIZE} and {CONSTANTS.EVOLUTION.MAX_POPULATION_SIZE} (got {population})[/red]")
                console.print(f"\n[yellow]Valid range:[/yellow] {CONSTANTS.EVOLUTION.MIN_POPULATION_SIZE} to {CONSTANTS.EVOLUTION.MAX_POPULATION_SIZE}")
                console.print("Example: msa \"Your question\" --evolve --population 5")
                ctx.exit(1)

            if generations < CONSTANTS.EVOLUTION.MIN_GENERATIONS or generations > CONSTANTS.EVOLUTION.MAX_GENERATIONS:
                console.print(f"[red]Error: Generations must be between {CONSTANTS.EVOLUTION.MIN_GENERATIONS} and {CONSTANTS.EVOLUTION.MAX_GENERATIONS} (got {generations})[/red]")
                console.print(f"\n[yellow]Valid range:[/yellow] {CONSTANTS.EVOLUTION.MIN_GENERATIONS} to {CONSTANTS.EVOLUTION.MAX_GENERATIONS}")
                console.print("Example: msa \"Your question\" --evolve --generations 3")
                ctx.exit(1)

        # Check API key
        if not google_key:
            console.print("[red]Error: GOOGLE_API_KEY not found[/red]")
            console.print("\n[yellow]To fix this:[/yellow]")
            console.print("1. Get a Google API key from: https://makersuite.google.com/app/apikey")
            console.print("2. Set environment variable: export GOOGLE_API_KEY='your-key'")
            console.print("3. Or create .env file: echo 'GOOGLE_API_KEY=your-key' > .env")
            ctx.exit(1)

        # Run QADI analysis
        _run_qadi_sync(
            input,
            temperature=temperature,
            verbose=verbose,
            evolve=evolve,
            generations=generations,
            population=population,
            traditional=traditional,
            diversity_method=diversity_method,
            image_paths=image,
            document_paths=document,
            urls=url,
            output_file=output,
            export_format=export_format
        )
    else:
        # Subcommand will be invoked
        # Initialize LLM provider for subcommands that need it
        if google_key:
            async def init_llm() -> None:
                await setup_llm_providers(google_api_key=google_key)

            try:
                try:
                    loop = asyncio.get_running_loop()
                    if verbose:
                        console.print("[yellow]Warning: Cannot initialize LLM providers in running event loop[/yellow]")
                except RuntimeError:
                    try:
                        asyncio.run(init_llm())
                    except Exception as e:
                        if verbose:
                            console.print(f"[red]Error: LLM provider initialization failed: {e}[/red]")
            except Exception as e:
                if verbose:
                    console.print(f"[red]Unexpected error during LLM initialization: {e}[/red]")
        elif verbose:
            console.print("[yellow]Info: GOOGLE_API_KEY not found, LLM features disabled[/yellow]")


# ===== QADI ANALYSIS IMPLEMENTATION =====

def _run_qadi_sync(
    user_input: str,
    temperature: Optional[float] = None,
    verbose: bool = False,
    evolve: bool = False,
    generations: int = 3,
    population: int = 12,
    traditional: bool = False,
    diversity_method: str = "jaccard",
    image_paths: tuple = (),
    document_paths: tuple = (),
    urls: tuple = (),
    output_file: Optional[str] = None,
    export_format: str = "json"
) -> None:
    """Synchronous wrapper for QADI analysis - handles event loop properly."""
    asyncio.run(_run_qadi_analysis(
        user_input,
        temperature=temperature,
        verbose=verbose,
        evolve=evolve,
        generations=generations,
        population=population,
        traditional=traditional,
        diversity_method=diversity_method,
        image_paths=image_paths,
        document_paths=document_paths,
        urls=urls,
        output_file=output_file,
        export_format=export_format
    ))


async def _run_qadi_analysis(
    user_input: str,
    temperature: Optional[float] = None,
    verbose: bool = False,
    evolve: bool = False,
    generations: int = 3,
    population: int = 12,
    traditional: bool = False,
    diversity_method: str = "jaccard",
    image_paths: tuple = (),
    document_paths: tuple = (),
    urls: tuple = (),
    output_file: Optional[str] = None,
    export_format: str = "json"
) -> None:
    """Run QADI analysis with simplified Phase 1 and optional evolution."""

    print("🧠 Simplified QADI Analysis")
    print("=" * 50 + "\n")

    # Display user input clearly
    print(f"📝 User Input: {user_input}\n")
    print("─" * 50)

    # Process multimodal inputs
    multimodal_inputs = []

    # Process images
    for img_path in image_paths:
        mime_type, _ = mimetypes.guess_type(img_path)
        if not mime_type or not mime_type.startswith("image/"):
            mime_type = "image/png"  # Default to PNG

        img_size = Path(img_path).stat().st_size
        multimodal_inputs.append(
            MultimodalInput(
                input_type=MultimodalInputType.IMAGE,
                source_type=MultimodalSourceType.FILE_PATH,
                data=str(Path(img_path).absolute()),
                mime_type=mime_type,
                file_size=img_size,
            )
        )

    # Process documents
    for doc_path in document_paths:
        mime_type, _ = mimetypes.guess_type(doc_path)

        # Validate document type - currently only PDF supported
        if mime_type != "application/pdf":
            if not doc_path.lower().endswith('.pdf'):
                raise ValueError(
                    f"Unsupported document type for {doc_path}. "
                    f"Only PDF files are currently supported. "
                    f"Detected type: {mime_type or 'unknown'}"
                )
            mime_type = "application/pdf"

        doc_size = Path(doc_path).stat().st_size
        multimodal_inputs.append(
            MultimodalInput(
                input_type=MultimodalInputType.DOCUMENT,
                source_type=MultimodalSourceType.FILE_PATH,
                data=str(Path(doc_path).absolute()),
                mime_type=mime_type,
                file_size=doc_size,
            )
        )

    # Convert URLs tuple to list
    url_list = list(urls) if urls else None

    # Setup LLM providers
    google_key = os.getenv("GOOGLE_API_KEY")
    if not google_key:
        print("❌ Error: GOOGLE_API_KEY not found")
        return

    await setup_llm_providers(google_api_key=google_key)

    # Create orchestrator with optional temperature override and num_hypotheses for evolution
    num_hypotheses = population if evolve else 3
    orchestrator = SimpleQADIOrchestrator(temperature_override=temperature, num_hypotheses=num_hypotheses)

    start_time = time.time()

    try:
        # Run QADI cycle
        result = await orchestrator.run_qadi_cycle(
            user_input,
            multimodal_inputs=multimodal_inputs if multimodal_inputs else None,
            urls=url_list,
        )

        # Extract key solutions for summary
        key_solutions = extract_key_solutions(
            result.hypotheses or [],
            result.action_plan or [],
        )

        # When evolving, just show a brief note
        if evolve and key_solutions:
            print("\n*Generating initial hypotheses for evolution...*")
            logger.info("Synthesized ideas for evolution:")
            for idx, idea in enumerate(result.synthesized_ideas[:3]):
                logger.info(f"Idea {idx+1} length: {len(idea.content)} chars")

        # Show phases in verbose mode
        if verbose:
            print("\n## 🎯 Phase 1: Question Clarification\n")
            render_markdown(f"**Core Question:** {result.core_question}")

            print("\n## 💡 Phase 2: Hypothesis Generation (Abduction)\n")
            for i, hypothesis in enumerate(result.hypotheses):
                # Clean ANSI codes from hypothesis
                hypothesis_clean = clean_ansi_codes(hypothesis)

                # Remove existing "Approach X:" prefix and duplicate numbering
                approach_prefix_pattern = r'^Approach\s+\d+:\s*'
                hypothesis_clean = re.sub(approach_prefix_pattern, '', hypothesis_clean, flags=re.IGNORECASE)

                # Remove duplicate numbering like "1. " at the start
                duplicate_number_pattern = r'^\d+\.\s*'
                hypothesis_clean = re.sub(duplicate_number_pattern, '', hypothesis_clean, flags=re.MULTILINE)

                # Split the hypothesis to separate title from description
                title_match = re.match(r'^([^.]+\.)', hypothesis_clean)
                if title_match:
                    title = title_match.group(1).strip()
                    description = hypothesis_clean[len(title):].strip()
                    render_markdown(f"{i+1}. {title}")
                    if description:
                        render_markdown(description)
                else:
                    render_markdown(f"{i+1}. {hypothesis_clean}")

                # Add extra line break between approaches
                if i < len(result.hypotheses) - 1:
                    print()

            print("\n## 🔍 Phase 3: Logical Analysis (Deduction)\n")

            # Show evaluation scores if verbose
            if result.hypothesis_scores:
                print("### Evaluation Scores:\n")
                formatted_scores = format_evaluation_scores(
                    result.hypotheses, result.hypothesis_scores
                )
                print(formatted_scores)

        # Main output - focused on solutions
        print("\n## 🔍 Analysis: Comparing the Approaches\n")
        render_markdown(result.final_answer)

        if result.action_plan:
            print("\n## 🎯 Your Recommended Path (Final Synthesis)\n")
            for i, action in enumerate(result.action_plan):
                render_markdown(f"{i+1}. {action}")

        # Examples section
        if result.verification_examples and (verbose or len(result.verification_examples) <= 2):
            print("\n## 💡 Real-World Examples\n")

            examples_to_show = result.verification_examples if verbose else result.verification_examples[:2]

            for i, example in enumerate(examples_to_show, 1):
                formatted = format_example_output(example, i)
                print(formatted)

        # Show conclusion only in verbose mode
        if verbose and result.verification_conclusion and result.verification_conclusion.strip():
            print("\n### Conclusion\n")
            cleaned_conclusion = result.verification_conclusion.strip()
            cleaned_conclusion = re.sub(r'^\*{1,2}\s*', '', cleaned_conclusion)
            cleaned_conclusion = re.sub(r'\*{3,}', '**', cleaned_conclusion)
            render_markdown(cleaned_conclusion)

        # Compact summary at the end
        elapsed_time = time.time() - start_time

        # Display multimodal processing stats if any were processed
        if result.total_images_processed > 0 or result.total_pages_processed > 0 or result.total_urls_processed > 0:
            multimodal_stats = []
            if result.total_images_processed > 0:
                multimodal_stats.append(f"{result.total_images_processed} images")
            if result.total_pages_processed > 0:
                multimodal_stats.append(f"{result.total_pages_processed} pages")
            if result.total_urls_processed > 0:
                multimodal_stats.append(f"{result.total_urls_processed} URLs")
            print(f"\n[dim]📎 Processed: {', '.join(multimodal_stats)}[/dim]")

        if not evolve:  # Show summary now if not evolving
            print("\n" + "─" * 50)
            print(f"⏱️  Time: {elapsed_time:.1f}s | 💰 Cost: ${result.total_llm_cost:.4f}")

        # Evolution phase if requested
        evolution_result = None
        if evolve and result.synthesized_ideas:
            evolution_result = await _run_evolution(
                result, user_input, elapsed_time, generations, population,
                traditional, diversity_method, verbose
            )

        # Export results if output file specified
        if output_file:
            try:
                if export_format.lower() == 'md':
                    export_to_markdown(result, output_file, evolution_result=evolution_result)
                else:  # default to json
                    export_to_json(result, output_file, evolution_result=evolution_result)

                print(f"\n💾 Results exported to: {output_file}")
            except Exception as export_error:
                print(f"\n⚠️  Export failed: {export_error}")
                if verbose:
                    import traceback
                    traceback.print_exc()

    except Exception as e:
        print(f"\n❌ Error: {e}")
        if verbose:
            import traceback
            traceback.print_exc()


async def _run_evolution(
    qadi_result: SimpleQADIResult,
    user_input: str,
    qadi_time: float,
    generations: int,
    population: int,
    traditional: bool,
    diversity_method: str,
    verbose: bool,
) -> Optional[EvolutionResult]:
    """Run evolution phase after QADI analysis.

    Returns:
        EvolutionResult if successful, None if failed or timed out
    """
    print("\n" + "═" * 50)
    print(f"🧬 Evolving ideas ({generations} generations, {population} population)...")
    print("─" * 50)

    # Check if we have fewer ideas than requested
    actual_population = min(population, len(qadi_result.synthesized_ideas))
    if actual_population < population:
        print(f"   (Note: Generated {len(qadi_result.synthesized_ideas)} hypotheses, but {population} were requested)")
        print(f"   (Using all {actual_population} available ideas for evolution)")

    # Configure logging to suppress debug messages during evolution
    evolution_logger = logging.getLogger('mad_spark_alt.evolution')
    original_level = evolution_logger.level
    evolution_logger.setLevel(logging.INFO)

    try:
        # Get LLM provider for semantic operators unless --traditional is used
        if traditional:
            llm_provider = None
            print("🧬 Evolution operators: TRADITIONAL (faster but less creative)")
            print("   (Use without --traditional for semantic operators)")
        else:
            llm_provider = get_google_provider()
            print("🧬 Evolution operators: SEMANTIC (LLM-powered for better creativity)")
            print("   (Use --traditional for faster traditional operators)")

        # Display diversity method information
        if diversity_method.lower() == "semantic":
            print("🧬 Diversity calculation: SEMANTIC (embedding-based, more accurate)")
            print("   (Use --diversity-method jaccard for faster word-based calculation)")
        else:
            print("🧬 Diversity calculation: JACCARD (word-based, faster)")
            print("   (Use --diversity-method semantic for more accurate embedding-based calculation)")

        # Create genetic algorithm instance
        ga = GeneticAlgorithm(
            use_cache=True,
            cache_ttl=3600,
            llm_provider=llm_provider
        )

        # Configure evolution
        mutation_rate = 0.5 if actual_population <= 3 else 0.3

        # Configure semantic operators based on --traditional flag
        use_semantic = not traditional
        # Set threshold to 0.0 when disabled (prevents semantic ops from triggering)
        semantic_threshold = 0.9 if use_semantic else 0.0

        config = EvolutionConfig(
            population_size=actual_population,
            generations=generations,
            mutation_rate=mutation_rate,
            crossover_rate=0.75,
            elite_size=min(2, max(1, actual_population // 3)),
            selection_strategy=SelectionStrategy.TOURNAMENT,
            parallel_evaluation=True,
            max_parallel_evaluations=min(8, actual_population),
            use_semantic_operators=use_semantic,
            semantic_operator_threshold=semantic_threshold,
            diversity_method=DiversityMethod.SEMANTIC if diversity_method.lower() == "semantic" else DiversityMethod.JACCARD,
        )

        request = EvolutionRequest(
            initial_population=qadi_result.synthesized_ideas[:config.population_size],
            config=config,
            context=user_input,
        )

        # Calculate adaptive timeout
        evolution_timeout = calculate_evolution_timeout(generations, actual_population)
        print(f"⏱️  Evolution timeout: {evolution_timeout:.0f}s (adjust --generations or --population if needed)")

        # Progress indicator
        async def show_progress(start_time: float, timeout: float) -> None:
            try:
                elapsed = 0.0
                while elapsed < timeout:
                    await asyncio.sleep(10)
                    elapsed = time.time() - start_time
                    remaining = max(0, timeout - elapsed)
                    print(f"   ...evolving ({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)", end='\r')
            except asyncio.CancelledError:
                pass

        # Run evolution with timeout protection
        evolution_start = time.time()
        progress_task = asyncio.create_task(show_progress(evolution_start, evolution_timeout))

        try:
            evolution_result = await asyncio.wait_for(
                ga.evolve(request),
                timeout=evolution_timeout
            )
            evolution_time = time.time() - evolution_start
            progress_task.cancel()
            print()
        except asyncio.TimeoutError:
            progress_task.cancel()
            evolution_time = time.time() - evolution_start
            print()
            print(f"\n❌ Evolution timed out after {evolution_time:.1f}s")
            print("💡 Try reducing --generations or --population for faster results")
            print("   Example: --evolve --generations 2 --population 5")
            return None

        if evolution_result.success:
            print(f"\n✅ Evolution completed in {evolution_time:.1f}s")

            # Display results
            _display_evolution_results(evolution_result, qadi_result, verbose)

            # Final summary with total time and cost
            total_time = qadi_time + evolution_time
            print("\n" + "═" * 50)
            print(f"⏱️  Total time: {total_time:.1f}s | 💰 Total cost: ${qadi_result.total_llm_cost:.4f}")

            return evolution_result
        else:
            print(f"\n❌ Evolution failed: {evolution_result.error_message}")
            return None

    except Exception as e:
        print(f"\n❌ Evolution error: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return None
    finally:
        evolution_logger.setLevel(original_level)


def _display_evolution_results(
    evolution_result: EvolutionResult,
    qadi_result: SimpleQADIResult,
    verbose: bool,
) -> None:
    """Display evolution results with deduplication."""
    print("\n" + "═" * 50)
    print("## 🧬 Evolution Results: Enhanced Solutions\n")
    print("*The initial hypotheses have been evolved and refined:*\n")

    def is_similar(a: str, b: str, threshold: Optional[float] = None) -> bool:
        """Check if two strings are similar above threshold."""
        if threshold is None:
            threshold = CONSTANTS.SIMILARITY.DEDUP_THRESHOLD
        return SequenceMatcher(None, a.lower(), b.lower()).ratio() > threshold

    # Get ALL individuals from all generations + initial QADI hypotheses
    all_individuals = []

    # Add individuals from all evolution generations
    all_individuals.extend(evolution_result.get_all_individuals())

    # Add initial QADI hypotheses as IndividualFitness objects
    if qadi_result.hypotheses and qadi_result.hypothesis_scores:
        for hypothesis, score in zip(qadi_result.hypotheses, qadi_result.hypothesis_scores):
            qadi_individual = IndividualFitness(
                idea=GeneratedIdea(
                    content=hypothesis,
                    thinking_method=ThinkingMethod.ABDUCTION,
                    agent_name="qadi",
                    generation_prompt="initial analysis"
                ),
                impact=score.impact,
                feasibility=score.feasibility,
                accessibility=score.accessibility,
                sustainability=score.sustainability,
                scalability=score.scalability,
                overall_fitness=score.overall
            )
            all_individuals.append(qadi_individual)

    # Collect unique ideas with fuzzy matching
    unique_individuals: List[IndividualFitness] = []
    for ind in sorted(all_individuals, key=lambda x: x.overall_fitness, reverse=True):
        normalized_content = ind.idea.content.strip() if ind.idea.content else ""

        is_duplicate = False
        for existing in unique_individuals:
            existing_content = existing.idea.content.strip() if existing.idea.content else ""
            similarity = SequenceMatcher(None, normalized_content.lower(), existing_content.lower()).ratio()
            if similarity > CONSTANTS.SIMILARITY.DEDUP_THRESHOLD:
                is_duplicate = True
                break

        if not is_duplicate:
            unique_individuals.append(ind)
            if len(unique_individuals) >= CONSTANTS.TEXT.MAX_DISPLAY_IDEAS:
                break

    # Display evolved ideas
    print("## 🏆 High Score Approaches\n")
    print("*Top-rated approaches from comprehensive analysis:*\n")

    displayed_contents: Set[str] = set()
    display_count = 0

    for individual in unique_individuals:
        idea = individual.idea
        content_normalized = idea.content.strip().lower() if idea.content else ""

        # Check for duplicates
        is_duplicate = False
        for displayed in displayed_contents:
            if SequenceMatcher(None, content_normalized, displayed).ratio() > 0.9:
                is_duplicate = True
                break

        if is_duplicate:
            continue

        displayed_contents.add(content_normalized)
        display_count += 1

        # Create score display
        score_dict = individual.get_scores_dict()
        score_display = f"[Overall: {score_dict['overall']:.2f} | Impact: {score_dict['impact']:.2f} | Feasibility: {score_dict['feasibility']:.2f} | Accessibility: {score_dict['accessibility']:.2f} | Sustainability: {score_dict['sustainability']:.2f} | Scalability: {score_dict['scalability']:.2f}]"

        print(f"**{display_count}. High Score Approach** {score_display}")
        cleaned_content = clean_evolution_output(idea.content)
        render_markdown(cleaned_content)
        print()

        if display_count >= CONSTANTS.TEXT.MAX_TOP_IDEAS_DISPLAY:
            break

    # Show metrics
    print("\n## 📊 Development Process & Metrics\n")
    print("**Process:** QADI Analysis → Genetic Evolution → Enhanced Solutions")

    metrics = evolution_result.evolution_metrics
    print(f"• **Improvement:** {metrics.get('fitness_improvement_percent', 0):.1f}% fitness increase")
    print(f"• **Evaluation:** {metrics.get('total_ideas_evaluated', 0)} total ideas tested")

    if verbose:
        cache_stats = metrics.get("cache_stats")
        if cache_stats:
            print(f"\n💾 Cache Performance:")
            print(f"   • Hit rate: {cache_stats.get('hit_rate', 0):.1%}")
            print(f"   • LLM calls saved: {cache_stats.get('hits', 0)}")


# ===== EVALUATION SUBCOMMANDS =====

@main.command()
def list_evaluators() -> None:
    """List all registered evaluators with usage examples."""
    evaluators = registry.list_evaluators()

    if not evaluators:
        console.print("[yellow]No evaluators registered.[/yellow]")
        return

    table = Table(title="Registered Evaluators")
    table.add_column("Name", style="cyan")
    table.add_column("Layer", style="green")
    table.add_column("Supported Types", style="blue")

    for name, info in evaluators.items():
        table.add_row(name, info["layer"], ", ".join(info["supported_output_types"]))

    console.print(table)
    console.print("\n[bold]Usage Examples:[/bold]")
    console.print("  # Use a single evaluator:")
    console.print("  msa evaluate 'text' --evaluators diversity_evaluator")
    console.print("\n  # Use multiple evaluators:")
    console.print("  msa evaluate 'text' --evaluators diversity_evaluator,quality_evaluator")
    console.print("\n  # Use all evaluators (default):")
    console.print("  msa evaluate 'text'")


@main.command()
@click.argument("text", required=False)
@click.option("--file", "-f", type=click.Path(exists=True), help="Read text from file")
@click.option("--model", "-m", default="test-model", help="Model name for the output")
@click.option("--output-type", "-t", type=click.Choice(["text", "code"]), default="text", help="Output type")
@click.option("--evaluators", "-e", help="Comma-separated list of evaluators to use")
@click.option("--layers", "-l", help="Comma-separated list of layers (quantitative,llm_judge,human)")
@click.option("--output", "-o", type=click.Path(), help="Save results to file")
@click.option("--format", "output_format", type=click.Choice(["json", "table"]), default="table", help="Output format")
def evaluate(
    text: Optional[str],
    file: Optional[str],
    model: str,
    output_type: str,
    evaluators: Optional[str],
    layers: Optional[str],
    output: Optional[str],
    output_format: str,
) -> None:
    """Evaluate creativity of AI output."""

    # Get input text
    if file:
        with open(file, "r") as f:
            input_text = f.read()
    elif text:
        input_text = text
    elif not sys.stdin.isatty():
        input_text = sys.stdin.read().strip()
    else:
        console.print("[red]Error: Provide text via argument, --file option, or stdin[/red]")
        sys.exit(1)

    # Parse output type
    try:
        output_type_enum = OutputType(output_type)
    except ValueError:
        console.print(f"[red]Error: Invalid output type '{output_type}'[/red]")
        sys.exit(1)

    # Parse layers
    target_layers = []
    if layers:
        layer_names = [l.strip() for l in layers.split(",")]
        for layer_name in layer_names:
            try:
                target_layers.append(EvaluationLayer(layer_name))
            except ValueError:
                console.print(f"[red]Error: Invalid layer '{layer_name}'[/red]")
                sys.exit(1)

    # Parse and validate evaluator names
    evaluator_names = None
    if evaluators:
        evaluator_names = [e.strip() for e in evaluators.split(",")]
        available_evaluators = registry.list_evaluators()
        invalid_names = [name for name in evaluator_names if name not in available_evaluators]
        if invalid_names:
            console.print(f"[red]Error: Unknown evaluators: {', '.join(invalid_names)}[/red]")
            console.print("[yellow]Available evaluators:[/yellow]")
            for name in sorted(available_evaluators.keys()):
                console.print(f"  - {name}")
            sys.exit(1)

    # Create model output
    model_output = ModelOutput(
        content=input_text,
        output_type=output_type_enum,
        model_name=model,
    )

    # Create evaluation request
    request = EvaluationRequest(
        outputs=[model_output],
        target_layers=target_layers,
        evaluator_names=evaluator_names,
    )

    # Run evaluation
    asyncio.run(_run_evaluation(request, output, output_format))


@main.command()
@click.argument("files", nargs=-1, type=click.Path(exists=True), required=True)
@click.option("--model", "-m", default="test-model", help="Model name for the outputs")
@click.option("--output-type", "-t", type=click.Choice(["text", "code"]), default="text", help="Output type")
@click.option("--output", "-o", type=click.Path(), help="Save results to file")
@click.option("--format", "output_format", type=click.Choice(["json", "table"]), default="table", help="Output format")
def batch_evaluate(
    files: List[str],
    model: str,
    output_type: str,
    output: Optional[str],
    output_format: str,
) -> None:
    """Evaluate creativity of multiple AI outputs from files."""

    # Parse output type
    try:
        output_type_enum = OutputType(output_type)
    except ValueError:
        console.print(f"[red]Error: Invalid output type '{output_type}'[/red]")
        sys.exit(1)

    # Read all files
    model_outputs = []
    for file_path in files:
        with open(file_path, "r") as f:
            content = f.read()

        model_outputs.append(
            ModelOutput(
                content=content,
                output_type=output_type_enum,
                model_name=model,
                metadata={"source_file": file_path},
            )
        )

    # Create evaluation request
    request = EvaluationRequest(outputs=model_outputs)

    # Run evaluation
    asyncio.run(_run_evaluation(request, output, output_format))


@main.command()
@click.argument("prompt")
@click.option("--responses", "-r", multiple=True, required=True, help="Multiple responses to compare")
@click.option("--model", "-m", default="test-model", help="Model name")
@click.option("--output", "-o", type=click.Path(), help="Save results to file")
def compare(
    prompt: str,
    responses: List[str],
    model: str,
    output: Optional[str],
) -> None:
    """Compare creativity of multiple responses to the same prompt."""

    if len(responses) < 2:
        console.print("[red]Error: Need at least 2 responses to compare[/red]")
        sys.exit(1)

    # Create model outputs
    model_outputs = []
    for i, response in enumerate(responses):
        model_outputs.append(
            ModelOutput(
                content=response,
                output_type=OutputType.TEXT,
                model_name=model,
                prompt=prompt,
                metadata={"response_index": i},
            )
        )

    # Create evaluation request
    request = EvaluationRequest(
        outputs=model_outputs, task_context=f"Comparing responses to: {prompt}"
    )

    # Run evaluation with focus on diversity
    asyncio.run(_run_evaluation(request, output, "table", compare_mode=True))


async def _run_evaluation(
    request: EvaluationRequest,
    output_file: Optional[str],
    output_format: str,
    compare_mode: bool = False,
) -> None:
    """Run the evaluation and display results."""

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:

        task = progress.add_task("Evaluating creativity...", total=None)

        # Create evaluator and run evaluation
        evaluator = CreativityEvaluator()
        summary = await evaluator.evaluate(request)

        progress.update(task, completed=True)

    # Display results
    if output_format == "json":
        _display_json_results(summary, output_file)
    else:
        _display_table_results(summary, compare_mode)

        if output_file:
            _save_json_results(summary, output_file)


def _display_json_results(
    summary: EvaluationSummary, output_file: Optional[str]
) -> None:
    """Display results in JSON format."""
    result_data = _summary_to_dict(summary)

    if output_file:
        with open(output_file, "w") as f:
            json.dump(result_data, f, indent=2)
        console.print(f"[green]Results saved to {output_file}[/green]")
    else:
        console.print(JSON.from_data(result_data))


def _display_table_results(
    summary: EvaluationSummary, compare_mode: bool = False
) -> None:
    """Display results in table format."""

    # Summary panel
    overall_score = summary.get_overall_creativity_score()
    overall_score_str = f"{overall_score:.3f}" if overall_score is not None else "N/A"

    summary_text = f"""
    [bold]Evaluation Summary[/bold]

    • Total outputs: {summary.total_outputs}
    • Total evaluators: {summary.total_evaluators}
    • Execution time: {summary.execution_time:.2f}s
    • Overall creativity score: {overall_score_str}
    """

    console.print(Panel(summary_text, title="Results", expand=False))

    # Results by layer
    for layer, results in summary.layer_results.items():
        if not results:
            continue

        table = Table(title=f"{layer.value.title()} Layer Results")
        table.add_column("Output", style="cyan")
        table.add_column("Evaluator", style="green")
        table.add_column("Scores", style="yellow")

        for result in results:
            # Format scores
            score_strs = []
            for metric, score in result.scores.items():
                if isinstance(score, float):
                    score_strs.append(f"{metric}: {score:.3f}")
                else:
                    score_strs.append(f"{metric}: {score}")

            # Get output identifier
            output_id = result.metadata.get("output_index", 0)
            if compare_mode:
                output_id = f"Response {output_id + 1}"
            else:
                output_id = f"Output {output_id + 1}"

            table.add_row(str(output_id), result.evaluator_name, "\n".join(score_strs))

        console.print(table)
        console.print()


def _save_json_results(summary: EvaluationSummary, output_file: str) -> None:
    """Save results to JSON file."""
    result_data = _summary_to_dict(summary)
    with open(output_file, "w") as f:
        json.dump(result_data, f, indent=2)
    console.print(f"[green]Results saved to {output_file}[/green]")


def _summary_to_dict(summary: EvaluationSummary) -> Dict[str, Any]:
    """Convert evaluation summary to dictionary."""
    return {
        "request_id": summary.request_id,
        "total_outputs": summary.total_outputs,
        "total_evaluators": summary.total_evaluators,
        "execution_time": summary.execution_time,
        "overall_creativity_score": summary.get_overall_creativity_score(),
        "aggregate_scores": summary.aggregate_scores,
        "layer_results": {
            layer.value: [
                {
                    "evaluator_name": result.evaluator_name,
                    "scores": result.scores,
                    "explanations": result.explanations,
                    "metadata": result.metadata,
                }
                for result in results
            ]
            for layer, results in summary.layer_results.items()
        },
    }


if __name__ == "__main__":
    main()
