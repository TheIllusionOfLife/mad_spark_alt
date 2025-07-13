"""
Enhanced QADI Orchestrator with Answer Extraction.

This orchestrator extends the Smart QADI system with proper answer extraction
that converts abstract QADI insights into direct, actionable user answers.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .smart_orchestrator import SmartQADIOrchestrator, SmartQADICycleResult
from .smart_registry import SmartAgentRegistry
from .answer_extractor import EnhancedAnswerExtractor, AnswerExtractionResult
from .interfaces import GeneratedIdea, ThinkingMethod

logger = logging.getLogger(__name__)


@dataclass
class EnhancedQADICycleResult(SmartQADICycleResult):
    """Enhanced QADI result with extracted answers."""

    extracted_answers: Optional[AnswerExtractionResult] = None
    answer_extraction_time: float = 0.0


class EnhancedQADIOrchestrator(SmartQADIOrchestrator):
    """
    Enhanced QADI orchestrator that provides direct answers to user questions.

    This orchestrator:
    1. Runs the complete QADI cycle
    2. Analyzes the user's question type
    3. Extracts direct, actionable answers from QADI insights
    4. Provides both raw QADI output and user-friendly answers
    """

    def __init__(
        self, registry: Optional[SmartAgentRegistry] = None, auto_setup: bool = True
    ) -> None:
        super().__init__(registry=registry, auto_setup=auto_setup)
        self.answer_extractor = EnhancedAnswerExtractor()

    async def run_qadi_cycle_with_answers(
        self,
        problem_statement: str,
        context: Optional[str] = None,
        cycle_config: Optional[Dict[str, Any]] = None,
        extract_answers: bool = True,
        max_answers: Optional[int] = None,
    ) -> EnhancedQADICycleResult:
        """
        Run QADI cycle and extract direct answers to user questions.

        Args:
            problem_statement: The user's question or problem
            context: Optional additional context
            cycle_config: Configuration for QADI cycle
            extract_answers: Whether to extract direct answers
            max_answers: Maximum number of answers to extract

        Returns:
            Enhanced result with both QADI insights and direct answers
        """
        start_time = time.time()

        logger.info(f"Running enhanced QADI cycle for: {problem_statement[:100]}...")

        # Run standard QADI cycle
        base_result = await super().run_qadi_cycle(
            problem_statement=problem_statement,
            context=context,
            cycle_config=cycle_config,
        )

        # Convert to enhanced result by copying relevant attributes
        # We need to be explicit about which attributes to copy to avoid
        # issues with mock objects in tests
        enhanced_result = EnhancedQADICycleResult(
            problem_statement=base_result.problem_statement,
            cycle_id=base_result.cycle_id,
            phases=base_result.phases,
            synthesized_ideas=base_result.synthesized_ideas,
            execution_time=base_result.execution_time,
            metadata=base_result.metadata,
            timestamp=base_result.timestamp,
            agent_types=base_result.agent_types,
            llm_cost=base_result.llm_cost,
            setup_time=base_result.setup_time,
            conclusion=base_result.conclusion,
        )

        # Extract direct answers if requested
        if extract_answers and enhanced_result.synthesized_ideas:
            extraction_start = time.time()

            # Group ideas by phase for extraction
            ideas_by_phase = self._group_ideas_by_phase(
                enhanced_result.synthesized_ideas
            )

            # Extract answers
            enhanced_result.extracted_answers = (
                await self.answer_extractor.extract_answers(
                    question=problem_statement,
                    qadi_results=ideas_by_phase,
                    max_answers=max_answers,
                )
            )

            enhanced_result.answer_extraction_time = time.time() - extraction_start

            logger.info(
                f"Extracted {len(enhanced_result.extracted_answers.direct_answers)} "
                f"direct answers in {enhanced_result.answer_extraction_time:.3f}s"
            )

        total_time = time.time() - start_time
        enhanced_result.execution_time = total_time

        return enhanced_result

    def _group_ideas_by_phase(
        self, ideas: List[GeneratedIdea]
    ) -> Dict[str, List[GeneratedIdea]]:
        """Group synthesized ideas by their originating phase."""
        ideas_by_phase: Dict[str, List[GeneratedIdea]] = {}

        for idea in ideas:
            # Handle cases where metadata might be None
            phase = (
                idea.metadata.get("phase", "unknown") if idea.metadata else "unknown"
            )
            if phase not in ideas_by_phase:
                ideas_by_phase[phase] = []
            ideas_by_phase[phase].append(idea)

        return ideas_by_phase

    async def get_direct_answers(
        self, question: str, context: Optional[str] = None, max_answers: int = 5
    ) -> List[str]:
        """
        Get direct answers to a user question (convenience method).

        Args:
            question: User's question
            context: Optional context
            max_answers: Maximum answers to return

        Returns:
            List of direct answer strings
        """
        result = await self.run_qadi_cycle_with_answers(
            problem_statement=question, context=context, max_answers=max_answers
        )

        if result.extracted_answers:
            return [
                answer.content for answer in result.extracted_answers.direct_answers
            ]
        else:
            return ["Unable to extract answers from QADI analysis."]

    def display_enhanced_results(self, result: EnhancedQADICycleResult) -> None:
        """Display both QADI insights and extracted answers."""

        print("🔄 Enhanced QADI Results")
        print("=" * 70)
        print(f"Question: {result.problem_statement}")
        print(f"Total Time: {result.execution_time:.3f}s")
        print(f"LLM Cost: ${result.llm_cost:.4f}")
        print(f"Agent Types: {result.agent_types}")

        # Display extracted answers first (most useful for users)
        if result.extracted_answers:
            print(f"\n✅ DIRECT ANSWERS ({result.extracted_answers.question_type}):")
            print("-" * 50)

            for i, answer in enumerate(result.extracted_answers.direct_answers, 1):
                print(f"{i}. {answer.content}")
                if answer.reasoning:
                    print(f"   💭 Source: {answer.source_phase} phase")

            print(f"\n📊 Extraction Summary: {result.extracted_answers.summary}")

        # Display raw QADI output for reference
        print(f"\n🔬 RAW QADI INSIGHTS:")
        print("-" * 50)

        for phase_name, phase_result in result.phases.items():
            if phase_result and phase_result.generated_ideas:
                agent_type = result.agent_types.get(phase_name, "unknown")
                print(f"\n🔸 {phase_name.upper()} ({agent_type}):")
                for i, idea in enumerate(phase_result.generated_ideas, 1):
                    print(f"  {i}. {idea.content}")


# Global instance - removed to reduce coupling
# Create instances where needed instead of using a global
