"""
Robust QADI Orchestrator that prevents timeouts and handles all edge cases.
"""

import asyncio
import logging
import time
import uuid
from typing import Any, Dict, List, Optional, Set, Tuple

from .interfaces import (
    GeneratedIdea,
    IdeaGenerationRequest,
    IdeaGenerationResult,
    OutputType,
    ThinkingAgentInterface,
    ThinkingMethod,
)
from .timeout_wrapper import TimeoutManager, with_timeout, TimeoutError
from .robust_json_handler import safe_parse_ideas_array
from .smart_orchestrator import SmartQADIOrchestrator, SmartQADICycleResult
from .smart_registry import SmartAgentRegistry

logger = logging.getLogger(__name__)


class RobustQADIOrchestrator(SmartQADIOrchestrator):
    """
    Robust version of SmartQADIOrchestrator with comprehensive timeout handling.
    """
    
    def __init__(
        self,
        agents: Optional[List[ThinkingAgentInterface]] = None,
        registry: Optional[SmartAgentRegistry] = None,
        default_timeout: float = 300.0,  # 5 minutes total
        phase_timeout: float = 75.0,     # 75 seconds per phase (for 4 phases)
    ):
        super().__init__(agents, registry)
        self.default_timeout = default_timeout
        self.phase_timeout = phase_timeout
        
    async def run_qadi_cycle(
        self,
        problem_statement: str,
        context: Optional[str] = None,
        cycle_config: Optional[Dict[str, Any]] = None,
    ) -> SmartQADICycleResult:
        """
        Run complete QADI cycle with robust timeout handling.
        """
        # Create timeout manager
        timeout_mgr = TimeoutManager(self.default_timeout)
        
        # Generate cycle ID
        cycle_id = str(uuid.uuid4())[:8]
        start_time = time.time()
        
        # Initialize tracking
        phases = {}
        synthesized_ideas = []
        llm_cost = 0.0
        agent_types = {}
        
        logger.info(f"Starting QADI cycle {cycle_id} with {timeout_mgr.total_timeout}s timeout")
        
        # Ensure agents are ready (with timeout)
        try:
            await with_timeout(
                self.ensure_agents_ready(),
                timeout=min(10, timeout_mgr.get_remaining_time()),
                phase="agent_setup"
            )
        except TimeoutError:
            logger.error("Agent setup timed out, using template agents")
            # Force template agents on timeout
            self._force_template_agents()
        
        # Define QADI phases
        qadi_phases = [
            ThinkingMethod.QUESTIONING,
            ThinkingMethod.ABDUCTION,
            ThinkingMethod.DEDUCTION,
            ThinkingMethod.INDUCTION,
        ]
        
        # Run each phase with individual timeout
        for phase_method in qadi_phases:
            phase_name = phase_method.value
            
            # Check if we have time left
            remaining_time = timeout_mgr.get_remaining_time()
            if remaining_time <= 0:
                logger.warning(f"Skipping {phase_name} - total timeout exceeded")
                phases[phase_name] = self._create_timeout_result(phase_method)
                continue
            
            # Calculate phase timeout
            phase_timeout = min(self.phase_timeout, remaining_time)
            
            # Run phase with timeout
            phase_start = time.time()
            try:
                phase_result = await with_timeout(
                    self._run_robust_phase(
                        phase_method,
                        problem_statement,
                        context,
                        cycle_config,
                        previous_results=synthesized_ideas
                    ),
                    timeout=phase_timeout,
                    phase=f"QADI_{phase_name}",
                    fallback=None  # Let the timeout handler deal with this
                )
                
                # Record phase time
                phase_duration = time.time() - phase_start
                timeout_mgr.record_phase_time(phase_name, phase_duration)
                
                # Process result
                if phase_result and phase_result != "TIMEOUT":
                    phases[phase_name] = phase_result[0]
                    agent_types[phase_name] = phase_result[1]
                    synthesized_ideas.extend(phase_result[0].generated_ideas)
                    llm_cost += phase_result[0].generation_metadata.get("llm_cost", 0.0)
                elif phase_result == "TIMEOUT":
                    # Create timeout fallback result
                    fallback_result = self._create_fallback_result(phase_method)
                    phases[phase_name] = fallback_result[0]
                    agent_types[phase_name] = fallback_result[1]
                    synthesized_ideas.extend(fallback_result[0].generated_ideas)
                else:
                    phases[phase_name] = self._create_empty_result(phase_method)
                    agent_types[phase_name] = "template"
                    
            except Exception as e:
                logger.error(f"Error in {phase_name}: {e}")
                phases[phase_name] = self._create_error_result(phase_method, str(e))
                agent_types[phase_name] = "error"
        
        # Create final result
        execution_time = time.time() - start_time
        
        return SmartQADICycleResult(
            problem_statement=problem_statement,
            cycle_id=cycle_id,
            phases=phases,
            synthesized_ideas=synthesized_ideas,
            execution_time=execution_time,
            llm_cost=llm_cost,
            agent_types=agent_types,
            metadata={
                "timeout_summary": timeout_mgr.get_summary(),
                "config": cycle_config or {}
            }
        )
    
    async def _run_robust_phase(
        self,
        method: ThinkingMethod,
        problem_statement: str,
        context: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        previous_results: Optional[List[GeneratedIdea]] = None,
    ) -> Optional[Tuple[IdeaGenerationResult, str]]:
        """Run a single phase with robust error handling."""
        try:
            # Get agent for this method
            agent = self.registry.get_preferred_agent(method)
            if not agent:
                logger.warning(f"No agent found for {method.value}")
                return None
            
            # Create request with previous context
            request = IdeaGenerationRequest(
                problem_statement=problem_statement,
                context=self._build_enhanced_context(context, method, previous_results),
                max_ideas_per_method=config.get("max_ideas_per_method", 3) if config else 3,
                require_reasoning=config.get("require_reasoning", False) if config else False,
                output_type=OutputType.STRUCTURED,
                metadata={"phase": method.value}
            )
            
            # Run with timeout
            result = await with_timeout(
                agent.generate_ideas(request),
                timeout=60,  # 60 seconds per agent call
                phase=f"agent_{method.value}"
            )
            
            # Determine agent type
            agent_type = "LLM" if hasattr(agent, "is_llm_powered") and agent.is_llm_powered else "template"
            
            return (result, agent_type)
            
        except Exception as e:
            logger.error(f"Phase {method.value} failed: {e}")
            return None
    
    def _force_template_agents(self):
        """Force the use of template agents."""
        # Clear any LLM agent preferences
        self.registry._agent_preferences.clear()
        
        # Re-register template agents
        for agent_class in [
            QuestioningAgent,
            AbductionAgent,
            DeductionAgent,
            InductionAgent,
        ]:
            try:
                agent = agent_class()
                self.registry.base_registry.register(agent)
            except Exception as e:
                logger.error(f"Failed to register {agent_class.__name__}: {e}")
    
    def _create_timeout_result(self, method: ThinkingMethod) -> IdeaGenerationResult:
        """Create a result for a timed-out phase."""
        return IdeaGenerationResult(
            agent_name="timeout",
            thinking_method=method,
            generated_ideas=[],
            execution_time=0.0,
            error_message="Phase skipped due to timeout",
            generation_metadata={"timeout": True}
        )
    
    def _create_fallback_result(self, method: ThinkingMethod) -> Tuple[IdeaGenerationResult, str]:
        """Create a fallback result for timeout."""
        result = IdeaGenerationResult(
            agent_name="fallback",
            thinking_method=method,
            generated_ideas=[
                GeneratedIdea(
                    content=f"[Timeout fallback] Generated idea for {method.value}",
                    thinking_method=method,
                    agent_name="fallback",
                    generation_prompt="timeout_fallback",
                    confidence_score=0.1,
                    metadata={"fallback": True}
                )
            ],
            execution_time=0.0,
            generation_metadata={"fallback": True}
        )
        return (result, "fallback")
    
    def _create_empty_result(self, method: ThinkingMethod) -> IdeaGenerationResult:
        """Create an empty result."""
        return IdeaGenerationResult(
            agent_name="empty",
            thinking_method=method,
            generated_ideas=[],
            execution_time=0.0,
            generation_metadata={"empty": True}
        )
    
    def _create_error_result(self, method: ThinkingMethod, error: str) -> IdeaGenerationResult:
        """Create an error result."""
        return IdeaGenerationResult(
            agent_name="error",
            thinking_method=method,
            generated_ideas=[],
            execution_time=0.0,
            error_message=error,
            generation_metadata={"error": True}
        )


# Import at the bottom to avoid circular imports
from ..agents import (
    QuestioningAgent,
    AbductionAgent,
    DeductionAgent,
    InductionAgent,
)