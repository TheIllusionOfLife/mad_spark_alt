"""
Tests for improved partial result collection strategy with asyncio.shield().
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mad_spark_alt.core import ThinkingMethod
from mad_spark_alt.core.interfaces import GeneratedIdea, IdeaGenerationResult
from mad_spark_alt.core.smart_orchestrator import SmartQADIOrchestrator
from mad_spark_alt.core.smart_registry import SmartAgentRegistry


class TestPartialResultCollection:
    """Test cases for improved partial result collection during timeouts."""

    @pytest.fixture
    def mock_registry(self):
        """Create a mock smart agent registry."""
        registry = MagicMock(spec=SmartAgentRegistry)
        registry.setup_intelligent_agents = AsyncMock(return_value={
            "questioning": "LLM agent ready",
            "abduction": "LLM agent ready",
            "deduction": "LLM agent ready",
            "induction": "LLM agent ready",
        })
        return registry

    @pytest.fixture
    def orchestrator(self, mock_registry):
        """Create orchestrator with short timeouts for testing."""
        return SmartQADIOrchestrator(
            registry=mock_registry,
            phase_timeout=0.5,
            parallel_timeout=1.0,
            conclusion_timeout=0.5,
        )

    @pytest.mark.asyncio
    async def test_current_timeout_behavior_loses_results(self, orchestrator, mock_registry):
        """Baseline test: current implementation loses partial results on timeout."""
        # Create agents with varying completion times
        fast_agent = self._create_timed_agent("FastAgent", 0.1, "Fast result")
        medium_agent = self._create_timed_agent("MediumAgent", 0.8, "Medium result")
        slow_agent = self._create_timed_agent("SlowAgent", 2.0, "Never completes")
        
        def get_agent(method):
            if method == ThinkingMethod.QUESTIONING:
                return fast_agent
            elif method == ThinkingMethod.ABDUCTION:
                return medium_agent
            else:
                return slow_agent
        
        mock_registry.get_preferred_agent.side_effect = get_agent
        
        # Run parallel generation with timeout
        results = await orchestrator.run_parallel_generation(
            "Test problem",
            [
                ThinkingMethod.QUESTIONING,
                ThinkingMethod.ABDUCTION,
                ThinkingMethod.DEDUCTION,
            ],
        )
        
        # Fast agent should complete
        assert ThinkingMethod.QUESTIONING in results
        q_result, _ = results[ThinkingMethod.QUESTIONING]
        assert len(q_result.generated_ideas) == 1
        assert q_result.generated_ideas[0].content == "Fast result"
        
        # Medium agent might complete or timeout
        if ThinkingMethod.ABDUCTION in results:
            a_result, _ = results[ThinkingMethod.ABDUCTION]
            # If it completed, should have result
            if not a_result.error_message:
                assert len(a_result.generated_ideas) == 1
        
        # Slow agent should timeout
        if ThinkingMethod.DEDUCTION in results:
            d_result, _ = results[ThinkingMethod.DEDUCTION]
            assert d_result.error_message is not None

    @pytest.mark.asyncio
    async def test_shielded_tasks_preserve_results(self):
        """Test that using asyncio.shield() preserves partial results."""
        # This test demonstrates the improved approach
        tasks = []
        results = []
        
        async def fast_task():
            await asyncio.sleep(0.1)
            return "Fast result"
        
        async def slow_task():
            await asyncio.sleep(2.0)
            return "Slow result"
        
        # Create shielded tasks
        fast_future = asyncio.create_task(fast_task())
        slow_future = asyncio.create_task(slow_task())
        
        # Shield the slow task to prevent cancellation
        shielded_slow = asyncio.shield(slow_future)
        
        try:
            # Wait with timeout
            results = await asyncio.wait_for(
                asyncio.gather(fast_future, shielded_slow),
                timeout=0.5
            )
        except asyncio.TimeoutError:
            # Collect completed results
            if fast_future.done():
                results.append(fast_future.result())
            
            # The slow task is still running in background (shielded)
            # In real implementation, we'd store reference for later collection
            assert not slow_future.done()  # Still running
        
        assert len(results) == 1
        assert results[0] == "Fast result"

    @pytest.mark.asyncio
    async def test_gather_with_return_exceptions(self):
        """Test using gather with return_exceptions for partial results."""
        async def fast_task():
            await asyncio.sleep(0.1)
            return "Fast result"
        
        async def failing_task():
            await asyncio.sleep(0.2)
            raise Exception("Task failed")
        
        async def slow_task():
            await asyncio.sleep(2.0)
            return "Slow result"
        
        tasks = [
            asyncio.create_task(fast_task()),
            asyncio.create_task(failing_task()),
            asyncio.create_task(slow_task()),
        ]
        
        try:
            # Use return_exceptions to get partial results
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=0.5
            )
        except asyncio.TimeoutError:
            # Collect whatever completed
            results = []
            for task in tasks:
                if task.done():
                    try:
                        results.append(task.result())
                    except asyncio.CancelledError:
                        results.append(asyncio.CancelledError("Task cancelled"))
                    except Exception as e:
                        results.append(e)
                else:
                    results.append(asyncio.TimeoutError("Task timed out"))
                    task.cancel()  # Clean up pending task
        
        # Should have results from fast and failing tasks
        assert len(results) >= 2
        assert results[0] == "Fast result"
        assert isinstance(results[1], Exception)

    @pytest.mark.asyncio
    async def test_wait_with_timeout_pattern(self):
        """Test using asyncio.wait() with timeout for better control."""
        async def create_delayed_result(delay: float, content: str):
            await asyncio.sleep(delay)
            return IdeaGenerationResult(
                agent_name=f"Agent-{delay}",
                thinking_method=ThinkingMethod.QUESTIONING,
                generated_ideas=[GeneratedIdea(
                    content=content,
                    thinking_method=ThinkingMethod.QUESTIONING,
                    agent_name=f"Agent-{delay}",
                    generation_prompt="Test",
                )],
            )
        
        # Create tasks with different delays
        tasks = [
            asyncio.create_task(create_delayed_result(0.1, "Fast")),
            asyncio.create_task(create_delayed_result(0.5, "Medium")),
            asyncio.create_task(create_delayed_result(2.0, "Slow")),
        ]
        
        # Wait with timeout, collecting completed tasks
        done, pending = await asyncio.wait(tasks, timeout=0.8)
        
        # Should have 2 completed, 1 pending
        assert len(done) == 2
        assert len(pending) == 1
        
        # Collect results from completed tasks
        results = []
        for task in done:
            try:
                results.append(task.result())
            except Exception:
                pass
        
        assert len(results) == 2
        assert any(r.generated_ideas[0].content == "Fast" for r in results)
        assert any(r.generated_ideas[0].content == "Medium" for r in results)
        
        # Cancel pending tasks
        for task in pending:
            task.cancel()

    def _create_timed_agent(self, name: str, delay: float, result_content: str):
        """Helper to create an agent that takes specific time to complete."""
        agent = MagicMock()
        agent.is_llm_powered = True
        agent.name = name
        
        async def generate(*args, **kwargs):
            await asyncio.sleep(delay)
            return IdeaGenerationResult(
                agent_name=name,
                thinking_method=ThinkingMethod.QUESTIONING,
                generated_ideas=[GeneratedIdea(
                    content=result_content,
                    thinking_method=ThinkingMethod.QUESTIONING,
                    agent_name=name,
                    generation_prompt="Test",
                )],
            )
        
        agent.generate_ideas = generate
        return agent

    @pytest.mark.asyncio 
    async def test_improved_collect_partial_results(self, orchestrator):
        """Test the improved _collect_partial_results method."""
        # Create mock tasks with different states
        completed_task = MagicMock()  # Use MagicMock for consistency
        completed_task.done.return_value = True
        completed_task.result.return_value = (
            IdeaGenerationResult(
                agent_name="CompletedAgent",
                thinking_method=ThinkingMethod.QUESTIONING,
                generated_ideas=[GeneratedIdea(
                    content="Completed result",
                    thinking_method=ThinkingMethod.QUESTIONING,
                    agent_name="CompletedAgent",
                    generation_prompt="Test",
                )],
            ),
            "LLM"
        )
        
        pending_task = MagicMock()
        pending_task.done.return_value = False
        
        # Test the collection directly
        results = await orchestrator._collect_partial_results(
            [completed_task, pending_task],
            [ThinkingMethod.QUESTIONING, ThinkingMethod.ABDUCTION],
            timeout_per_task=0.001  # Very short timeout
        )
        
        # Should have result for completed task
        assert ThinkingMethod.QUESTIONING in results
        q_result, q_type = results[ThinkingMethod.QUESTIONING]
        assert len(q_result.generated_ideas) == 1
        assert q_type == "LLM"
        
        # Should have timeout for pending task
        assert ThinkingMethod.ABDUCTION in results
        a_result, a_type = results[ThinkingMethod.ABDUCTION]
        assert a_result.error_message is not None
        assert "timed out" in a_result.error_message
        assert a_type == "timeout"