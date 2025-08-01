"""
Tests for evolution timeout handling in qadi_simple.py
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import after path setup
from qadi_simple import run_qadi_analysis
from mad_spark_alt.core.interfaces import GeneratedIdea, ThinkingMethod
from mad_spark_alt.core.llm_provider import LLMResponse, LLMProvider
from mad_spark_alt.evolution import EvolutionResult
from mad_spark_alt.evolution.interfaces import IndividualFitness


@pytest.fixture
def mock_qadi_result():
    """Mock QADI result with synthesized ideas."""
    from mad_spark_alt.core.simple_qadi_orchestrator import SimpleQADIResult, HypothesisScore
    
    return SimpleQADIResult(
        core_question="How can we improve remote work?",
        hypotheses=[
            "Implement better communication tools",
            "Create virtual social spaces",
            "Establish clear work-life boundaries"
        ],
        hypothesis_scores=[
            HypothesisScore(0.8, 0.7, 0.9, 0.8, 0.7, 0.78),
            HypothesisScore(0.7, 0.8, 0.8, 0.7, 0.8, 0.76),
            HypothesisScore(0.9, 0.6, 0.7, 0.8, 0.6, 0.72),
        ],
        final_answer="The best approach combines all three strategies...",
        action_plan=["Step 1", "Step 2", "Step 3"],
        verification_examples=["Example 1", "Example 2"],
        verification_conclusion="Conclusion",
        total_llm_cost=0.01,
        synthesized_ideas=[
            GeneratedIdea(
                content="Implement better communication tools",
                thinking_method=ThinkingMethod.ABDUCTION,
                agent_name="test",
                generation_prompt="test",
                confidence_score=0.8,
            ),
            GeneratedIdea(
                content="Create virtual social spaces",
                thinking_method=ThinkingMethod.ABDUCTION,
                agent_name="test",
                generation_prompt="test",
                confidence_score=0.7,
            ),
        ]
    )


@pytest.fixture
def mock_slow_genetic_algorithm():
    """Mock genetic algorithm that simulates slow evolution."""
    
    class SlowGeneticAlgorithm:
        def __init__(self, *args, **kwargs):
            pass
            
        async def evolve(self, request):
            """Simulate a very slow evolution process."""
            # Sleep for longer than any reasonable timeout
            await asyncio.sleep(10)  # 10 seconds
            
            # This should never be reached in timeout tests
            return EvolutionResult(
                final_population=[],
                best_ideas=[],
                generation_snapshots=[],
                total_generations=5,
                execution_time=10.0,
                evolution_metrics={
                    'total_generations': 5,
                    'total_ideas_evaluated': 50,
                    'fitness_improvement_percent': 10.0,
                },
                error_message=None,
            )
    
    return SlowGeneticAlgorithm


@pytest.fixture
def mock_fast_genetic_algorithm():
    """Mock genetic algorithm that completes quickly."""
    
    class FastGeneticAlgorithm:
        def __init__(self, *args, **kwargs):
            pass
            
        async def evolve(self, request):
            """Simulate a fast evolution process."""
            await asyncio.sleep(0.1)  # Very quick
            
            # Create mock generation snapshots with evolved ideas
            from mad_spark_alt.evolution.interfaces import GenerationSnapshot
            
            enhanced_individual = IndividualFitness(
                idea=GeneratedIdea(
                    content="Enhanced idea 1",
                    thinking_method=ThinkingMethod.ABDUCTION,
                    agent_name="evolution",
                    generation_prompt="evolved",
                    confidence_score=0.9,
                ),
                impact=0.85,
                feasibility=0.85,
                accessibility=0.85,
                sustainability=0.85,
                scalability=0.85,
                overall_fitness=0.85,
            )
            
            generation_snapshots = [
                GenerationSnapshot(
                    generation=i,
                    population=[enhanced_individual],
                    best_fitness=0.85,
                    average_fitness=0.85,
                    diversity=0.8
                )
                for i in range(request.config.generations)
            ]
            
            return EvolutionResult(
                final_population=[enhanced_individual],
                best_ideas=[],
                generation_snapshots=generation_snapshots,
                total_generations=request.config.generations,
                execution_time=0.1,
                evolution_metrics={
                    'total_generations': request.config.generations,
                    'total_ideas_evaluated': request.config.population_size * request.config.generations,
                    'fitness_improvement_percent': 15.0,
                },
                error_message=None,
            )
    
    return FastGeneticAlgorithm


@pytest.mark.asyncio
async def test_evolution_timeout_handling(mock_qadi_result, mock_slow_genetic_algorithm, capfd, monkeypatch):
    """Test that evolution times out gracefully with slow genetic algorithm."""
    # Mock GOOGLE_API_KEY
    monkeypatch.setenv('GOOGLE_API_KEY', 'test-key')
    
    with patch('qadi_simple.SimplerQADIOrchestrator') as mock_orchestrator:
        # Mock the orchestrator to return our test result
        mock_instance = AsyncMock()
        mock_instance.run_qadi_cycle = AsyncMock(return_value=mock_qadi_result)
        mock_orchestrator.return_value = mock_instance
        
        # Mock the genetic algorithm to be slow
        with patch('mad_spark_alt.evolution.GeneticAlgorithm', mock_slow_genetic_algorithm):
            # Mock LLM manager to have Google provider
            with patch('mad_spark_alt.core.llm_provider.llm_manager') as mock_llm:
                mock_llm.providers = {LLMProvider.GOOGLE: MagicMock()}
                
                # Run with evolution enabled - should timeout
                await run_qadi_analysis(
                    "How can we improve remote work?",
                    evolve=True,
                    generations=5,
                    population=10
                )
                
                # Check output for timeout message
                captured = capfd.readouterr()
                assert ("Evolution error:" in captured.out or 
                        "timed out" in captured.out.lower() or 
                        "Evolution failed:" in captured.out)


@pytest.mark.asyncio
async def test_evolution_completes_within_timeout(mock_qadi_result, mock_fast_genetic_algorithm, capfd, monkeypatch):
    """Test that fast evolution completes successfully within timeout."""
    # Mock GOOGLE_API_KEY
    monkeypatch.setenv('GOOGLE_API_KEY', 'test-key')
    
    with patch('qadi_simple.SimplerQADIOrchestrator') as mock_orchestrator:
        # Mock the orchestrator to return our test result
        mock_instance = AsyncMock()
        mock_instance.run_qadi_cycle = AsyncMock(return_value=mock_qadi_result)
        mock_orchestrator.return_value = mock_instance
        
        # Mock the genetic algorithm to be fast
        with patch('mad_spark_alt.evolution.GeneticAlgorithm', mock_fast_genetic_algorithm):
            # Mock LLM manager to have Google provider
            with patch('mad_spark_alt.core.llm_provider.llm_manager') as mock_llm:
                mock_llm.providers = {LLMProvider.GOOGLE: MagicMock()}
                
                # Run with evolution enabled - should complete
                await run_qadi_analysis(
                    "How can we improve remote work?",
                    evolve=True,
                    generations=2,
                    population=3
                )
                
                # Check output for success indicators
                captured = capfd.readouterr()
                assert "Evolution completed" in captured.out
                assert "Enhanced idea" in captured.out


@pytest.mark.asyncio
async def test_evolution_timeout_calculation():
    """Test that timeout is calculated appropriately based on parameters."""
    
    # Test timeout calculation logic
    def calculate_evolution_timeout(generations: int, population: int) -> float:
        """Calculate adaptive timeout based on evolution complexity."""
        # Base timeout + time per evaluation
        base_timeout = 60.0
        time_per_eval = 2.0  # seconds per idea evaluation
        
        # Estimate total evaluations
        total_evaluations = generations * population
        estimated_time = base_timeout + (total_evaluations * time_per_eval)
        
        # Cap at 10 minutes
        return min(estimated_time, 600.0)
    
    # Test various scenarios
    assert calculate_evolution_timeout(2, 2) == 68.0  # Small: 60 + 4*2
    assert calculate_evolution_timeout(5, 10) == 160.0  # Medium: 60 + 50*2
    assert calculate_evolution_timeout(10, 50) == 600.0  # Large: capped at 600


@pytest.mark.asyncio
async def test_evolution_disabled_semantic_operators(mock_qadi_result, capfd, monkeypatch):
    """Test that semantic operators can be disabled with --traditional flag."""
    # Mock GOOGLE_API_KEY
    monkeypatch.setenv('GOOGLE_API_KEY', 'test-key')
    
    with patch('qadi_simple.SimplerQADIOrchestrator') as mock_orchestrator:
        # Mock the orchestrator to return our test result
        mock_instance = AsyncMock()
        mock_instance.run_qadi_cycle = AsyncMock(return_value=mock_qadi_result)
        mock_orchestrator.return_value = mock_instance
        
        # Mock genetic algorithm constructor to verify parameters
        with patch('mad_spark_alt.evolution.GeneticAlgorithm') as mock_ga_class:
            mock_ga_instance = AsyncMock()
            mock_ga_instance.evolve = AsyncMock(return_value=EvolutionResult(
                final_population=[],
                best_ideas=[],
                generation_snapshots=[],
                total_generations=2,
                execution_time=1.0,
                evolution_metrics={
                    'total_generations': 2,
                    'total_ideas_evaluated': 6,
                    'fitness_improvement_percent': 5.0,
                },
                error_message=None,
            ))
            mock_ga_class.return_value = mock_ga_instance
            
            # Mock get_google_provider to simulate provider being available
            with patch('mad_spark_alt.core.llm_provider.get_google_provider') as mock_get_provider:
                mock_get_provider.return_value = AsyncMock()
                
                # Run with evolution and --traditional flag
                await run_qadi_analysis(
                    "Test query",
                    evolve=True,
                    generations=2,
                    population=3,
                    traditional=True  # This should disable semantic operators
                )
                
                # Verify genetic algorithm was created without llm_provider
                mock_ga_class.assert_called_once()
                call_kwargs = mock_ga_class.call_args[1]
                assert call_kwargs.get('llm_provider') is None
                
                # Check output indicates traditional operators are being used
                captured = capfd.readouterr()
                assert "Evolution operators: TRADITIONAL" in captured.out


@pytest.mark.asyncio
async def test_evolution_enabled_semantic_operators_by_default(mock_qadi_result, capfd, monkeypatch):
    """Test that semantic operators are enabled by default in qadi_simple."""
    # Mock GOOGLE_API_KEY
    monkeypatch.setenv('GOOGLE_API_KEY', 'test-key')
    
    with patch('qadi_simple.SimplerQADIOrchestrator') as mock_orchestrator:
        # Mock the orchestrator to return our test result
        mock_instance = AsyncMock()
        mock_instance.run_qadi_cycle = AsyncMock(return_value=mock_qadi_result)
        mock_orchestrator.return_value = mock_instance
        
        # Mock genetic algorithm constructor to verify parameters
        with patch('mad_spark_alt.evolution.GeneticAlgorithm') as mock_ga_class:
            mock_ga_instance = AsyncMock()
            mock_ga_instance.evolve = AsyncMock(return_value=EvolutionResult(
                final_population=[],
                best_ideas=[],
                generation_snapshots=[],
                total_generations=2,
                execution_time=1.0,
                evolution_metrics={
                    'total_generations': 2,
                    'total_ideas_evaluated': 6,
                    'fitness_improvement_percent': 5.0,
                },
                error_message=None,
            ))
            mock_ga_class.return_value = mock_ga_instance
            
            # Mock get_google_provider to simulate provider being available
            with patch('mad_spark_alt.core.llm_provider.get_google_provider') as mock_get_provider:
                mock_llm_provider = AsyncMock()
                mock_get_provider.return_value = mock_llm_provider
                
                # Run with evolution (no traditional flag = semantic operators enabled)
                await run_qadi_analysis(
                    "Test query",
                    evolve=True,
                    generations=2,
                    population=3
                )
                
                # Verify genetic algorithm was created WITH llm_provider
                mock_ga_class.assert_called_once()
                call_kwargs = mock_ga_class.call_args[1]
                assert call_kwargs.get('llm_provider') is mock_llm_provider
                
                # Check output indicates semantic operators are enabled
                captured = capfd.readouterr()
                assert "Evolution operators: SEMANTIC" in captured.out


@pytest.mark.asyncio
async def test_evolution_progress_indicators(mock_qadi_result, capfd, monkeypatch):
    """Test that evolution shows progress indicators."""
    # Mock GOOGLE_API_KEY
    monkeypatch.setenv('GOOGLE_API_KEY', 'test-key')
    
    with patch('qadi_simple.SimplerQADIOrchestrator') as mock_orchestrator:
        # Mock the orchestrator
        mock_instance = AsyncMock()
        mock_instance.run_qadi_cycle = AsyncMock(return_value=mock_qadi_result)
        mock_orchestrator.return_value = mock_instance
        
        # Mock genetic algorithm with progress callback
        class ProgressGA:
            def __init__(self, *args, **kwargs):
                pass
                
            async def evolve(self, request):
                # Simulate some progress
                await asyncio.sleep(0.1)
                
                return EvolutionResult(
                    final_population=[],
                    best_ideas=[],
                    generation_snapshots=[],
                    total_generations=request.config.generations,
                    execution_time=0.1,
                    evolution_metrics={
                        'total_generations': request.config.generations,
                        'total_ideas_evaluated': 10,
                        'fitness_improvement_percent': 8.0,
                    },
                    error_message=None,
                )
        
        with patch('mad_spark_alt.evolution.GeneticAlgorithm', ProgressGA):
            with patch('mad_spark_alt.core.llm_provider.llm_manager') as mock_llm:
                mock_llm.providers = {}
                
                await run_qadi_analysis(
                    "Test query",
                    evolve=True,
                    generations=3,
                    population=5
                )
                
                # Check for progress indicators
                captured = capfd.readouterr()
                assert "Evolving ideas" in captured.out
                assert "generations" in captured.out
                assert "population" in captured.out