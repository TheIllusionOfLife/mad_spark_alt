"""
Tests for evolution checkpointing and recovery functionality.

This module tests the ability to save and restore evolution state,
enabling recovery from failures and resumption of long-running evolutions.
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mad_spark_alt.core import GeneratedIdea, ThinkingMethod
from mad_spark_alt.evolution import (
    EvolutionConfig,
    EvolutionRequest,
    GeneticAlgorithm,
    SelectionStrategy,
)
from mad_spark_alt.evolution.checkpointing import (
    EvolutionCheckpoint,
    EvolutionCheckpointer,
)
from mad_spark_alt.evolution.interfaces import IndividualFitness, PopulationSnapshot


class TestEvolutionCheckpointer:
    """Test suite for EvolutionCheckpointer."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        # Create temporary directory for checkpoints
        self.temp_dir = tempfile.mkdtemp()
        self.checkpointer = EvolutionCheckpointer(checkpoint_dir=self.temp_dir)
        
        # Create test data
        self.test_ideas = [
            GeneratedIdea(
                content=f"Test idea {i}",
                thinking_method=ThinkingMethod.QUESTIONING,
                agent_name="TestAgent",
                generation_prompt="Test prompt",
                metadata={"generation": 1},
            )
            for i in range(5)
        ]
        
        self.test_population = [
            IndividualFitness(
                idea=idea,
                creativity_score=0.5 + i * 0.1,
                diversity_score=0.5 + i * 0.1,
                quality_score=0.5 + i * 0.1,
                overall_fitness=0.5 + i * 0.1,
            )
            for i, idea in enumerate(self.test_ideas)
        ]
        
        self.test_config = EvolutionConfig(
            population_size=5,
            generations=10,
            mutation_rate=0.15,
            crossover_rate=0.75,
            selection_strategy=SelectionStrategy.TOURNAMENT,
        )

    def teardown_method(self) -> None:
        """Clean up test fixtures."""
        # Remove temporary directory
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_checkpoint_save_and_load(self) -> None:
        """Test basic checkpoint save and load functionality."""
        # Create checkpoint
        checkpoint = EvolutionCheckpoint(
            generation=3,
            population=self.test_population,
            config=self.test_config,
            generation_snapshots=[
                PopulationSnapshot.from_population(i, self.test_population)
                for i in range(3)
            ],
            context="Test evolution context",
            random_state={"seed": 42},
        )
        
        # Save checkpoint
        checkpoint_id = self.checkpointer.save_checkpoint(checkpoint)
        assert checkpoint_id is not None
        
        # Verify file exists
        checkpoint_path = Path(self.temp_dir) / f"{checkpoint_id}.json"
        assert checkpoint_path.exists()
        
        # Load checkpoint
        loaded = self.checkpointer.load_checkpoint(checkpoint_id)
        assert loaded is not None
        assert loaded.generation == checkpoint.generation
        assert len(loaded.population) == len(checkpoint.population)
        assert loaded.config.population_size == checkpoint.config.population_size
        assert loaded.context == checkpoint.context

    def test_checkpoint_serialization(self) -> None:
        """Test checkpoint serialization handles all data types correctly."""
        checkpoint = EvolutionCheckpoint(
            generation=5,
            population=self.test_population,
            config=self.test_config,
            generation_snapshots=[],
            context="Complex context with special chars: 'quotes' and \"double quotes\"",
            metadata={
                "start_time": "2025-07-15T10:00:00",
                "total_evaluations": 150,
                "cache_hits": 45,
            },
        )
        
        # Save and load
        checkpoint_id = self.checkpointer.save_checkpoint(checkpoint)
        loaded = self.checkpointer.load_checkpoint(checkpoint_id)
        
        # Verify complex data preserved
        assert loaded.context == checkpoint.context
        assert loaded.metadata["total_evaluations"] == 150
        assert loaded.metadata["cache_hits"] == 45
        
        # Verify population data
        for i, ind in enumerate(loaded.population):
            assert ind.idea.content == self.test_ideas[i].content
            assert ind.overall_fitness == self.test_population[i].overall_fitness

    def test_list_checkpoints(self) -> None:
        """Test listing available checkpoints."""
        # Initially empty
        checkpoints = self.checkpointer.list_checkpoints()
        assert len(checkpoints) == 0
        
        # Save multiple checkpoints
        checkpoint_ids = []
        for i in range(3):
            checkpoint = EvolutionCheckpoint(
                generation=i,
                population=self.test_population,
                config=self.test_config,
                generation_snapshots=[],
            )
            checkpoint_id = self.checkpointer.save_checkpoint(checkpoint)
            checkpoint_ids.append(checkpoint_id)
        
        # List checkpoints
        checkpoints = self.checkpointer.list_checkpoints()
        assert len(checkpoints) == 3
        
        # Verify all saved checkpoints are listed
        listed_ids = [cp["id"] for cp in checkpoints]
        for checkpoint_id in checkpoint_ids:
            assert checkpoint_id in listed_ids

    def test_delete_checkpoint(self) -> None:
        """Test checkpoint deletion."""
        # Save checkpoint
        checkpoint = EvolutionCheckpoint(
            generation=1,
            population=self.test_population,
            config=self.test_config,
            generation_snapshots=[],
        )
        checkpoint_id = self.checkpointer.save_checkpoint(checkpoint)
        
        # Verify exists
        assert self.checkpointer.load_checkpoint(checkpoint_id) is not None
        
        # Delete checkpoint
        success = self.checkpointer.delete_checkpoint(checkpoint_id)
        assert success
        
        # Verify deleted
        assert self.checkpointer.load_checkpoint(checkpoint_id) is None
        
        # Verify not in list
        checkpoints = self.checkpointer.list_checkpoints()
        assert checkpoint_id not in [cp["id"] for cp in checkpoints]

    def test_auto_checkpoint_naming(self) -> None:
        """Test automatic checkpoint ID generation."""
        checkpoint = EvolutionCheckpoint(
            generation=2,
            population=self.test_population,
            config=self.test_config,
            generation_snapshots=[],
        )
        
        # Save without specifying ID
        checkpoint_id = self.checkpointer.save_checkpoint(checkpoint)
        
        # Verify ID format (should include generation and timestamp)
        assert checkpoint_id.startswith("evolution_gen2_")
        assert len(checkpoint_id) > 20  # Includes timestamp

    def test_checkpoint_with_invalid_data(self) -> None:
        """Test handling of invalid checkpoint data."""
        # Create invalid checkpoint file
        invalid_path = Path(self.temp_dir) / "invalid_checkpoint.json"
        with open(invalid_path, "w") as f:
            f.write("invalid json data")
        
        # Attempt to load
        loaded = self.checkpointer.load_checkpoint("invalid_checkpoint")
        assert loaded is None
        
        # Create checkpoint with missing required fields
        incomplete_path = Path(self.temp_dir) / "incomplete_checkpoint.json"
        with open(incomplete_path, "w") as f:
            json.dump({"generation": 1}, f)  # Missing population, config
        
        loaded = self.checkpointer.load_checkpoint("incomplete_checkpoint")
        assert loaded is None

    def test_checkpoint_directory_creation(self) -> None:
        """Test automatic creation of checkpoint directory."""
        # Use non-existent directory
        new_dir = os.path.join(self.temp_dir, "new_checkpoint_dir")
        checkpointer = EvolutionCheckpointer(checkpoint_dir=new_dir)
        
        # Directory should be created on first save
        checkpoint = EvolutionCheckpoint(
            generation=1,
            population=self.test_population,
            config=self.test_config,
            generation_snapshots=[],
        )
        checkpoint_id = checkpointer.save_checkpoint(checkpoint)
        
        # Verify directory was created
        assert os.path.exists(new_dir)
        assert checkpoint_id is not None


class TestGeneticAlgorithmCheckpointing:
    """Test genetic algorithm integration with checkpointing."""

    @pytest.mark.asyncio
    async def test_evolution_with_checkpointing(self) -> None:
        """Test evolution runs with periodic checkpointing."""
        # Create mock fitness evaluator
        mock_evaluator = MagicMock()
        mock_evaluator.evaluate_population = AsyncMock()
        mock_evaluator.calculate_population_diversity = AsyncMock(return_value=0.75)
        
        # Set up evaluator to return incrementing fitness
        async def mock_evaluate(population, config, context=None):
            return [
                IndividualFitness(
                    idea=idea,
                    overall_fitness=0.5 + i * 0.05,
                    creativity_score=0.5,
                    diversity_score=0.5,
                    quality_score=0.5,
                )
                for i, idea in enumerate(population)
            ]
        
        mock_evaluator.evaluate_population.side_effect = mock_evaluate
        
        # Create GA with checkpointing
        with tempfile.TemporaryDirectory() as temp_dir:
            ga = GeneticAlgorithm(
                fitness_evaluator=mock_evaluator,
                checkpoint_dir=temp_dir,
                checkpoint_interval=2,  # Checkpoint every 2 generations
            )
            
            # Create test request
            initial_ideas = [
                GeneratedIdea(
                    content=f"Initial idea {i}",
                    thinking_method=ThinkingMethod.QUESTIONING,
                    agent_name="TestAgent",
                    generation_prompt="Test",
                )
                for i in range(5)
            ]
            
            request = EvolutionRequest(
                initial_population=initial_ideas,
                config=EvolutionConfig(
                    population_size=5,
                    generations=5,
                    mutation_rate=0.1,
                    crossover_rate=0.7,
                ),
            )
            
            # Run evolution
            result = await ga.evolve(request)
            
            # Verify checkpoints were created
            checkpointer = EvolutionCheckpointer(checkpoint_dir=temp_dir)
            checkpoints = checkpointer.list_checkpoints()
            
            # Should have checkpoints at generations 2 and 4
            assert len(checkpoints) >= 2
            
            # Verify checkpoint contents
            for checkpoint_info in checkpoints:
                checkpoint = checkpointer.load_checkpoint(checkpoint_info["id"])
                assert checkpoint is not None
                assert checkpoint.generation in [2, 4]
                assert len(checkpoint.population) == 5

    @pytest.mark.asyncio
    async def test_resume_from_checkpoint(self) -> None:
        """Test resuming evolution from a checkpoint."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create and save a checkpoint at generation 3
            checkpointer = EvolutionCheckpointer(checkpoint_dir=temp_dir)
            
            test_population = [
                IndividualFitness(
                    idea=GeneratedIdea(
                        content=f"Checkpoint idea {i}",
                        thinking_method=ThinkingMethod.ABDUCTION,
                        agent_name="TestAgent",
                        generation_prompt="Test",
                        metadata={"generation": 3},
                    ),
                    overall_fitness=0.6 + i * 0.05,
                )
                for i in range(5)
            ]
            
            checkpoint = EvolutionCheckpoint(
                generation=3,
                population=test_population,
                config=EvolutionConfig(
                    population_size=5,
                    generations=10,  # Total 10 generations
                ),
                generation_snapshots=[],
                context="Resume test context",
            )
            
            checkpoint_id = checkpointer.save_checkpoint(checkpoint)
            
            # Create GA and resume
            mock_evaluator = MagicMock()
            mock_evaluator.evaluate_population = AsyncMock()
            mock_evaluator.calculate_population_diversity = AsyncMock(return_value=0.75)
            
            ga = GeneticAlgorithm(
                fitness_evaluator=mock_evaluator,
                checkpoint_dir=temp_dir,
            )
            
            # Resume evolution
            result = await ga.resume_evolution(checkpoint_id)
            
            assert result.success
            # Should complete remaining generations (4-9)
            assert result.total_generations == 10
            # Should include pre-checkpoint generations
            assert len(result.generation_snapshots) >= 7  # generations 3-9