"""
Checkpointing and recovery for genetic evolution.

This module provides functionality to save and restore evolution state,
enabling recovery from failures and resumption of long-running evolutions.
"""

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from mad_spark_alt.core.interfaces import GeneratedIdea, ThinkingMethod
from mad_spark_alt.evolution.interfaces import (
    EvolutionConfig,
    IndividualFitness,
    PopulationSnapshot,
    SelectionStrategy,
)

logger = logging.getLogger(__name__)


@dataclass
class EvolutionCheckpoint:
    """
    Represents a saved state of evolution at a specific generation.

    This checkpoint contains all necessary information to resume
    evolution from where it left off.
    """

    generation: int
    population: List[IndividualFitness]
    config: EvolutionConfig
    generation_snapshots: List[PopulationSnapshot]
    context: Optional[str] = None
    random_state: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    checkpoint_time: Optional[datetime] = None

    def __post_init__(self) -> None:
        """Set checkpoint time if not provided."""
        if self.checkpoint_time is None:
            self.checkpoint_time = datetime.now()


class EvolutionCheckpointer:
    """
    Manages checkpointing for genetic evolution processes.

    This class handles saving and loading evolution state to/from disk,
    enabling recovery and resumption of evolution processes.
    """

    def __init__(
        self, 
        checkpoint_dir: str = ".evolution_checkpoints",
        max_checkpoints: int = 50,
        max_age_days: int = 30
    ):
        """
        Initialize checkpointer.

        Args:
            checkpoint_dir: Directory to store checkpoint files
            max_checkpoints: Maximum number of checkpoints to keep (oldest deleted first)
            max_age_days: Maximum age of checkpoints in days (older deleted)
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.max_checkpoints = max_checkpoints
        self.max_age_days = max_age_days
        self._ensure_checkpoint_dir()

    def _ensure_checkpoint_dir(self) -> None:
        """Ensure checkpoint directory exists."""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(
        self,
        checkpoint: EvolutionCheckpoint,
        checkpoint_id: Optional[str] = None,
    ) -> str:
        """
        Save an evolution checkpoint to disk.

        Args:
            checkpoint: The checkpoint to save
            checkpoint_id: Optional custom ID (auto-generated if None)

        Returns:
            The checkpoint ID
        """
        # Generate checkpoint ID if not provided
        if checkpoint_id is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_id = f"evolution_gen{checkpoint.generation}_{timestamp}"

        # Prepare checkpoint data
        checkpoint_data = {
            "generation": checkpoint.generation,
            "population": self._serialize_population(checkpoint.population),
            "config": self._serialize_config(checkpoint.config),
            "generation_snapshots": self._serialize_snapshots(
                checkpoint.generation_snapshots
            ),
            "context": checkpoint.context,
            "random_state": checkpoint.random_state,
            "metadata": checkpoint.metadata,
            "checkpoint_time": (
                checkpoint.checkpoint_time.isoformat()
                if checkpoint.checkpoint_time
                else None
            ),
        }

        # Validate checkpoint_id to prevent path traversal
        if not checkpoint_id or not isinstance(checkpoint_id, str):
            raise ValueError("checkpoint_id must be a non-empty string")
        
        # Remove any path separators and potentially dangerous characters
        safe_checkpoint_id = "".join(
            c for c in checkpoint_id if c.isalnum() or c in ".-_"
        )
        if not safe_checkpoint_id:
            raise ValueError("checkpoint_id contains no valid characters")
        
        # Save to file
        checkpoint_path = self.checkpoint_dir / f"{safe_checkpoint_id}.json"
        try:
            with open(checkpoint_path, "w") as f:
                json.dump(checkpoint_data, f, indent=2)
            logger.info(
                f"Saved checkpoint {checkpoint_id} at generation {checkpoint.generation}"
            )
            
            # Clean up old checkpoints
            self._cleanup_old_checkpoints()
            
            return checkpoint_id
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            raise

    def load_checkpoint(self, checkpoint_id: str) -> Optional[EvolutionCheckpoint]:
        """
        Load a checkpoint from disk.

        Args:
            checkpoint_id: ID of the checkpoint to load

        Returns:
            The loaded checkpoint or None if not found/invalid
        """
        # Validate checkpoint_id to prevent path traversal
        if not checkpoint_id or not isinstance(checkpoint_id, str):
            logger.error("checkpoint_id must be a non-empty string")
            return None
        
        # Remove any path separators and potentially dangerous characters
        safe_checkpoint_id = "".join(
            c for c in checkpoint_id if c.isalnum() or c in ".-_"
        )
        if not safe_checkpoint_id:
            logger.error("checkpoint_id contains no valid characters")
            return None
        
        checkpoint_path = self.checkpoint_dir / f"{safe_checkpoint_id}.json"

        if not checkpoint_path.exists():
            logger.warning(f"Checkpoint {checkpoint_id} not found")
            return None

        try:
            with open(checkpoint_path, "r") as f:
                data = json.load(f)

            # Deserialize components
            population = self._deserialize_population(data.get("population", []))
            config = self._deserialize_config(data.get("config", {}))
            snapshots = self._deserialize_snapshots(
                data.get("generation_snapshots", [])
            )

            # Create checkpoint
            checkpoint = EvolutionCheckpoint(
                generation=data["generation"],
                population=population,
                config=config,
                generation_snapshots=snapshots,
                context=data.get("context"),
                random_state=data.get("random_state"),
                metadata=data.get("metadata", {}),
                checkpoint_time=(
                    datetime.fromisoformat(data["checkpoint_time"])
                    if data.get("checkpoint_time")
                    else None
                ),
            )

            logger.info(
                f"Loaded checkpoint {checkpoint_id} from generation {checkpoint.generation}"
            )
            return checkpoint

        except Exception as e:
            logger.error(f"Failed to load checkpoint {checkpoint_id}: {e}")
            return None

    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """
        List all available checkpoints.

        Returns:
            List of checkpoint information dictionaries
        """
        checkpoints = []

        for checkpoint_file in self.checkpoint_dir.glob("*.json"):
            checkpoint_id = checkpoint_file.stem

            # Try to load basic info without full deserialization
            try:
                with open(checkpoint_file, "r") as f:
                    data = json.load(f)

                checkpoints.append(
                    {
                        "id": checkpoint_id,
                        "generation": data.get("generation", 0),
                        "checkpoint_time": data.get("checkpoint_time"),
                        "metadata": data.get("metadata", {}),
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to read checkpoint {checkpoint_id}: {e}")

        # Sort by generation (newest first)
        checkpoints.sort(key=lambda x: x["generation"], reverse=True)
        return checkpoints

    def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """
        Delete a checkpoint.

        Args:
            checkpoint_id: ID of the checkpoint to delete

        Returns:
            True if deleted successfully
        """
        # Validate checkpoint_id to prevent path traversal
        if not checkpoint_id or not isinstance(checkpoint_id, str):
            logger.error("checkpoint_id must be a non-empty string")
            return False
        
        # Remove any path separators and potentially dangerous characters
        safe_checkpoint_id = "".join(
            c for c in checkpoint_id if c.isalnum() or c in ".-_"
        )
        if not safe_checkpoint_id:
            logger.error("checkpoint_id contains no valid characters")
            return False
        
        checkpoint_path = self.checkpoint_dir / f"{safe_checkpoint_id}.json"

        if checkpoint_path.exists():
            try:
                checkpoint_path.unlink()
                logger.info(f"Deleted checkpoint {checkpoint_id}")
                return True
            except Exception as e:
                logger.error(f"Failed to delete checkpoint {checkpoint_id}: {e}")
                return False
        return False

    def _cleanup_old_checkpoints(self) -> None:
        """Clean up old checkpoints based on configured limits."""
        import time
        from datetime import datetime, timedelta
        
        if not self.checkpoint_dir.exists():
            return
        
        checkpoints = []
        current_time = time.time()
        cutoff_time = current_time - (self.max_age_days * 24 * 3600)
        
        # Collect checkpoint files with their metadata
        for checkpoint_file in self.checkpoint_dir.glob("*.json"):
            try:
                stat = checkpoint_file.stat()
                checkpoints.append({
                    "path": checkpoint_file,
                    "mtime": stat.st_mtime,
                    "name": checkpoint_file.stem
                })
            except Exception as e:
                logger.warning(f"Failed to stat checkpoint {checkpoint_file}: {e}")
        
        # Remove checkpoints older than max_age_days
        aged_out = []
        for checkpoint in checkpoints:
            if checkpoint["mtime"] < cutoff_time:
                aged_out.append(checkpoint)
        
        for checkpoint in aged_out:
            try:
                checkpoint["path"].unlink()
                logger.info(f"Deleted aged checkpoint {checkpoint['name']}")
            except Exception as e:
                logger.warning(f"Failed to delete aged checkpoint {checkpoint['name']}: {e}")
        
        # Remove aged checkpoints from list
        checkpoints = [cp for cp in checkpoints if cp not in aged_out]
        
        # Enforce max_checkpoints limit by removing oldest
        if len(checkpoints) > self.max_checkpoints:
            # Sort by modification time (oldest first)
            checkpoints.sort(key=lambda x: x["mtime"])
            
            # Remove oldest checkpoints
            to_remove = checkpoints[:len(checkpoints) - self.max_checkpoints]
            for checkpoint in to_remove:
                try:
                    checkpoint["path"].unlink()
                    logger.info(f"Deleted excess checkpoint {checkpoint['name']}")
                except Exception as e:
                    logger.warning(f"Failed to delete excess checkpoint {checkpoint['name']}: {e}")

    def _serialize_population(
        self, population: List[IndividualFitness]
    ) -> List[Dict[str, Any]]:
        """Serialize population for storage."""
        serialized = []

        for individual in population:
            # Serialize idea
            idea_data = {
                "content": individual.idea.content,
                "thinking_method": individual.idea.thinking_method.value,
                "agent_name": individual.idea.agent_name,
                "generation_prompt": individual.idea.generation_prompt,
                "confidence_score": individual.idea.confidence_score,
                "reasoning": individual.idea.reasoning,
                "parent_ideas": individual.idea.parent_ideas,
                "metadata": individual.idea.metadata,
                "timestamp": individual.idea.timestamp,
            }

            # Serialize fitness
            fitness_data = {
                "idea": idea_data,
                "creativity_score": individual.creativity_score,
                "diversity_score": individual.diversity_score,
                "quality_score": individual.quality_score,
                "overall_fitness": individual.overall_fitness,
                "evaluation_metadata": individual.evaluation_metadata,
                "evaluated_at": (
                    individual.evaluated_at.isoformat()
                    if individual.evaluated_at
                    else None
                ),
            }

            serialized.append(fitness_data)

        return serialized

    def _deserialize_population(
        self, data: List[Dict[str, Any]]
    ) -> List[IndividualFitness]:
        """Deserialize population from storage."""
        population = []

        for fitness_data in data:
            # Deserialize idea
            idea_data = fitness_data["idea"]
            idea = GeneratedIdea(
                content=idea_data["content"],
                thinking_method=ThinkingMethod(idea_data["thinking_method"]),
                agent_name=idea_data["agent_name"],
                generation_prompt=idea_data["generation_prompt"],
                confidence_score=idea_data.get("confidence_score"),
                reasoning=idea_data.get("reasoning"),
                parent_ideas=idea_data.get("parent_ideas", []),
                metadata=idea_data.get("metadata", {}),
                timestamp=idea_data.get("timestamp"),
            )

            # Deserialize fitness
            individual = IndividualFitness(
                idea=idea,
                creativity_score=fitness_data.get("creativity_score", 0.0),
                diversity_score=fitness_data.get("diversity_score", 0.0),
                quality_score=fitness_data.get("quality_score", 0.0),
                overall_fitness=fitness_data.get("overall_fitness", 0.0),
                evaluation_metadata=fitness_data.get("evaluation_metadata", {}),
                evaluated_at=(
                    datetime.fromisoformat(fitness_data["evaluated_at"])
                    if fitness_data.get("evaluated_at")
                    else None
                ),
            )

            population.append(individual)

        return population

    def _serialize_snapshots(
        self, snapshots: List[PopulationSnapshot]
    ) -> List[Dict[str, Any]]:
        """Serialize generation snapshots."""
        serialized = []

        for snapshot in snapshots:
            snapshot_data = {
                "generation": snapshot.generation,
                "population": self._serialize_population(snapshot.population),
                "best_fitness": snapshot.best_fitness,
                "average_fitness": snapshot.average_fitness,
                "worst_fitness": snapshot.worst_fitness,
                "diversity_score": snapshot.diversity_score,
                "timestamp": (
                    snapshot.timestamp.isoformat() if snapshot.timestamp else None
                ),
            }
            serialized.append(snapshot_data)

        return serialized

    def _deserialize_snapshots(
        self, data: List[Dict[str, Any]]
    ) -> List[PopulationSnapshot]:
        """Deserialize generation snapshots."""
        snapshots = []

        for snapshot_data in data:
            snapshot = PopulationSnapshot(
                generation=snapshot_data["generation"],
                population=self._deserialize_population(snapshot_data["population"]),
                best_fitness=snapshot_data["best_fitness"],
                average_fitness=snapshot_data["average_fitness"],
                worst_fitness=snapshot_data["worst_fitness"],
                diversity_score=snapshot_data["diversity_score"],
                timestamp=(
                    datetime.fromisoformat(snapshot_data["timestamp"])
                    if snapshot_data.get("timestamp")
                    else None
                ),
            )
            snapshots.append(snapshot)

        return snapshots

    def _serialize_config(self, config: EvolutionConfig) -> Dict[str, Any]:
        """Serialize evolution configuration."""
        config_dict = asdict(config)
        # Convert enum to string
        if "selection_strategy" in config_dict:
            config_dict["selection_strategy"] = config_dict["selection_strategy"].value
        return config_dict

    def _deserialize_config(self, data: Dict[str, Any]) -> EvolutionConfig:
        """Deserialize evolution configuration."""
        # Handle selection strategy enum
        if "selection_strategy" in data:
            data["selection_strategy"] = SelectionStrategy(data["selection_strategy"])

        return EvolutionConfig(**data)
