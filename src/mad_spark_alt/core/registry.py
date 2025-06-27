"""
Plugin registry system for managing evaluators.
"""

from typing import Dict, List, Optional, Type, Set, Any
import logging
from ..core.interfaces import EvaluatorInterface, EvaluationLayer, OutputType

logger = logging.getLogger(__name__)


class EvaluatorRegistry:
    """Registry for managing evaluator plugins."""

    def __init__(self) -> None:
        self._evaluators: Dict[str, Type[EvaluatorInterface]] = {}
        self._instances: Dict[str, EvaluatorInterface] = {}
        self._layer_index: Dict[EvaluationLayer, Set[str]] = {
            layer: set() for layer in EvaluationLayer
        }
        self._output_type_index: Dict[OutputType, Set[str]] = {
            output_type: set() for output_type in OutputType
        }

    def register(self, evaluator_class: Type[EvaluatorInterface]) -> None:
        """
        Register an evaluator class.

        Args:
            evaluator_class: The evaluator class to register
        """
        # Create temporary instance to get metadata
        temp_instance = evaluator_class()
        name = temp_instance.name

        if name in self._evaluators:
            logger.warning(f"Overriding existing evaluator: {name}")

        self._evaluators[name] = evaluator_class

        # Update indices
        self._layer_index[temp_instance.layer].add(name)
        for output_type in temp_instance.supported_output_types:
            self._output_type_index[output_type].add(name)

        logger.info(f"Registered evaluator: {name}")

    def get_evaluator(self, name: str) -> Optional[EvaluatorInterface]:
        """
        Get an evaluator instance by name.

        Args:
            name: Name of the evaluator

        Returns:
            Evaluator instance or None if not found
        """
        if name not in self._evaluators:
            return None

        # Use singleton pattern for evaluator instances
        if name not in self._instances:
            self._instances[name] = self._evaluators[name]()

        return self._instances[name]

    def get_evaluators_by_layer(
        self, layer: EvaluationLayer
    ) -> List[EvaluatorInterface]:
        """
        Get all evaluators for a specific layer.

        Args:
            layer: The evaluation layer

        Returns:
            List of evaluator instances
        """
        evaluator_names = self._layer_index[layer]
        evaluators = []
        for name in evaluator_names:
            evaluator = self.get_evaluator(name)
            if evaluator is not None:
                evaluators.append(evaluator)
        return evaluators

    def get_evaluators_by_output_type(
        self, output_type: OutputType
    ) -> List[EvaluatorInterface]:
        """
        Get all evaluators that support a specific output type.

        Args:
            output_type: The output type

        Returns:
            List of evaluator instances
        """
        evaluator_names = self._output_type_index[output_type]
        evaluators = []
        for name in evaluator_names:
            evaluator = self.get_evaluator(name)
            if evaluator is not None:
                evaluators.append(evaluator)
        return evaluators

    def get_compatible_evaluators(
        self,
        layer: Optional[EvaluationLayer] = None,
        output_type: Optional[OutputType] = None,
    ) -> List[EvaluatorInterface]:
        """
        Get evaluators matching the given criteria.

        Args:
            layer: Optional layer filter
            output_type: Optional output type filter

        Returns:
            List of matching evaluator instances
        """
        candidates = set(self._evaluators.keys())

        if layer is not None:
            candidates &= self._layer_index[layer]

        if output_type is not None:
            candidates &= self._output_type_index[output_type]

        evaluators = []
        for name in candidates:
            evaluator = self.get_evaluator(name)
            if evaluator is not None:
                evaluators.append(evaluator)
        return evaluators

    def list_evaluators(self) -> Dict[str, Dict[str, Any]]:
        """
        List all registered evaluators with their metadata.

        Returns:
            Dictionary mapping evaluator names to metadata
        """
        result = {}
        for name in self._evaluators:
            evaluator = self.get_evaluator(name)
            if evaluator:
                result[name] = {
                    "layer": evaluator.layer.value,
                    "supported_output_types": [
                        ot.value for ot in evaluator.supported_output_types
                    ],
                }
        return result

    def unregister(self, name: str) -> bool:
        """
        Unregister an evaluator.

        Args:
            name: Name of the evaluator to unregister

        Returns:
            True if unregistered, False if not found
        """
        if name not in self._evaluators:
            return False

        evaluator = self.get_evaluator(name)
        if evaluator:
            # Remove from indices
            self._layer_index[evaluator.layer].discard(name)
            for output_type in evaluator.supported_output_types:
                self._output_type_index[output_type].discard(name)

        # Remove from registry
        del self._evaluators[name]
        if name in self._instances:
            del self._instances[name]

        logger.info(f"Unregistered evaluator: {name}")
        return True


# Global registry instance
registry = EvaluatorRegistry()


def register_evaluator(evaluator_class: Type[EvaluatorInterface]) -> None:
    """Convenience function to register an evaluator with the global registry."""
    registry.register(evaluator_class)
