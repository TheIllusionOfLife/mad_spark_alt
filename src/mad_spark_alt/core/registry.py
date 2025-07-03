"""
Plugin registry system for managing evaluators and thinking agents.
"""

import logging
from typing import Any, Dict, List, Optional, Set, Type, Union

from ..core.interfaces import (
    EvaluationLayer,
    EvaluatorInterface,
    OutputType,
    ThinkingAgentInterface,
    ThinkingMethod,
)

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


class ThinkingAgentRegistry:
    """Registry for managing thinking agent plugins."""

    def __init__(self) -> None:
        self._agents: Dict[str, Type[ThinkingAgentInterface]] = {}
        self._instances: Dict[str, ThinkingAgentInterface] = {}
        self._method_index: Dict[ThinkingMethod, Set[str]] = {
            method: set() for method in ThinkingMethod
        }
        self._output_type_index: Dict[OutputType, Set[str]] = {
            output_type: set() for output_type in OutputType
        }

    def register(self, agent_class: Type[ThinkingAgentInterface]) -> None:
        """
        Register a thinking agent class.

        Args:
            agent_class: The thinking agent class to register
        """
        # Create temporary instance to get metadata
        temp_instance = agent_class()
        name = temp_instance.name

        if name in self._agents:
            logger.warning(f"Overriding existing thinking agent: {name}")

        self._agents[name] = agent_class

        # Update indices
        self._method_index[temp_instance.thinking_method].add(name)
        for output_type in temp_instance.supported_output_types:
            self._output_type_index[output_type].add(name)

        logger.info(f"Registered thinking agent: {name}")

    def get_agent(self, name: str) -> Optional[ThinkingAgentInterface]:
        """
        Get a thinking agent instance by name.

        Args:
            name: Name of the thinking agent

        Returns:
            Thinking agent instance or None if not found
        """
        if name not in self._agents:
            return None

        # Use singleton pattern for agent instances
        if name not in self._instances:
            self._instances[name] = self._agents[name]()

        return self._instances[name]

    def get_agents_by_method(
        self, method: ThinkingMethod
    ) -> List[ThinkingAgentInterface]:
        """
        Get all agents for a specific thinking method.

        Args:
            method: The thinking method

        Returns:
            List of thinking agent instances
        """
        agent_names = self._method_index[method]
        agents = []
        for name in agent_names:
            agent = self.get_agent(name)
            if agent is not None:
                agents.append(agent)
        return agents

    def get_agent_by_method(
        self, method: ThinkingMethod
    ) -> Optional[ThinkingAgentInterface]:
        """
        Get the first available agent for a specific thinking method.

        Args:
            method: The thinking method

        Returns:
            First available thinking agent instance or None
        """
        agents = self.get_agents_by_method(method)
        return agents[0] if agents else None

    def get_agents_by_output_type(
        self, output_type: OutputType
    ) -> List[ThinkingAgentInterface]:
        """
        Get all agents that support a specific output type.

        Args:
            output_type: The output type

        Returns:
            List of thinking agent instances
        """
        agent_names = self._output_type_index[output_type]
        agents = []
        for name in agent_names:
            agent = self.get_agent(name)
            if agent is not None:
                agents.append(agent)
        return agents

    def get_all_agents(self) -> List[ThinkingAgentInterface]:
        """Get all registered thinking agents."""
        agents = []
        for name in self._agents:
            agent = self.get_agent(name)
            if agent is not None:
                agents.append(agent)
        return agents

    def list_agents(self) -> Dict[str, Dict[str, Any]]:
        """
        List all registered thinking agents with their metadata.

        Returns:
            Dictionary mapping agent names to metadata
        """
        result = {}
        for name in self._agents:
            agent = self.get_agent(name)
            if agent:
                result[name] = {
                    "thinking_method": agent.thinking_method.value,
                    "supported_output_types": [
                        ot.value for ot in agent.supported_output_types
                    ],
                }
        return result

    def unregister(self, name: str) -> bool:
        """
        Unregister a thinking agent.

        Args:
            name: Name of the agent to unregister

        Returns:
            True if unregistered, False if not found
        """
        if name not in self._agents:
            return False

        agent = self.get_agent(name)
        if agent:
            # Remove from indices
            self._method_index[agent.thinking_method].discard(name)
            for output_type in agent.supported_output_types:
                self._output_type_index[output_type].discard(name)

        # Remove from registry
        del self._agents[name]
        if name in self._instances:
            del self._instances[name]

        logger.info(f"Unregistered thinking agent: {name}")
        return True

    def clear(self) -> None:
        """Clear all registered agents (useful for testing)."""
        self._agents.clear()
        self._instances.clear()
        for method_set in self._method_index.values():
            method_set.clear()
        for type_set in self._output_type_index.values():
            type_set.clear()


class UnifiedRegistry:
    """Unified registry managing both evaluators and thinking agents."""

    def __init__(self) -> None:
        # Reference global registry instances to avoid duplication
        self.evaluators: Optional[EvaluatorRegistry] = None
        self.agents: Optional[ThinkingAgentRegistry] = None

    def _ensure_registries(self) -> None:
        """Ensure registry references are initialized."""
        if self.evaluators is None:
            global registry
            self.evaluators = registry
        if self.agents is None:
            global agent_registry
            self.agents = agent_registry

    def get_evaluators(self) -> List[EvaluatorInterface]:
        """Get all registered evaluators."""
        self._ensure_registries()
        assert self.evaluators is not None
        evaluators = []
        for name in self.evaluators._evaluators:
            evaluator = self.evaluators.get_evaluator(name)
            if evaluator is not None:
                evaluators.append(evaluator)
        return evaluators

    def get_agents(self) -> List[ThinkingAgentInterface]:
        """Get all registered thinking agents."""
        self._ensure_registries()
        assert self.agents is not None
        return self.agents.get_all_agents()

    def clear_all(self) -> None:
        """Clear all registries (useful for testing)."""
        self._ensure_registries()
        assert self.evaluators is not None
        assert self.agents is not None
        # Clear evaluator registry
        self.evaluators._evaluators.clear()
        self.evaluators._instances.clear()
        for layer_set in self.evaluators._layer_index.values():
            layer_set.clear()
        for type_set in self.evaluators._output_type_index.values():
            type_set.clear()

        # Clear agent registry
        self.agents.clear()


# Global registry instances
registry = EvaluatorRegistry()
agent_registry = ThinkingAgentRegistry()
unified_registry = UnifiedRegistry()


def register_evaluator(evaluator_class: Type[EvaluatorInterface]) -> None:
    """Convenience function to register an evaluator with the global registry."""
    registry.register(evaluator_class)


def register_agent(agent_class: Type[ThinkingAgentInterface]) -> None:
    """Convenience function to register a thinking agent with the global registry."""
    agent_registry.register(agent_class)
