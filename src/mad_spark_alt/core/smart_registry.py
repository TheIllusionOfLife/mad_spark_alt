"""
Smart Agent Registry with automatic LLM agent preference and fallback.

This module provides intelligent agent registration that automatically prefers
LLM-powered agents when API keys are available and falls back to template
agents when they're not.
"""

import asyncio
import logging
import os
from typing import Dict, List, Optional, Set

from .interfaces import ThinkingAgentInterface, ThinkingMethod
from .llm_provider import LLMProvider, setup_llm_providers
from .registry import ThinkingAgentRegistry

logger = logging.getLogger(__name__)


class SmartAgentRegistry:
    """
    Smart registry that automatically configures agents based on available resources.
    
    Features:
    - Automatic LLM agent preference when API keys are available
    - Graceful fallback to template agents when LLM services fail
    - Environment-based configuration
    - Intelligent agent selection
    """

    def __init__(self, base_registry: Optional[ThinkingAgentRegistry] = None):
        """
        Initialize the smart registry.
        
        Args:
            base_registry: Base registry to use (creates new if None)
        """
        self.base_registry = base_registry or ThinkingAgentRegistry()
        self._llm_availability: Dict[LLMProvider, bool] = {}
        self._llm_setup_attempted = False
        self._agent_preferences: Dict[ThinkingMethod, str] = {}
        
    async def setup_intelligent_agents(self) -> Dict[str, str]:
        """
        Setup agents intelligently based on available API keys and services.
        
        Returns:
            Dictionary with setup status for each thinking method
        """
        setup_status = {}
        
        # Detect available LLM providers
        available_providers = await self._detect_llm_providers()
        
        if available_providers:
            # Try to setup LLM providers
            llm_setup_success = await self._setup_llm_providers()
            
            if llm_setup_success:
                # Register LLM agents
                setup_status.update(await self._register_llm_agents(available_providers))
            else:
                logger.warning("LLM setup failed, falling back to template agents")
                setup_status.update(self._register_template_agents())
        else:
            logger.info("No LLM API keys detected, using template agents")
            setup_status.update(self._register_template_agents())
            
        return setup_status
    
    async def _detect_llm_providers(self) -> List[LLMProvider]:
        """
        Detect available LLM providers based on environment variables.
        
        Returns:
            List of available LLM providers
        """
        available = []
        
        # Check OpenAI
        if os.getenv("OPENAI_API_KEY"):
            available.append(LLMProvider.OPENAI)
            self._llm_availability[LLMProvider.OPENAI] = True
            
        # Check Anthropic
        if os.getenv("ANTHROPIC_API_KEY"):
            available.append(LLMProvider.ANTHROPIC)
            self._llm_availability[LLMProvider.ANTHROPIC] = True
            
        # Check Google
        if os.getenv("GOOGLE_API_KEY"):
            available.append(LLMProvider.GOOGLE)
            self._llm_availability[LLMProvider.GOOGLE] = True
            
        logger.info(f"Detected LLM providers: {[p.value for p in available]}")
        return available
    
    async def _setup_llm_providers(self) -> bool:
        """
        Setup LLM providers and test connectivity.
        
        Returns:
            True if setup was successful, False otherwise
        """
        if self._llm_setup_attempted:
            return any(self._llm_availability.values())
            
        self._llm_setup_attempted = True
        
        try:
            await setup_llm_providers(
                openai_api_key=os.getenv("OPENAI_API_KEY"),
                anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
                google_api_key=os.getenv("GOOGLE_API_KEY")
            )
            logger.info("LLM providers setup successful")
            return True
        except Exception as e:
            logger.error(f"LLM provider setup failed: {e}")
            # Mark all as unavailable
            for provider in self._llm_availability:
                self._llm_availability[provider] = False
            return False
    
    async def _register_llm_agents(self, available_providers: List[LLMProvider]) -> Dict[str, str]:
        """
        Register LLM-powered agents for all thinking methods.
        
        Args:
            available_providers: List of available LLM providers
            
        Returns:
            Dictionary with registration status for each thinking method
        """
        from ..agents import (
            LLMQuestioningAgent,
            LLMAbductiveAgent, 
            LLMDeductiveAgent,
            LLMInductiveAgent
        )
        
        status = {}
        
        # Determine preferred provider (prefer Anthropic > OpenAI > Google)
        preferred_provider = None
        if LLMProvider.ANTHROPIC in available_providers:
            preferred_provider = LLMProvider.ANTHROPIC
        elif LLMProvider.OPENAI in available_providers:
            preferred_provider = LLMProvider.OPENAI
        elif LLMProvider.GOOGLE in available_providers:
            preferred_provider = LLMProvider.GOOGLE
            
        logger.info(f"Using preferred LLM provider: {preferred_provider.value if preferred_provider else 'None'}")
        
        # Register LLM agents
        llm_agents = [
            (LLMQuestioningAgent, ThinkingMethod.QUESTIONING, "LLMQuestioningAgent"),
            (LLMAbductiveAgent, ThinkingMethod.ABDUCTION, "LLMAbductiveAgent"),
            (LLMDeductiveAgent, ThinkingMethod.DEDUCTION, "LLMDeductiveAgent"),
            (LLMInductiveAgent, ThinkingMethod.INDUCTION, "LLMInductiveAgent"),
        ]
        
        for agent_class, method, name in llm_agents:
            try:
                # Create agent with preferred provider
                agent_instance = agent_class(preferred_provider=preferred_provider)
                self.base_registry.register(agent_class)
                self._agent_preferences[method] = name
                status[method.value] = f"LLM agent registered ({preferred_provider.value})"
                logger.info(f"Registered LLM agent for {method.value}")
            except Exception as e:
                logger.error(f"Failed to register LLM agent for {method.value}: {e}")
                status[method.value] = f"LLM registration failed: {e}"
                
        return status
    
    def _register_template_agents(self) -> Dict[str, str]:
        """
        Register template-based agents as fallback.
        
        Returns:
            Dictionary with registration status for each thinking method
        """
        from ..agents import (
            QuestioningAgent,
            AbductionAgent,
            DeductionAgent,
            InductionAgent
        )
        
        status = {}
        
        template_agents = [
            (QuestioningAgent, ThinkingMethod.QUESTIONING, "QuestioningAgent"),
            (AbductionAgent, ThinkingMethod.ABDUCTION, "AbductionAgent"),
            (DeductionAgent, ThinkingMethod.DEDUCTION, "DeductionAgent"),
            (InductionAgent, ThinkingMethod.INDUCTION, "InductionAgent"),
        ]
        
        for agent_class, method, name in template_agents:
            try:
                self.base_registry.register(agent_class)
                self._agent_preferences[method] = name
                status[method.value] = "Template agent registered"
                logger.info(f"Registered template agent for {method.value}")
            except Exception as e:
                logger.error(f"Failed to register template agent for {method.value}: {e}")
                status[method.value] = f"Template registration failed: {e}"
                
        return status
    
    def get_preferred_agent(self, method: ThinkingMethod) -> Optional[ThinkingAgentInterface]:
        """
        Get the preferred agent for a thinking method.
        
        Args:
            method: The thinking method
            
        Returns:
            Preferred agent instance or None if not available
        """
        preferred_name = self._agent_preferences.get(method)
        if preferred_name:
            return self.base_registry.get_agent(preferred_name)
        
        # Fallback to any available agent for the method
        return self.base_registry.get_agent_by_method(method)
    
    def get_agent_status(self) -> Dict[str, Dict[str, str]]:
        """
        Get status of all registered agents.
        
        Returns:
            Dictionary with agent status information
        """
        status = {
            "llm_providers": {
                provider.value: available 
                for provider, available in self._llm_availability.items()
            },
            "agent_preferences": {
                method.value: name 
                for method, name in self._agent_preferences.items()
            },
            "registered_agents": self.base_registry.list_agents()
        }
        
        return status
    
    async def test_agent_connectivity(self) -> Dict[str, bool]:
        """
        Test connectivity of registered agents.
        
        Returns:
            Dictionary with connectivity test results
        """
        results = {}
        
        for method in ThinkingMethod:
            agent = self.get_preferred_agent(method)
            if agent:
                try:
                    # Try a simple generation request to test connectivity
                    from .interfaces import IdeaGenerationRequest
                    
                    test_request = IdeaGenerationRequest(
                        problem_statement="Test connectivity",
                        max_ideas_per_method=1
                    )
                    
                    result = await agent.generate_ideas(test_request)
                    results[method.value] = not result.error_message
                except Exception as e:
                    logger.warning(f"Connectivity test failed for {method.value}: {e}")
                    results[method.value] = False
            else:
                results[method.value] = False
                
        return results
    
    def clear_registry(self):
        """Clear the registry (useful for testing)."""
        self.base_registry.clear()
        self._llm_availability.clear()
        self._agent_preferences.clear()
        self._llm_setup_attempted = False


# Global smart registry instance
smart_registry = SmartAgentRegistry()


async def setup_smart_agents() -> Dict[str, str]:
    """
    Convenience function to setup agents intelligently.
    
    Returns:
        Dictionary with setup status for each thinking method
    """
    return await smart_registry.setup_intelligent_agents()


def get_smart_agent(method: ThinkingMethod) -> Optional[ThinkingAgentInterface]:
    """
    Convenience function to get the preferred agent for a thinking method.
    
    Args:
        method: The thinking method
        
    Returns:
        Preferred agent instance or None if not available
    """
    return smart_registry.get_preferred_agent(method)