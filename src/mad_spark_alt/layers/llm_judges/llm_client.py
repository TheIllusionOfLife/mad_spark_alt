"""
LLM client abstraction for different AI model providers.
"""

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
import os

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Response from an LLM API call."""
    
    content: str
    model: str
    usage: Dict[str, int]  # tokens, cost info, etc.
    metadata: Dict[str, Any]
    error: Optional[str] = None


class LLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    @abstractmethod
    async def generate(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1000,
        **kwargs
    ) -> LLMResponse:
        """Generate a response from the LLM."""
        pass
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Get the model name."""
        pass
    
    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the client is available (has API key, etc)."""
        pass


class OpenAIClient(LLMClient):
    """OpenAI GPT model client."""
    
    def __init__(self, model: str = "gpt-4", api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._client = None
        
        if self.is_available:
            try:
                import openai
                self._client = openai.AsyncOpenAI(api_key=self.api_key)
            except ImportError:
                logger.warning("OpenAI package not available")
    
    @property
    def model_name(self) -> str:
        return self.model
    
    @property
    def is_available(self) -> bool:
        return self.api_key is not None and self._client is not None
    
    async def generate(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1000,
        **kwargs
    ) -> LLMResponse:
        """Generate response using OpenAI API."""
        if not self.is_available:
            return LLMResponse(
                content="",
                model=self.model,
                usage={},
                metadata={},
                error="OpenAI client not available"
            )
        
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = await self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            
            return LLMResponse(
                content=response.choices[0].message.content,
                model=self.model,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                },
                metadata={"finish_reason": response.choices[0].finish_reason}
            )
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return LLMResponse(
                content="",
                model=self.model,
                usage={},
                metadata={},
                error=str(e)
            )


class AnthropicClient(LLMClient):
    """Anthropic Claude model client."""
    
    def __init__(self, model: str = "claude-3-sonnet-20240229", api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self._client = None
        
        if self.is_available:
            try:
                import anthropic
                self._client = anthropic.AsyncAnthropic(api_key=self.api_key)
            except ImportError:
                logger.warning("Anthropic package not available")
    
    @property
    def model_name(self) -> str:
        return self.model
    
    @property
    def is_available(self) -> bool:
        return self.api_key is not None and self._client is not None
    
    async def generate(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1000,
        **kwargs
    ) -> LLMResponse:
        """Generate response using Anthropic API."""
        if not self.is_available:
            return LLMResponse(
                content="",
                model=self.model,
                usage={},
                metadata={},
                error="Anthropic client not available"
            )
        
        try:
            response = await self._client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt or "",
                messages=[{"role": "user", "content": prompt}],
                **kwargs
            )
            
            return LLMResponse(
                content=response.content[0].text,
                model=self.model,
                usage={
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
                },
                metadata={"stop_reason": response.stop_reason}
            )
            
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            return LLMResponse(
                content="",
                model=self.model,
                usage={},
                metadata={},
                error=str(e)
            )


class MockLLMClient(LLMClient):
    """Mock LLM client for testing and when API keys are not available."""
    
    def __init__(self, model: str = "mock-model"):
        self.model = model
    
    @property
    def model_name(self) -> str:
        return self.model
    
    @property
    def is_available(self) -> bool:
        return True
    
    async def generate(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1000,
        **kwargs
    ) -> LLMResponse:
        """Generate mock response for testing."""
        # Simulate API delay
        await asyncio.sleep(0.1)
        
        # Generate mock creativity scores based on prompt content
        mock_scores = {
            "novelty": 0.7 + len(prompt) % 3 * 0.1,
            "usefulness": 0.6 + len(prompt) % 4 * 0.1, 
            "feasibility": 0.8 + len(prompt) % 2 * 0.1,
            "elaboration": 0.5 + len(prompt) % 5 * 0.1,
        }
        
        mock_content = json.dumps({
            "creativity_scores": mock_scores,
            "overall_score": sum(mock_scores.values()) / len(mock_scores),
            "rationale": f"Mock evaluation of content with {len(prompt)} characters. Shows moderate creativity across dimensions with some novel elements.",
            "strengths": ["Original concept", "Clear expression"],
            "weaknesses": ["Could be more detailed", "Limited practical application"]
        }, indent=2)
        
        return LLMResponse(
            content=mock_content,
            model=self.model,
            usage={
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": len(mock_content.split()),
                "total_tokens": len(prompt.split()) + len(mock_content.split()),
            },
            metadata={"finish_reason": "stop"}
        )


def create_llm_client(model: str) -> LLMClient:
    """Factory function to create LLM clients."""
    if model.startswith("gpt"):
        return OpenAIClient(model)
    elif model.startswith("claude"):
        return AnthropicClient(model)
    else:
        logger.warning(f"Unknown model {model}, using mock client")
        return MockLLMClient(model)