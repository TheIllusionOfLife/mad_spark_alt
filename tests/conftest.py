"""
Pytest configuration and fixtures for test suite.

This module provides shared fixtures and configuration for all tests,
including proper LLM provider state management and test isolation.
"""

import gc
import os
from typing import AsyncGenerator

import pytest

from mad_spark_alt.core.llm_provider import LLMManager, llm_manager, setup_llm_providers


@pytest.fixture
async def mock_llm_setup() -> AsyncGenerator[LLMManager, None]:
    """
    Setup mock LLM providers for tests that need them.

    This fixture should be used by tests that require LLM providers
    to be registered. It sets up a mock provider and cleans up after.

    Usage:
        async def test_something(mock_llm_setup):
            # LLM providers are now available
            result = await orchestrator.run_qadi_cycle("test")
    """
    # Skip if real API key is available (integration tests)
    api_key = os.getenv("GOOGLE_API_KEY")
    if api_key:
        await setup_llm_providers(api_key)
    else:
        # For unit tests without API keys, tests should mock the LLM calls themselves
        # This fixture just ensures the llm_manager is in a clean state
        pass

    yield llm_manager

    # Clean up after test
    llm_manager.providers.clear()


@pytest.fixture(autouse=True)
def isolate_llm_state(request):
    """
    Ensure each test starts with clean LLM provider state.

    This runs before every test automatically to prevent state leakage
    between tests. Integration tests are skipped to allow class-scoped
    fixtures to register and maintain real LLM providers across tests.

    Tests that need providers should either:
    1. Use the mock_llm_setup fixture (unit tests)
    2. Set up their own mocked providers (unit tests)
    3. Be marked as @pytest.mark.integration (integration tests)

    Note: mock_llm_setup provides LLM providers for tests that need them.
    This fixture (autouse) ensures cleanup for all other tests.
    """
    # Skip isolation for integration tests that manage their own provider state
    if "integration" in request.keywords:
        yield
        return

    # Clear providers before each test
    llm_manager.providers.clear()

    yield

    # Clear providers after each test
    llm_manager.providers.clear()

    # Force garbage collection to clean up lingering mock objects and weak references
    # This is necessary because AsyncMock objects can hold references that prevent
    # proper cleanup, leading to ResourceWarnings in tests
    gc.collect()
