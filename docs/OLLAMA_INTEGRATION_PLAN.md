# Ollama Integration Plan - Local Model Support for Cost Savings

## Executive Summary

**Goal**: Integrate Ollama for local inference to reduce costs and scale to many more agents.

**Strategy**: Hybrid architecture - Use Ollama for high-volume operations, keep Gemini for complex reasoning.

**Expected Cost Reduction**: 70-90% depending on workload distribution.

**Timeline**: 2-3 days for basic integration, 1 week for production-ready.

## Current State Analysis

### Existing Architecture

The system currently uses **GoogleProvider** exclusively:
- REST API calls to `https://generativelanguage.googleapis.com/v1beta`
- Gemini-specific features: structured output, multimodal, URL context
- Cost: ~$0.075 per 1M input tokens, ~$0.30 per 1M output tokens

### Cost Breakdown (Typical QADI + Evolution Run)

| Operation | Calls | Tokens/Call | Cost/Run | Annual Cost (1000 runs) |
|-----------|-------|-------------|----------|------------------------|
| Hypothesis Generation | 1 | 500 in, 1000 out | $0.0004 | $0.40 |
| Deduction (Scoring) | 1 | 1000 in, 1500 out | $0.0005 | $0.50 |
| Evolution (5 gen × 10 pop) | 50 | 800 in, 600 out | $0.012 | $12.00 |
| Fitness Evaluations | 50 | 600 in, 400 out | $0.009 | $9.00 |
| **TOTAL** | **102** | **~90K tokens** | **$0.022** | **$22.00** |

**With 100 agents in workflow**: $2.20 per run × 1000 runs = **$2,200/year**

### Local Model Alternative (Ollama)

**Hardware Requirements**:
- Gemma 3 12B: 8-12GB VRAM (RTX 3090, RTX 4090, A4000, etc.)
- Gemma 3 12B-QAT: 6-8GB VRAM (more efficient, same quality)
- CPU fallback: Slower but possible (32GB+ RAM recommended)

**Cost**: $0/token (hardware amortized separately)

## Implementation Plan

### Phase 1: Foundation (Day 1) ✅

#### 1.1 Update LLMProvider Enum

**File**: `src/mad_spark_alt/core/llm_provider.py:51-54`

```python
class LLMProvider(Enum):
    """Supported LLM providers."""

    GOOGLE = "google"
    OLLAMA = "ollama"  # NEW
```

#### 1.2 Create OllamaProvider Class

**File**: `src/mad_spark_alt/core/llm_provider.py` (after GoogleProvider)

```python
class OllamaProvider(LLMProviderInterface):
    """
    Ollama local inference provider.

    Supports text-only generation via Ollama API.
    Multimodal features (images, PDFs, URLs) NOT supported.

    API Documentation: https://github.com/ollama/ollama/blob/main/docs/api.md
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        default_model: str = "gemma3:12b-it-qat",
        retry_config: Optional[RetryConfig] = None
    ):
        self.base_url = base_url
        self.default_model = default_model
        self._session: Optional[aiohttp.ClientSession] = None
        self.retry_config = retry_config or RetryConfig()
        self.circuit_breaker = CircuitBreaker()

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """
        Generate text using Ollama API.

        NOTE: Multimodal inputs (images, PDFs, URLs) are IGNORED.
        Ollama GGUF format does not support vision inputs currently.
        """
        session = await self._get_session()

        # Warn if multimodal inputs provided
        if request.multimodal_inputs or request.urls:
            logger.warning(
                "Ollama does not support multimodal inputs. "
                "Images, PDFs, and URLs will be ignored."
            )

        # Get model config
        model_config = request.model_configuration or self._get_default_model_config()

        # Build prompt (combine system + user)
        prompt_parts = []
        if request.system_prompt:
            prompt_parts.append(f"System: {request.system_prompt}")
        prompt_parts.append(f"User: {request.user_prompt}")
        full_prompt = "\n\n".join(prompt_parts)

        # Prepare Ollama API request
        url = f"{self.base_url}/api/generate"

        payload = {
            "model": model_config.model_name,
            "prompt": full_prompt,
            "stream": False,  # Get complete response
            "options": {
                "temperature": request.temperature,
                "top_p": request.top_p,
                "num_predict": request.max_tokens,
            }
        }

        # Add structured output if requested (Ollama supports JSON mode)
        if request.response_mime_type == "application/json":
            payload["format"] = "json"
            # Note: Ollama doesn't enforce schema, just JSON format
            if request.response_schema:
                logger.warning(
                    "Ollama JSON mode does not enforce schemas. "
                    "Response may not match expected structure."
                )

        start_time = time.time()

        try:
            response_data = await safe_aiohttp_request(
                session=session,
                method="POST",
                url=url,
                json=payload,
                retry_config=self.retry_config,
                circuit_breaker=self.circuit_breaker,
                timeout=300,
            )
        except Exception as e:
            raise LLMError(f"Ollama API request failed: {str(e)}", ErrorType.API_ERROR)

        end_time = time.time()

        # Extract response
        try:
            content = response_data["response"]

            # Extract token counts (if available)
            prompt_tokens = response_data.get("prompt_eval_count", 0)
            completion_tokens = response_data.get("eval_count", 0)
            total_tokens = prompt_tokens + completion_tokens

        except (KeyError, IndexError) as e:
            raise LLMError(
                f"Invalid response format from Ollama API: {e}",
                ErrorType.API_ERROR,
            ) from e

        # Cost is $0 for local inference (track tokens for metrics)
        total_cost = 0.0

        return LLMResponse(
            content=content,
            provider=LLMProvider.OLLAMA,
            model=model_config.model_name,
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            },
            cost=total_cost,
            response_time=end_time - start_time,
        )

    def _get_default_model_config(self) -> ModelConfig:
        """Get default model configuration for Ollama."""
        return ModelConfig(
            provider=LLMProvider.OLLAMA,
            model_name=self.default_model,
            model_size=ModelSize.MEDIUM,
            input_cost_per_1k=0.0,  # Local inference is free
            output_cost_per_1k=0.0,
            max_tokens=4096,
            temperature=0.7,
            top_p=0.9,
        )

    def calculate_cost(
        self, prompt_tokens: int, completion_tokens: int, model_config: ModelConfig
    ) -> float:
        """Cost is always $0 for local Ollama inference."""
        return 0.0

    async def get_embeddings(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Get embeddings using Ollama."""
        session = await self._get_session()
        url = f"{self.base_url}/api/embeddings"

        embeddings = []
        total_tokens = 0

        for text in request.texts:
            payload = {
                "model": request.model,  # e.g., "nomic-embed-text"
                "prompt": text,
            }

            try:
                response_data = await safe_aiohttp_request(
                    session=session,
                    method="POST",
                    url=url,
                    json=payload,
                    retry_config=self.retry_config,
                    circuit_breaker=self.circuit_breaker,
                    timeout=60,
                )

                embedding = response_data["embedding"]
                embeddings.append(embedding)

                # Estimate tokens (Ollama doesn't always return this)
                total_tokens += len(text.split()) * TOKEN_ESTIMATION_FACTOR

            except Exception as e:
                raise LLMError(
                    f"Ollama embedding request failed: {str(e)}",
                    ErrorType.API_ERROR
                )

        return EmbeddingResponse(
            embeddings=embeddings,
            model=request.model,
            usage={"total_tokens": int(total_tokens)},
            cost=0.0,  # Local inference is free
        )

    async def close(self) -> None:
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
```

#### 1.3 Update LLMManager to Support Provider Selection

**File**: `src/mad_spark_alt/core/llm_provider.py` (LLMManager class)

```python
class LLMManager:
    """Enhanced manager supporting multiple LLM providers."""

    def __init__(
        self,
        primary_provider: LLMProviderInterface,
        fallback_provider: Optional[LLMProviderInterface] = None,
    ):
        self.primary_provider = primary_provider
        self.fallback_provider = fallback_provider
        self.usage_stats: Dict[str, UsageStats] = {}

    async def generate(
        self,
        request: LLMRequest,
        use_fallback: bool = False
    ) -> LLMResponse:
        """Generate text using primary or fallback provider."""
        provider = self.fallback_provider if use_fallback else self.primary_provider

        # Validate multimodal constraints per provider
        if isinstance(provider, OllamaProvider):
            if request.multimodal_inputs or request.urls:
                logger.warning(
                    "Ollama does not support multimodal. "
                    "Falling back to Gemini if available."
                )
                if self.fallback_provider and isinstance(
                    self.fallback_provider, GoogleProvider
                ):
                    provider = self.fallback_provider
                else:
                    raise ValueError(
                        "Multimodal input requires GoogleProvider. "
                        "Configure fallback_provider or remove multimodal inputs."
                    )

        response = await provider.generate(request)

        # Track usage
        key = f"{response.provider.value}:{response.model}"
        if key not in self.usage_stats:
            self.usage_stats[key] = UsageStats(
                provider=response.provider,
                model=response.model
            )

        self.usage_stats[key].add_usage(
            response.usage["prompt_tokens"],
            response.usage["completion_tokens"],
            response.cost,
        )

        return response
```

### Phase 2: Configuration & Environment (Day 1-2) ✅

#### 2.1 Update .env Configuration

**File**: `.env`

```bash
# Google Gemini API
GOOGLE_API_KEY=your_gemini_key_here

# Ollama Configuration (NEW)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_DEFAULT_MODEL=gemma3:12b-it-qat

# Provider Selection (NEW)
# Options: "google", "ollama", "hybrid"
LLM_PROVIDER=hybrid

# Hybrid Mode Configuration (NEW)
# Which operations use which provider
HYBRID_SIMPLE_OPERATIONS=ollama  # mutations, fitness
HYBRID_COMPLEX_OPERATIONS=google  # synthesis, final answer
```

#### 2.2 Update pyproject.toml Dependencies

**File**: `pyproject.toml`

```toml
[project]
dependencies = [
    # ... existing dependencies ...
    "ollama>=0.1.0",  # NEW: Optional for higher-level API
]

[project.optional-dependencies]
ollama = [
    "ollama>=0.1.0",
]
```

### Phase 3: Orchestrator Integration (Day 2) ✅

#### 3.1 Add Provider Selection to SimpleQADIOrchestrator

**File**: `src/mad_spark_alt/core/simple_qadi_orchestrator.py`

```python
class SimpleQADIOrchestrator:
    """QADI orchestrator with multi-provider support."""

    def __init__(
        self,
        llm_manager: LLMManager,
        use_local_for_simple: bool = False,  # NEW
    ):
        self.llm_manager = llm_manager
        self.use_local_for_simple = use_local_for_simple

    async def _generate_hypotheses(self, core_question: str) -> List[str]:
        """
        Generate hypotheses.

        COST OPTIMIZATION: Use local model if enabled (simple task).
        """
        request = LLMRequest(
            user_prompt=f"Generate 3 hypotheses for: {core_question}",
            max_tokens=1000,
        )

        # Use local model for simple generation if configured
        response = await self.llm_manager.generate(
            request,
            use_fallback=not self.use_local_for_simple
        )

        # Parse hypotheses...
        return hypotheses
```

### Phase 4: CLI Integration (Day 2) ✅

#### 4.1 Update CLI Arguments

**File**: `src/mad_spark_alt/unified_cli.py`

```python
@click.command()
@click.argument("question", required=False)
@click.option("--provider", type=click.Choice(["google", "ollama", "hybrid"]),
              default="google", help="LLM provider to use")
@click.option("--ollama-url", default="http://localhost:11434",
              help="Ollama API base URL")
@click.option("--ollama-model", default="gemma3:12b-it-qat",
              help="Ollama model name")
@click.option("--evolve", is_flag=True, help="Enable evolution")
# ... other options ...
def main(question, provider, ollama_url, ollama_model, evolve, ...):
    """Mad Spark Alt - Multi-provider QADI system."""

    # Initialize providers
    google_provider = None
    ollama_provider = None

    if provider in ["google", "hybrid"]:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            click.echo("Error: GOOGLE_API_KEY not set", err=True)
            sys.exit(1)
        google_provider = GoogleProvider(api_key)

    if provider in ["ollama", "hybrid"]:
        ollama_provider = OllamaProvider(
            base_url=ollama_url,
            default_model=ollama_model
        )

        # Test Ollama connection
        if not await test_ollama_connection(ollama_url):
            click.echo(
                f"Error: Cannot connect to Ollama at {ollama_url}\\n"
                f"Make sure Ollama is running: ollama serve",
                err=True
            )
            sys.exit(1)

    # Create LLMManager
    if provider == "google":
        llm_manager = LLMManager(primary_provider=google_provider)
    elif provider == "ollama":
        llm_manager = LLMManager(primary_provider=ollama_provider)
    else:  # hybrid
        llm_manager = LLMManager(
            primary_provider=ollama_provider,  # Use Ollama by default
            fallback_provider=google_provider  # Fall back to Gemini
        )

    # Run QADI...
```

### Phase 5: Testing (Day 3) ✅

#### 5.1 Create OllamaProvider Tests

**File**: `tests/test_ollama_provider.py`

```python
"""Tests for OllamaProvider."""

import pytest
from unittest.mock import AsyncMock, patch

from mad_spark_alt.core.llm_provider import (
    OllamaProvider,
    LLMRequest,
    LLMProvider,
)

@pytest.fixture
def ollama_provider():
    return OllamaProvider(base_url="http://localhost:11434")

@pytest.mark.asyncio
async def test_ollama_generate_text(ollama_provider):
    """Test basic text generation with Ollama."""
    mock_response = {
        "response": "Hypothesis 1: Test hypothesis",
        "prompt_eval_count": 100,
        "eval_count": 50,
    }

    with patch("mad_spark_alt.core.llm_provider.safe_aiohttp_request") as mock:
        mock.return_value = mock_response

        request = LLMRequest(user_prompt="Generate a hypothesis")
        response = await ollama_provider.generate(request)

        assert response.provider == LLMProvider.OLLAMA
        assert response.cost == 0.0
        assert "Hypothesis" in response.content

@pytest.mark.asyncio
async def test_ollama_ignores_multimodal(ollama_provider):
    """Test that Ollama warns about and ignores multimodal inputs."""
    from mad_spark_alt.core.multimodal import MultimodalInput

    request = LLMRequest(
        user_prompt="Test",
        multimodal_inputs=[MultimodalInput(...)]  # Should be ignored
    )

    with patch("mad_spark_alt.core.llm_provider.safe_aiohttp_request"):
        with pytest.warns(UserWarning, match="does not support multimodal"):
            await ollama_provider.generate(request)

@pytest.mark.integration
@pytest.mark.asyncio
async def test_ollama_real_api():
    """Test with real Ollama API (requires Ollama running locally)."""
    provider = OllamaProvider()
    request = LLMRequest(user_prompt="Say hello in one word")

    response = await provider.generate(request)

    assert response.content
    assert response.cost == 0.0
    assert response.usage["total_tokens"] > 0
```

### Phase 6: Documentation (Day 3) ✅

#### 6.1 Create Ollama Setup Guide

**File**: `docs/OLLAMA_SETUP.md`

```markdown
# Ollama Local Inference Setup Guide

## Installation

### macOS / Linux
\`\`\`bash
curl -fsSL https://ollama.com/install.sh | sh
\`\`\`

### Windows
Download from https://ollama.com/download/windows

## Model Download

\`\`\`bash
# Recommended: Gemma 3 12B with Quantization
ollama pull gemma3:12b-it-qat  # 6-8GB VRAM

# Alternative: Standard Gemma 3 12B
ollama pull gemma3:12b  # 12GB VRAM

# Alternative: Smaller Gemma 3 4B
ollama pull gemma3:4b  # 4GB VRAM
\`\`\`

## Start Ollama Server

\`\`\`bash
ollama serve
\`\`\`

## Test Installation

\`\`\`bash
# Test with CLI
ollama run gemma3:12b-it-qat "Hello, how are you?"

# Test with API
curl http://localhost:11434/api/generate -d '{
  "model": "gemma3:12b-it-qat",
  "prompt": "Hello, how are you?",
  "stream": false
}'
\`\`\`

## Usage with Mad Spark Alt

\`\`\`bash
# Pure Ollama (text-only)
msa "Your question" --provider ollama

# Hybrid (Ollama for simple, Gemini for complex)
msa "Your question" --provider hybrid

# With custom model
msa "Your question" --provider ollama --ollama-model gemma3:12b
\`\`\`

## Limitations

- **No multimodal support**: Images, PDFs, URLs not supported in GGUF format
- **No schema enforcement**: JSON mode doesn't enforce Pydantic schemas
- **Local hardware required**: GPU recommended for good performance
- **Model size**: 6-12GB VRAM depending on quantization

## Performance Expectations

| Model | VRAM | Tokens/sec | Quality vs Gemini |
|-------|------|------------|-------------------|
| gemma3:4b | 4GB | ~50-80 | 70% |
| gemma3:12b-it-qat | 6-8GB | ~30-50 | 85% |
| gemma3:12b | 12GB | ~30-50 | 90% |

## Cost Comparison

| Scenario | Gemini Cost | Ollama Cost | Savings |
|----------|-------------|-------------|---------|
| 1000 QADI runs | $22 | $0 | 100% |
| 1000 Evolution runs | $12,000 | $0 | 100% |
| Hardware amortization | - | $1000-2000/year | - |

**Break-even**: ~50-100 QADI+Evolution runs per month

## Expected Benefits

### Cost Savings

**Current (Gemini only)**:
- 1000 runs/month × $0.022 = $22/month
- With 100 agents: $2,200/month

**With Ollama Hybrid (70% local, 30% Gemini)**:
- Local: 700 runs × $0 = $0
- Gemini: 300 runs × $0.022 = $6.60/month
- **Savings: 70% = $15.40/month**

**With 100 agents**:
- Gemini only: $2,200/month
- Hybrid: $660/month
- **Savings: $1,540/month = $18,480/year**

### Scaling

**Agents per hour (with rate limits)**:
- Gemini: ~60 requests/min = 3,600/hour
- Ollama (local): ~unlimited (hardware constrained)
- **10x-100x more agents possible with local inference**

## Next Steps

1. **Implement OllamaProvider** (follow plan above)
2. **Test with sample workload** (compare quality vs Gemini)
3. **Tune hybrid routing** (which operations use which provider)
4. **Benchmark performance** (tokens/sec, quality, cost)
5. **Scale gradually** (start 10% local, increase to 70%)
