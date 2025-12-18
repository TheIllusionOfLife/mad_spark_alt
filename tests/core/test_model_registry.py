"""
Unit tests for centralized model registry.

Tests cover:
- Model specification lookups
- Default model retrieval
- Provider filtering
- Registry immutability
- Edge cases and error handling
"""

import pytest

from mad_spark_alt.core.model_registry import (
    DEFAULT_MODELS,
    MODELS,
    ModelSpec,
    ProviderType,
    get_default_model,
    get_model_spec,
    get_models_by_provider,
)


class TestModelSpec:
    """Tests for ModelSpec dataclass."""

    def test_model_spec_is_frozen(self):
        """ModelSpec instances should be immutable."""
        spec = ModelSpec(
            id="test-model",
            provider=ProviderType.GEMINI,
            input_cost_per_1k=0.001,
            output_cost_per_1k=0.002,
        )
        with pytest.raises(AttributeError):
            spec.id = "modified"  # type: ignore[misc]

    def test_model_spec_default_values(self):
        """ModelSpec should have sensible defaults."""
        spec = ModelSpec(
            id="test-model",
            provider=ProviderType.GEMINI,
            input_cost_per_1k=0.001,
            output_cost_per_1k=0.002,
        )
        assert spec.max_output_tokens == 8192
        assert spec.token_multiplier == 1.0
        assert spec.supports_structured_output is True
        assert spec.supports_multimodal is False

    def test_model_spec_custom_values(self):
        """ModelSpec should accept custom values."""
        spec = ModelSpec(
            id="custom-model",
            provider=ProviderType.OLLAMA,
            input_cost_per_1k=0.0,
            output_cost_per_1k=0.0,
            max_output_tokens=16384,
            token_multiplier=2.5,
            supports_structured_output=False,
            supports_multimodal=True,
        )
        assert spec.id == "custom-model"
        assert spec.provider == ProviderType.OLLAMA
        assert spec.max_output_tokens == 16384
        assert spec.token_multiplier == 2.5
        assert spec.supports_structured_output is False
        assert spec.supports_multimodal is True


class TestGetModelSpec:
    """Tests for get_model_spec() function."""

    def test_get_existing_model(self):
        """Should return ModelSpec for existing model."""
        spec = get_model_spec("gemini-3-flash-preview")
        assert spec is not None
        assert spec.id == "gemini-3-flash-preview"
        assert spec.provider == ProviderType.GEMINI

    def test_get_legacy_model(self):
        """Should return ModelSpec for legacy model."""
        spec = get_model_spec("gemini-2.5-flash")
        assert spec is not None
        assert spec.id == "gemini-2.5-flash"
        assert spec.provider == ProviderType.GEMINI

    def test_get_ollama_model(self):
        """Should return ModelSpec for Ollama model."""
        spec = get_model_spec("gemma3:12b")
        assert spec is not None
        assert spec.id == "gemma3:12b"
        assert spec.provider == ProviderType.OLLAMA
        assert spec.input_cost_per_1k == 0.0  # Local inference is free

    def test_get_nonexistent_model(self):
        """Should return None for nonexistent model."""
        spec = get_model_spec("nonexistent-model")
        assert spec is None

    def test_get_empty_string_model(self):
        """Should return None for empty string."""
        spec = get_model_spec("")
        assert spec is None


class TestGetDefaultModel:
    """Tests for get_default_model() function."""

    def test_get_default_gemini_model(self):
        """Should return default Gemini model ID."""
        model_id = get_default_model(ProviderType.GEMINI)
        assert model_id == "gemini-3-flash-preview"

    def test_get_default_ollama_model(self):
        """Should return default Ollama model ID."""
        model_id = get_default_model(ProviderType.OLLAMA)
        assert model_id == "gemma3:12b"

    def test_default_model_exists_in_registry(self):
        """Default model should exist in MODELS registry."""
        for provider in ProviderType:
            model_id = get_default_model(provider)
            spec = get_model_spec(model_id)
            assert spec is not None, f"Default model '{model_id}' for {provider} not in registry"
            assert spec.provider == provider

    def test_all_providers_have_default_model(self):
        """All ProviderType values should have a default model configured."""
        for provider in ProviderType:
            # Should not raise KeyError
            model_id = get_default_model(provider)
            assert model_id, f"No default model for {provider}"


class TestGetModelsByProvider:
    """Tests for get_models_by_provider() function."""

    def test_get_gemini_models(self):
        """Should return all Gemini models."""
        models = get_models_by_provider(ProviderType.GEMINI)
        assert len(models) >= 2  # At least gemini-3-flash-preview and gemini-2.5-flash
        for spec in models:
            assert spec.provider == ProviderType.GEMINI

    def test_get_ollama_models(self):
        """Should return all Ollama models."""
        models = get_models_by_provider(ProviderType.OLLAMA)
        assert len(models) >= 1  # At least gemma3:12b
        for spec in models:
            assert spec.provider == ProviderType.OLLAMA

    def test_models_are_model_spec_instances(self):
        """Returned models should be ModelSpec instances."""
        models = get_models_by_provider(ProviderType.GEMINI)
        for spec in models:
            assert isinstance(spec, ModelSpec)


class TestRegistryImmutability:
    """Tests for registry immutability."""

    def test_models_registry_immutable(self):
        """MODELS registry should be immutable."""
        with pytest.raises(TypeError):
            MODELS["new-model"] = ModelSpec(  # type: ignore[index]
                id="new-model",
                provider=ProviderType.GEMINI,
                input_cost_per_1k=0.001,
                output_cost_per_1k=0.002,
            )

    def test_default_models_registry_immutable(self):
        """DEFAULT_MODELS registry should be immutable."""
        with pytest.raises(TypeError):
            DEFAULT_MODELS[ProviderType.GEMINI] = "different-model"  # type: ignore[index]


class TestRegistryIntegrity:
    """Tests for registry data integrity."""

    def test_all_models_have_valid_provider(self):
        """All models should have a valid ProviderType."""
        for model_id, spec in MODELS.items():
            assert isinstance(spec.provider, ProviderType)

    def test_all_models_have_non_negative_costs(self):
        """All models should have non-negative costs."""
        for model_id, spec in MODELS.items():
            assert spec.input_cost_per_1k >= 0, f"{model_id} has negative input cost"
            assert spec.output_cost_per_1k >= 0, f"{model_id} has negative output cost"

    def test_all_models_have_positive_max_tokens(self):
        """All models should have positive max_output_tokens."""
        for model_id, spec in MODELS.items():
            assert spec.max_output_tokens > 0, f"{model_id} has non-positive max_output_tokens"

    def test_all_models_have_positive_token_multiplier(self):
        """All models should have positive token_multiplier."""
        for model_id, spec in MODELS.items():
            assert spec.token_multiplier > 0, f"{model_id} has non-positive token_multiplier"

    def test_model_id_matches_dict_key(self):
        """Each model's spec.id should match its dictionary key."""
        for model_id, spec in MODELS.items():
            assert spec.id == model_id, f"Key '{model_id}' doesn't match spec.id '{spec.id}'"

    def test_all_default_models_exist_in_registry(self):
        """All default models should exist in MODELS registry."""
        for provider, model_id in DEFAULT_MODELS.items():
            assert model_id in MODELS, f"Default model '{model_id}' for {provider} not in MODELS"

    def test_default_model_provider_matches(self):
        """Default model's provider should match the key provider."""
        for provider, model_id in DEFAULT_MODELS.items():
            spec = MODELS[model_id]
            assert spec.provider == provider, (
                f"Default model '{model_id}' has provider {spec.provider}, expected {provider}"
            )


class TestGemini3FlashPricing:
    """Tests for Gemini 3 Flash specific pricing."""

    def test_gemini_3_flash_pricing(self):
        """Gemini 3 Flash should have correct pricing."""
        spec = get_model_spec("gemini-3-flash-preview")
        assert spec is not None
        # $0.50 per million = $0.0005 per 1K
        assert spec.input_cost_per_1k == 0.0005
        # $3.00 per million = $0.003 per 1K
        assert spec.output_cost_per_1k == 0.003

    def test_gemini_3_flash_capabilities(self):
        """Gemini 3 Flash should have correct capabilities."""
        spec = get_model_spec("gemini-3-flash-preview")
        assert spec is not None
        assert spec.max_output_tokens == 65536  # 64k output context
        assert spec.token_multiplier == 3.0  # Reasoning overhead
        assert spec.supports_structured_output is True
        assert spec.supports_multimodal is True

    def test_gemini_3_flash_is_default(self):
        """Gemini 3 Flash should be the default Gemini model."""
        default = get_default_model(ProviderType.GEMINI)
        assert default == "gemini-3-flash-preview"
