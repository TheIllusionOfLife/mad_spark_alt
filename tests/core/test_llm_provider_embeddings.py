"""
Test suite for LLM provider embedding functionality.

This module tests the embedding request/response structures and the
GoogleProvider's embedding support.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import aiohttp
import json

from mad_spark_alt.core.llm_provider import (
    EmbeddingRequest, 
    EmbeddingResponse,
    GoogleProvider,
    ModelConfig,
    LLMProvider,
    ModelSize
)


class TestEmbeddingDataStructures:
    """Test embedding request and response structures."""
    
    def test_embedding_request_defaults(self):
        """Test EmbeddingRequest has correct defaults."""
        request = EmbeddingRequest(texts=["test text"])

        assert request.texts == ["test text"]
        assert request.model == "gemini-embedding-001"
        assert request.task_type == "SEMANTIC_SIMILARITY"
        assert request.output_dimensionality == 768
        assert request.title is None
        
    def test_embedding_request_custom_values(self):
        """Test EmbeddingRequest with custom values."""
        request = EmbeddingRequest(
            texts=["text1", "text2"],
            model="models/text-embedding-exp-03-07",
            task_type="CLUSTERING",
            output_dimensionality=1536,
            title="My embeddings"
        )
        
        assert request.texts == ["text1", "text2"]
        assert request.model == "models/text-embedding-exp-03-07"
        assert request.task_type == "CLUSTERING"
        assert request.output_dimensionality == 1536
        assert request.title == "My embeddings"
        
    def test_embedding_response_structure(self):
        """Test EmbeddingResponse structure."""
        response = EmbeddingResponse(
            embeddings=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            model="text-embedding-004",
            usage={"total_tokens": 100},
            cost=0.001
        )
        
        assert len(response.embeddings) == 2
        assert len(response.embeddings[0]) == 3
        assert response.model == "text-embedding-004"
        assert response.usage["total_tokens"] == 100
        assert response.cost == 0.001


class TestGoogleProviderEmbeddings:
    """Test GoogleProvider embedding support."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.api_key = "test-api-key"
        self.provider = GoogleProvider(api_key=self.api_key)
        
    @pytest.mark.asyncio
    async def test_get_embeddings_single_text(self):
        """Test getting embeddings for single text."""
        # Mock response
        mock_response = {
            "embeddings": [
                {
                    "values": [0.1, 0.2, 0.3]
                }
            ]
        }
        
        # Mock the safe_aiohttp_request function directly
        with patch('mad_spark_alt.core.llm_provider.safe_aiohttp_request', new=AsyncMock(return_value=mock_response)) as mock_request:
            request = EmbeddingRequest(texts=["hello world"])
            response = await self.provider.get_embeddings(request)
            
            # Verify API call
            mock_request.assert_called_once()
            call_args = mock_request.call_args
            
            # Check URL parameter
            assert "gemini-embedding-001:batchEmbedContents" in call_args[1]['url']
            
            # Check headers
            headers = call_args[1]['headers']
            assert headers['x-goog-api-key'] == self.api_key
            
            # Check request body - json parameter is used now, not data
            data = call_args[1]['json']
            assert data['requests'][0]['content']['parts'][0]['text'] == "hello world"
            assert data['requests'][0]['taskType'] == "SEMANTIC_SIMILARITY"
            assert data['requests'][0]['outputDimensionality'] == 768
            
            # Check response
            assert len(response.embeddings) == 1
            assert response.embeddings[0] == [0.1, 0.2, 0.3]
            assert response.model == "gemini-embedding-001"
            
    @pytest.mark.asyncio
    async def test_get_embeddings_batch(self):
        """Test getting embeddings for multiple texts."""
        # Mock response with multiple embeddings
        mock_response = {
            "embeddings": [
                {"values": [0.1, 0.2, 0.3]},
                {"values": [0.4, 0.5, 0.6]},
                {"values": [0.7, 0.8, 0.9]}
            ]
        }
        
        with patch('mad_spark_alt.core.llm_provider.safe_aiohttp_request', new=AsyncMock(return_value=mock_response)) as mock_request:
            request = EmbeddingRequest(
                texts=["text one", "text two", "text three"]
            )
            response = await self.provider.get_embeddings(request)
            
            # Check request body has all texts
            data = mock_request.call_args[1]['json']
            assert len(data['requests']) == 3
            assert data['requests'][0]['content']['parts'][0]['text'] == "text one"
            assert data['requests'][1]['content']['parts'][0]['text'] == "text two"
            assert data['requests'][2]['content']['parts'][0]['text'] == "text three"
            
            # Check response
            assert len(response.embeddings) == 3
            assert response.embeddings[0] == [0.1, 0.2, 0.3]
            assert response.embeddings[1] == [0.4, 0.5, 0.6]
            assert response.embeddings[2] == [0.7, 0.8, 0.9]
            
    @pytest.mark.asyncio
    async def test_get_embeddings_custom_dimensions(self):
        """Test embeddings with custom dimensions."""
        mock_response = {
            "embeddings": [
                {"values": [0.1] * 1536}  # 1536 dimensions
            ]
        }
        
        with patch('mad_spark_alt.core.llm_provider.safe_aiohttp_request', new=AsyncMock(return_value=mock_response)) as mock_request:
            request = EmbeddingRequest(
                texts=["test"],
                output_dimensionality=1536
            )
            response = await self.provider.get_embeddings(request)
            
            # Check request has correct dimensions
            data = mock_request.call_args[1]['json']
            assert data['requests'][0]['outputDimensionality'] == 1536
            
            # Check response
            assert len(response.embeddings[0]) == 1536
            
    @pytest.mark.asyncio
    async def test_get_embeddings_token_limit_enforcement(self):
        """Test that long texts are handled properly."""
        # Create a very long text (simulating >2048 tokens)
        long_text = " ".join(["word"] * 5000)  # Way over limit
        
        request = EmbeddingRequest(texts=[long_text])
        
        # For now, we'll just pass it through and let the API handle limits
        # In a real implementation, we might want to truncate or error
        mock_response = {
            "embeddings": [
                {"values": [0.1, 0.2, 0.3]}
            ]
        }
        
        with patch('mad_spark_alt.core.llm_provider.safe_aiohttp_request', new=AsyncMock(return_value=mock_response)) as mock_request:
            response = await self.provider.get_embeddings(request)
            
            # Should still work (API will handle truncation)
            assert len(response.embeddings) == 1
            
    @pytest.mark.asyncio
    async def test_get_embeddings_error_handling(self):
        """Test error handling for embedding requests."""
        # Mock safe_aiohttp_request to raise an exception
        from mad_spark_alt.core.retry import NetworkError
        
        with patch('mad_spark_alt.core.llm_provider.safe_aiohttp_request') as mock_request:
            mock_request.side_effect = NetworkError("Embedding API error 400: Bad request")
            
            request = EmbeddingRequest(texts=["test"])
            
            with pytest.raises(NetworkError) as exc_info:
                await self.provider.get_embeddings(request)
                
            assert "400" in str(exc_info.value)
            
    @pytest.mark.asyncio
    async def test_embedding_cost_calculation(self):
        """Test that embedding costs are calculated correctly."""
        mock_response = {
            "embeddings": [
                {"values": [0.1, 0.2, 0.3]}
            ],
            "usageMetadata": {
                "totalTokenCount": 10
            }
        }
        
        with patch('mad_spark_alt.core.llm_provider.safe_aiohttp_request', new=AsyncMock(return_value=mock_response)) as mock_request:
            request = EmbeddingRequest(texts=["hello world"])
            response = await self.provider.get_embeddings(request)
            
            # Check usage is tracked
            assert response.usage.get("total_tokens") == 10
            
            # Cost should be calculated (embedding costs are typically lower)
            # Exact cost depends on model pricing
            assert response.cost >= 0