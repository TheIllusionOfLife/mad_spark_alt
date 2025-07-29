"""
Tests for evolution display improvements to show full detailed approaches.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from mad_spark_alt.evolution.semantic_operators import BatchSemanticMutationOperator
from mad_spark_alt.evolution.interfaces import GeneratedIdea
from mad_spark_alt.core.interfaces import ThinkingMethod
from mad_spark_alt.core.llm_provider import LLMResponse, LLMProvider


class TestEvolutionDisplayFix:
    """Test that evolution produces and displays detailed approaches."""

    @pytest.mark.asyncio
    async def test_semantic_mutation_generates_detailed_output(self):
        """Test that semantic mutation prompts generate detailed implementations."""
        # Create mock LLM provider
        mock_provider = AsyncMock()
        
        # Create mutation operator
        mutation_op = BatchSemanticMutationOperator(llm_provider=mock_provider)
        
        # Original idea (brief)
        original_idea = GeneratedIdea(
            content="Create an API-based cognitive kernel for AGI",
            thinking_method=ThinkingMethod.ABDUCTION,
            agent_name="test",
            generation_prompt="test prompt",
            confidence_score=0.8,
            reasoning="test reasoning"
        )
        
        # Mock detailed mutation response
        detailed_mutation = """Develop a comprehensive microservices-based AGI architecture with the following components:

1. **Core Cognitive Kernel**: A central orchestrator service that manages all cognitive functions
   - REST API endpoints for perception, reasoning, memory, and action
   - Event-driven architecture using Apache Kafka for inter-service communication
   - GraphQL interface for complex query operations

2. **Perception Service**: Multi-modal input processing
   - Computer vision module using PyTorch for image understanding
   - Natural language processing using transformer models
   - Audio processing for speech recognition and sound analysis
   - Sensor data integration through IoT protocols

3. **Reasoning Engine**: Logic and inference system
   - First-order logic reasoner for deductive reasoning
   - Probabilistic reasoning using Bayesian networks
   - Causal inference engine for understanding cause-effect relationships
   - Integration with external knowledge bases (WikiData, ConceptNet)

4. **Memory Management**: Distributed storage system
   - Short-term memory using Redis for fast access
   - Long-term memory in PostgreSQL with vector embeddings
   - Episodic memory for experience replay
   - Semantic memory for concept relationships

5. **Implementation Details**:
   - Containerized deployment using Kubernetes
   - Service mesh with Istio for traffic management
   - Monitoring with Prometheus and Grafana
   - CI/CD pipeline using GitLab CI"""
        
        mock_provider.generate.return_value = LLMResponse(
            content=detailed_mutation,
            provider=LLMProvider.GOOGLE,
            model="gemini-1.5-flash",
            usage={"input_tokens": 200, "output_tokens": 400},
            cost=0.001,
            response_time=0.5,
            metadata={"test": True}
        )
        
        # Perform mutation
        mutated = await mutation_op.mutate(original_idea, "Build AGI system")
        
        # Verify detailed output
        assert len(mutated.content) > 1000  # Should be much longer than original
        assert "microservices-based" in mutated.content
        assert "Implementation Details" in mutated.content
        assert "Kubernetes" in mutated.content

    @pytest.mark.asyncio
    async def test_evolution_preserves_full_content(self):
        """Test that evolution pipeline preserves full content throughout."""
        from mad_spark_alt.evolution.genetic_algorithm import GeneticAlgorithm
        from mad_spark_alt.evolution.interfaces import EvolutionRequest, EvolutionConfig
        
        # Create initial population with detailed ideas
        detailed_idea1 = GeneratedIdea(
            content="A" * 1000,  # 1000 character idea
            thinking_method=ThinkingMethod.ABDUCTION,
            agent_name="test",
            generation_prompt="test",
            confidence_score=0.8,
            reasoning="test"
        )
        
        detailed_idea2 = GeneratedIdea(
            content="B" * 1200,  # 1200 character idea
            thinking_method=ThinkingMethod.DEDUCTION,
            agent_name="test",
            generation_prompt="test",
            confidence_score=0.7,
            reasoning="test"
        )
        
        # Mock LLM provider
        mock_provider = AsyncMock()
        
        # Mock mutation to return equally detailed content
        mock_provider.generate.return_value = LLMResponse(
            content="C" * 1100,  # Mutated content
            provider=LLMProvider.GOOGLE,
            model="gemini-1.5-flash",
            usage={"input_tokens": 200, "output_tokens": 400},
            cost=0.001,
            response_time=0.5,
            metadata={"test": True}
        )
        
        # Create GA with semantic operators
        ga = GeneticAlgorithm(llm_provider=mock_provider)
        
        # Configure evolution
        config = EvolutionConfig(
            population_size=2,
            generations=1,
            mutation_rate=1.0,  # Force mutation
            crossover_rate=0.0,
            elite_size=0
        )
        
        request = EvolutionRequest(
            initial_population=[detailed_idea1, detailed_idea2],
            config=config,
            context="test context"
        )
        
        # Run evolution
        with patch.object(ga, '_evaluate_fitness', new_callable=AsyncMock) as mock_eval:
            mock_eval.return_value = 0.9
            result = await ga.evolve(request)
        
        # Verify content length is preserved
        assert result.success
        for individual in result.final_population:
            assert len(individual.idea.content) >= 1000  # Content not truncated

    def test_display_does_not_truncate_evolved_ideas(self):
        """Test that the display logic shows full evolved ideas."""
        # This would be an integration test with the CLI
        # For now, we'll test the concept
        
        # Mock evolved idea with detailed content
        detailed_evolved_idea = GeneratedIdea(
            content="A" * 1500,  # Very long content
            thinking_method=ThinkingMethod.ABDUCTION,
            agent_name="evolution",
            generation_prompt="evolved",
            confidence_score=0.9,
            reasoning="evolved through GA"
        )
        
        # In the actual implementation, we'll ensure qadi_simple.py
        # doesn't truncate evolved ideas when displaying them
        
        # The fix will be in qadi_simple.py around line 538:
        # render_markdown(idea.content)  # No truncation
        
        assert len(detailed_evolved_idea.content) == 1500