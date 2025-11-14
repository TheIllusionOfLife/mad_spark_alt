"""
Semantic mutation operators for evolution.

This module implements LLM-powered mutation operators that create
semantically meaningful variations of ideas using different strategies
like perspective shifts and mechanism changes.
"""

import json
import logging
import random
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

from pydantic import ValidationError

from mad_spark_alt.core.interfaces import GeneratedIdea
from mad_spark_alt.core.llm_provider import GoogleProvider, LLMRequest
from mad_spark_alt.core.schemas import BatchMutationResponse
from mad_spark_alt.core.system_constants import CONSTANTS
from mad_spark_alt.evolution.interfaces import MutationInterface, EvaluationContext

from .operator_cache import SemanticOperatorCache
from .semantic_utils import (
    get_mutation_schema,
    is_likely_truncated,
    _prepare_operator_contexts,
    _prepare_cache_key_with_context,
)

logger = logging.getLogger(__name__)


class BatchSemanticMutationOperator(MutationInterface):
    """
    LLM-powered mutation operator with batch processing and caching.

    Creates semantically meaningful variations of ideas using different
    mutation strategies like perspective shifts and mechanism changes.
    """

    # Mutation prompt templates
    MUTATION_SYSTEM_PROMPT = """You are a genetic mutation operator for idea evolution.
Your role is to create meaningful variations of ideas while preserving the core goal.

IMPORTANT: Generate comprehensive, detailed implementations with specific steps, technologies, and methodologies.
When returning results, follow the exact format requested in the prompt (JSON or plain text)."""

    BREAKTHROUGH_SYSTEM_PROMPT = """You are an advanced breakthrough mutation operator for high-performing ideas.
Your role is to create REVOLUTIONARY variations that push beyond conventional boundaries while maintaining feasibility.

BREAKTHROUGH MODE: Push creative limits, explore cutting-edge technologies, and create game-changing innovations.
Generate comprehensive, detailed implementations with specific steps, technologies, and methodologies.
When returning results, follow the exact format requested in the prompt (JSON or plain text)."""

    BREAKTHROUGH_BATCH_PROMPT = """These are HIGH-PERFORMING ideas that deserve REVOLUTIONARY mutations.
Context: {context}
{evaluation_context}

High-performing ideas to revolutionize:
{ideas_list}

For each idea, create ONE breakthrough mutation that:
- Represents a PARADIGM SHIFT in approach
- Pushes beyond conventional boundaries
- Explores cutting-edge technologies and methodologies
- Creates transformational change while maintaining feasibility
- Provides REVOLUTIONARY implementation (minimum 200 words per variation)
- Dramatically improves target criteria mentioned above

Use one of these breakthrough mutation types for each idea:
- paradigm_shift: Completely reframe the problem and solution space
- system_integration: Connect with broader ecosystems for exponential impact
- scale_amplification: Transform local solutions into global movements
- future_forward: Leverage emerging technologies for next-generation solutions

Generate revolutionary, detailed solutions that include:
- Groundbreaking implementation strategies
- Cutting-edge technologies and innovations
- Transformational resource allocation
- Game-changing expected outcomes
- How it REVOLUTIONIZES the improvement of target criteria

Return mutations as JSON array with objects containing:
- "id": The index (0-based)
- "content": The revolutionary mutation (minimum 200 words)
- "mutation_type": The breakthrough type used"""

    SINGLE_MUTATION_PROMPT = """Original idea: {idea}
Problem context: {context}

{evaluation_context}

Create a semantically different variation that:
1. Addresses the same core problem
2. Uses a fundamentally different approach or mechanism
3. Maintains feasibility but explores new solution space
4. Provides DETAILED implementation with specific steps, technologies, and methodologies
5. Includes at least 150-200 words of comprehensive explanation
6. PRIORITIZES improvements to any target criteria mentioned above

Mutation type: {mutation_type}
- perspective_shift: Change viewpoint (individual→community, local→global, etc.)
- mechanism_change: Use different methods to achieve the same goal
- constraint_variation: Add or remove constraints
- abstraction_shift: Make more concrete or more abstract

Generate a complete, detailed solution that includes:
- Specific implementation steps
- Technologies or tools to be used
- Resources required
- Expected outcomes and benefits
- How it addresses the core problem
- Improvements to target criteria (if specified)

Return the mutated idea as JSON with the field "mutated_content" containing the detailed solution (minimum 150 words)."""

    BREAKTHROUGH_MUTATION_PROMPT = """BREAKTHROUGH MUTATION - High-Performance Idea Enhancement

Original high-scoring idea: {idea}
Problem context: {context}

{evaluation_context}

🚀 BREAKTHROUGH MODE: Create a REVOLUTIONARY variation that:
1. Leverages cutting-edge technologies and methodologies
2. Explores unconventional approaches beyond traditional solutions
3. Maintains core feasibility while pushing creative boundaries
4. Integrates multiple advanced systems for maximum impact
5. Provides COMPREHENSIVE implementation (200+ words)
6. MAXIMALLY IMPROVES target criteria through innovative approaches

BREAKTHROUGH mutation type: {mutation_type}
- paradigm_shift: Fundamentally new approach using emerging technologies
- system_integration: Combine multiple advanced systems for synergy
- scale_amplification: Dramatically increase scope and impact
- future_forward: Incorporate predictive and adaptive capabilities

Generate a REVOLUTIONARY, detailed solution that includes:
- Advanced technologies and cutting-edge methodologies
- Innovative integration of multiple systems
- Scalable architecture for maximum impact
- Future-proof design with adaptive capabilities
- Expected transformational outcomes
- How it REVOLUTIONIZES the target criteria improvement

Return the breakthrough mutation as JSON with "mutated_content" containing the revolutionary solution (minimum 200 words)."""

    BATCH_MUTATION_PROMPT = """Generate diverse variations for these ideas. Each variation should use a different approach.

Context: {context}

{evaluation_context}

Ideas to mutate:
{ideas_list}

For each idea, provide ONE variation that:
- Is semantically different but addresses the same goal
- Uses different mutation strategies: perspective shift, mechanism change, or abstraction level
- Provides DETAILED implementation (minimum 150 words per variation)
- Includes specific steps, technologies, methodologies, and resources
- PRIORITIZES improvements to any target criteria mentioned above

Generate complete, detailed solutions that include:
- Specific implementation steps
- Technologies or tools to be used
- Resources required
- Expected outcomes and benefits
- Improvements to target criteria (if specified)

Return JSON with mutations array containing id and content for each idea."""

    def __init__(
        self,
        llm_provider: GoogleProvider,
        cache_ttl: int = 3600
    ):
        """
        Initialize batch mutation operator.

        Args:
            llm_provider: LLM provider for generating mutations
            cache_ttl: Cache time-to-live in seconds
        """
        self.llm_provider = llm_provider
        self.cache = SemanticOperatorCache(ttl_seconds=cache_ttl)
        self.mutation_types = [
            "perspective_shift",
            "mechanism_change",
            "constraint_variation",
            "abstraction_shift"
        ]

        # Breakthrough mutation types for high-performing ideas
        self.breakthrough_mutation_types = [
            "paradigm_shift",
            "system_integration",
            "scale_amplification",
            "future_forward"
        ]

        # Threshold for determining if an idea qualifies for breakthrough mutation
        self.breakthrough_threshold = CONSTANTS.LLM.BREAKTHROUGH_FITNESS_THRESHOLD  # Ideas with high fitness get breakthrough mutations

    def _is_high_scoring_idea(self, idea: GeneratedIdea) -> bool:
        """
        Determine if an idea qualifies for breakthrough mutation based on fitness score.

        Args:
            idea: GeneratedIdea to evaluate

        Returns:
            True if idea qualifies for breakthrough mutation
        """
        # Check multiple sources for fitness indicators

        # 1. Check metadata for overall fitness score
        fitness = idea.metadata.get("overall_fitness")
        if fitness is not None and isinstance(fitness, (int, float)) and fitness >= self.breakthrough_threshold:
            return True

        # Also check "fitness" key used in tests
        fitness = idea.metadata.get("fitness")
        if fitness is not None and isinstance(fitness, (int, float)) and fitness >= self.breakthrough_threshold:
            return True

        # 2. Check confidence score as proxy (high confidence + later generation)
        if idea.confidence_score and idea.confidence_score >= CONSTANTS.LLM.BREAKTHROUGH_CONFIDENCE_THRESHOLD:
            generation = idea.metadata.get("generation", 0)
            if generation >= 1:  # Must be from evolution, not initial generation
                return True

        # 3. Check for explicit high-performance indicators
        if "high_performance" in idea.metadata:
            return bool(idea.metadata["high_performance"])

        return False

    @property
    def name(self) -> str:
        return "batch_semantic_mutation"

    def validate_config(self, config: Dict) -> bool:
        """Validate mutation configuration."""
        return True  # No specific validation needed

    async def mutate(
        self,
        idea: GeneratedIdea,
        mutation_rate: float,
        context: Union[Optional[str], EvaluationContext] = None
    ) -> GeneratedIdea:
        """
        Mutate a single idea using LLM.

        Note: When semantic operators are enabled, this method always applies
        mutation (ignores mutation_rate) since the decision to use semantic
        operators has already been made.

        Args:
            idea: Idea to mutate
            mutation_rate: Probability of mutation (ignored for semantic operators)
            context: Optional context for mutation

        Returns:
            Mutated idea
        """
        # Always apply mutation - semantic operators are used when available
        # and enabled, ignoring mutation_rate probability
        return await self.mutate_single(idea, context)

    async def mutate_single(
        self,
        idea: GeneratedIdea,
        context: Optional[Union[str, "EvaluationContext"]] = None
    ) -> GeneratedIdea:
        """
        Mutate a single idea with caching.

        Args:
            idea: Idea to mutate
            context: Optional context

        Returns:
            Mutated idea
        """
        # Determine if this is a high-scoring idea that qualifies for breakthrough mutation
        is_breakthrough = self._is_high_scoring_idea(idea)

        # Create cache key that includes EvaluationContext AND breakthrough status
        base_cache_key = _prepare_cache_key_with_context(idea.content, context)
        cache_key = f"{base_cache_key}||breakthrough:{is_breakthrough}"

        # Check cache first
        cached_result = self.cache.get(cache_key)
        cached_content = None
        cached_mutation_type = None

        if cached_result:
            # Extract content and original mutation type from cached data
            if isinstance(cached_result, dict):
                cached_content = cached_result.get("content")
                if not isinstance(cached_content, str):
                    # If content is missing or not a string, skip cache
                    cached_result = None
                else:
                    cached_mutation_type = cached_result.get("mutation_type")
            else:
                # For backward compatibility with string cache entries
                cached_content = cached_result
                cached_mutation_type = None

        if cached_result and cached_content is not None:

            # Use cached mutation type if available, otherwise select appropriate type
            if cached_mutation_type:
                mutation_type = cached_mutation_type
            else:
                mutation_type = random.choice(self.breakthrough_mutation_types if is_breakthrough else self.mutation_types)

            return self._create_mutated_idea(
                idea,
                cached_content,
                0.0,
                mutation_type,
                is_breakthrough
            )

        if is_breakthrough:
            # Use breakthrough mutation for high-scoring ideas
            mutation_type = random.choice(self.breakthrough_mutation_types)
            system_prompt = self.BREAKTHROUGH_SYSTEM_PROMPT
            user_prompt_template = self.BREAKTHROUGH_MUTATION_PROMPT
            temperature = CONSTANTS.LLM.BREAKTHROUGH_TEMPERATURE  # Higher temperature for breakthrough creativity
            max_tokens = CONSTANTS.LLM.SEMANTIC_MUTATION_MAX_TOKENS * CONSTANTS.LLM.BREAKTHROUGH_TOKEN_MULTIPLIER  # More tokens for detailed breakthrough solutions
        else:
            # Use regular mutation
            mutation_type = random.choice(self.mutation_types)
            system_prompt = self.MUTATION_SYSTEM_PROMPT
            user_prompt_template = self.SINGLE_MUTATION_PROMPT
            temperature = 0.8  # Standard temperature for creativity
            max_tokens = CONSTANTS.LLM.SEMANTIC_MUTATION_MAX_TOKENS

        # Create single mutation schema
        single_schema = {
            "type": "OBJECT",
            "properties": {
                "mutated_content": {"type": "STRING"}
            },
            "required": ["mutated_content"]
        }

        # Prepare context and evaluation context
        context_str, evaluation_context_str = _prepare_operator_contexts(
            context, idea.generation_prompt, "general improvement"
        )

        request = LLMRequest(
            system_prompt=system_prompt,
            user_prompt=user_prompt_template.format(
                idea=idea.content,
                context=context_str,
                evaluation_context=evaluation_context_str,
                mutation_type=mutation_type
            ),
            max_tokens=max_tokens,
            temperature=temperature,
            response_schema=single_schema,
            response_mime_type="application/json"
        )

        response = await self.llm_provider.generate(request)

        # Try to parse as JSON first (structured output)
        mutated_content: Optional[str] = None
        try:
            data = json.loads(response.content)
            if "mutated_content" in data:
                mutated_content = data["mutated_content"]
                logger.debug("Successfully parsed single mutation from structured output")
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.debug("Structured output parsing failed for single mutation, using raw content: %s", e)
            mutated_content = response.content

        # Ensure we have content (fallback to response.content if needed)
        if mutated_content is None:
            mutated_content = response.content

        # Check for truncation
        if is_likely_truncated(mutated_content):
            logger.warning("Mutation response appears truncated, may need higher token limit")

        # Cache the result with metadata using the same key structure
        cache_data = {
            "content": mutated_content,
            "mutation_type": mutation_type
        }
        self.cache.put(cache_key, cache_data)

        return self._create_mutated_idea(
            idea,
            mutated_content,
            response.cost,
            mutation_type,
            is_breakthrough
        )

    async def mutate_batch(
        self,
        ideas: List[GeneratedIdea],
        context: Union[Optional[str], EvaluationContext] = None
    ) -> List[GeneratedIdea]:
        """
        Mutate multiple ideas in a single LLM call.

        Args:
            ideas: List of ideas to mutate
            context: Optional context

        Returns:
            List of mutated ideas
        """
        # Separate cached and uncached ideas using context-aware cache keys
        cached_results = {}
        uncached_ideas = []
        cache_keys = {}

        for idea in ideas:
            # Include breakthrough status in cache key
            is_breakthrough = self._is_high_scoring_idea(idea)
            base_cache_key = _prepare_cache_key_with_context(idea.content, context)
            cache_key = f"{base_cache_key}||breakthrough:{is_breakthrough}"
            cache_keys[idea.content] = cache_key

            cached_result = self.cache.get(cache_key)
            if cached_result:
                cached_results[idea.content] = cached_result
            else:
                uncached_ideas.append(idea)

        # If all are cached, return immediately
        if not uncached_ideas:
            results = []
            for idea in ideas:
                cached_data = cached_results[idea.content]
                is_breakthrough = self._is_high_scoring_idea(idea)

                # Extract content and mutation type from cached data
                if isinstance(cached_data, dict):
                    cached_content = cached_data.get("content")
                    if not isinstance(cached_content, str):
                        # Fallback to original content if cache is invalid
                        cached_content = idea.content
                        logger.debug(f"Invalid cache entry for '{idea.content[:50]}...', using original content")
                    cached_mutation_type = cached_data.get("mutation_type", "batch_mutation")
                else:
                    # Backward compatibility
                    cached_content = cached_data
                    cached_mutation_type = "batch_mutation"

                results.append(
                    self._create_mutated_idea(
                        idea,
                        cached_content,
                        0.0,
                        cached_mutation_type,
                        is_breakthrough
                    )
                )
            return results

        # Batch process uncached ideas - SEPARATE into breakthrough and regular batches
        # Split ideas into breakthrough and regular groups
        breakthrough_ideas = []
        breakthrough_indices = []
        regular_ideas = []
        regular_indices = []

        for i, idea in enumerate(uncached_ideas):
            if self._is_high_scoring_idea(idea):
                breakthrough_ideas.append(idea)
                breakthrough_indices.append(i)
            else:
                regular_ideas.append(idea)
                regular_indices.append(i)

        # Process batches separately
        all_mutations: List[Optional[Dict[str, Any]]] = [None] * len(uncached_ideas)  # Preserve order
        total_cost = 0.0

        # Process breakthrough ideas with special parameters
        if breakthrough_ideas:
            logger.info(f"Processing {len(breakthrough_ideas)} breakthrough ideas with revolutionary parameters")

            # Create breakthrough batch prompt
            ideas_list = "\n".join([
                f"IDEA_{i+1}: {idea.content}"
                for i, idea in enumerate(breakthrough_ideas)
            ])

            # Prepare context and evaluation context
            context_str, evaluation_context_str = _prepare_operator_contexts(
                context, "", "revolutionary breakthrough"
            )

            # Create breakthrough schema with mutation type
            breakthrough_schema = {
                "type": "OBJECT",
                "properties": {
                    "mutations": {
                        "type": "ARRAY",
                        "items": {
                            "type": "OBJECT",
                            "properties": {
                                "id": {"type": "INTEGER"},
                                "content": {"type": "STRING"},
                                "mutation_type": {"type": "STRING"}
                            },
                            "required": ["id", "content", "mutation_type"]
                        }
                    }
                },
                "required": ["mutations"]
            }

            # Create breakthrough request
            request = LLMRequest(
                system_prompt=self.BREAKTHROUGH_SYSTEM_PROMPT,
                user_prompt=self.BREAKTHROUGH_BATCH_PROMPT.format(
                    context=context_str,
                    evaluation_context=evaluation_context_str,
                    ideas_list=ideas_list
                ),
                max_tokens=min(
                    CONSTANTS.LLM.SEMANTIC_BATCH_MUTATION_BASE_TOKENS * len(breakthrough_ideas) * CONSTANTS.LLM.BREAKTHROUGH_TOKEN_MULTIPLIER,
                    CONSTANTS.LLM.SEMANTIC_BATCH_MUTATION_MAX_TOKENS
                ),
                temperature=CONSTANTS.LLM.BREAKTHROUGH_TEMPERATURE,
                response_schema=breakthrough_schema,
                response_mime_type="application/json"
            )

            try:
                response = await self.llm_provider.generate(request)
                total_cost += response.cost

                # Parse breakthrough mutations
                breakthrough_mutations = self._parse_mutation_response(
                    response, len(breakthrough_ideas), breakthrough_ideas, is_breakthrough=True
                )

                # Place breakthrough mutations in correct positions
                for idx, (orig_idx, mutation_data) in enumerate(zip(breakthrough_indices, breakthrough_mutations)):
                    all_mutations[orig_idx] = mutation_data
            except Exception as e:
                logger.warning(f"Breakthrough batch processing failed: {e}, using fallback mutations")
                # Use fallback mutations for breakthrough ideas
                for idx, (orig_idx, idea) in enumerate(zip(breakthrough_indices, breakthrough_ideas)):
                    all_mutations[orig_idx] = self._create_fallback_mutation(idea, is_breakthrough=True)

        # Process regular ideas with standard parameters
        if regular_ideas:
            logger.info(f"Processing {len(regular_ideas)} regular ideas with standard parameters")

            # Create regular batch prompt
            ideas_list = "\n".join([
                f"IDEA_{i+1}: {idea.content}"
                for i, idea in enumerate(regular_ideas)
            ])

            # Prepare context and evaluation context
            context_str, evaluation_context_str = _prepare_operator_contexts(
                context, "", "general improvement"
            )

            # Create standard request
            schema = get_mutation_schema()
            request = LLMRequest(
                system_prompt=self.MUTATION_SYSTEM_PROMPT,
                user_prompt=self.BATCH_MUTATION_PROMPT.format(
                    context=context_str,
                    evaluation_context=evaluation_context_str,
                    ideas_list=ideas_list
                ),
                max_tokens=min(CONSTANTS.LLM.SEMANTIC_BATCH_MUTATION_BASE_TOKENS * len(regular_ideas), CONSTANTS.LLM.SEMANTIC_BATCH_MUTATION_MAX_TOKENS),
                temperature=CONSTANTS.LLM.REGULAR_MUTATION_TEMPERATURE,
                response_schema=schema,
                response_mime_type="application/json"
            )

            try:
                response = await self.llm_provider.generate(request)
                total_cost += response.cost

                # Parse regular mutations
                regular_mutations = self._parse_mutation_response(
                    response, len(regular_ideas), regular_ideas, is_breakthrough=False
                )

                # Place regular mutations in correct positions
                for idx, (orig_idx, mutation_data) in enumerate(zip(regular_indices, regular_mutations)):
                    all_mutations[orig_idx] = mutation_data
            except Exception as e:
                logger.warning(f"Regular batch processing failed: {e}, using fallback mutations")
                # Use fallback mutations for regular ideas
                for idx, (orig_idx, idea) in enumerate(zip(regular_indices, regular_ideas)):
                    all_mutations[orig_idx] = self._create_fallback_mutation(idea, is_breakthrough=False)

        # Check for truncation and cache results using context-aware keys
        for idea, cached_mutation_data in zip(uncached_ideas, all_mutations):
            if cached_mutation_data is not None:
                if is_likely_truncated(cached_mutation_data["content"]):
                    logger.warning("Batch mutation appears truncated for idea: %s...", idea.content[:50])
                # Use the context-aware cache key for this idea
                cache_key = cache_keys[idea.content]
                # Cache the mutation data including type
                self.cache.put(cache_key, cached_mutation_data)

        # Distribute cost across mutations
        cost_per_mutation = total_cost / len(uncached_ideas) if uncached_ideas else 0

        # Create result list maintaining original order
        results = []
        uncached_index = 0

        for idea in ideas:
            # Check if this idea qualifies for breakthrough
            is_breakthrough = self._is_high_scoring_idea(idea)

            if idea.content in cached_results:
                cached_data = cached_results[idea.content]

                # Extract content and mutation type from cached data
                if isinstance(cached_data, dict):
                    cached_content = cached_data.get("content")
                    if not isinstance(cached_content, str):
                        # Use original content if cache is invalid
                        cached_content = idea.content
                    cached_mutation_type = cached_data.get("mutation_type", "batch_mutation")
                else:
                    # Backward compatibility
                    cached_content = cached_data
                    cached_mutation_type = "batch_mutation"

                results.append(
                    self._create_mutated_idea(
                        idea,
                        cached_content,
                        0.0,
                        cached_mutation_type,
                        is_breakthrough
                    )
                )
            else:
                # Get the mutation data from all_mutations
                current_mutation_data = all_mutations[uncached_index]
                if current_mutation_data is not None:
                    results.append(
                        self._create_mutated_idea(
                            idea,
                            current_mutation_data["content"],
                            cost_per_mutation,
                            current_mutation_data["mutation_type"],
                            is_breakthrough
                        )
                    )
                else:
                    # Fallback if mutation data is missing
                    results.append(
                        self._create_mutated_idea(
                            idea,
                            idea.content,  # Use original content
                            0.0,
                            "fallback",
                            is_breakthrough
                        )
                    )
                uncached_index += 1

        return results

    def _parse_mutation_response(
        self,
        response: Any,
        expected_count: int,
        original_ideas: List[GeneratedIdea],
        is_breakthrough: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Parse mutation response from LLM.

        Args:
            response: LLM response object
            expected_count: Number of mutations expected
            original_ideas: Original ideas for fallback content
            is_breakthrough: Whether these are breakthrough mutations

        Returns:
            List of mutation data dictionaries with content and mutation_type
        """
        mutations: List[Dict[str, Any]] = []

        # Try Pydantic validation first (Phase 4)
        try:
            if hasattr(response, 'content'):
                response_text = response.content
            else:
                response_text = str(response)

            # Validate using Pydantic model
            validated_response = BatchMutationResponse.model_validate_json(response_text)

            # Initialize ordered mutations list with None placeholders
            ordered_mutations: List[Optional[Dict[str, Any]]] = [None] * expected_count

            # Process validated mutations with type-safe access
            for mutation_result in validated_response.mutations:
                # IDs are 1-based, convert to 0-based index
                idx = mutation_result.id - 1
                if 0 <= idx < expected_count:
                    ordered_mutations[idx] = {
                        "content": mutation_result.mutated_idea,
                        "mutation_type": mutation_result.mutation_type if mutation_result.mutation_type else (
                            "paradigm_shift" if is_breakthrough else "batch_mutation"
                        )
                    }
                else:
                    logger.warning(f"Mutation ID {mutation_result.id} out of range")

            # Fill any None entries with fallbacks
            for i in range(expected_count):
                if ordered_mutations[i] is None:
                    ordered_mutations[i] = self._create_fallback_mutation(
                        original_ideas[i] if i < len(original_ideas) else None,
                        is_breakthrough
                    )

            # Add ordered mutations to results, filtering out None
            for mutation in ordered_mutations:
                if mutation is not None:
                    mutations.append(mutation)

            # Fill any missing mutations with fallbacks
            while len(mutations) < expected_count:
                idx = len(mutations)
                mutations.append(self._create_fallback_mutation(
                    original_ideas[idx] if idx < len(original_ideas) else None,
                    is_breakthrough
                ))

            logger.debug("Successfully parsed mutation response with Pydantic validation")
            return [m for m in mutations if m is not None]

        except (ValidationError, json.JSONDecodeError) as e:
            logger.debug(f"Pydantic validation failed for mutation response, falling back to manual parsing: {e}")

        # Fall back to manual JSON parsing
        try:
            if hasattr(response, 'content'):
                data = json.loads(response.content)
            else:
                data = response

            if isinstance(data, dict) and "mutations" in data:
                raw_mutations = data["mutations"]
            elif isinstance(data, list):
                raw_mutations = data
            else:
                raise ValueError("Unexpected response format")

            # Process mutations - create a list with correct ordering based on ID
            # Initialize ordered mutations list with None placeholders
            fallback_mutations: List[Optional[Dict[str, Any]]] = [None] * expected_count

            for mutation in raw_mutations:
                if isinstance(mutation, dict) and "id" in mutation and "content" in mutation:
                    # IDs are 1-based, convert to 0-based index
                    idx = mutation["id"] - 1
                    if 0 <= idx < expected_count:
                        fallback_mutations[idx] = {
                            "content": mutation["content"],
                            "mutation_type": mutation.get("mutation_type",
                                "paradigm_shift" if is_breakthrough else "batch_mutation")
                        }
                    else:
                        logger.warning(f"Mutation ID {mutation['id']} out of range")

            # Fill any None entries with fallbacks
            for i in range(expected_count):
                if fallback_mutations[i] is None:
                    fallback_mutations[i] = self._create_fallback_mutation(
                        original_ideas[i] if i < len(original_ideas) else None,
                        is_breakthrough
                    )

            # Add ordered mutations to results, filtering out None
            for mutation in fallback_mutations:
                if mutation is not None:
                    mutations.append(mutation)

            # Fill any missing mutations with fallbacks
            while len(mutations) < expected_count:
                idx = len(mutations)
                mutations.append(self._create_fallback_mutation(
                    original_ideas[idx] if idx < len(original_ideas) else None,
                    is_breakthrough
                ))

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.debug(f"Failed to parse mutation response as JSON: {e}, falling back to text parsing")

            # Fall back to text parsing
            response_text = response.content if hasattr(response, 'content') else str(response)
            lines = response_text.strip().split('\n')

            for i in range(expected_count):
                # Look for IDEA_N_MUTATION: pattern
                pattern = f"IDEA_{i+1}_MUTATION:"
                mutation_found = False

                for line in lines:
                    if pattern in line:
                        # Extract mutation text after the pattern
                        content = line.split(pattern, 1)[1].strip()
                        mutations.append({
                            "content": content,
                            "mutation_type": "paradigm_shift" if is_breakthrough else "batch_mutation"
                        })
                        mutation_found = True
                        break

                if not mutation_found:
                    # Use fallback
                    mutations.append(self._create_fallback_mutation(
                        original_ideas[i] if i < len(original_ideas) else None,
                        is_breakthrough
                    ))

        # Return non-None mutations
        return [m for m in mutations if m is not None]

    def _create_fallback_mutation(
        self,
        original_idea: Optional[GeneratedIdea],
        is_breakthrough: bool
    ) -> Dict[str, Any]:
        """
        Create a fallback mutation when parsing fails.

        Args:
            original_idea: Original idea to base mutation on
            is_breakthrough: Whether this is a breakthrough mutation

        Returns:
            Dictionary with fallback mutation content and type
        """
        if is_breakthrough:
            mutation_type = "paradigm_shift"
            if original_idea:
                content = (
                    f"[FALLBACK TEXT] Revolutionary transformation of original concept: "
                    f"Taking '{original_idea.content[:50]}...' to new heights, this breakthrough "
                    f"mutation introduces paradigm-shifting approaches that fundamentally reframe "
                    f"the problem space. By leveraging cutting-edge technologies and revolutionary "
                    f"methodologies, this variation creates transformational change while maintaining "
                    f"feasibility. The approach integrates advanced systems thinking, exponential "
                    f"scaling strategies, and next-generation solutions to achieve unprecedented "
                    f"impact and dramatically improve target criteria."
                )
            else:
                content = (
                    "[FALLBACK TEXT] Revolutionary breakthrough variation: This paradigm-shifting "
                    "mutation explores transformational approaches that fundamentally reframe the "
                    "problem space and solution methodology. By integrating cutting-edge technologies, "
                    "exponential scaling strategies, and next-generation innovations, this variation "
                    "creates game-changing impact while maintaining practical feasibility. The approach "
                    "leverages advanced systems thinking and revolutionary frameworks to achieve "
                    "unprecedented improvements in target criteria."
                )
        else:
            mutation_type = "batch_mutation"
            if original_idea:
                content = (
                    f"[FALLBACK TEXT] Enhanced variation of original concept: Building upon "
                    f"'{original_idea.content[:50]}...', this mutation explores alternative "
                    f"implementation strategies while maintaining the core objective. The approach "
                    f"introduces different methodologies, tools, and frameworks to achieve similar "
                    f"outcomes through a varied pathway. By shifting perspectives on scale, audience, "
                    f"or technological approach, this variation demonstrates how the fundamental "
                    f"concept can be realized through innovative alternatives."
                )
            else:
                content = (
                    "[FALLBACK TEXT] Enhanced variation exploring alternative approaches: This mutation "
                    "investigates different implementation strategies while maintaining the core "
                    "objective of the original idea. The approach introduces alternative methodologies, "
                    "tools, and frameworks to achieve similar outcomes through a different pathway. "
                    "By exploring varied perspectives such as changing the scale of implementation, "
                    "shifting target audience, or adopting different technological foundations, this "
                    "variation demonstrates how the same fundamental problem can be addressed through "
                    "multiple viable and innovative solutions."
                )

        return {
            "content": content,
            "mutation_type": mutation_type
        }

    def _create_mutated_idea(
        self,
        original: GeneratedIdea,
        mutated_content: str,
        llm_cost: float,
        mutation_type: Optional[str] = None,
        is_breakthrough: bool = False
    ) -> GeneratedIdea:
        """Create a mutated idea object."""
        # Adjust confidence based on mutation type
        confidence_multiplier = CONSTANTS.LLM.BREAKTHROUGH_CONFIDENCE_MULTIPLIER if is_breakthrough else CONSTANTS.LLM.REGULAR_CONFIDENCE_MULTIPLIER
        base_confidence = original.confidence_score or 0.5
        new_confidence = min(1.0, base_confidence * confidence_multiplier)

        # Create enhanced metadata
        metadata = {
            "operator": "breakthrough_semantic_mutation" if is_breakthrough else "semantic_mutation",
            "mutation_type": mutation_type or "semantic_variation",
            "is_breakthrough": is_breakthrough,
            "llm_cost": llm_cost,
            "generation": original.metadata.get("generation", 0) + 1,
        }

        # Include original fitness information if available
        parent_fitness = original.metadata.get("overall_fitness")
        if parent_fitness is not None:
            metadata["parent_fitness"] = parent_fitness

        generation_description = "BREAKTHROUGH mutation" if is_breakthrough else "Semantic mutation"

        return GeneratedIdea(
            content=mutated_content,
            thinking_method=original.thinking_method,
            agent_name="BatchSemanticMutationOperator",
            generation_prompt=f"{generation_description} of: '{original.content[:50]}...'",
            confidence_score=new_confidence,
            reasoning=f"Applied {generation_description.lower()} to create {'revolutionary' if is_breakthrough else 'semantic'} variation",
            parent_ideas=[original.content],
            metadata=metadata,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
