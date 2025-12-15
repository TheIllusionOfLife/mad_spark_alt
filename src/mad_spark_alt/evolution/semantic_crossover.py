"""
Semantic crossover operators for evolution.

This module implements LLM-powered crossover operators that meaningfully
combine concepts from two parent ideas into offspring with synergy.
"""

import json
import logging
from datetime import datetime, timezone
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import ValidationError

from mad_spark_alt.core.interfaces import GeneratedIdea
from mad_spark_alt.core.llm_provider import GoogleProvider, LLMRequest
from mad_spark_alt.core.schemas import CrossoverResponse
from mad_spark_alt.core.system_constants import CONSTANTS
from mad_spark_alt.evolution.interfaces import CrossoverInterface, EvaluationContext

from .operator_cache import SemanticOperatorCache
from .semantic_utils import (
    generate_crossover_fallback_text,
    get_crossover_schema,
    is_likely_truncated,
    _prepare_operator_contexts,
    _prepare_cache_key_with_context,
)

logger = logging.getLogger(__name__)


class SemanticCrossoverOperator(CrossoverInterface):
    """
    LLM-powered crossover operator that meaningfully combines parent ideas.

    Uses LLM to understand key concepts from both parents and create
    offspring that integrate these concepts synergistically.
    """

    CROSSOVER_SYSTEM_PROMPT = """You are a genetic crossover operator for idea evolution.
Your role is to meaningfully combine concepts from two parent ideas into offspring.

IMPORTANT: Generate comprehensive, detailed implementations with specific steps, technologies, and methodologies.
When returning results, follow the exact format requested in the prompt (JSON or plain text)."""

    CROSSOVER_PROMPT = """First Approach: {parent1}
Second Approach: {parent2}
Context: {context}

{evaluation_context}

Analyze the key concepts in each parent and create TWO offspring ideas that:
1. Meaningfully integrate concepts from BOTH parents
2. Are not just concatenations or word swaps
3. Create synergy between the parent concepts
4. Maintain coherence and feasibility
5. Provide DETAILED implementation (minimum 150 words per offspring)
6. Include specific steps, technologies, methodologies, and resources
7. CRITICAL: Each offspring must be SUBSTANTIALLY DIFFERENT from the other
8. PRIORITIZES improvements to any target criteria mentioned above

IMPORTANT: The two offspring MUST take different approaches to combining the parent concepts:
- Offspring 1: Focus on how the first approach's strengths can enhance the second approach
- Offspring 2: Focus on how the second approach's strengths can enhance the first approach
- Ensure NO shared sentences or paragraphs between offspring
- Use different implementation strategies, technologies, and methodologies

Generate complete, detailed solutions that include:
- Specific implementation steps combining elements from both parents
- Technologies or tools from both approaches
- Resources required for the hybrid solution
- Expected outcomes showing synergy
- How it leverages strengths of both parent ideas
- Improvements to target criteria (if specified)

Return two detailed offspring ideas as JSON with offspring_1 and offspring_2 fields."""

    def __init__(
        self,
        llm_provider: GoogleProvider,
        cache_ttl: int = 3600
    ):
        """
        Initialize semantic crossover operator.

        Args:
            llm_provider: LLM provider for generating crossovers
            cache_ttl: Cache time-to-live in seconds
        """
        self.llm_provider = llm_provider
        self.cache = SemanticOperatorCache(ttl_seconds=cache_ttl)

    @property
    def name(self) -> str:
        return "semantic_crossover"

    def validate_config(self, config: Dict) -> bool:
        """Validate crossover configuration."""
        return True

    async def crossover(
        self,
        parent1: GeneratedIdea,
        parent2: GeneratedIdea,
        context: Optional[Union[str, EvaluationContext]] = None
    ) -> Tuple[GeneratedIdea, GeneratedIdea]:
        """
        Perform semantic crossover between two parent ideas.

        Args:
            parent1: First parent idea
            parent2: Second parent idea
            context: Optional context for crossover

        Returns:
            Tuple of two offspring ideas
        """
        # Create a canonical, order-independent cache key from sorted parent content
        # Create cache key that includes EvaluationContext for context-aware caching
        sorted_contents = sorted([parent1.content, parent2.content])
        base_cache_key = f"{sorted_contents[0]}||{sorted_contents[1]}"
        cache_key = _prepare_cache_key_with_context(base_cache_key, context)

        cached_result = self.cache.get(cache_key, operation_type="crossover", return_dict=False)

        if cached_result:
            # Parse cached result
            offspring_contents = cached_result.split("||") if isinstance(cached_result, str) else []
            if len(offspring_contents) >= 2:
                return (
                    self._create_offspring(parent1, parent2, offspring_contents[0], 0.0),
                    self._create_offspring(parent2, parent1, offspring_contents[1], 0.0)
                )

        # Prepare context and evaluation context for crossover
        context_str, evaluation_context_str = _prepare_operator_contexts(
            context, "", "general optimization"
        )

        # Generate crossover using LLM with structured output
        schema = get_crossover_schema()
        request = LLMRequest(
            system_prompt=self.CROSSOVER_SYSTEM_PROMPT,
            user_prompt=self.CROSSOVER_PROMPT.format(
                parent1=parent1.content,
                parent2=parent2.content,
                context=context_str,
                evaluation_context=evaluation_context_str
            ),
            max_tokens=CONSTANTS.LLM.SEMANTIC_CROSSOVER_MAX_TOKENS,
            temperature=0.7,  # Moderate temperature for balanced creativity
            response_schema=schema,
            response_mime_type="application/json"
        )

        response = await self.llm_provider.generate(request)

        # Try Pydantic validation first (Phase 4)
        offspring1_content = None
        offspring2_content = None

        try:
            # Validate using Pydantic model
            validated_response = CrossoverResponse.model_validate_json(response.content)
            offspring1_content = validated_response.offspring1
            offspring2_content = validated_response.offspring2
            logger.debug("Successfully parsed crossover with Pydantic validation")
        except (ValidationError, json.JSONDecodeError) as e:
            logger.debug("Pydantic validation failed for crossover, falling back to manual parsing: %s", e)

            # Fall back to manual JSON parsing
            try:
                data = json.loads(response.content)
                if "offspring_1" in data and "offspring_2" in data:
                    offspring1_content = data["offspring_1"]
                    offspring2_content = data["offspring_2"]
                    logger.debug("Successfully parsed crossover from manual JSON parsing")
                else:
                    raise KeyError("offspring_1 or offspring_2 not found in JSON")
            except (json.JSONDecodeError, KeyError, TypeError) as e_manual:
                logger.debug("Manual JSON parsing failed, falling back to text parsing: %s", e_manual)
                # Fall back to text parsing
                offspring1_content, offspring2_content = self._parse_crossover_response(
                    response.content, parent1, parent2
                )

        # Check for truncation
        if is_likely_truncated(offspring1_content):
            logger.warning("Offspring 1 appears truncated, may need higher token limit")
        if is_likely_truncated(offspring2_content):
            logger.warning("Offspring 2 appears truncated, may need higher token limit")

        # Check for excessive duplication between offspring
        similarity = self._calculate_similarity(offspring1_content, offspring2_content)
        if similarity > CONSTANTS.SIMILARITY.CROSSOVER_THRESHOLD:  # Similar parents produce similar offspring
            logger.warning(f"High similarity detected between offspring: {similarity:.2f}")
            # Use fallback generation for more diverse offspring
            offspring1_content = self._generate_crossover_fallback(parent1, parent2, is_first=True)
            offspring2_content = self._generate_crossover_fallback(parent1, parent2, is_first=False)

        # Cache the result (as string for backward compatibility)
        self.cache.put(cache_key, f"{offspring1_content}||{offspring2_content}", operation_type="crossover")

        # Distribute cost
        cost_per_offspring = response.cost / 2

        return (
            self._create_offspring(parent1, parent2, offspring1_content, cost_per_offspring),
            self._create_offspring(parent2, parent1, offspring2_content, cost_per_offspring)
        )

    def _parse_crossover_response(self, response: str, parent1: Optional[GeneratedIdea] = None, parent2: Optional[GeneratedIdea] = None) -> Tuple[str, str]:
        """
        Parse crossover response from LLM.

        Args:
            response: LLM response text
            parent1: First parent idea (for fallback)
            parent2: Second parent idea (for fallback)

        Returns:
            Tuple of two offspring contents
        """
        lines = response.strip().split('\n')
        offspring1 = None
        offspring2 = None

        for line in lines:
            if "OFFSPRING_1:" in line:
                offspring1 = line.split("OFFSPRING_1:", 1)[1].strip()
            elif "OFFSPRING_2:" in line:
                offspring2 = line.split("OFFSPRING_2:", 1)[1].strip()

        # Fallback if parsing fails - create meaningful combinations based on parent content
        if not offspring1:
            logger.debug("Using fallback text for offspring 1 - LLM parsing failed")
            offspring1 = self._generate_crossover_fallback(parent1, parent2, is_first=True)
        if not offspring2:
            logger.debug("Using fallback text for offspring 2 - LLM parsing failed")
            offspring2 = self._generate_crossover_fallback(parent1, parent2, is_first=False)

        return offspring1, offspring2

    def _calculate_similarity(self, content1: str, content2: str) -> float:
        """
        Calculate similarity ratio between two content strings.

        Args:
            content1: First content string
            content2: Second content string

        Returns:
            Similarity ratio between 0.0 and 1.0
        """
        # Normalize content for comparison
        norm1 = content1.strip().lower()
        norm2 = content2.strip().lower()

        # Calculate similarity ratio
        return SequenceMatcher(None, norm1, norm2).ratio()

    def _generate_crossover_fallback(
        self,
        parent1: Optional[GeneratedIdea],
        parent2: Optional[GeneratedIdea],
        is_first: bool
    ) -> str:
        """
        Generate fallback text for crossover offspring.

        Args:
            parent1: First parent idea
            parent2: Second parent idea
            is_first: Whether this is the first offspring

        Returns:
            Fallback text for offspring
        """
        return generate_crossover_fallback_text(parent1, parent2, is_first)

    def _create_offspring(
        self,
        primary_parent: GeneratedIdea,
        secondary_parent: GeneratedIdea,
        content: str,
        llm_cost: float
    ) -> GeneratedIdea:
        """Create an offspring idea object."""
        return GeneratedIdea(
            content=content,
            thinking_method=primary_parent.thinking_method,
            agent_name="SemanticCrossoverOperator",
            generation_prompt=f"Crossover of: '{primary_parent.content[:30]}...' and '{secondary_parent.content[:30]}...'",
            confidence_score=(
                (primary_parent.confidence_score or 0.5) +
                (secondary_parent.confidence_score or 0.5)
            ) / 2,
            reasoning="Semantic integration of parent concepts",
            parent_ideas=[primary_parent.content, secondary_parent.content],
            metadata={
                "operator": "semantic_crossover",
                "llm_cost": llm_cost,
                "generation": max(
                    primary_parent.metadata.get("generation", 0),
                    secondary_parent.metadata.get("generation", 0)
                ) + 1,
            },
            timestamp=datetime.now(timezone.utc).isoformat(),
        )


class BatchSemanticCrossoverOperator(CrossoverInterface):
    """
    Batch LLM-powered crossover operator that processes multiple parent pairs in a single call.

    This operator significantly improves performance by batching all crossover operations
    from a generation into a single LLM request, reducing API calls and improving consistency.
    """

    def __init__(
        self,
        llm_provider: GoogleProvider,
        cache_ttl: int = 3600
    ):
        """
        Initialize batch semantic crossover operator.

        Args:
            llm_provider: LLM provider for generating crossovers
            cache_ttl: Cache time-to-live in seconds
        """
        self.llm_provider = llm_provider
        self.cache = SemanticOperatorCache(ttl_seconds=cache_ttl)
        self._sequential_operator = SemanticCrossoverOperator(llm_provider, cache_ttl)
        self.structured_output_enabled = True

    @property
    def name(self) -> str:
        return "batch_semantic_crossover"

    def validate_config(self, config: Dict) -> bool:
        """Validate crossover configuration."""
        return True

    async def crossover(
        self,
        parent1: GeneratedIdea,
        parent2: GeneratedIdea,
        context: Optional[Union[str, EvaluationContext]] = None
    ) -> Tuple[GeneratedIdea, GeneratedIdea]:
        """
        Perform crossover between two parent ideas.

        This method implements the abstract method from CrossoverInterface
        by delegating to the batch processing method for a single pair.

        Args:
            parent1: First parent idea
            parent2: Second parent idea
            context: Optional context for crossover

        Returns:
            Tuple of two offspring ideas
        """
        # Use batch processing for single pair
        results = await self.crossover_batch([(parent1, parent2)], context)
        return results[0] if results else (parent1, parent2)

    async def crossover_batch(
        self,
        parent_pairs: List[Tuple[GeneratedIdea, GeneratedIdea]],
        context: Optional[Union[str, EvaluationContext]] = None
    ) -> List[Tuple[GeneratedIdea, GeneratedIdea]]:
        """
        Perform batch semantic crossover on multiple parent pairs.

        Args:
            parent_pairs: List of (parent1, parent2) tuples
            context: Optional context for crossover

        Returns:
            List of (offspring1, offspring2) tuples
        """
        if not parent_pairs:
            return []

        # Check cache for each pair and separate cached/uncached
        cached_results = []
        uncached_pairs = []
        uncached_indices = []

        for i, (parent1, parent2) in enumerate(parent_pairs):
            # Create canonical cache key
            sorted_contents = sorted([parent1.content, parent2.content])
            base_cache_key = f"{sorted_contents[0]}||{sorted_contents[1]}"
            cache_key = _prepare_cache_key_with_context(base_cache_key, context)

            cached_result = self.cache.get(cache_key, operation_type="batch_crossover", return_dict=True)

            if cached_result and isinstance(cached_result, dict):
                # Parse cached result
                offspring1_content = cached_result.get("offspring1", "")
                offspring2_content = cached_result.get("offspring2", "")
                if offspring1_content and offspring2_content:
                    cached_results.append((
                        self._create_offspring(parent1, parent2, offspring1_content, 0.0, from_cache=True),
                        self._create_offspring(parent2, parent1, offspring2_content, 0.0, from_cache=True)
                    ))
                    continue

            # Not cached, add to uncached list
            uncached_pairs.append((parent1, parent2))
            uncached_indices.append(i)

        # If all are cached, return cached results
        if not uncached_pairs:
            return cached_results

        # Prepare context for batch processing
        context_str, evaluation_context_str = _prepare_operator_contexts(
            context, "", "general optimization"
        )

        # Generate batch crossover prompt
        batch_prompt = self._create_batch_prompt(uncached_pairs, context_str, evaluation_context_str)

        try:
            # Use structured output for batch crossover
            schema = self._get_batch_crossover_schema()
            request = LLMRequest(
                system_prompt=SemanticCrossoverOperator.CROSSOVER_SYSTEM_PROMPT,
                user_prompt=batch_prompt,
                max_tokens=min(CONSTANTS.LLM.SEMANTIC_CROSSOVER_MAX_TOKENS * len(uncached_pairs), CONSTANTS.LLM.SEMANTIC_BATCH_MUTATION_MAX_TOKENS),
                temperature=CONSTANTS.LLM.REGULAR_MUTATION_TEMPERATURE,
                response_schema=schema,
                response_mime_type="application/json"
            )

            result = await self.llm_provider.generate(request)
            llm_cost = result.cost if hasattr(result, 'cost') else 0.0

            # Parse batch results
            # Initialize results list with None placeholders to maintain order
            batch_results: List[Optional[Tuple[GeneratedIdea, GeneratedIdea]]] = [None] * len(uncached_pairs)
            # Parse JSON from response content
            try:
                data = json.loads(result.content)
                logger.debug(f"Parsed data type: {type(data)}, keys: {data.keys() if isinstance(data, dict) else 'not a dict'}")
                if isinstance(data, dict) and "crossovers" in data:
                    crossovers = data["crossovers"]

                    # Validate we got the right number of results
                    if len(crossovers) != len(uncached_pairs):
                        logger.warning(f"Expected {len(uncached_pairs)} crossovers but got {len(crossovers)}")

                    # Process each crossover result using pair_id for correct mapping
                    for crossover_data in crossovers:
                        pair_id = crossover_data.get("pair_id")
                        if pair_id is None:
                            logger.warning("Missing pair_id in crossover result")
                            continue

                        # pair_id is 1-based as per the prompt
                        idx = pair_id - 1
                        if idx < 0 or idx >= len(uncached_pairs):
                            logger.warning(f"Crossover pair_id {pair_id} out of range")
                            continue

                        parent1, parent2 = uncached_pairs[idx]
                        offspring1_content = str(crossover_data.get("offspring1", ""))
                        offspring2_content = str(crossover_data.get("offspring2", ""))

                        # Validate offspring content
                        if not offspring1_content or not offspring2_content:
                            logger.warning(f"Empty offspring content for pair_id {pair_id}")
                            # Use fallback
                            offspring1_content = self._generate_fallback_offspring(parent1, parent2, True)
                            offspring2_content = self._generate_fallback_offspring(parent1, parent2, False)

                        # Create offspring ideas
                        offspring1 = self._create_offspring(parent1, parent2, offspring1_content, llm_cost / len(uncached_pairs))
                        offspring2 = self._create_offspring(parent2, parent1, offspring2_content, llm_cost / len(uncached_pairs))

                        # Place result at correct index
                        batch_results[idx] = (offspring1, offspring2)

                        # Cache the result
                        sorted_contents = sorted([parent1.content, parent2.content])
                        base_cache_key = f"{sorted_contents[0]}||{sorted_contents[1]}"
                        cache_key = _prepare_cache_key_with_context(base_cache_key, context)
                        cache_data = {
                            "offspring1": offspring1_content,
                            "offspring2": offspring2_content
                        }
                        self.cache.put(cache_key, cache_data, operation_type="batch_crossover")

                    # Check for any None values in batch_results (missing pair_ids)
                    for i, batch_result in enumerate(batch_results):
                        if batch_result is None:
                            parent1, parent2 = uncached_pairs[i]
                            logger.warning(f"Missing result for pair {i+1}, using fallback")
                            offspring1_content = self._generate_fallback_offspring(parent1, parent2, True)
                            offspring2_content = self._generate_fallback_offspring(parent1, parent2, False)
                            offspring1 = self._create_offspring(parent1, parent2, offspring1_content, 0.0)
                            offspring2 = self._create_offspring(parent2, parent1, offspring2_content, 0.0)
                            batch_results[i] = (offspring1, offspring2)

                else:
                    raise ValueError("Invalid response format - missing crossovers")

            except (json.JSONDecodeError, KeyError, ValueError) as e:
                # Fallback to sequential processing if batch fails
                logger.warning(f"Batch crossover failed: {e}, falling back to sequential processing")
                batch_results = []
                for parent1, parent2 in uncached_pairs:
                    offspring1, offspring2 = await self._sequential_operator.crossover(parent1, parent2, context)
                    batch_results.append((offspring1, offspring2))

        except Exception as e:
            logger.error(f"Batch crossover error: {e}")
            # Fallback to sequential processing
            batch_results = []
            for parent1, parent2 in uncached_pairs:
                try:
                    offspring1, offspring2 = await self._sequential_operator.crossover(parent1, parent2, context)
                    batch_results.append((offspring1, offspring2))
                except Exception as seq_error:
                    logger.error(f"Sequential crossover also failed: {seq_error}")
                    # Use fallback content
                    offspring1_content = self._generate_fallback_offspring(parent1, parent2, True)
                    offspring2_content = self._generate_fallback_offspring(parent1, parent2, False)
                    offspring1 = self._create_offspring(parent1, parent2, offspring1_content, 0.0)
                    offspring2 = self._create_offspring(parent2, parent1, offspring2_content, 0.0)
                    batch_results.append((offspring1, offspring2))

        # Combine cached and new results in original order
        final_results: List[Tuple[GeneratedIdea, GeneratedIdea]] = []
        cached_idx = 0
        batch_idx = 0

        for i in range(len(parent_pairs)):
            if i in uncached_indices:
                crossover_result = batch_results[batch_idx]
                if crossover_result is not None:
                    final_results.append(crossover_result)
                batch_idx += 1
            else:
                final_results.append(cached_results[cached_idx])
                cached_idx += 1

        return final_results

    def _create_batch_prompt(
        self,
        parent_pairs: List[Tuple[GeneratedIdea, GeneratedIdea]],
        context_str: str,
        evaluation_context_str: str
    ) -> str:
        """Create batch crossover prompt for multiple parent pairs."""
        pairs_text = ""
        for i, (parent1, parent2) in enumerate(parent_pairs):
            pairs_text += f"\nPair {i + 1}:\nParent 1: {parent1.content}\nParent 2: {parent2.content}\n"

        return f"""Context: {context_str}

{evaluation_context_str}

Generate crossover offspring for the following {len(parent_pairs)} parent pairs.
For each pair, create TWO distinct offspring that meaningfully integrate concepts from BOTH parents.

{pairs_text}

For each pair, the offspring must:
1. Meaningfully integrate concepts from BOTH parents
2. Create synergy between parent concepts
3. Provide DETAILED implementation (minimum 150 words per offspring)
4. Be SUBSTANTIALLY DIFFERENT from each other
5. PRIORITIZE improvements to any target criteria mentioned above

Return the results as JSON with a "crossovers" array containing objects with:
- "pair_id": index of the parent pair (1-based, matching Pair 1, Pair 2, etc.)
- "offspring1": detailed first offspring (min 150 words)
- "offspring2": detailed second offspring (min 150 words)"""

    def _get_batch_crossover_schema(self) -> Dict[str, Any]:
        """Get JSON schema for batch crossover structured output.

        Uses standard JSON Schema format (lowercase types) for Ollama compatibility.
        Google Gemini also accepts lowercase types.
        """
        return {
            "type": "object",
            "properties": {
                "crossovers": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "pair_id": {"type": "integer"},
                            "offspring1": {"type": "string"},
                            "offspring2": {"type": "string"}
                        },
                        "required": ["pair_id", "offspring1", "offspring2"]
                    }
                }
            },
            "required": ["crossovers"]
        }

    def _generate_fallback_offspring(
        self,
        parent1: GeneratedIdea,
        parent2: GeneratedIdea,
        is_first: bool
    ) -> str:
        """Generate fallback text for batch crossover offspring."""
        # Use shared utility with extra detail for batch variant
        extra_detail = " The hybrid nature allows for greater flexibility and robustness in addressing various scenarios." if is_first else " The focus is on creating emergent properties that neither parent could achieve alone."
        return generate_crossover_fallback_text(parent1, parent2, is_first, extra_detail)

    def _create_offspring(
        self,
        primary_parent: GeneratedIdea,
        secondary_parent: GeneratedIdea,
        content: str,
        llm_cost: float,
        from_cache: bool = False
    ) -> GeneratedIdea:
        """Create an offspring idea object."""
        return GeneratedIdea(
            content=content,
            thinking_method=primary_parent.thinking_method,
            agent_name="BatchSemanticCrossoverOperator",
            generation_prompt=f"Batch crossover of: '{primary_parent.content[:30]}...' and '{secondary_parent.content[:30]}...'",
            confidence_score=(
                (primary_parent.confidence_score or 0.5) +
                (secondary_parent.confidence_score or 0.5)
            ) / 2,
            reasoning="Batch semantic integration of parent concepts",
            parent_ideas=[primary_parent.content, secondary_parent.content],
            metadata={
                "operator": "semantic_batch_crossover",
                "crossover_type": "semantic_batch",
                "llm_cost": llm_cost,
                "from_cache": from_cache,
                "parent_ids": [
                    primary_parent.metadata.get("id", ""),
                    secondary_parent.metadata.get("id", "")
                ],
                "generation": max(
                    primary_parent.metadata.get("generation", 0),
                    secondary_parent.metadata.get("generation", 0)
                ) + 1,
            },
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
