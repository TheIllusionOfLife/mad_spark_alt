"""
Semantic (LLM-powered) genetic operators for evolution.

This module implements mutation and crossover operators that use LLMs to create
meaningful variations and combinations of ideas, with caching and batch processing.
"""

import hashlib
import json
import logging
import random
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from mad_spark_alt.core.interfaces import GeneratedIdea
from mad_spark_alt.core.llm_provider import GoogleProvider, LLMRequest, LLMResponse
from mad_spark_alt.evolution.interfaces import CrossoverInterface, MutationInterface

logger = logging.getLogger(__name__)

# Cache configuration constants
_CACHE_MAX_SIZE = 500  # Maximum number of cache entries
_SIMILARITY_KEY_LENGTH = 16  # Length of similarity hash key
_SIMILARITY_CONTENT_PREFIX_LENGTH = 50  # Characters to use for similarity matching
_SIMILARITY_WORDS_COUNT = 10  # Number of meaningful words for similarity key
_SESSION_TTL_EXTENSION_RATE = 0.1  # Rate of TTL extension during session
_MAX_SESSION_TTL_EXTENSION = 3600  # Maximum TTL extension in seconds

# Stop words for similarity matching
_STOP_WORDS = {'the', 'a', 'an', 'and', 'or', 'but', 'by', 'for', 'with', 'to', 'of', 'in', 'on', 'at'}


def get_mutation_schema() -> Dict[str, Any]:
    """Get JSON schema for structured mutation output.
        
    Returns:
        JSON schema dictionary for Gemini structured output
    """
    return {
        "type": "OBJECT",
        "properties": {
            "mutations": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "idea_id": {"type": "INTEGER"},
                        "mutated_content": {"type": "STRING"},
                    },
                    "required": ["idea_id", "mutated_content"]
                }
            }
        },
        "required": ["mutations"]
    }


def get_crossover_schema() -> Dict[str, Any]:
    """Get JSON schema for structured crossover output.
    
    Returns:
        JSON schema dictionary for Gemini structured output
    """
    return {
        "type": "OBJECT",
        "properties": {
            "offspring_1": {"type": "STRING"},
            "offspring_2": {"type": "STRING"}
        },
        "required": ["offspring_1", "offspring_2"]
    }


class SemanticOperatorCache:
    """
    Enhanced in-memory cache for semantic operator results with session-based TTL.
    
    Reduces redundant LLM calls by caching mutation and crossover results with
    intelligent cache key clustering and extended session-based TTL.
    """
    
    def __init__(self, ttl_seconds: int = 7200):  # Extended to 2 hours for longer sessions
        """
        Initialize cache with enhanced session-based time-to-live.
        
        Args:
            ttl_seconds: Time-to-live for cache entries in seconds (default: 2 hours)
        """
        self.ttl_seconds = ttl_seconds
        self._cache: Dict[str, Tuple[str, float]] = {}  # key -> (value, timestamp)
        self._similarity_index: Dict[str, List[str]] = {}  # Map similarity keys to cache keys
        self._session_start = time.time()  # Track session for extended caching
        
    def _get_cache_key(self, content: str, operation_type: str = "default") -> str:
        """
        Generate consistent cache key with operation type for better clustering.
        
        Args:
            content: Content to generate key for
            operation_type: Type of operation (mutation, crossover, etc.) for clustering
        """
        # Include operation type in key for better cache organization
        combined_content = f"{operation_type}:{content}"
        return hashlib.md5(combined_content.encode()).hexdigest()
    
    def _get_similarity_key(self, content: str) -> str:
        """
        Generate similarity-based key for cache clustering.
        Uses first 50 characters of normalized content for similarity matching.
        """
        # Normalize content for similarity matching
        normalized = content.lower().strip()[:_SIMILARITY_CONTENT_PREFIX_LENGTH]
        # Remove common words that don't affect semantic meaning
        words = [w for w in normalized.split() if w not in _STOP_WORDS]
        key_content = ' '.join(words[:_SIMILARITY_WORDS_COUNT])
        return hashlib.md5(key_content.encode()).hexdigest()[:_SIMILARITY_KEY_LENGTH]
        
    def _get_effective_ttl(self, current_time: float) -> float:
        """Calculate effective TTL with session-based extension."""
        session_duration = current_time - self._session_start
        return self.ttl_seconds + min(
            session_duration * _SESSION_TTL_EXTENSION_RATE,
            _MAX_SESSION_TTL_EXTENSION
        )
    
    def get(self, content: str, operation_type: str = "default") -> Optional[str]:
        """
        Get cached result with enhanced lookup including similarity matching.
        
        Args:
            content: Original content to look up
            operation_type: Operation type for cache clustering
            
        Returns:
            Cached mutation result or None if not found/expired
        """
        # Try exact match first
        exact_key = self._get_cache_key(content, operation_type)
        
        if exact_key in self._cache:
            value, timestamp = self._cache[exact_key]
            
            # Check if expired (extended session-based TTL)
            current_time = time.time()
            effective_ttl = self._get_effective_ttl(current_time)
            
            if current_time - timestamp < effective_ttl:
                logger.debug(f"Cache exact hit for {operation_type} hash {exact_key[:8]}")
                return value
            else:
                # Remove expired entry
                del self._cache[exact_key]
                logger.debug(f"Cache expired for {operation_type} hash {exact_key[:8]}")
        
        # Try similarity-based lookup for mutation operations
        # Store similarity keys separately for efficient lookup
        if operation_type == "mutation" and hasattr(self, '_similarity_index'):
            similarity_key = self._get_similarity_key(content)
            if similarity_key in self._similarity_index:
                # Get all cache keys with this similarity
                for cache_key in self._similarity_index[similarity_key]:
                    if cache_key in self._cache:
                        cached_value, timestamp = self._cache[cache_key]
                        if current_time - timestamp < effective_ttl:  # Reuse calculated TTL
                            logger.debug(f"Cache similarity hit for {operation_type} hash {cache_key[:8]}")
                            return cached_value
                
        return None
        
    def put(self, content: str, result: str, operation_type: str = "default") -> None:
        """
        Store result in enhanced cache with operation type clustering.
        
        Args:
            content: Original content
            result: Mutation/crossover result to cache
            operation_type: Operation type for cache clustering
        """
        key = self._get_cache_key(content, operation_type)
        self._cache[key] = (result, time.time())
        logger.debug(f"Cached {operation_type} result for hash {key[:8]}")
        
        # Update similarity index for mutation operations
        if operation_type == "mutation":
            similarity_key = self._get_similarity_key(content)
            if similarity_key not in self._similarity_index:
                self._similarity_index[similarity_key] = []
            self._similarity_index[similarity_key].append(key)
        
        # Periodic cache cleanup to prevent memory growth
        if len(self._cache) > _CACHE_MAX_SIZE:
            self._cleanup_expired_entries()
    
    def _cleanup_expired_entries(self) -> None:
        """Clean up expired cache entries to manage memory usage."""
        current_time = time.time()
        effective_ttl = self._get_effective_ttl(current_time)
        expired_keys = []
        
        for key, (_, timestamp) in self._cache.items():
            if current_time - timestamp >= effective_ttl:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self._cache[key]
        
        # Clean up similarity index
        for sim_key in list(self._similarity_index.keys()):
            self._similarity_index[sim_key] = [k for k in self._similarity_index[sim_key] if k not in expired_keys]
            if not self._similarity_index[sim_key]:
                del self._similarity_index[sim_key]
        
        logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics for monitoring performance."""
        current_time = time.time()
        valid_entries = sum(1 for _, (_, timestamp) in self._cache.items() 
                           if current_time - timestamp < self.ttl_seconds)
        
        return {
            "total_entries": len(self._cache),
            "valid_entries": valid_entries,
            "expired_entries": len(self._cache) - valid_entries,
            "session_duration_minutes": int((current_time - self._session_start) / 60),
        }


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
Return ONLY the mutated idea text, no explanations or metadata."""

    SINGLE_MUTATION_PROMPT = """Original idea: {idea}
Problem context: {context}

Create a semantically different variation that:
1. Addresses the same core problem
2. Uses a fundamentally different approach or mechanism
3. Maintains feasibility but explores new solution space
4. Provides DETAILED implementation with specific steps, technologies, and methodologies
5. Includes at least 150-200 words of comprehensive explanation

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

Output only the detailed mutated idea (minimum 150 words):"""

    BATCH_MUTATION_PROMPT = """Generate diverse variations for these ideas. Each variation should use a different approach.

Context: {context}

Ideas to mutate:
{ideas_list}

For each idea, provide ONE variation that:
- Is semantically different but addresses the same goal
- Uses different mutation strategies: perspective shift, mechanism change, or abstraction level
- Provides DETAILED implementation (minimum 150 words per variation)
- Includes specific steps, technologies, methodologies, and resources

Generate complete, detailed solutions that include:
- Specific implementation steps
- Technologies or tools to be used
- Resources required
- Expected outcomes and benefits

Return JSON with mutations array containing idea_id and mutated_content for each idea."""
    
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
        context: Optional[str] = None
    ) -> GeneratedIdea:
        """
        Mutate a single idea using LLM.
        
        Note: When used as a semantic operator selected by SmartOperatorSelector,
        this method always applies mutation (ignores mutation_rate) since the
        selector has already decided to use semantic mutation.
        
        Args:
            idea: Idea to mutate
            mutation_rate: Probability of mutation (ignored for semantic operators)
            context: Optional context for mutation
            
        Returns:
            Mutated idea
        """
        # Always apply mutation - the SmartOperatorSelector has already
        # decided to use semantic mutation based on diversity and fitness
        return await self.mutate_single(idea, context)
        
    async def mutate_single(
        self,
        idea: GeneratedIdea,
        context: Optional[str] = None
    ) -> GeneratedIdea:
        """
        Mutate a single idea with caching.
        
        Args:
            idea: Idea to mutate
            context: Optional context
            
        Returns:
            Mutated idea
        """
        # Check cache first
        cached_result = self.cache.get(idea.content)
        if cached_result:
            return self._create_mutated_idea(idea, cached_result, 0.0)
            
        # Generate mutation using LLM
        mutation_type = random.choice(self.mutation_types)
        
        request = LLMRequest(
            system_prompt=self.MUTATION_SYSTEM_PROMPT,
            user_prompt=self.SINGLE_MUTATION_PROMPT.format(
                idea=idea.content,
                context=context or idea.generation_prompt,
                mutation_type=mutation_type
            ),
            max_tokens=200,
            temperature=0.8  # Higher temperature for creativity
        )
        
        response = await self.llm_provider.generate(request)
        
        # Cache the result
        self.cache.put(idea.content, response.content)
        
        return self._create_mutated_idea(
            idea, 
            response.content,
            response.cost,
            mutation_type
        )
        
    async def mutate_batch(
        self,
        ideas: List[GeneratedIdea],
        context: Optional[str] = None
    ) -> List[GeneratedIdea]:
        """
        Mutate multiple ideas in a single LLM call.
        
        Args:
            ideas: List of ideas to mutate
            context: Optional context
            
        Returns:
            List of mutated ideas
        """
        # Separate cached and uncached ideas
        cached_results = {}
        uncached_ideas = []
        
        for idea in ideas:
            cached_result = self.cache.get(idea.content)
            if cached_result:
                cached_results[idea.content] = cached_result
            else:
                uncached_ideas.append(idea)
                
        # If all are cached, return immediately
        if not uncached_ideas:
            return [
                self._create_mutated_idea(idea, cached_results[idea.content], 0.0)
                for idea in ideas
            ]
            
        # Batch process uncached ideas
        # Create batch prompt
        ideas_list = "\n".join([
            f"IDEA_{i+1}: {idea.content}"
            for i, idea in enumerate(uncached_ideas)
        ])
        
        # Create request with structured output
        schema = get_mutation_schema()
        request = LLMRequest(
            system_prompt=self.MUTATION_SYSTEM_PROMPT,
            user_prompt=self.BATCH_MUTATION_PROMPT.format(
                context=context or "general improvement",
                ideas_list=ideas_list
            ),
            max_tokens=min(200 * len(uncached_ideas), 1000),
            temperature=0.8,
            response_schema=schema,
            response_mime_type="application/json"
        )
        
        response = await self.llm_provider.generate(request)
        
        # Try to parse as JSON first (structured output)
        mutations = []
        try:
            data = json.loads(response.content)
            if "mutations" in data and isinstance(data["mutations"], list):
                # Extract mutations and sort by idea_id to handle any ordering
                mutation_list = []
                for mut in data["mutations"]:
                    if isinstance(mut, dict) and "idea_id" in mut and "mutated_content" in mut:
                        mutation_list.append((mut["idea_id"], mut["mutated_content"]))
                
                # Sort by idea_id to ensure correct order
                mutation_list.sort(key=lambda x: x[0])
                
                # Extract just the content in the correct order
                # Handle both 0-based and 1-based indexing by using array position
                mutations = [content for _, content in mutation_list]
                
                # Ensure we have the right number of mutations
                if len(mutations) == len(uncached_ideas):
                    logger.debug("Successfully parsed %d mutations from structured output", len(mutations))
                else:
                    logger.warning("Mutation count mismatch: expected %d, got %d", 
                                 len(uncached_ideas), len(mutations))
                    # If count doesn't match, fall back to text parsing
                    mutations = []
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.debug("Structured output parsing failed, falling back to text parsing: %s", e)
            # Fall back to text parsing
            mutations = self._parse_batch_response(response.content, len(uncached_ideas), uncached_ideas)
        
        # Cache results
        for idea, mutation in zip(uncached_ideas, mutations):
            self.cache.put(idea.content, mutation)
            
        # Distribute cost across mutations
        cost_per_mutation = response.cost / len(mutations) if mutations else 0
        
        # Create result list maintaining original order
        results = []
        uncached_index = 0
        
        for idea in ideas:
            if idea.content in cached_results:
                results.append(
                    self._create_mutated_idea(idea, cached_results[idea.content], 0.0)
                )
            else:
                results.append(
                    self._create_mutated_idea(
                        idea,
                        mutations[uncached_index],
                        cost_per_mutation
                    )
                )
                uncached_index += 1
                
        return results
            
    def _parse_batch_response(self, response: str, expected_count: int, original_ideas: Optional[List[GeneratedIdea]] = None) -> List[str]:
        """
        Parse batch mutation response from LLM.
        
        Args:
            response: LLM response text
            expected_count: Expected number of mutations
            original_ideas: Original ideas for context-aware fallback
            
        Returns:
            List of mutation texts
        """
        mutations = []
        lines = response.strip().split('\n')
        
        for i in range(expected_count):
            # Look for IDEA_N_MUTATION: pattern
            pattern = f"IDEA_{i+1}_MUTATION:"
            
            for line in lines:
                if pattern in line:
                    # Extract mutation text after the pattern
                    mutation = line.split(pattern, 1)[1].strip()
                    mutations.append(mutation)
                    break
            else:
                # Fallback if pattern not found - create meaningful variation
                logger.warning(f"Could not find mutation for IDEA_{i+1}, using fallback")
                if original_ideas and i < len(original_ideas):
                    original_content = original_ideas[i].content
                    mutations.append(f"Enhanced variation of original concept: Building upon '{original_content[:50]}...', this mutation explores alternative implementation strategies while maintaining the core objective. The approach introduces different methodologies, tools, and frameworks to achieve similar outcomes through a varied pathway. By shifting perspectives on scale, audience, or technological approach, this variation demonstrates how the fundamental concept can be realized through innovative alternatives.")
                else:
                    mutations.append(f"Enhanced variation exploring alternative approaches: This mutation investigates different implementation strategies while maintaining the core objective of the original idea. The approach introduces alternative methodologies, tools, and frameworks to achieve similar outcomes through a different pathway. By exploring varied perspectives such as changing the scale of implementation, shifting target audience, or adopting different technological foundations, this variation demonstrates how the same fundamental problem can be addressed through multiple viable and innovative solutions.")
                
        return mutations
        
    def _create_mutated_idea(
        self,
        original: GeneratedIdea,
        mutated_content: str,
        llm_cost: float,
        mutation_type: Optional[str] = None
    ) -> GeneratedIdea:
        """Create a mutated idea object."""
        return GeneratedIdea(
            content=mutated_content,
            thinking_method=original.thinking_method,
            agent_name="BatchSemanticMutationOperator",
            generation_prompt=f"Semantic mutation of: '{original.content[:50]}...'",
            confidence_score=(original.confidence_score or 0.5) * 0.95,
            reasoning=f"Applied semantic mutation to create variation",
            parent_ideas=[original.content],
            metadata={
                "operator": "semantic_mutation",
                "mutation_type": mutation_type or "semantic_variation",
                "llm_cost": llm_cost,
                "generation": original.metadata.get("generation", 0) + 1,
            },
            timestamp=datetime.now(timezone.utc).isoformat(),
        )


class SemanticCrossoverOperator(CrossoverInterface):
    """
    LLM-powered crossover operator that meaningfully combines parent ideas.
    
    Uses LLM to understand key concepts from both parents and create
    offspring that integrate these concepts synergistically.
    """
    
    CROSSOVER_SYSTEM_PROMPT = """You are a genetic crossover operator for idea evolution.
Your role is to meaningfully combine concepts from two parent ideas into offspring.

IMPORTANT: Generate comprehensive, detailed implementations with specific steps, technologies, and methodologies.
Return ONLY the offspring idea texts, no explanations."""

    CROSSOVER_PROMPT = """Parent Idea 1: {parent1}
Parent Idea 2: {parent2}
Context: {context}

Analyze the key concepts in each parent and create TWO offspring ideas that:
1. Meaningfully integrate concepts from BOTH parents
2. Are not just concatenations or word swaps
3. Create synergy between the parent concepts
4. Maintain coherence and feasibility
5. Provide DETAILED implementation (minimum 150 words per offspring)
6. Include specific steps, technologies, methodologies, and resources

Generate complete, detailed solutions that include:
- Specific implementation steps combining elements from both parents
- Technologies or tools from both approaches
- Resources required for the hybrid solution
- Expected outcomes showing synergy
- How it leverages strengths of both parent ideas

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
        context: Optional[str] = None
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
        sorted_contents = sorted([parent1.content, parent2.content])
        cache_key = f"{sorted_contents[0]}||{sorted_contents[1]}"
        cached_result = self.cache.get(cache_key)
        
        if cached_result:
            # Parse cached result
            offspring_contents = cached_result.split("||")
            if len(offspring_contents) >= 2:
                return (
                    self._create_offspring(parent1, parent2, offspring_contents[0], 0.0),
                    self._create_offspring(parent2, parent1, offspring_contents[1], 0.0)
                )
                
        # Generate crossover using LLM with structured output
        schema = get_crossover_schema()
        request = LLMRequest(
            system_prompt=self.CROSSOVER_SYSTEM_PROMPT,
            user_prompt=self.CROSSOVER_PROMPT.format(
                parent1=parent1.content,
                parent2=parent2.content,
                context=context or "general optimization"
            ),
            max_tokens=400,
            temperature=0.7,  # Moderate temperature for balanced creativity
            response_schema=schema,
            response_mime_type="application/json"
        )
        
        response = await self.llm_provider.generate(request)
        
        # Try to parse as JSON first (structured output)
        offspring1_content = None
        offspring2_content = None
        
        try:
            data = json.loads(response.content)
            if "offspring_1" in data and "offspring_2" in data:
                offspring1_content = data["offspring_1"]
                offspring2_content = data["offspring_2"]
                logger.debug("Successfully parsed crossover from structured output")
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.debug("Structured output parsing failed, falling back to text parsing: %s", e)
        
        # Fall back to text parsing if needed
        if not offspring1_content or not offspring2_content:
            offspring1_content, offspring2_content = self._parse_crossover_response(
                response.content, parent1, parent2
            )
        
        # Cache the result
        self.cache.put(cache_key, f"{offspring1_content}||{offspring2_content}")
        
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
            offspring1 = self._generate_crossover_fallback(parent1, parent2, is_first=True)
        if not offspring2:
            offspring2 = self._generate_crossover_fallback(parent1, parent2, is_first=False)
            
        return offspring1, offspring2
    
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
        if parent1 and parent2:
            if is_first:
                return f"Hybrid approach combining elements from both parent ideas: This solution integrates key aspects from '{parent1.content[:50]}...' and '{parent2.content[:50]}...'. The approach combines the structural framework of the first concept with the innovative mechanisms of the second, creating a comprehensive solution that addresses the same core problem through multiple complementary strategies. Implementation would involve adapting the proven methodologies from both approaches while ensuring seamless integration and enhanced effectiveness."
            else:
                return f"Alternative integration emphasizing synergy: This variation explores a different combination pattern by merging the core principles from '{parent1.content[:50]}...' with the practical implementation strategies from '{parent2.content[:50]}...'. The resulting solution maintains the strengths of both parent approaches while introducing novel elements that emerge from their interaction. This alternative demonstrates how the same foundational concepts can yield distinctly different yet equally valuable outcomes through strategic recombination."
        else:
            if is_first:
                return "Integrated solution combining complementary strengths: This approach synthesizes the core methodologies from both parent concepts, creating a hybrid solution that leverages their respective advantages. By merging the foundational elements with enhanced scalability features, this combination addresses limitations present in either approach alone. The integration focuses on creating synergy between different implementation strategies while maintaining practical feasibility."
            else:
                return "Alternative fusion emphasizing innovation: This variation explores a different integration pattern by prioritizing the innovative aspects of one approach while using the structural framework of the other. The result is a solution that pushes boundaries while remaining grounded in proven methodologies. This alternative path demonstrates how the same parent concepts can yield distinctly different yet equally valuable outcomes through creative recombination."
        
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