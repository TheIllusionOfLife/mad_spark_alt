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
from typing import Any, Dict, List, Optional, Tuple, Union

from mad_spark_alt.core.interfaces import GeneratedIdea
from mad_spark_alt.core.llm_provider import GoogleProvider, LLMRequest
from mad_spark_alt.evolution.interfaces import CrossoverInterface, MutationInterface, EvaluationContext

logger = logging.getLogger(__name__)

# Token limits for semantic operators (optimized for performance)
SEMANTIC_MUTATION_MAX_TOKENS = 1500  # Increased to reduce truncation warnings
SEMANTIC_BATCH_MUTATION_BASE_TOKENS = 1500  # Base tokens per idea in batch
SEMANTIC_BATCH_MUTATION_MAX_TOKENS = 6000  # Maximum tokens for batch mutation
SEMANTIC_CROSSOVER_MAX_TOKENS = 2000  # Increased for better synthesis


def _prepare_operator_contexts(
    context: Union[Optional[str], EvaluationContext],
    idea_prompt: str,
    default_context: str
) -> Tuple[str, str]:
    """
    Prepare string and evaluation contexts for semantic operators.
    
    Args:
        context: Either string context or EvaluationContext object
        idea_prompt: The idea's generation prompt as fallback
        default_context: Default context if all else fails
        
    Returns:
        Tuple of (context_str, evaluation_context_str)
    """
    if isinstance(context, EvaluationContext):
        context_str = context.original_question or idea_prompt or default_context
        evaluation_context_str = format_evaluation_context(context)
    else:
        context_str = context or idea_prompt or default_context
        evaluation_context_str = "No specific evaluation context provided."
    
    return context_str, evaluation_context_str


def _prepare_cache_key_with_context(
    base_key: str,
    context: Union[Optional[str], EvaluationContext]
) -> str:
    """
    Prepare cache key that includes EvaluationContext for context-aware caching.
    
    Args:
        base_key: Base cache key (e.g., idea content or parent combination)
        context: Either string context or EvaluationContext object
        
    Returns:
        Cache key that includes context information if applicable
    """
    if isinstance(context, EvaluationContext):
        # Include target improvements and current scores in cache key
        context_hash = hash(frozenset([
            (k, v) for k, v in context.current_best_scores.items()
        ] + [tuple(context.target_improvements)]))
        return f"{base_key}||ctx:{context_hash}"
    else:
        return base_key


def format_evaluation_context(context: EvaluationContext) -> str:
    """
    Format evaluation context for inclusion in prompts.
    
    Args:
        context: EvaluationContext with scoring information
        
    Returns:
        Formatted context string for prompts
    """
    context_parts = [
        f"Original Question: {context.original_question}",
        ""
    ]
    
    if context.current_best_scores:
        context_parts.append("Current Best Scores:")
        for criterion, score in context.current_best_scores.items():
            context_parts.append(f"  {criterion.title()}: {score:.1f}")
        context_parts.append("")
    
    if context.target_improvements:
        context_parts.append(f"Target Improvements: {', '.join(context.target_improvements)}")
        context_parts.append("")
    
    context_parts.append("FOCUS: Create variations that improve the target criteria while maintaining strengths.")
    
    return "\n".join(context_parts)


def is_likely_truncated(text: str) -> bool:
    """
    Detect if text appears to be truncated.
    
    Args:
        text: Text to check for truncation
        
    Returns:
        True if text appears truncated
    """
    if not text:
        return False
        
    # Check for common truncation indicators
    text = text.strip()
    
    if not text:
        return False
    
    # Check if ends with ellipsis
    if text.endswith('...'):
        return True
    
    # Check for incomplete JSON
    if text.startswith('{') and text.count('{') != text.count('}'):
        return True
    if text.startswith('[') and text.count('[') != text.count(']'):
        return True
    
    # Check if ends mid-sentence (no proper ending punctuation)
    if text[-1] not in '.!?"\'':
        words = text.split()
        if words:
            last_word = words[-1]
            # Check for comma or colon at end (likely truncated)
            if last_word.endswith(',') or last_word.endswith(':'):
                return True
            # Check if last word is very short (likely a determiner or preposition)
            # Common truncation patterns: "the", "a", "an", "and", "or", "with", etc.
            if len(last_word) <= 3 and last_word.lower() in {'a', 'an', 'the', 'and', 'or', 
                                                              'but', 'for', 'with', 'to', 'of', 
                                                              'in', 'on', 'at', 'by', 'is', 'was'}:
                return True
            # Check for other common incomplete endings
            if last_word.lower() in {'without', 'within', 'through', 'before', 'after', 'during'}:
                return True
            # Check if it appears to be mid-word (e.g., "previous" without context)
            if len(words) >= 3 and last_word == "previous":
                return True
        
    return False

# Cache configuration constants  
_CACHE_MAX_SIZE = 1000  # Increased maximum number of cache entries for better performance
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
    
    def __init__(self, ttl_seconds: int = 10800):  # Extended to 3 hours for longer evolution sessions
        """
        Initialize cache with enhanced session-based time-to-live.
        
        Args:
            ttl_seconds: Time-to-live for cache entries in seconds (default: 2 hours)
        """
        self.ttl_seconds = ttl_seconds
        self._cache: Dict[str, Tuple[Dict[str, Any], float]] = {}  # key -> (result_data, timestamp)
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
    
    def get(self, content: str, operation_type: str = "default", return_dict: bool = True) -> Optional[Union[str, Dict[str, Any]]]:
        """
        Get cached result with enhanced lookup including similarity matching.
        
        Args:
            content: Original content to look up
            operation_type: Operation type for cache clustering
            return_dict: If False, return just content string for backward compatibility
            
        Returns:
            Cached result (dict with metadata or string based on return_dict) or None if not found/expired
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
                # Return based on requested format
                if return_dict:
                    return value
                else:
                    # Backward compatibility - return just content string
                    if isinstance(value, dict):
                        return str(value.get("content", ""))
                    else:
                        return str(value)
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
                            # Return based on requested format
                            if return_dict:
                                return cached_value
                            else:
                                if isinstance(cached_value, dict):
                                    return str(cached_value.get("content", ""))
                                else:
                                    return str(cached_value)
                
        return None
        
    def put(self, content: str, result: Union[str, Dict[str, Any]], operation_type: str = "default") -> None:
        """
        Store result in enhanced cache with operation type clustering.
        
        Args:
            content: Original content
            result: Result data (string for backward compatibility or dict with metadata)
            operation_type: Operation type for cache clustering
        """
        key = self._get_cache_key(content, operation_type)
        
        # Convert string to dict format for consistency
        if isinstance(result, str):
            result_data = {"content": result, "mutation_type": operation_type}
        else:
            result_data = result
            
        self._cache[key] = (result_data, time.time())
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
    
    # Constants for breakthrough mutations and temperature settings
    BREAKTHROUGH_TEMPERATURE = 0.95
    REGULAR_MUTATION_TEMPERATURE = 0.8
    BREAKTHROUGH_TOKEN_MULTIPLIER = 2
    BREAKTHROUGH_CONFIDENCE_PROXY_THRESHOLD = 0.85
    BREAKTHROUGH_CONFIDENCE_MULTIPLIER = 1.05
    REGULAR_CONFIDENCE_MULTIPLIER = 0.95
    
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
        
        # Breakthrough mutation types for high-performing ideas
        self.breakthrough_mutation_types = [
            "paradigm_shift",
            "system_integration",
            "scale_amplification", 
            "future_forward"
        ]
        
        # Threshold for determining if an idea qualifies for breakthrough mutation
        self.breakthrough_threshold = 0.8  # Ideas with fitness >= 0.8 get breakthrough mutations
        
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
        if fitness is not None and isinstance(fitness, (int, float)) and fitness >= 0.8:
            return True
            
        # Also check "fitness" key used in tests
        fitness = idea.metadata.get("fitness")
        if fitness is not None and isinstance(fitness, (int, float)) and fitness >= 0.8:
            return True
                    
        # 2. Check confidence score as proxy (high confidence + later generation)
        if idea.confidence_score and idea.confidence_score >= self.BREAKTHROUGH_CONFIDENCE_PROXY_THRESHOLD:
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
            temperature = self.BREAKTHROUGH_TEMPERATURE  # Higher temperature for breakthrough creativity
            max_tokens = SEMANTIC_MUTATION_MAX_TOKENS * self.BREAKTHROUGH_TOKEN_MULTIPLIER  # More tokens for detailed breakthrough solutions
        else:
            # Use regular mutation
            mutation_type = random.choice(self.mutation_types)
            system_prompt = self.MUTATION_SYSTEM_PROMPT
            user_prompt_template = self.SINGLE_MUTATION_PROMPT
            temperature = 0.8  # Standard temperature for creativity
            max_tokens = SEMANTIC_MUTATION_MAX_TOKENS
        
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
        all_mutations = [None] * len(uncached_ideas)  # Preserve order
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
                    SEMANTIC_BATCH_MUTATION_BASE_TOKENS * len(breakthrough_ideas) * self.BREAKTHROUGH_TOKEN_MULTIPLIER,
                    SEMANTIC_BATCH_MUTATION_MAX_TOKENS
                ),
                temperature=self.BREAKTHROUGH_TEMPERATURE,
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
                max_tokens=min(SEMANTIC_BATCH_MUTATION_BASE_TOKENS * len(regular_ideas), SEMANTIC_BATCH_MUTATION_MAX_TOKENS),
                temperature=self.REGULAR_MUTATION_TEMPERATURE,
                response_schema=schema,
                response_mime_type="application/json"
            )
            
            response = await self.llm_provider.generate(request)
            total_cost += response.cost
            
            # Parse regular mutations
            regular_mutations = self._parse_mutation_response(
                response, len(regular_ideas), regular_ideas, is_breakthrough=False
            )
            
            # Place regular mutations in correct positions
            for idx, (orig_idx, mutation_data) in enumerate(zip(regular_indices, regular_mutations)):
                all_mutations[orig_idx] = mutation_data
        
        # Check for truncation and cache results using context-aware keys
        for idea, mutation_data in zip(uncached_ideas, all_mutations):
            if mutation_data and is_likely_truncated(mutation_data["content"]):
                logger.warning("Batch mutation appears truncated for idea: %s...", idea.content[:50])
            # Use the context-aware cache key for this idea
            cache_key = cache_keys[idea.content]
            # Cache the mutation data including type
            self.cache.put(cache_key, mutation_data)
            
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
                mutation_data = all_mutations[uncached_index]
                results.append(
                    self._create_mutated_idea(
                        idea,
                        mutation_data["content"],
                        cost_per_mutation,
                        mutation_data["mutation_type"],
                        is_breakthrough
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
                logger.debug(f"Could not find mutation for IDEA_{i+1}, using fallback")
                if original_ideas and i < len(original_ideas):
                    original_content = original_ideas[i].content
                    mutations.append(f"[FALLBACK TEXT] Enhanced variation of original concept: Building upon '{original_content[:50]}...', this mutation explores alternative implementation strategies while maintaining the core objective. The approach introduces different methodologies, tools, and frameworks to achieve similar outcomes through a varied pathway. By shifting perspectives on scale, audience, or technological approach, this variation demonstrates how the fundamental concept can be realized through innovative alternatives.")
                else:
                    mutations.append("[FALLBACK TEXT] Enhanced variation exploring alternative approaches: This mutation investigates different implementation strategies while maintaining the core objective of the original idea. The approach introduces alternative methodologies, tools, and frameworks to achieve similar outcomes through a different pathway. By exploring varied perspectives such as changing the scale of implementation, shifting target audience, or adopting different technological foundations, this variation demonstrates how the same fundamental problem can be addressed through multiple viable and innovative solutions.")
                
        return mutations
        
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
        mutations = []
        
        # Try to parse as JSON first (structured output)
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
                
            # Sort by ID to ensure correct order
            if all(isinstance(m, dict) and "id" in m for m in raw_mutations):
                raw_mutations = sorted(raw_mutations, key=lambda x: x["id"])
                
            for i, mutation in enumerate(raw_mutations[:expected_count]):
                if isinstance(mutation, dict) and "content" in mutation:
                    mutations.append({
                        "content": mutation["content"],
                        "mutation_type": mutation.get("mutation_type", 
                            "paradigm_shift" if is_breakthrough else "batch_mutation")
                    })
                else:
                    # Fallback for malformed mutation
                    mutations.append(self._create_fallback_mutation(
                        original_ideas[i] if i < len(original_ideas) else None,
                        is_breakthrough
                    ))
                    
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
                    
        return mutations
        
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
        confidence_multiplier = self.BREAKTHROUGH_CONFIDENCE_MULTIPLIER if is_breakthrough else self.REGULAR_CONFIDENCE_MULTIPLIER
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


class SemanticCrossoverOperator(CrossoverInterface):
    """
    LLM-powered crossover operator that meaningfully combines parent ideas.
    
    Uses LLM to understand key concepts from both parents and create
    offspring that integrate these concepts synergistically.
    """
    
    # Similarity threshold for detecting duplicate offspring
    SIMILARITY_THRESHOLD = 0.7
    
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
            max_tokens=SEMANTIC_CROSSOVER_MAX_TOKENS,
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
        # Note: Check for None explicitly, empty string is valid content
        if offspring1_content is None or offspring2_content is None:
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
        if similarity > self.SIMILARITY_THRESHOLD:  # More than 70% similar
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
        from difflib import SequenceMatcher
        
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
        if parent1 and parent2:
            if is_first:
                return f"[FALLBACK TEXT] Hybrid approach combining elements from both parent ideas: This solution integrates key aspects from '{parent1.content[:50]}...' and '{parent2.content[:50]}...'. The approach combines the structural framework of the first concept with the innovative mechanisms of the second, creating a comprehensive solution that addresses the same core problem through multiple complementary strategies. Implementation would involve adapting the proven methodologies from both approaches while ensuring seamless integration and enhanced effectiveness."
            else:
                return f"[FALLBACK TEXT] Alternative integration emphasizing synergy: This variation explores a different combination pattern by merging the core principles from '{parent1.content[:50]}...' with the practical implementation strategies from '{parent2.content[:50]}...'. The resulting solution maintains the strengths of both parent approaches while introducing novel elements that emerge from their interaction. This alternative demonstrates how the same foundational concepts can yield distinctly different yet equally valuable outcomes through strategic recombination."
        else:
            if is_first:
                return "[FALLBACK TEXT] Integrated solution combining complementary strengths: This approach synthesizes the core methodologies from both parent concepts, creating a hybrid solution that leverages their respective advantages. By merging the foundational elements with enhanced scalability features, this combination addresses limitations present in either approach alone. The integration focuses on creating synergy between different implementation strategies while maintaining practical feasibility."
            else:
                return "[FALLBACK TEXT] Alternative fusion emphasizing innovation: This variation explores a different integration pattern by prioritizing the innovative aspects of one approach while using the structural framework of the other. The result is a solution that pushes boundaries while remaining grounded in proven methodologies. This alternative path demonstrates how the same parent concepts can yield distinctly different yet equally valuable outcomes through creative recombination."
        
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
                max_tokens=SEMANTIC_CROSSOVER_MAX_TOKENS * len(uncached_pairs),
                temperature=0.8,
                response_schema=schema,
                response_mime_type="application/json"
            )
            
            result = await self.llm_provider.generate(request)
            llm_cost = result.cost if hasattr(result, 'cost') else 0.0
            
            # Parse batch results
            batch_results = []
            # Parse JSON from response content
            try:
                data = json.loads(result.content)
                if isinstance(data, dict) and "crossovers" in data:
                    crossovers = data["crossovers"]
                    
                    # Validate we got the right number of results
                    if len(crossovers) != len(uncached_pairs):
                        logger.warning(f"Expected {len(uncached_pairs)} crossovers but got {len(crossovers)}")
                    
                    # Process each crossover result
                    for i, crossover_data in enumerate(crossovers):
                        if i >= len(uncached_pairs):
                            break
                            
                        parent1, parent2 = uncached_pairs[i]
                        offspring1_content = crossover_data.get("offspring1", "")
                        offspring2_content = crossover_data.get("offspring2", "")
                        
                        # Validate offspring content
                        if not offspring1_content or not offspring2_content:
                            logger.warning(f"Empty offspring content for pair {i}")
                            # Use fallback
                            offspring1_content = self._generate_fallback_offspring(parent1, parent2, True)
                            offspring2_content = self._generate_fallback_offspring(parent1, parent2, False)
                        
                        # Create offspring ideas
                        offspring1 = self._create_offspring(parent1, parent2, offspring1_content, llm_cost / len(uncached_pairs))
                        offspring2 = self._create_offspring(parent2, parent1, offspring2_content, llm_cost / len(uncached_pairs))
                        
                        batch_results.append((offspring1, offspring2))
                        
                        # Cache the result
                        sorted_contents = sorted([parent1.content, parent2.content])
                        base_cache_key = f"{sorted_contents[0]}||{sorted_contents[1]}"
                        cache_key = _prepare_cache_key_with_context(base_cache_key, context)
                        cache_data = {
                            "offspring1": offspring1_content,
                            "offspring2": offspring2_content
                        }
                        self.cache.put(cache_key, cache_data, operation_type="batch_crossover")
                        
                else:
                    raise ValueError("Invalid response format - missing crossovers")
                    
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                # Fallback to sequential processing if batch fails
                logger.warning("Batch crossover failed, falling back to sequential processing")
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
        final_results = []
        cached_idx = 0
        batch_idx = 0
        
        for i in range(len(parent_pairs)):
            if i in uncached_indices:
                final_results.append(batch_results[batch_idx])
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
- "pair_id": index of the parent pair (0-based)
- "offspring1": detailed first offspring (min 150 words)
- "offspring2": detailed second offspring (min 150 words)"""
        
    def _get_batch_crossover_schema(self) -> Dict[str, Any]:
        """Get JSON schema for batch crossover structured output."""
        return {
            "type": "OBJECT",
            "properties": {
                "crossovers": {
                    "type": "ARRAY",
                    "items": {
                        "type": "OBJECT",
                        "properties": {
                            "pair_id": {"type": "INTEGER"},
                            "offspring1": {"type": "STRING"},
                            "offspring2": {"type": "STRING"}
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
        if is_first:
            return f"[FALLBACK TEXT] Hybrid approach combining elements from both parent ideas: This solution integrates key aspects from '{parent1.content[:50]}...' and '{parent2.content[:50]}...'. The approach combines the structural framework of the first concept with the innovative mechanisms of the second, creating a comprehensive solution that addresses the same core problem through multiple complementary strategies. Implementation would involve adapting the proven methodologies from both approaches while ensuring seamless integration and enhanced effectiveness. The hybrid nature allows for greater flexibility and robustness in addressing various scenarios."
        else:
            return f"[FALLBACK TEXT] Alternative integration emphasizing synergy: This variation explores a different combination pattern by merging the core principles from '{parent1.content[:50]}...' with the practical implementation strategies from '{parent2.content[:50]}...'. The resulting solution maintains the strengths of both parent approaches while introducing novel elements that emerge from their interaction. This alternative demonstrates how the same foundational concepts can yield distinctly different yet equally valuable outcomes through strategic recombination. The focus is on creating emergent properties that neither parent could achieve alone."
            
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