# Thinking Method Agents Architecture

## Overview

This document details the architecture for implementing the four core thinking method agents based on "Shin Logical Thinking" methodology. Each agent specializes in a different cognitive approach while collaborating through the QADI (Question → Abduction → Deduction → Induction) cycle.

## Agent Design Philosophy

### Core Principles:
1. **Specialized Cognitive Focus** - Each agent excels at one thinking method
2. **Collaborative Integration** - Agents work together through defined protocols
3. **LLM-Powered Reasoning** - Leverage existing LLM infrastructure for cognitive processing
4. **Extensible Architecture** - Plugin-based design allows for additional thinking methods

### Shared Infrastructure:
- **Base Class**: `ThinkingAgentInterface` (extends existing `EvaluatorInterface` pattern)
- **Registry System**: Leverages current `EvaluatorRegistry` for agent management
- **Async Processing**: Uses existing async patterns for concurrent agent operations
- **Data Models**: Extends `ModelOutput` → `GeneratedIdea` for idea representation

## Agent Specifications

### 1. Questioning Agent (質問生成エージェント)

**Purpose**: Generate diverse questions and problem framings to explore different perspectives.

**Core Capabilities:**
- **Problem Reframing**: Transform problem statements into multiple viewpoints
- **Question Hierarchies**: Generate broad to specific question sequences  
- **Perspective Shifting**: Approach problems from different stakeholder viewpoints
- **Context Expansion**: Identify related domains and analogies

**Implementation Structure:**
```
agents/questioning/
├── __init__.py
├── agent.py              # QuestioningAgent main implementation
├── techniques.py         # Specific questioning methodologies
├── prompts.py           # LLM prompt templates
├── frameworks.py        # Question categorization frameworks
└── tests/
    └── test_questioning.py
```

**Key Methods:**
```python
class QuestioningAgent(ThinkingAgentInterface):
    @property
    def thinking_method(self) -> ThinkingMethod:
        return ThinkingMethod.QUESTIONING
    
    async def generate_ideas(self, request: IdeaGenerationRequest) -> List[GeneratedIdea]:
        """Primary interface method - generates diverse questions as ideas"""
        questions = await self.generate_questions(request.problem_statement)
        reframings = await self.reframe_problem(request.problem_statement)
        hierarchies = await self.generate_question_hierarchy(request.problem_statement)
        
        # Convert questions to GeneratedIdea objects
        ideas = []
        for question in questions + reframings:
            ideas.append(GeneratedIdea(
                content=question,
                thinking_method=ThinkingMethod.QUESTIONING,
                generation_context=request.generation_config
            ))
        return ideas
    
    def can_collaborate_with(self, other_method: ThinkingMethod) -> bool:
        """Question generation collaborates well with all other methods"""
        return True
    
    # Internal helper methods
    async def generate_questions(self, problem: str) -> List[str]:
        """Generate diverse questions about the problem"""
        
    async def reframe_problem(self, problem: str) -> List[str]:
        """Create alternative problem framings"""
        
    async def generate_question_hierarchy(self, focus: str) -> Dict[str, List[str]]:
        """Create structured question sequences"""
```

**Question Categories:**
- **Clarification**: "What exactly do we mean by...?"
- **Assumption**: "What are we assuming here?"
- **Evidence**: "What evidence supports this?"
- **Perspective**: "How would X stakeholder view this?"
- **Implication**: "What are the consequences if...?"
- **Alternative**: "What other ways could we...?"

### 2. Abductive Reasoning Agent (仮説推論エージェント)

**Purpose**: Generate creative hypotheses and explanations from observations and questions.

**Core Capabilities:**
- **Hypothesis Generation**: Create plausible explanations for observations
- **Creative Leaps**: Make non-obvious connections between concepts
- **Pattern Recognition**: Identify underlying patterns in complex information
- **Analogical Reasoning**: Draw insights from similar situations in other domains

**Implementation Structure:**
```
agents/abduction/
├── __init__.py
├── agent.py              # AbductiveAgent main implementation
├── hypothesis_generator.py # Core hypothesis creation logic
├── pattern_matcher.py    # Pattern recognition utilities
├── analogy_engine.py     # Cross-domain analogy generation
├── prompts.py           # LLM prompt templates
└── tests/
    └── test_abduction.py
```

**Key Methods:**
```python
class AbductiveAgent(ThinkingAgentInterface):
    @property
    def thinking_method(self) -> ThinkingMethod:
        return ThinkingMethod.ABDUCTION
    
    async def generate_ideas(self, request: IdeaGenerationRequest) -> List[GeneratedIdea]:
        """Primary interface method - generates hypotheses as ideas"""
        observations = [request.problem_statement]  # Extract observations from context
        hypotheses = await self.generate_hypotheses(observations)
        patterns = await self.find_patterns(request.generation_config.get('data', []))
        analogies = await self.generate_analogies(
            request.problem_statement, 
            request.generation_config.get('domains', [])
        )
        
        # Convert hypotheses to GeneratedIdea objects
        ideas = []
        for hypothesis in hypotheses + patterns + analogies:
            ideas.append(GeneratedIdea(
                content=hypothesis,
                thinking_method=ThinkingMethod.ABDUCTION,
                generation_context=request.generation_config
            ))
        return ideas
    
    def can_collaborate_with(self, other_method: ThinkingMethod) -> bool:
        """Abduction works especially well with questioning and induction"""
        return other_method in [ThinkingMethod.QUESTIONING, ThinkingMethod.INDUCTION, ThinkingMethod.DEDUCTION]
    
    # Internal helper methods
    async def generate_hypotheses(self, observations: List[str]) -> List[str]:
        """Generate explanatory hypotheses"""
        
    async def find_patterns(self, data: List[Dict]) -> List[str]:
        """Identify underlying patterns"""
        
    async def generate_analogies(self, concept: str, domains: List[str]) -> List[str]:
        """Create cross-domain analogies"""
```

**Reasoning Patterns:**
- **Best Explanation**: "The most likely explanation is..."
- **Multiple Hypotheses**: "Several possibilities include..."
- **Creative Connections**: "This might relate to X because..."
- **Pattern Emergence**: "The underlying pattern seems to be..."

### 3. Deductive Analysis Agent (演繹分析エージェント)

**Purpose**: Systematically validate hypotheses and derive logical consequences.

**Core Capabilities:**
- **Logical Validation**: Test hypothesis consistency and logical soundness
- **Consequence Derivation**: Determine what follows if hypotheses are true
- **Systematic Analysis**: Break down complex ideas into logical components
- **Constraint Identification**: Find limitations and boundary conditions

**Implementation Structure:**
```
agents/deduction/
├── __init__.py
├── agent.py              # DeductiveAgent main implementation
├── logic_validator.py    # Logical consistency checking
├── consequence_engine.py # Derive implications and results
├── constraint_analyzer.py # Identify limitations and boundaries
├── prompts.py           # LLM prompt templates
└── tests/
    └── test_deduction.py
```

**Key Methods:**
```python
class DeductiveAgent(ThinkingAgentInterface):
    @property
    def thinking_method(self) -> ThinkingMethod:
        return ThinkingMethod.DEDUCTION
    
    async def generate_ideas(self, request: IdeaGenerationRequest) -> List[GeneratedIdea]:
        """Primary interface method - generates validated ideas through deductive analysis"""
        # Process existing ideas from request context or generate from problem
        base_ideas = request.generation_config.get('input_ideas', [])
        if not base_ideas:
            # Create initial idea from problem statement for analysis
            base_ideas = [GeneratedIdea(content=request.problem_statement, thinking_method=ThinkingMethod.QUESTIONING)]
        
        validated_ideas = []
        for idea in base_ideas:
            validation = await self.validate_hypothesis(idea)
            consequences = await self.derive_consequences(idea)
            constraints = await self.identify_constraints(idea)
            
            # Create refined ideas based on deductive analysis
            for consequence in consequences:
                validated_ideas.append(GeneratedIdea(
                    content=consequence,
                    thinking_method=ThinkingMethod.DEDUCTION,
                    generation_context=request.generation_config,
                    parent_ideas=[idea.content] if hasattr(idea, 'content') else []
                ))
        
        return validated_ideas
    
    def can_collaborate_with(self, other_method: ThinkingMethod) -> bool:
        """Deduction works with all methods, especially abduction and induction"""
        return True
    
    # Internal helper methods
    async def validate_hypothesis(self, hypothesis: GeneratedIdea) -> Dict[str, Any]:
        """Check logical consistency and validity"""
        
    async def derive_consequences(self, hypothesis: GeneratedIdea) -> List[str]:
        """Determine logical implications"""
        
    async def identify_constraints(self, idea: GeneratedIdea) -> List[str]:
        """Find limitations and boundary conditions"""
```

**Analysis Frameworks:**
- **Logical Consistency**: "If A is true, then B must follow..."
- **Contradiction Detection**: "This conflicts with..."
- **Constraint Analysis**: "This approach is limited by..."
- **Feasibility Assessment**: "This is possible if and only if..."

### 4. Inductive Synthesis Agent (帰納統合エージェント)

**Purpose**: Synthesize patterns, generalize insights, and create unified understanding.

**Core Capabilities:**
- **Pattern Generalization**: Extract general principles from specific examples
- **Insight Synthesis**: Combine multiple perspectives into unified insights
- **Rule Formation**: Develop general rules from observed patterns
- **Creative Integration**: Merge different ideas into novel combinations

**Implementation Structure:**
```
agents/induction/
├── __init__.py
├── agent.py              # InductiveAgent main implementation
├── pattern_generalizer.py # Extract general principles
├── insight_synthesizer.py # Combine multiple perspectives
├── rule_former.py        # Generate general rules
├── creative_integrator.py # Novel idea combinations
├── prompts.py           # LLM prompt templates
└── tests/
    └── test_induction.py
```

**Key Methods:**
```python
class InductiveAgent(ThinkingAgentInterface):
    @property
    def thinking_method(self) -> ThinkingMethod:
        return ThinkingMethod.INDUCTION
    
    async def generate_ideas(self, request: IdeaGenerationRequest) -> List[GeneratedIdea]:
        """Primary interface method - generates synthesized insights through inductive reasoning"""
        # Get input ideas from previous agents in the QADI cycle
        input_ideas = request.generation_config.get('input_ideas', [])
        observations = request.generation_config.get('observations', [request.problem_statement])
        
        # Synthesize insights through inductive reasoning
        patterns = await self.generalize_patterns(input_ideas)
        synthesis = await self.synthesize_insights(input_ideas)
        rules = await self.form_rules(observations)
        
        # Create final synthesized ideas
        synthesized_ideas = []
        
        # Add pattern-based ideas
        for pattern in patterns:
            synthesized_ideas.append(GeneratedIdea(
                content=pattern,
                thinking_method=ThinkingMethod.INDUCTION,
                generation_context=request.generation_config,
                parent_ideas=[idea.content for idea in input_ideas if hasattr(idea, 'content')]
            ))
        
        # Add the main synthesis
        if synthesis:
            synthesized_ideas.append(synthesis)
        
        # Add rule-based insights
        for rule in rules:
            synthesized_ideas.append(GeneratedIdea(
                content=rule,
                thinking_method=ThinkingMethod.INDUCTION,
                generation_context=request.generation_config
            ))
        
        return synthesized_ideas
    
    def can_collaborate_with(self, other_method: ThinkingMethod) -> bool:
        """Induction synthesizes output from all other thinking methods"""
        return True
    
    # Internal helper methods
    async def generalize_patterns(self, examples: List[GeneratedIdea]) -> List[str]:
        """Extract general principles"""
        
    async def synthesize_insights(self, ideas: List[GeneratedIdea]) -> GeneratedIdea:
        """Combine multiple perspectives"""
        
    async def form_rules(self, observations: List[str]) -> List[str]:
        """Develop general principles"""
```

**Synthesis Patterns:**
- **Pattern Extraction**: "The common pattern across these is..."
- **Principle Formation**: "The underlying principle seems to be..."
- **Creative Combination**: "Combining X and Y suggests..."
- **Insight Integration**: "Taking all perspectives together..."

## QADI Cycle Orchestration

### Workflow Implementation:

```python
class QADIOrchestrator:
    def __init__(self):
        self.questioning_agent = QuestioningAgent()
        self.abductive_agent = AbductiveAgent()
        self.deductive_agent = DeductiveAgent()
        self.inductive_agent = InductiveAgent()
    
    async def run_cycle(self, problem_statement: str) -> List[GeneratedIdea]:
        # Q: Question Phase
        questions = await self.questioning_agent.generate_ideas(
            IdeaGenerationRequest(
                problem_statement=problem_statement,
                thinking_methods=[ThinkingMethod.QUESTIONING]
            )
        )
        
        # A: Abduction Phase  
        hypotheses = await self.abductive_agent.generate_ideas(
            IdeaGenerationRequest(
                problem_statement=problem_statement,
                generation_context={"questions": questions}
            )
        )
        
        # D: Deduction Phase
        validated_ideas = []
        for hypothesis in hypotheses:
            validation = await self.deductive_agent.validate_hypothesis(hypothesis)
            if validation["is_valid"]:
                hypothesis.metadata.update(validation)
                validated_ideas.append(hypothesis)
        
        # I: Induction Phase
        final_insights = await self.inductive_agent.synthesize_insights(validated_ideas)
        
        return final_insights
```

## Agent Collaboration Patterns

### 1. Sequential Collaboration (QADI Cycle)
- **Linear Flow**: Q → A → D → I
- **Context Passing**: Each agent receives output from previous stage
- **Cumulative Refinement**: Ideas become more refined through the cycle

### 2. Parallel Processing
- **Simultaneous Generation**: Multiple agents work on same problem concurrently
- **Perspective Diversity**: Different cognitive approaches to same challenge
- **Result Synthesis**: Combine outputs for richer solution space

### 3. Iterative Refinement
- **Multiple Cycles**: Run QADI cycle repeatedly
- **Progressive Enhancement**: Each iteration builds on previous results
- **Convergence Tracking**: Monitor idea quality improvement over iterations

## Integration with Existing Infrastructure

### Leveraging Current Strengths:

1. **Registry System** (`core/registry.py`)
   ```python
   # Register thinking agents
   registry.register(QuestioningAgent)
   registry.register(AbductiveAgent)
   registry.register(DeductiveAgent)
   registry.register(InductiveAgent)
   
   # Get agents by thinking method
   questioning_agents = registry.get_agents_by_thinking_method(ThinkingMethod.QUESTIONING)
   ```

2. **Async Processing** (existing patterns)
   ```python
   # Concurrent agent operations
   tasks = [
       agent.generate_ideas(request) for agent in agents
   ]
   results = await asyncio.gather(*tasks)
   ```

3. **Evaluation Framework** (fitness scoring)
   ```python
   # Use existing evaluation for idea fitness
   evaluator = CreativityEvaluator()
   fitness_scores = await evaluator.evaluate(
       EvaluationRequest(outputs=generated_ideas)
   )
   ```

## Testing Strategy

### Unit Tests:
- **Individual Agent Testing**: Each agent tested independently
- **Method-Specific Tests**: Test core capabilities of each thinking method
- **Mock LLM Responses**: Controlled testing without API costs

### Integration Tests:
- **QADI Cycle Testing**: End-to-end cycle execution
- **Agent Collaboration**: Multi-agent coordination patterns
- **Performance Testing**: Concurrent agent operations

### Example Test Structure:
```python
class TestQuestioningAgent:
    async def test_generate_questions(self):
        agent = QuestioningAgent()
        request = IdeaGenerationRequest(
            problem_statement="How can we reduce traffic congestion?"
        )
        questions = await agent.generate_ideas(request)
        
        assert len(questions) > 0
        assert all(q.thinking_method == ThinkingMethod.QUESTIONING for q in questions)
        assert any("stakeholder" in q.content.lower() for q in questions)
```

## Performance Considerations

### Optimization Strategies:
1. **LLM Call Efficiency**: Batch similar requests where possible
2. **Caching**: Cache agent responses for similar problem contexts  
3. **Parallel Processing**: Leverage async patterns for concurrent operations
4. **Resource Management**: Monitor and limit concurrent LLM API calls

### Scalability Patterns:
- **Agent Pool Management**: Dynamic agent instance creation
- **Load Balancing**: Distribute requests across available agents
- **Resource Monitoring**: Track LLM usage and costs

This architecture provides a solid foundation for implementing the thinking method agents while leveraging the existing evaluation infrastructure as the foundation for a revolutionary multi-agent idea generation system.