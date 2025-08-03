# Mad Spark Alt Transformation Roadmap

## From AI Creativity Evaluation to Multi-Agent Idea Generation

### Project Vision Reference
**Issue #1:** 本プロジェクトは、「シン・ロジカルシンキング」の概念に基づき、多様な思考法を持つ複数のAIエージェントが協調し、革新的なアイデアを生成・評価し、遺伝的アルゴリズム（GA）を通じてアイデアを進化させていく「マルチエージェント・アイデア生成システム」を構築することを目的とします。

## Current Foundation Strengths ✅

### Architecture Advantages We Can Leverage:
1. **Plugin Registry System** (`core/registry.py`)
   - Perfect pattern for managing different thinking method agents
   - Dynamic discovery and registration already implemented
   - Indexing by capability (layer/output type) → extends to thinking methods

2. **Async Processing Framework** 
   - Essential for multi-agent coordination
   - Existing patterns in `CreativityEvaluator` for concurrent evaluation
   - Ready to scale to agent orchestration

3. **Evaluation Infrastructure**
   - Critical for genetic algorithm fitness evaluation
   - Existing diversity/quality metrics perfect for idea scoring
   - Multi-dimensional scoring already implemented

4. **Data Models & Interfaces**
   - Strong foundation in `interfaces.py` with `EvaluatorInterface`
   - `ModelOutput` can be extended to `GeneratedIdea`
   - `EvaluationRequest` can become `IdeaGenerationRequest`

## Transformation Strategy

### Phase 1: Core Architecture Evolution (Weeks 1-3)

#### 1.1 Extend Core Interfaces (`core/interfaces.py`)

**New Enums:**
```python
class ThinkingMethod(Enum):
    QUESTIONING = "questioning"
    ABDUCTION = "abduction" 
    DEDUCTION = "deduction"
    INDUCTION = "induction"
    QADI_CYCLE = "qadi_cycle"

class IdeaStatus(Enum):
    GENERATED = "generated"
    VALIDATED = "validated"
    EVOLVED = "evolved"
    SELECTED = "selected"
```

**New Data Models:**
```python
@dataclass
class GeneratedIdea(ModelOutput):
    """Extends ModelOutput with idea-specific metadata"""
    thinking_method: ThinkingMethod
    generation_context: Dict[str, Any]
    fitness_scores: Dict[str, float] = field(default_factory=dict)
    evolution_history: List[str] = field(default_factory=list)
    parent_ideas: List[str] = field(default_factory=list)

@dataclass  
class IdeaGenerationRequest:
    """Request for multi-agent idea generation"""
    problem_statement: str
    thinking_methods: List[ThinkingMethod]
    generation_config: Dict[str, Any] = field(default_factory=dict)
    human_context: Optional[str] = None
```

**New Interface:**
```python
class ThinkingAgentInterface(ABC):
    """Abstract base for thinking method agents"""
    
    @property
    @abstractmethod
    def thinking_method(self) -> ThinkingMethod:
        pass
    
    @abstractmethod
    async def generate_ideas(self, request: IdeaGenerationRequest) -> List[GeneratedIdea]:
        pass
    
    @abstractmethod
    def can_collaborate_with(self, other_method: ThinkingMethod) -> bool:
        pass
```

#### 1.2 Enhance Registry System (`core/registry.py`)

**Add ThinkingAgentRegistry:**
- Extend existing `EvaluatorRegistry` pattern
- Index agents by thinking method and collaboration capabilities
- Support for agent discovery and orchestration patterns

#### 1.3 Build Orchestration Engine (`core/orchestrator.py`)

**QADIOrchestrator:**
```python
class QADIOrchestrator:
    """Orchestrates Question → Abduction → Deduction → Induction cycle"""
    
    async def run_cycle(self, problem: str) -> List[GeneratedIdea]:
        # Q: Generate diverse questions about the problem
        questions = await self.questioning_agent.generate_ideas(...)
        
        # A: Generate hypotheses from questions  
        hypotheses = await self.abduction_agent.generate_ideas(...)
        
        # D: Validate and refine hypotheses
        validated = await self.deduction_agent.generate_ideas(...)
        
        # I: Synthesize patterns and insights
        insights = await self.induction_agent.generate_ideas(...)
        
        return insights
```

### Phase 2: Thinking Method Agents (Weeks 4-7)

#### 2.1 Question Generation Agent (`agents/questioning/`)

**Core Capabilities:**
- Implement questioning techniques from "Shin Logical Thinking"
- Generate diverse problem frames and perspectives
- Use existing LLM integration patterns

**Implementation Structure:**
```
agents/questioning/
├── __init__.py
├── agent.py           # QuestioningAgent implementation
├── techniques.py      # Specific questioning methods
└── prompts.py         # Question generation prompts
```

#### 2.2 Abductive Reasoning Agent (`agents/abduction/`)

**Core Capabilities:**
- Hypothesis generation from observations
- Creative leap identification
- Pattern-based inference

#### 2.3 Deductive Analysis Agent (`agents/deduction/`)

**Core Capabilities:**
- Logical consequence derivation
- Systematic validation of hypotheses
- Structured reasoning chains

#### 2.4 Inductive Synthesis Agent (`agents/induction/`)

**Core Capabilities:**
- Pattern generalization from examples
- Insight extraction and rule formation
- Creative pattern combination

### Phase 3: Genetic Evolution Engine (Weeks 8-10)

#### 3.1 Evolution Engine (`evolution/genetic_algorithm.py`)

**Leverage Existing Evaluation:**
- Use current `CreativityEvaluator` for fitness scoring
- Implement genetic operators (crossover, mutation, selection)
- Population management and evolution tracking

**Core Components:**
```python
class GeneticEvolution:
    def __init__(self, fitness_evaluator: CreativityEvaluator):
        self.fitness_evaluator = fitness_evaluator
    
    async def evolve(self, population: List[GeneratedIdea], generations: int) -> List[GeneratedIdea]:
        for generation in range(generations):
            # Evaluate fitness using existing evaluation framework
            fitness_scores = await self.evaluate_population(population)
            
            # Apply genetic operators
            population = self.selection(population, fitness_scores)
            population = await self.crossover(population)
            population = await self.mutate(population)
        
        return population
```

#### 3.2 Fitness Evaluation (`evolution/fitness.py`)

**Repurpose Current Metrics:**
- Diversity metrics → Idea novelty scoring
- Quality metrics → Idea feasibility scoring  
- Multi-dimensional fitness evaluation

### Phase 4: Human-AI Collaboration (Weeks 11-12)

#### 4.1 Interactive Interface (`collaboration/interface.py`)

**Extend Current CLI:**
- Session-based ideation workflows
- Real-time human feedback integration
- Progress visualization and idea tracking

#### 4.2 Web Interface (Optional)

**Browser-based Platform:**
- Visual idea evolution tracking
- Multi-user ideation sessions
- Collaboration workspace

## Implementation Benefits

### Architectural Strengths We Keep:
1. **Minimal Breaking Changes** - Current evaluation APIs remain functional
2. **Plugin Architecture** - Existing pattern perfect for agent management
3. **Async Patterns** - Transfer directly to agent coordination
4. **Strong Foundation** - Registry, interfaces, and data models extend naturally

### New Capabilities We Gain:
1. **Multi-Agent QADI Cycle** - Automated ideation workflows
2. **Genetic Algorithm Evolution** - Idea population optimization
3. **Human-AI Collaboration** - Interactive innovation sessions
4. **Scalable Agent Architecture** - Modular thinking method implementations

## Success Metrics

### Month 1 (Weeks 1-4):
- [ ] Core interfaces extended for idea generation
- [ ] Basic thinking method agents operational
- [ ] QADI cycle orchestration working

### Month 2 (Weeks 5-8):
- [ ] All four thinking method agents implemented
- [ ] Agent collaboration patterns established
- [ ] Genetic algorithm foundation built

### Month 3 (Weeks 9-12):
- [ ] Full genetic evolution system operational
- [ ] Human-AI collaboration interface working
- [ ] End-to-end ideation workflows complete

## Risk Mitigation

### Technical Risks:
1. **Agent Coordination Complexity** → Leverage existing async patterns
2. **LLM Integration Challenges** → Build on current evaluation infrastructure
3. **Performance Scaling** → Use proven plugin architecture

### Project Risks:
1. **Scope Creep** → Focus on QADI cycle as core workflow
2. **Integration Complexity** → Maintain backward compatibility with evaluation APIs
3. **Timeline Pressure** → Prioritize core functionality over advanced features

## Next Immediate Actions

### Week 1 Priority Tasks:
1. [ ] Extend `ThinkingMethod` enum in `interfaces.py`
2. [ ] Create `GeneratedIdea` data model
3. [ ] Build basic `ThinkingAgentInterface`
4. [ ] Start `QuestioningAgent` implementation

This roadmap leverages the existing strong foundation while systematically transforming the system into a revolutionary multi-agent idea generation platform.