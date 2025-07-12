#!/usr/bin/env python3
"""
Performance profiling script for QADI system.
Measures execution time for each phase and component.
"""

import asyncio
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import json
from contextlib import asynccontextmanager

from mad_spark_alt.core.smart_orchestrator import SmartQADIOrchestrator
from mad_spark_alt.core.smart_registry import SmartAgentRegistry
from mad_spark_alt.core.interfaces import (
    IdeaGenerationRequest, 
    IdeaGenerationResult,
    ThinkingMethod
)


@dataclass
class TimingEvent:
    """Represents a single timing event."""
    name: str
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def complete(self):
        """Mark the event as complete and calculate duration."""
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time


class PerformanceProfiler:
    """Profiles QADI system performance."""
    
    def __init__(self):
        self.events: List[TimingEvent] = []
        self.current_events: Dict[str, TimingEvent] = {}
        
    def start_event(self, name: str, **metadata) -> TimingEvent:
        """Start timing an event."""
        event = TimingEvent(name=name, start_time=time.time(), metadata=metadata)
        self.current_events[name] = event
        return event
        
    def end_event(self, name: str) -> Optional[TimingEvent]:
        """End timing an event."""
        if name in self.current_events:
            event = self.current_events.pop(name)
            event.complete()
            self.events.append(event)
            return event
        return None
        
    @asynccontextmanager
    async def measure(self, name: str, **metadata):
        """Context manager for measuring async operations."""
        event = self.start_event(name, **metadata)
        try:
            yield event
        finally:
            event.complete()
            self.events.append(event)
            if name in self.current_events:
                del self.current_events[name]
                
    def get_report(self) -> Dict[str, Any]:
        """Generate a timing report."""
        if not self.events:
            return {"error": "No events recorded"}
            
        # Group events by phase
        phases = {}
        for event in self.events:
            phase = event.metadata.get("phase", "other")
            if phase not in phases:
                phases[phase] = []
            phases[phase].append(event)
            
        # Calculate statistics
        total_duration = sum(e.duration for e in self.events if e.duration)
        
        report = {
            "total_duration": total_duration,
            "event_count": len(self.events),
            "phases": {},
            "detailed_events": []
        }
        
        # Phase summaries
        for phase, phase_events in phases.items():
            phase_duration = sum(e.duration for e in phase_events if e.duration)
            report["phases"][phase] = {
                "duration": phase_duration,
                "percentage": (phase_duration / total_duration * 100) if total_duration > 0 else 0,
                "event_count": len(phase_events),
                "events": [
                    {
                        "name": e.name,
                        "duration": e.duration,
                        "metadata": e.metadata
                    }
                    for e in sorted(phase_events, key=lambda x: x.duration or 0, reverse=True)
                ]
            }
            
        # All events sorted by duration
        report["detailed_events"] = [
            {
                "name": e.name,
                "duration": e.duration,
                "start_time": e.start_time,
                "metadata": e.metadata
            }
            for e in sorted(self.events, key=lambda x: x.duration or 0, reverse=True)
        ]
        
        return report


class ProfiledSmartQADIOrchestrator(SmartQADIOrchestrator):
    """Instrumented version of SmartQADIOrchestrator for profiling."""
    
    def __init__(self, profiler: PerformanceProfiler, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.profiler = profiler
        
    async def run_qadi_cycle(self, problem_statement: str, context: Optional[str] = None, 
                             cycle_config: Optional[Dict[str, Any]] = None):
        """Run a complete QADI cycle with profiling."""
        async with self.profiler.measure("total_qadi_cycle"):
            result = await super().run_qadi_cycle(problem_statement, context, cycle_config)
            return result
            
    async def ensure_agents_ready(self) -> Dict[str, str]:
        """Override to add profiling to agent setup."""
        async with self.profiler.measure("agent_setup", phase="init"):
            return await super().ensure_agents_ready()
            
    async def _run_smart_phase(self, method: ThinkingMethod, problem_statement: str,
                              context: Optional[str], config: Dict[str, Any]) -> Tuple[IdeaGenerationResult, str]:
        """Override to add profiling to each phase."""
        phase_name = f"{method.value}_phase"
        
        async with self.profiler.measure(phase_name, phase=method.value):
            # Get agent with profiling
            async with self.profiler.measure(f"{method.value}_get_agent", phase=method.value):
                agent = self.registry.get_preferred_agent(method)
                if not agent:
                    return await super()._run_smart_phase(method, problem_statement, context, config)
                    
            # Track agent type
            is_llm = agent.is_llm_powered
            agent_type = "LLM" if is_llm else "template"
            
            # Execute phase with profiling
            async with self.profiler.measure(f"{method.value}_agent_execution", 
                                           phase=method.value,
                                           is_llm=is_llm,
                                           agent_type=agent_type):
                result, actual_agent_type = await super()._run_smart_phase(
                    method, problem_statement, context, config
                )
                
                # Extract LLM cost if available
                if result.generated_ideas:
                    total_cost = sum(
                        idea.metadata.get('llm_cost', 0)
                        for idea in result.generated_ideas
                    )
                    if total_cost > 0:
                        self.profiler.events[-1].metadata['llm_cost'] = total_cost
                        
                return result, actual_agent_type


async def profile_qadi_system(prompt: str = "what is life?"):
    """Profile the QADI system with a given prompt."""
    print(f"Starting QADI performance profiling...")
    print(f"Prompt: '{prompt}'")
    print("-" * 80)
    
    profiler = PerformanceProfiler()
    
    # Setup phase
    async with profiler.measure("setup", phase="setup"):
        # Initialize registry
        async with profiler.measure("registry_initialization", phase="setup"):
            registry = SmartAgentRegistry()
            
        # Create orchestrator
        async with profiler.measure("orchestrator_initialization", phase="setup"):
            orchestrator = ProfiledSmartQADIOrchestrator(
                profiler=profiler,
                registry=registry
            )
    
    # Run the QADI cycle
    start_time = time.time()
    result = await orchestrator.run_qadi_cycle(
        problem_statement=prompt,
        context=None,
        cycle_config={
            "max_ideas_per_method": 3,
            "require_reasoning": True
        }
    )
    end_time = time.time()
    
    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")
    
    # Generate and display report
    report = profiler.get_report()
    
    print("\n" + "=" * 80)
    print("PERFORMANCE REPORT")
    print("=" * 80)
    
    # Phase breakdown
    print("\nPHASE BREAKDOWN:")
    print("-" * 60)
    print(f"{'Phase':<20} {'Duration (s)':<15} {'Percentage':<15} {'Events':<10}")
    print("-" * 60)
    
    for phase, data in sorted(report["phases"].items(), 
                             key=lambda x: x[1]["duration"], reverse=True):
        print(f"{phase:<20} {data['duration']:<15.3f} {data['percentage']:<15.1f}% {data['event_count']:<10}")
    
    # Top 10 slowest operations
    print("\n\nTOP 10 SLOWEST OPERATIONS:")
    print("-" * 80)
    print(f"{'Operation':<40} {'Duration (s)':<15} {'Phase':<15} {'Metadata'}")
    print("-" * 80)
    
    for event in report["detailed_events"][:10]:
        metadata_str = ""
        if event['metadata'].get('is_llm'):
            metadata_str += "LLM-powered"
        if event['metadata'].get('llm_cost'):
            metadata_str += f" (${event['metadata']['llm_cost']:.4f})"
            
        print(f"{event['name']:<40} {event['duration']:<15.3f} "
              f"{event['metadata'].get('phase', 'N/A'):<15} {metadata_str}")
    
    # Analysis
    print("\n\nANALYSIS:")
    print("-" * 80)
    
    # Check for sequential vs parallel execution
    phase_events = []
    for phase in ['questioning', 'abduction', 'deduction', 'induction']:
        phase_key = f"{phase}_phase"
        for event in report["detailed_events"]:
            if event["name"] == phase_key:
                phase_events.append((phase, event["start_time"], event["duration"]))
                
    if len(phase_events) > 1:
        print("\nPhase Execution Timeline:")
        phase_events.sort(key=lambda x: x[1])  # Sort by start time
        
        for i, (phase, start, duration) in enumerate(phase_events):
            rel_start = start - phase_events[0][1]
            print(f"  {phase:<15} Start: {rel_start:>6.2f}s  Duration: {duration:>6.2f}s")
            
            # Check for overlap with next phase
            if i < len(phase_events) - 1:
                next_start = phase_events[i + 1][1] - phase_events[0][1]
                if rel_start + duration > next_start:
                    print(f"    -> Overlaps with next phase (parallel execution)")
                else:
                    gap = next_start - (rel_start + duration)
                    print(f"    -> Gap before next phase: {gap:.2f}s")
    
    # LLM call analysis
    llm_events = [e for e in report["detailed_events"] 
                  if e["metadata"].get("is_llm")]
    if llm_events:
        total_llm_time = sum(e["duration"] for e in llm_events)
        print(f"\nLLM API Calls:")
        print(f"  Total calls: {len(llm_events)}")
        print(f"  Total time: {total_llm_time:.2f}s ({total_llm_time/report['total_duration']*100:.1f}% of total)")
        print(f"  Average per call: {total_llm_time/len(llm_events):.2f}s")
        
        total_cost = sum(e["metadata"].get("llm_cost", 0) for e in llm_events)
        if total_cost > 0:
            print(f"  Total cost: ${total_cost:.4f}")
    
    # Agent type analysis
    print("\n\nAGENT TYPES USED:")
    if hasattr(result, 'agent_types'):
        for method, agent_type in result.agent_types.items():
            print(f"  {method:<15} -> {agent_type}")
    
    # Identify bottlenecks
    print("\n\nBOTTLENECKS IDENTIFIED:")
    if report["detailed_events"]:
        top_event = report["detailed_events"][0]
        if top_event["duration"] > report["total_duration"] * 0.2:
            print(f"  - '{top_event['name']}' takes {top_event['duration']/report['total_duration']*100:.1f}% of total time")
            
    # Check for sequential execution
    sequential_time = sum(data["duration"] for phase, data in report["phases"].items() 
                         if phase in ['questioning', 'abduction', 'deduction', 'induction'])
    if sequential_time > report["total_duration"] * 0.95:
        print("  - Phases appear to be running sequentially (no parallelization)")
        phase_durations = [data["duration"] for phase, data in report["phases"].items() 
                          if phase in ['questioning', 'abduction', 'deduction', 'induction']]
        if phase_durations:
            potential_parallel_time = max(phase_durations)
            print(f"  - Potential time savings with parallelization: {sequential_time - potential_parallel_time:.2f}s")
    
    # Agent setup time
    setup_events = [e for e in report["detailed_events"] if "setup" in e["name"].lower()]
    if setup_events:
        total_setup_time = sum(e["duration"] for e in setup_events)
        print(f"  - Agent setup time: {total_setup_time:.2f}s ({total_setup_time/report['total_duration']*100:.1f}% of total)")
    
    # Save detailed report
    report_path = "/Users/yuyamukai/dev/mad_spark_alt/performance_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n\nDetailed report saved to: {report_path}")
    
    return report


if __name__ == "__main__":
    # Run the profiling
    asyncio.run(profile_qadi_system("what is life?"))