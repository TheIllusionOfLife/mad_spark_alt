"""
Conclusion Synthesizer for QADI Results.

This module synthesizes all generated ideas from QADI phases into
actionable conclusions and recommendations.
"""

import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

from .interfaces import GeneratedIdea, ThinkingMethod
from .llm_provider import LLMRequest, llm_manager, LLMProvider

logger = logging.getLogger(__name__)


@dataclass
class Conclusion:
    """Represents a synthesized conclusion from QADI results."""
    
    summary: str
    key_insights: List[str]
    actionable_recommendations: List[str]
    next_steps: List[str]
    confidence_level: float = 0.8
    metadata: Dict[str, Any] = field(default_factory=dict)
    

class ConclusionSynthesizer:
    """Synthesizes QADI ideas into actionable conclusions."""
    
    def __init__(self, use_llm: bool = True):
        self.use_llm = use_llm
        
    async def synthesize_conclusion(
        self,
        problem_statement: str,
        ideas_by_phase: Dict[str, List[GeneratedIdea]],
        context: Optional[str] = None
    ) -> Conclusion:
        """
        Synthesize all QADI ideas into a coherent conclusion.
        
        Args:
            problem_statement: The original problem/question
            ideas_by_phase: Ideas grouped by QADI phase
            context: Additional context
            
        Returns:
            Synthesized conclusion with insights and recommendations
        """
        if self.use_llm and self._has_llm_available():
            return await self._synthesize_with_llm(
                problem_statement, ideas_by_phase, context
            )
        else:
            return self._synthesize_with_template(
                problem_statement, ideas_by_phase
            )
    
    def _has_llm_available(self) -> bool:
        """Check if LLM is available."""
        try:
            return len(llm_manager.providers) > 0
        except:
            return False
    
    async def _synthesize_with_llm(
        self,
        problem_statement: str,
        ideas_by_phase: Dict[str, List[GeneratedIdea]],
        context: Optional[str]
    ) -> Conclusion:
        """Use LLM to synthesize a comprehensive conclusion."""
        
        # Prepare ideas summary
        ideas_text = self._format_ideas_for_llm(ideas_by_phase)
        
        prompt = f"""
You are a strategic advisor synthesizing insights from a QADI (Question-Abduction-Deduction-Induction) analysis.

Original Problem: {problem_statement}
Context: {context or "General analysis"}

Ideas Generated Through QADI Process:
{ideas_text}

Please synthesize these ideas into a comprehensive conclusion with:

1. EXECUTIVE SUMMARY (2-3 sentences capturing the essence)

2. KEY INSIGHTS (3-5 bullet points of the most important discoveries)

3. ACTIONABLE RECOMMENDATIONS (4-6 specific, practical steps)

4. NEXT STEPS (3-4 immediate actions to take)

Format your response as follows:
SUMMARY: [Your executive summary]

KEY_INSIGHTS:
- [Insight 1]
- [Insight 2]
- [Insight 3]

RECOMMENDATIONS:
- [Recommendation 1]
- [Recommendation 2]
- [Recommendation 3]
- [Recommendation 4]

NEXT_STEPS:
- [Step 1]
- [Step 2]
- [Step 3]

Focus on practical, actionable advice that directly addresses the original problem.
"""
        
        try:
            # Get the first available provider
            provider = next(iter(llm_manager.providers.keys()))
            
            request = LLMRequest(
                user_prompt=prompt,
                temperature=0.3,  # Lower temperature for more focused conclusions
                max_tokens=1000
            )
            
            response = await llm_manager.generate(request, provider)
            
            # Parse the response
            return self._parse_llm_conclusion(response.content)
            
        except Exception as e:
            logger.error(f"LLM conclusion synthesis failed: {e}")
            # Fallback to template
            return self._synthesize_with_template(problem_statement, ideas_by_phase)
    
    def _format_ideas_for_llm(self, ideas_by_phase: Dict[str, List[GeneratedIdea]]) -> str:
        """Format ideas for LLM processing."""
        formatted = []
        
        phase_names = {
            "questioning": "Questions that explore the problem space",
            "abduction": "Hypotheses and root causes",
            "deduction": "Logical conclusions and implications",
            "induction": "Patterns and principles"
        }
        
        for phase, ideas in ideas_by_phase.items():
            if ideas:
                formatted.append(f"\n{phase_names.get(phase, phase).upper()}:")
                for i, idea in enumerate(ideas, 1):
                    formatted.append(f"{i}. {idea.content}")
        
        return "\n".join(formatted)
    
    def _parse_llm_conclusion(self, llm_response: str) -> Conclusion:
        """Parse LLM response into Conclusion object."""
        # Try JSON parsing first with robust handler
        try:
            from .robust_json_handler import extract_json_from_response
            
            # Attempt to extract structured JSON response
            json_data = extract_json_from_response(
                llm_response,
                expected_keys=["summary", "key_insights", "recommendations", "next_steps"],
                fallback=None
            )
            
            if json_data and isinstance(json_data, dict):
                return Conclusion(
                    summary=json_data.get("summary", ""),
                    key_insights=json_data.get("key_insights", []),
                    actionable_recommendations=json_data.get("recommendations", []),
                    next_steps=json_data.get("next_steps", []),
                    metadata={"parsing_method": "json", "llm_cost": 0.0}
                )
        except Exception as e:
            logger.debug(f"JSON parsing failed, using text parsing fallback: {e}")
        
        # Fallback to text parsing
        summary = ""
        key_insights = []
        recommendations = []
        next_steps = []
        
        # Parse sections
        lines = llm_response.strip().split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if line.startswith("SUMMARY:"):
                summary = line.replace("SUMMARY:", "").strip()
                current_section = "summary"
            elif line.startswith("KEY_INSIGHTS:"):
                current_section = "insights"
            elif line.startswith("RECOMMENDATIONS:"):
                current_section = "recommendations"
            elif line.startswith("NEXT_STEPS:"):
                current_section = "next_steps"
            elif line.startswith("- "):
                item = line[2:].strip()
                if current_section == "insights":
                    key_insights.append(item)
                elif current_section == "recommendations":
                    recommendations.append(item)
                elif current_section == "next_steps":
                    next_steps.append(item)
            elif current_section == "summary" and not summary:
                summary = line
        
        # Ensure we have content
        if not summary:
            summary = "Analysis complete. See insights and recommendations below."
        if not key_insights:
            key_insights = ["Multiple perspectives analyzed through QADI methodology"]
        if not recommendations:
            recommendations = ["Consider the insights above for decision-making"]
        if not next_steps:
            next_steps = ["Review the recommendations and prioritize actions"]
        
        return Conclusion(
            summary=summary,
            key_insights=key_insights,
            actionable_recommendations=recommendations,
            next_steps=next_steps,
            confidence_level=0.85,
            metadata={"parsing_method": "text_fallback", "llm_cost": 0.0}
        )
    
    def _synthesize_with_template(
        self,
        problem_statement: str,
        ideas_by_phase: Dict[str, List[GeneratedIdea]]
    ) -> Conclusion:
        """Synthesize conclusion using templates when LLM unavailable."""
        
        # Count total ideas
        total_ideas = sum(len(ideas) for ideas in ideas_by_phase.values())
        
        # Extract key themes
        all_ideas = []
        for phase, ideas in ideas_by_phase.items():
            all_ideas.extend(ideas)
        
        # Create summary
        summary = (
            f"Analysis of '{problem_statement}' generated {total_ideas} ideas "
            f"across {len(ideas_by_phase)} thinking methods, revealing multiple "
            f"perspectives and potential solutions."
        )
        
        # Generate insights based on phase results
        key_insights = []
        
        if "questioning" in ideas_by_phase and ideas_by_phase["questioning"]:
            key_insights.append(
                "Key questions identified fundamental aspects of the problem that need addressing"
            )
        
        if "abduction" in ideas_by_phase and ideas_by_phase["abduction"]:
            key_insights.append(
                "Root causes and hypotheses reveal underlying systemic factors"
            )
        
        if "deduction" in ideas_by_phase and ideas_by_phase["deduction"]:
            key_insights.append(
                "Logical analysis provides clear pathways for implementation"
            )
        
        if "induction" in ideas_by_phase and ideas_by_phase["induction"]:
            key_insights.append(
                "Patterns identified suggest broader principles for long-term success"
            )
        
        # Generate recommendations
        recommendations = [
            "Prioritize addressing the root causes identified in the abductive phase",
            "Implement the logical solutions from deductive reasoning",
            "Apply the patterns and principles from inductive analysis",
            "Consider the critical questions raised for comprehensive coverage"
        ]
        
        # Next steps
        next_steps = [
            "Review all generated ideas and select top priorities",
            "Develop detailed implementation plans for selected solutions",
            "Establish metrics to measure progress and success"
        ]
        
        return Conclusion(
            summary=summary,
            key_insights=key_insights,
            actionable_recommendations=recommendations,
            next_steps=next_steps,
            confidence_level=0.7
        )