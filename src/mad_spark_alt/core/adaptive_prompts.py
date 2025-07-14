"""
Adaptive Prompt Templates for Dynamic QADI Analysis

This module provides specialized prompt templates for different question types
and complexity levels to optimize QADI analysis results.
"""

from typing import Dict, Any
from .prompt_classifier import QuestionType, ComplexityLevel, ClassificationResult


class AdaptivePromptGenerator:
    """Generates optimized prompts based on question classification."""
    
    def __init__(self):
        """Initialize the adaptive prompt generator with specialized templates."""
        
        # Technical question prompts - focus on implementation and architecture
        self.technical_prompts = {
            "questioning": {
                "regular": """As a technical architect, generate 2 strategic technical questions about: "{prompt}"
{previous_insights}
Focus on scalability, maintainability, and technical feasibility.
Consider architecture decisions, technology choices, and implementation challenges.
Format each question on a new line starting with "Q:".""",
                
                "concrete": """As a senior software engineer, generate 2 practical implementation questions about: "{prompt}"
{previous_insights}
Focus on specific technical requirements, development tools, and deployment concerns.
Consider performance, security, and integration requirements.
Format each question on a new line starting with "Q:"."""
            },
            
            "abduction": {
                "regular": """As a solutions architect, generate 2 innovative technical approaches for: "{prompt}"
{previous_insights}
Consider emerging technologies, architectural patterns, and creative solutions.
Think about microservices, cloud-native approaches, and modern development practices.
Format each approach on a new line starting with "H:".""",
                
                "concrete": """As a technical lead, generate 2 specific implementation solutions for: "{prompt}"
{previous_insights}
Provide concrete technology stacks, frameworks, and development approaches.
Include specific tools, libraries, and platforms (e.g., React, Node.js, AWS, Docker).
Format each solution on a new line starting with "H:"."""
            },
            
            "deduction": {
                "regular": """As a system architect, generate 2 logical technical conclusions about: "{prompt}"
{previous_insights}
Apply engineering principles and systematic reasoning to derive sound conclusions.
Consider performance implications, security requirements, and scalability factors.
Format each conclusion on a new line starting with "D:".""",
                
                "concrete": """As a project technical lead, generate 2 logical implementation steps for: "{prompt}"
{previous_insights}
Focus on development phases, deployment pipelines, and technical milestones.
Include specific actions like "Set up CI/CD", "Implement authentication", "Deploy to staging".
Format each step on a new line starting with "D:"."""
            },
            
            "induction": {
                "regular": """As a principal engineer, generate 2 technical patterns and best practices for: "{prompt}"
{previous_insights}
Identify proven architectural patterns, design principles, and engineering methodologies.
Consider industry standards, performance patterns, and maintainability principles.
Format each pattern on a new line starting with "I:".""",
                
                "concrete": """As a technical advisor, generate 2 concrete technical methodologies for: "{prompt}"
{previous_insights}
Identify specific development practices, testing strategies, and deployment patterns.
Include actionable practices like "Test-driven development", "Blue-green deployment", "Code reviews".
Format each methodology on a new line starting with "I:"."""
            }
        }
        
        # Business question prompts - focus on strategy and growth
        self.business_prompts = {
            "questioning": {
                "regular": """As a strategic business consultant, generate 2 insightful business questions about: "{prompt}"
{previous_insights}
Focus on market dynamics, competitive positioning, and strategic opportunities.
Consider customer needs, revenue models, and growth potential.
Format each question on a new line starting with "Q:".""",
                
                "concrete": """As a business development expert, generate 2 practical business questions about: "{prompt}"
{previous_insights}
Focus on implementation challenges, resource requirements, and measurable outcomes.
Consider budget constraints, timeline feasibility, and operational requirements.
Format each question on a new line starting with "Q:"."""
            },
            
            "abduction": {
                "regular": """As a business strategist, generate 2 innovative business hypotheses about: "{prompt}"
{previous_insights}
Consider disruptive business models, market opportunities, and strategic innovations.
Think about customer value propositions, competitive advantages, and growth strategies.
Format each hypothesis on a new line starting with "H:".""",
                
                "concrete": """As a business operations manager, generate 2 specific business solutions for: "{prompt}"
{previous_insights}
Provide concrete business strategies, operational approaches, and implementation tactics.
Include specific metrics, target markets, and actionable business plans.
Format each solution on a new line starting with "H:"."""
            },
            
            "deduction": {
                "regular": """As a business analyst, generate 2 logical business conclusions about: "{prompt}"
{previous_insights}
Apply business reasoning and market analysis to derive strategic insights.
Consider market trends, competitive dynamics, and financial implications.
Format each conclusion on a new line starting with "D:".""",
                
                "concrete": """As a business project manager, generate 2 logical business implementation steps for: "{prompt}"
{previous_insights}
Focus on actionable business processes, operational steps, and measurable milestones.
Include specific actions like "Conduct market research", "Develop pricing strategy", "Launch pilot program".
Format each step on a new line starting with "D:"."""
            },
            
            "induction": {
                "regular": """As a senior business strategist, generate 2 business patterns and principles for: "{prompt}"
{previous_insights}
Identify proven business models, strategic frameworks, and success patterns.
Consider industry best practices, growth methodologies, and strategic principles.
Format each pattern on a new line starting with "I:".""",
                
                "concrete": """As a business operations expert, generate 2 concrete business methodologies for: "{prompt}"
{previous_insights}
Identify specific business practices, operational frameworks, and implementation strategies.
Include actionable approaches like "Agile project management", "Customer feedback loops", "A/B testing".
Format each methodology on a new line starting with "I:"."""
            }
        }
        
        # Creative question prompts - focus on innovation and artistic thinking
        self.creative_prompts = {
            "questioning": {
                "regular": """As a creative director, generate 2 imaginative questions about: "{prompt}"
{previous_insights}
Focus on artistic possibilities, innovative approaches, and creative potential.
Consider aesthetic dimensions, user experience, and emotional impact.
Format each question on a new line starting with "Q:".""",
                
                "concrete": """As a design project manager, generate 2 practical creative questions about: "{prompt}"
{previous_insights}
Focus on design implementation, creative resources, and production feasibility.
Consider design tools, creative workflows, and deliverable requirements.
Format each question on a new line starting with "Q:"."""
            },
            
            "abduction": {
                "regular": """As a creative visionary, generate 2 breakthrough creative concepts for: "{prompt}"
{previous_insights}
Consider revolutionary artistic approaches, innovative design solutions, and creative breakthroughs.
Think about unconventional methods, artistic inspiration, and transformative ideas.
Format each concept on a new line starting with "H:".""",
                
                "concrete": """As a creative producer, generate 2 specific creative solutions for: "{prompt}"
{previous_insights}
Provide concrete design approaches, creative techniques, and production methods.
Include specific tools, materials, and creative processes (e.g., Figma, Adobe Creative Suite, prototyping).
Format each solution on a new line starting with "H:"."""
            },
            
            "deduction": {
                "regular": """As a design theorist, generate 2 logical creative conclusions about: "{prompt}"
{previous_insights}
Apply design principles and creative reasoning to derive artistic insights.
Consider aesthetic theory, user psychology, and creative impact.
Format each conclusion on a new line starting with "D:".""",
                
                "concrete": """As a creative project lead, generate 2 logical creative implementation steps for: "{prompt}"
{previous_insights}
Focus on design phases, creative milestones, and production workflows.
Include specific actions like "Create mood boards", "Develop prototypes", "Conduct user testing".
Format each step on a new line starting with "D:"."""
            },
            
            "induction": {
                "regular": """As a master creative practitioner, generate 2 creative patterns and principles for: "{prompt}"
{previous_insights}
Identify proven design methodologies, creative frameworks, and artistic principles.
Consider design thinking, creative processes, and innovation methodologies.
Format each pattern on a new line starting with "I:".""",
                
                "concrete": """As a creative operations expert, generate 2 concrete creative methodologies for: "{prompt}"
{previous_insights}
Identify specific design practices, creative workflows, and production techniques.
Include actionable approaches like "Design sprints", "User-centered design", "Iterative prototyping".
Format each methodology on a new line starting with "I:"."""
            }
        }
        
        # Research question prompts - focus on analysis and investigation
        self.research_prompts = {
            "questioning": {
                "regular": """As a research methodologist, generate 2 analytical research questions about: "{prompt}"
{previous_insights}
Focus on investigative approaches, research methodologies, and analytical frameworks.
Consider data sources, research validity, and systematic investigation methods.
Format each question on a new line starting with "Q:".""",
                
                "concrete": """As a research project coordinator, generate 2 practical research questions about: "{prompt}"
{previous_insights}
Focus on research implementation, data collection methods, and study feasibility.
Consider resource requirements, timeline constraints, and methodological practicality.
Format each question on a new line starting with "Q:"."""
            },
            
            "abduction": {
                "regular": """As a research theorist, generate 2 innovative research hypotheses about: "{prompt}"
{previous_insights}
Consider novel research approaches, theoretical frameworks, and investigative methods.
Think about interdisciplinary connections, emerging methodologies, and research innovations.
Format each hypothesis on a new line starting with "H:".""",
                
                "concrete": """As a research analyst, generate 2 specific research approaches for: "{prompt}"
{previous_insights}
Provide concrete research methods, data collection techniques, and analytical tools.
Include specific methodologies, software tools, and research protocols.
Format each approach on a new line starting with "H:"."""
            },
            
            "deduction": {
                "regular": """As a research analyst, generate 2 logical research conclusions about: "{prompt}"
{previous_insights}
Apply systematic reasoning and analytical thinking to derive research insights.
Consider evidence evaluation, logical inference, and research validity.
Format each conclusion on a new line starting with "D:".""",
                
                "concrete": """As a research project manager, generate 2 logical research implementation steps for: "{prompt}"
{previous_insights}
Focus on research phases, data collection milestones, and analytical procedures.
Include specific actions like "Design survey instrument", "Recruit participants", "Analyze data".
Format each step on a new line starting with "D:"."""
            },
            
            "induction": {
                "regular": """As a senior researcher, generate 2 research patterns and methodologies for: "{prompt}"
{previous_insights}
Identify proven research frameworks, analytical methods, and investigative principles.
Consider research best practices, methodological rigor, and analytical approaches.
Format each pattern on a new line starting with "I:".""",
                
                "concrete": """As a research operations specialist, generate 2 concrete research methodologies for: "{prompt}"
{previous_insights}
Identify specific research practices, analytical techniques, and data collection methods.
Include actionable approaches like "Systematic literature review", "Mixed-methods analysis", "Peer review process".
Format each methodology on a new line starting with "I:"."""
            }
        }
        
        # Planning question prompts - focus on organization and execution
        self.planning_prompts = {
            "questioning": {
                "regular": """As a strategic planning consultant, generate 2 insightful planning questions about: "{prompt}"
{previous_insights}
Focus on strategic considerations, planning methodologies, and organizational approaches.
Consider dependencies, resource allocation, and strategic alignment.
Format each question on a new line starting with "Q:".""",
                
                "concrete": """As a project planning specialist, generate 2 practical planning questions about: "{prompt}"
{previous_insights}
Focus on implementation logistics, resource requirements, and execution challenges.
Consider timeline constraints, budget limitations, and operational feasibility.
Format each question on a new line starting with "Q:"."""
            },
            
            "abduction": {
                "regular": """As a planning strategist, generate 2 innovative planning approaches for: "{prompt}"
{previous_insights}
Consider creative planning methodologies, strategic frameworks, and organizational innovations.
Think about agile approaches, collaborative planning, and adaptive strategies.
Format each approach on a new line starting with "H:".""",
                
                "concrete": """As a project coordinator, generate 2 specific planning solutions for: "{prompt}"
{previous_insights}
Provide concrete planning methods, organizational tools, and execution strategies.
Include specific planning software, methodologies, and management approaches.
Format each solution on a new line starting with "H:"."""
            },
            
            "deduction": {
                "regular": """As a planning analyst, generate 2 logical planning conclusions about: "{prompt}"
{previous_insights}
Apply systematic planning principles and organizational reasoning to derive insights.
Consider resource optimization, timeline efficiency, and strategic execution.
Format each conclusion on a new line starting with "D:".""",
                
                "concrete": """As a project execution manager, generate 2 logical planning implementation steps for: "{prompt}"
{previous_insights}
Focus on execution phases, milestone delivery, and operational procedures.
Include specific actions like "Create project charter", "Assign team roles", "Set up tracking systems".
Format each step on a new line starting with "D:"."""
            },
            
            "induction": {
                "regular": """As a master planning practitioner, generate 2 planning patterns and principles for: "{prompt}"
{previous_insights}
Identify proven planning methodologies, organizational frameworks, and execution principles.
Consider project management best practices, planning theories, and strategic approaches.
Format each pattern on a new line starting with "I:".""",
                
                "concrete": """As a planning operations expert, generate 2 concrete planning methodologies for: "{prompt}"
{previous_insights}
Identify specific planning practices, organizational tools, and execution techniques.
Include actionable approaches like "Gantt chart planning", "Agile sprints", "Risk management matrices".
Format each methodology on a new line starting with "I:"."""
            }
        }
        
        # Personal question prompts - focus on individual growth and development
        self.personal_prompts = {
            "questioning": {
                "regular": """As a personal development coach, generate 2 insightful personal questions about: "{prompt}"
{previous_insights}
Focus on self-reflection, personal growth opportunities, and individual potential.
Consider values alignment, personal goals, and development pathways.
Format each question on a new line starting with "Q:".""",
                
                "concrete": """As a life coach, generate 2 practical personal questions about: "{prompt}"
{previous_insights}
Focus on actionable steps, personal resources, and realistic goal achievement.
Consider time constraints, personal circumstances, and practical implementation.
Format each question on a new line starting with "Q:"."""
            },
            
            "abduction": {
                "regular": """As a personal growth strategist, generate 2 innovative personal development approaches for: "{prompt}"
{previous_insights}
Consider creative growth methods, personal transformation strategies, and development innovations.
Think about holistic approaches, mindfulness practices, and personal breakthroughs.
Format each approach on a new line starting with "H:".""",
                
                "concrete": """As a personal effectiveness coach, generate 2 specific personal solutions for: "{prompt}"
{previous_insights}
Provide concrete development methods, personal tools, and growth strategies.
Include specific techniques, resources, and personal development approaches.
Format each solution on a new line starting with "H:"."""
            },
            
            "deduction": {
                "regular": """As a personal development analyst, generate 2 logical personal conclusions about: "{prompt}"
{previous_insights}
Apply psychological principles and personal development reasoning to derive insights.
Consider behavior patterns, motivation factors, and personal effectiveness.
Format each conclusion on a new line starting with "D:".""",
                
                "concrete": """As a personal action coach, generate 2 logical personal implementation steps for: "{prompt}"
{previous_insights}
Focus on actionable personal steps, development milestones, and growth activities.
Include specific actions like "Create daily routine", "Set SMART goals", "Track progress weekly".
Format each step on a new line starting with "D:"."""
            },
            
            "induction": {
                "regular": """As a master personal development practitioner, generate 2 personal growth patterns and principles for: "{prompt}"
{previous_insights}
Identify proven development methodologies, personal growth frameworks, and success principles.
Consider psychology research, personal effectiveness theories, and growth best practices.
Format each pattern on a new line starting with "I:".""",
                
                "concrete": """As a personal development specialist, generate 2 concrete personal methodologies for: "{prompt}"
{previous_insights}
Identify specific development practices, personal tools, and growth techniques.
Include actionable approaches like "Habit stacking", "Mindfulness meditation", "Goal tracking systems".
Format each methodology on a new line starting with "I:"."""
            }
        }
        
        # Default fallback prompts (current system prompts)
        self.default_prompts = {
            "questioning": {
                "regular": """As a questioning specialist, generate 2 insightful questions about: "{prompt}"
{previous_insights}
Format each question on a new line starting with "Q:".""",
                
                "concrete": """As an implementation specialist, generate 2 practical questions about: "{prompt}"
{previous_insights}
Focus on implementation challenges, resource requirements, and feasibility concerns.
Format each question on a new line starting with "Q:"."""
            },
            
            "abduction": {
                "regular": """As a hypothesis specialist, generate 2 creative hypotheses about: "{prompt}"
{previous_insights}
Consider unexpected connections and possibilities.
Format each hypothesis on a new line starting with "H:".""",
                
                "concrete": """As a solution architect, generate 2 specific, implementable solutions for: "{prompt}"
{previous_insights}
Provide concrete approaches with specific tools, methods, or technologies.
Include real-world examples where possible.
Format each solution on a new line starting with "H:"."""
            },
            
            "deduction": {
                "regular": """As a logical reasoning specialist, generate 2 logical deductions about: "{prompt}"
{previous_insights}
Apply systematic reasoning and derive conclusions.
Format each deduction on a new line starting with "D:".""",
                
                "concrete": """As a project planner, generate 2 logical implementation steps for: "{prompt}"
{previous_insights}
Focus on step-by-step approaches, prerequisites, and concrete actions.
Format each step on a new line starting with "D:"."""
            },
            
            "induction": {
                "regular": """As a pattern synthesis specialist, generate 2 pattern-based insights about: "{prompt}"
{previous_insights}
Identify recurring themes and general principles.
Format each insight on a new line starting with "I:".""",
                
                "concrete": """As a best practices specialist, generate 2 concrete patterns or methodologies for: "{prompt}"
{previous_insights}
Identify proven approaches, specific frameworks, and actionable principles.
Format each pattern on a new line starting with "I:"."""
            }
        }
        
        # Map question types to their prompt sets
        self.prompt_sets = {
            QuestionType.TECHNICAL: self.technical_prompts,
            QuestionType.BUSINESS: self.business_prompts,
            QuestionType.CREATIVE: self.creative_prompts,
            QuestionType.RESEARCH: self.research_prompts,
            QuestionType.PLANNING: self.planning_prompts,
            QuestionType.PERSONAL: self.personal_prompts,
            QuestionType.UNKNOWN: self.default_prompts
        }
    
    def get_adaptive_prompt(
        self,
        phase_name: str,
        question_type: QuestionType,
        prompt: str,
        previous_insights: str = "",
        concrete_mode: bool = False
    ) -> str:
        """
        Get an adaptive prompt based on question classification.
        
        Args:
            phase_name: QADI phase name (questioning, abduction, deduction, induction)
            question_type: Classified question type
            prompt: Original question/prompt
            previous_insights: Previous phase results
            concrete_mode: Whether to use concrete mode
            
        Returns:
            Formatted prompt optimized for the question type
        """
        # Get the appropriate prompt set
        prompt_set = self.prompt_sets.get(question_type, self.default_prompts)
        
        # Get the phase prompts
        phase_prompts = prompt_set.get(phase_name, self.default_prompts[phase_name])
        
        # Choose regular or concrete mode
        mode = "concrete" if concrete_mode else "regular"
        template = phase_prompts.get(mode, phase_prompts["regular"])
        
        # Format the template with the provided values
        return template.format(
            prompt=prompt,
            previous_insights=previous_insights
        )
    
    def get_complexity_adjusted_params(self, complexity: ComplexityLevel) -> Dict[str, Any]:
        """
        Get LLM parameters adjusted for question complexity.
        
        Args:
            complexity: Detected complexity level
            
        Returns:
            Dictionary of adjusted parameters
        """
        if complexity == ComplexityLevel.SIMPLE:
            return {
                "max_tokens": 200,
                "temperature": 0.5
            }
        elif complexity == ComplexityLevel.COMPLEX:
            return {
                "max_tokens": 400,
                "temperature": 0.8
            }
        else:  # MEDIUM
            return {
                "max_tokens": 300,
                "temperature": 0.7
            }


# Global adaptive prompt generator instance
adaptive_generator = AdaptivePromptGenerator()


def get_adaptive_prompt(
    phase_name: str,
    classification_result: ClassificationResult,
    prompt: str,
    previous_insights: str = "",
    concrete_mode: bool = False
) -> str:
    """
    Convenience function to get an adaptive prompt.
    
    Args:
        phase_name: QADI phase name
        classification_result: Question classification result
        prompt: Original question/prompt
        previous_insights: Previous phase results
        concrete_mode: Whether to use concrete mode
        
    Returns:
        Formatted adaptive prompt
    """
    return adaptive_generator.get_adaptive_prompt(
        phase_name=phase_name,
        question_type=classification_result.question_type,
        prompt=prompt,
        previous_insights=previous_insights,
        concrete_mode=concrete_mode
    )


def get_complexity_adjusted_params(classification_result: ClassificationResult) -> Dict[str, Any]:
    """
    Convenience function to get complexity-adjusted parameters.
    
    Args:
        classification_result: Question classification result
        
    Returns:
        Dictionary of adjusted parameters
    """
    return adaptive_generator.get_complexity_adjusted_params(classification_result.complexity)