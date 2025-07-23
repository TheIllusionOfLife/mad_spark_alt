"""
Multi-Perspective QADI Prompts

This module provides perspective-specific prompts for the QADI methodology,
enabling analysis from environmental, personal, technical, business, scientific,
and philosophical viewpoints.
"""

from typing import Dict

from .intent_detector import QuestionIntent


class MultiPerspectivePrompts:
    """Perspective-specific prompts for QADI analysis."""

    def __init__(self) -> None:
        """Initialize with perspective-specific prompt templates."""
        self.perspective_prompts = {
            QuestionIntent.ENVIRONMENTAL: {
                "role": "environmental sustainability expert",
                "question_focus": "ecological impact and sustainable solutions",
                "hypothesis_focus": "environmentally responsible approaches",
                "evaluation_criteria": "environmental benefit, feasibility, and long-term sustainability",
                "verification_focus": "real-world environmental success stories",
            },
            QuestionIntent.PERSONAL: {
                "role": "personal development coach",
                "question_focus": "individual actions and personal growth",
                "hypothesis_focus": "practical personal strategies",
                "evaluation_criteria": "personal impact, ease of adoption, and lifestyle compatibility",
                "verification_focus": "individual success stories and behavioral changes",
            },
            QuestionIntent.TECHNICAL: {
                "role": "technical solutions architect",
                "question_focus": "technical implementation and system design",
                "hypothesis_focus": "technological solutions and architectures",
                "evaluation_criteria": "technical feasibility, scalability, and innovation",
                "verification_focus": "successful technical implementations",
            },
            QuestionIntent.BUSINESS: {
                "role": "strategic business consultant",
                "question_focus": "business value and strategic objectives",
                "hypothesis_focus": "business strategies and operational improvements",
                "evaluation_criteria": "ROI, market impact, and competitive advantage",
                "verification_focus": "business case studies and market examples",
            },
            QuestionIntent.SCIENTIFIC: {
                "role": "research scientist",
                "question_focus": "scientific understanding and evidence-based solutions",
                "hypothesis_focus": "research-backed approaches and methodologies",
                "evaluation_criteria": "scientific validity, empirical support, and reproducibility",
                "verification_focus": "peer-reviewed studies and research findings",
            },
            QuestionIntent.PHILOSOPHICAL: {
                "role": "philosophical thinker",
                "question_focus": "ethical implications and fundamental principles",
                "hypothesis_focus": "value-based approaches and ethical frameworks",
                "evaluation_criteria": "ethical soundness, societal benefit, and principled consistency",
                "verification_focus": "philosophical precedents and ethical applications",
            },
            QuestionIntent.GENERAL: {
                "role": "comprehensive analyst",
                "question_focus": "core challenges and opportunities",
                "hypothesis_focus": "diverse solution approaches",
                "evaluation_criteria": "overall effectiveness and practical implementation",
                "verification_focus": "varied real-world applications",
            },
        }

    def get_questioning_prompt(
        self, user_input: str, perspective: QuestionIntent
    ) -> str:
        """Get perspective-specific questioning prompt."""
        config = self.perspective_prompts[perspective]
        return f"""As a {config['role']}, identify THE single most important question focusing on {config['question_focus']}.

User's input:
{user_input}

Think about:
- What is the core {perspective.value} challenge or opportunity here?
- What question would provide maximum insight from a {perspective.value} perspective?
- What specific aspect needs the most attention?

Output exactly ONE core question.
Format: "Q: [Your core question]"
"""

    def get_abduction_prompt(
        self, user_input: str, core_question: str, perspective: QuestionIntent
    ) -> str:
        """Get perspective-specific hypothesis generation prompt."""
        config = self.perspective_prompts[perspective]
        return f"""As a {config['role']}, generate 3 specific hypotheses focusing on {config['hypothesis_focus']}.

Core Question: {core_question}

User's original input:
{user_input}

Each hypothesis should:
- Directly answer the question from a {perspective.value} perspective
- Be specific and actionable
- Focus on {config['hypothesis_focus']}

Format:
H1: [First hypothesis]
H2: [Second hypothesis]
H3: [Third hypothesis]
"""

    def get_deduction_prompt(
        self,
        user_input: str,
        core_question: str,
        hypotheses: str,
        perspective: QuestionIntent,
    ) -> str:
        """Get perspective-specific evaluation prompt."""
        config = self.perspective_prompts[perspective]

        # Adjust criteria labels based on perspective
        criteria = self._get_perspective_criteria(perspective)

        # Extract criteria names for the format template
        criteria_lines = [
            line.strip()
            for line in criteria.split("\n")
            if line.strip() and ":" in line
        ]
        criteria_names = []
        for line in criteria_lines:
            name = line.split(":")[0].replace("-", "").strip()
            criteria_names.append(name)

        # Build the format template
        criteria_format = "\n  ".join(
            [f"* {name}: [score] - [brief explanation]" for name in criteria_names]
        )

        return f"""As a {config['role']}, evaluate each hypothesis based on {config['evaluation_criteria']}.

Core Question: {core_question}

Hypotheses to evaluate:
{hypotheses}

User's original context:
{user_input}

Score each hypothesis from 0.0 to 1.0 on:
{criteria}

Format:
Analysis:
- H1: 
  {criteria_format}
  * Overall: [calculated weighted score]

- H2: 
  [Same format as H1]

- H3: 
  [Same format as H1]

ANSWER: [Your definitive answer from a {perspective.value} perspective]

Action Plan:
1. [First specific action from {perspective.value} perspective]
2. [Second concrete action]
3. [Third actionable step]
"""

    def get_induction_prompt(
        self,
        user_input: str,
        core_question: str,
        answer: str,
        perspective: QuestionIntent,
    ) -> str:
        """Get perspective-specific verification prompt."""
        config = self.perspective_prompts[perspective]
        return f"""As a {config['role']}, verify the answer by examining {config['verification_focus']}.

Core Question: {core_question}
Our Answer: {answer}

Original context:
{user_input}

Provide 3 examples that demonstrate this answer's validity:

Format:
Verification:
1. [Real-world example from {perspective.value} domain]
2. [Different context where this {perspective.value} principle applies]
3. [Future scenario where this would be effective]

Conclusion: [Is this answer valid from a {perspective.value} perspective? Any limitations?]
"""

    def _get_perspective_criteria(self, perspective: QuestionIntent) -> str:
        """Get evaluation criteria tailored to each perspective."""
        criteria_map = {
            QuestionIntent.ENVIRONMENTAL: """- Environmental Impact: How much positive environmental change? (0=harmful, 1=highly beneficial)
- Sustainability: How sustainable long-term? (0=unsustainable, 1=fully sustainable)
- Resource Efficiency: How efficient with natural resources? (0=wasteful, 1=highly efficient)
- Implementation Feasibility: How practical to implement? (0=very difficult, 1=easy)
- Ecosystem Benefits: Benefits to ecosystems/biodiversity? (0=none, 1=significant)""",
            QuestionIntent.PERSONAL: """- Personal Impact: How much positive personal change? (0=none, 1=transformative)
- Ease of Adoption: How easy for individuals to adopt? (0=very difficult, 1=effortless)
- Time Investment: Time required? (0=excessive, 1=minimal)
- Cost to Individual: Financial cost? (0=expensive, 1=free/saves money)
- Long-term Benefits: Sustained personal benefits? (0=temporary, 1=lasting)""",
            QuestionIntent.TECHNICAL: """- Technical Innovation: How innovative technically? (0=outdated, 1=cutting-edge)
- Implementation Complexity: Technical difficulty? (0=extremely complex, 1=simple)
- Scalability: How well does it scale? (0=doesn't scale, 1=infinitely scalable)
- Performance: Technical performance/efficiency? (0=poor, 1=excellent)
- Maintainability: Ease of maintenance? (0=nightmare, 1=self-maintaining)""",
            QuestionIntent.BUSINESS: """- Business Value: Revenue/profit potential? (0=loss-making, 1=highly profitable)
- Market Impact: Market differentiation? (0=no impact, 1=market-defining)
- Cost Efficiency: Resource requirements? (0=expensive, 1=cost-effective)
- Implementation Speed: Time to market? (0=years, 1=immediate)
- Risk Level: Business risks? (0=high risk, 1=low risk)""",
            QuestionIntent.SCIENTIFIC: """- Scientific Validity: Evidence support? (0=pseudoscience, 1=well-established)
- Research Impact: Contribution to knowledge? (0=none, 1=breakthrough)
- Reproducibility: Can findings be reproduced? (0=no, 1=highly reproducible)
- Practical Application: Real-world applicability? (0=theoretical only, 1=immediately applicable)
- Peer Acceptance: Scientific community support? (0=rejected, 1=widely accepted)""",
            QuestionIntent.PHILOSOPHICAL: """- Ethical Soundness: Alignment with ethical principles? (0=unethical, 1=highly ethical)
- Societal Benefit: Benefit to society? (0=harmful, 1=transformative)
- Universal Applicability: Works across cultures? (0=culturally specific, 1=universal)
- Principled Consistency: Internal consistency? (0=contradictory, 1=coherent)
- Human Flourishing: Promotes wellbeing? (0=diminishes, 1=enhances)""",
            QuestionIntent.GENERAL: """- Overall Effectiveness: How well does it work? (0=ineffective, 1=highly effective)
- Practical Feasibility: How practical to implement? (0=impractical, 1=easily done)
- Resource Requirements: Resources needed? (0=excessive, 1=minimal)
- Adoption Potential: Likelihood of adoption? (0=unlikely, 1=certain)
- Long-term Viability: Sustainable over time? (0=temporary, 1=permanent)""",
        }

        return criteria_map.get(perspective, criteria_map[QuestionIntent.GENERAL])
