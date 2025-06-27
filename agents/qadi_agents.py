# agents/qadi_agents.py
from .base_agent import BaseAgent

class QADIQuestionAgent(BaseAgent):
    def __init__(self, llm_client):
        super().__init__(llm_client, thinking_method_name="qadi_cycle")

    def execute(self, theme: str) -> str:
        # In a real scenario, self.llm_client would be used to call an LLM
        # For now, we simulate LLM call by returning the formatted prompt
        formatted_prompt = self._format_prompt("question", theme=theme)
        # Simulated LLM response
        simulated_llm_response = f"Generated questions for theme '{theme}' based on: {formatted_prompt}"
        print(f"QADIQuestionAgent: Called LLM with prompt: \n{formatted_prompt}")
        return simulated_llm_response


class QADIAnswerAgent(BaseAgent):
    def __init__(self, llm_client):
        super().__init__(llm_client, thinking_method_name="qadi_cycle")

    def execute(self, question: str) -> str:
        formatted_prompt = self._format_prompt("answer", question=question)
        simulated_llm_response = f"Generated answer for question '{question}' based on: {formatted_prompt}"
        print(f"QADIAnswerAgent: Called LLM with prompt: \n{formatted_prompt}")
        return simulated_llm_response

class QADIDeeperQuestionAgent(BaseAgent):
    def __init__(self, llm_client):
        super().__init__(llm_client, thinking_method_name="qadi_cycle")

    def execute(self, answer: str) -> str:
        formatted_prompt = self._format_prompt("deeper_question", answer=answer)
        simulated_llm_response = f"Generated deeper questions for answer '{answer}' based on: {formatted_prompt}"
        print(f"QADIDeeperQuestionAgent: Called LLM with prompt: \n{formatted_prompt}")
        return simulated_llm_response

class QADIInsightAgent(BaseAgent):
    def __init__(self, llm_client):
        super().__init__(llm_client, thinking_method_name="qadi_cycle")

    def execute(self, original_question: str, answer_to_original_question: str, deeper_question: str, answer_to_deeper_question: str) -> str:
        formatted_prompt = self._format_prompt("insight_idea",
                                                original_question=original_question,
                                                answer_to_original_question=answer_to_original_question,
                                                deeper_question=deeper_question,
                                                answer_to_deeper_question=answer_to_deeper_question)
        simulated_llm_response = f"Generated insight based on Q&A flow: {formatted_prompt}"
        print(f"QADIInsightAgent: Called LLM with prompt: \n{formatted_prompt}")
        return simulated_llm_response
