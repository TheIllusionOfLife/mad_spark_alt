# agents/base_agent.py
from abc import ABC, abstractmethod
import json
import os

class BaseAgent(ABC):
    def __init__(self, llm_client, thinking_method_name: str):
        self.llm_client = llm_client # In a real scenario, this would be an LLM API client
        self.thinking_method_name = thinking_method_name
        self.prompts = self._load_prompts()

    def _load_prompts(self):
        # Construct the path relative to this file's location
        current_dir = os.path.dirname(os.path.abspath(__file__))
        prompts_path = os.path.join(current_dir, '..', 'thinking_methods', 'prompts.json')
        try:
            with open(prompts_path, 'r', encoding='utf-8') as f:
                all_prompts = json.load(f)
            # Get prompts specific to the agent's thinking method, if defined
            # This allows an agent to potentially use multiple related prompts
            return all_prompts.get(self.thinking_method_name, {})
        except FileNotFoundError:
            print(f"Warning: Prompts file not found at {prompts_path}")
            return {} # Return empty dict if file not found
        except json.JSONDecodeError:
            print(f"Warning: Error decoding JSON from {prompts_path}")
            return {} # Return empty dict if JSON is invalid

    @abstractmethod
    def execute(self, **kwargs) -> str:
        '''
        Executes the agent's thinking process.
        kwargs will contain necessary inputs like 'theme', 'question', 'observation', etc.
        Returns the generated text (idea, question, hypothesis, etc.).
        '''
        pass

    def _get_prompt_template(self, prompt_key: str) -> str:
        '''
        Retrieves a specific prompt template for a given key (e.g., "question", "answer").
        '''
        if not self.prompts:
            # This case handles when prompts.json was not found or was empty/invalid.
            # It also handles if self.thinking_method_name isn't in prompts.json
            print(f"Warning: No prompts loaded for thinking method '{self.thinking_method_name}'.")
            return ""

        prompt_data = self.prompts.get(prompt_key)
        if prompt_data and isinstance(prompt_data, dict):
            return prompt_data.get("prompt_template", "")
        elif isinstance(prompt_data, str): # Older format perhaps, or direct prompt
             # This part is more for flexibility, assuming a simpler structure might exist
             # For the current prompts.json, this branch won't be hit for qadi_cycle.*
             # but it might for "abduction" if it was just a string.
             # However, prompts.json defines "abduction" as a dict with "description" and "prompt_template"
             # So, let's adjust to always expect the dict structure for consistency.
             # The current prompts.json has "abduction" as a dict with a "prompt_template" key.
             # So, this case is unlikely with the current prompts.json.
             # It would be better if the structure is consistent.
             # Let's assume for now that all main thinking methods (like 'abduction', 'qadi_cycle')
             # in prompts.json are dictionaries containing a 'prompt_template' for their primary prompt
             # or further sub-keys which then contain 'prompt_template'.
             # For direct access like AbductionAgent, it should access self.prompts['prompt_template']
             # if 'abduction' itself is the key holding the template.
             # The current structure of prompts.json is:
             # "abduction": { "description": "...", "prompt_template": "..." }
             # "qadi_cycle": { "question": {"description":"...","prompt_template":"..."}, ...}
             # So, an AbductionAgent would have self.thinking_method_name = "abduction"
             # and its main prompt would be self.prompts["prompt_template"]
             # A QADIQuestionAgent would have self.thinking_method_name = "qadi_cycle"
             # and would call _get_prompt_template("question") to get self.prompts["question"]["prompt_template"]
            return "" # Should not happen with current prompts.json structure
        return ""


    def _format_prompt(self, prompt_key: str, **kwargs) -> str:
        '''
        Formats the prompt template with the given arguments.
        '''
        template = self._get_prompt_template(prompt_key)
        if not template:
            # If the specific sub-prompt (like "question" for QADI) is not found,
            # try to use the main prompt_template of the thinking method (e.g., for Abduction)
            if 'prompt_template' in self.prompts:
                template = self.prompts.get('prompt_template', "")
            else:
                print(f"Error: Prompt template for key '{prompt_key}' or main prompt for '{self.thinking_method_name}' not found.")
                return "" # Or raise an error
        try:
            return template.format(**kwargs)
        except KeyError as e:
            print(f"Error: Missing key {e} for formatting prompt '{template}'")
            return "" # Or raise an error
