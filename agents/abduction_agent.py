# agents/abduction_agent.py
from .base_agent import BaseAgent

class AbductionAgent(BaseAgent):
    def __init__(self, llm_client):
        # The "abduction" prompt is directly under the "abduction" key in prompts.json
        super().__init__(llm_client, thinking_method_name="abduction")

    def execute(self, observation: str) -> str:
        # For AbductionAgent, the thinking_method_name "abduction" directly has the "prompt_template"
        # So, _format_prompt will try to get self.prompts["prompt_template"]
        formatted_prompt = self._format_prompt(prompt_key="prompt_template", observation=observation) # Pass a dummy key or adjust _format_prompt
                                                                                                   # Let's adjust _format_prompt to handle this.
                                                                                                   # Actually, BaseAgent._format_prompt was adjusted:
                                                                                                   # if not template: # (template for prompt_key not found)
                                                                                                   #    if 'prompt_template' in self.prompts:
                                                                                                   #        template = self.prompts.get('prompt_template', "")
                                                                                                   # So, we can pass any key if it's not found, it will fallback.
                                                                                                   # Or pass 'prompt_template' if that's the actual key inside "abduction"
                                                                                                   # The JSON is: "abduction": { "description": "...", "prompt_template": "..." }
                                                                                                   # So, self.prompts will be {"description": "...", "prompt_template": "..."}
                                                                                                   # And calling self._format_prompt("prompt_template", ...) will correctly use it.

        # Correct way if "abduction" prompt is self.prompts['prompt_template']
        # The `_format_prompt` method in BaseAgent has a fallback:
        # If `template = self._get_prompt_template(prompt_key)` is empty,
        # it tries `template = self.prompts.get('prompt_template', "")`.
        # So, we can call it with a key that won't be found in self.prompts (e.g. "main_prompt")
        # or directly with "prompt_template" if that's the key within the "abduction" part of the JSON.
        # Given `prompts.json` structure for "abduction":
        # "abduction": { "description": "...", "prompt_template": "..." }
        # `self.prompts` for `AbductionAgent` will be this dictionary.
        # So we need to access `self.prompts['prompt_template']`.
        # `_get_prompt_template('prompt_template')` would look for `self.prompts['prompt_template']['prompt_template']` which is wrong.
        # `_format_prompt` needs to handle the case where `self.prompts` *is* the dict containing `prompt_template`.

        # The current `_format_prompt` logic:
        # 1. `template = self._get_prompt_template(prompt_key)`
        #    `_get_prompt_template` gets `self.prompts.get(prompt_key)`. If `prompt_key` is "prompt_template",
        #    this becomes `self.prompts.get("prompt_template")` which is the actual template string.
        #    Then it tries `prompt_data.get("prompt_template", "")` which will fail for a string.
        #    This needs fixing in BaseAgent._get_prompt_template.

        # Let's refine BaseAgent's _get_prompt_template and _format_prompt in the subtask.
        # For now, let's assume the subtask will fix it as described in the comments within base_agent.py.
        # The fixed _format_prompt (as in the base_agent.py content of this subtask)
        # should correctly fetch the prompt if 'prompt_template' is passed as prompt_key
        # and self.prompts is the dict containing 'prompt_template'.

        formatted_prompt = self._format_prompt("prompt_template", observation=observation)


        simulated_llm_response = f"Generated abduction for observation '{observation}' based on: {formatted_prompt}"
        print(f"AbductionAgent: Called LLM with prompt: \n{formatted_prompt}")
        return simulated_llm_response
