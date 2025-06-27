# agents/llm_client.py

class MockLLMClient:
    def __init__(self, api_key="dummy_api_key"):
        self.api_key = api_key
        print(f"MockLLMClient initialized. (Note: This is a mock client and does not make real API calls.)")

    def generate_text(self, prompt: str, model: str = "mock-model") -> str:
        print(f"MockLLMClient: Received prompt for model '{model}':\n{prompt}")
        # Simulate a delay or some processing
        response = f"Mock response for: {prompt[:100]}..."
        if "question" in prompt.lower():
            response = f"Mock questions based on: {prompt[:100]}..."
        elif "answer" in prompt.lower():
            response = f"Mock answer based on: {prompt[:100]}..."
        elif "hypothesis" in prompt.lower() or "explanation" in prompt.lower() :
            response = f"Mock hypothesis for: {prompt[:100]}..."
        print(f"MockLLMClient: Sending back simulated response: {response}")
        return response

# Example usage (optional, for testing)
if __name__ == '__main__':
    client = MockLLMClient()
    test_prompt = "What are three interesting facts about the Roman Empire?"
    response = client.generate_text(test_prompt)
    print(f"Test response: {response}")
