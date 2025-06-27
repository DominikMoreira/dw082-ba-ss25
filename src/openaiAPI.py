from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class OpenAIClient:
    def __init__(self):
        self.client = OpenAI()

    def request(self, model, review, system_prompt):
        response = self.client.responses.create(
            model=model,
            input=review,
            instructions=system_prompt
        )
        return response.output_text
