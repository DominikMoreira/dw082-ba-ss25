from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class OpenAIClient:
    def __init__(self):
        self.client = OpenAI()

    def request(self, review, system_prompt):
        response = self.client.responses.create(
            model="gpt-4.1-nano",
            input=review,
            instructions=system_prompt
        )
        return response.output_text

