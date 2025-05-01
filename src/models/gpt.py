from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

client = OpenAI()

response = client.responses.create(
    model="gpt-4.1-nano",
    input="Write a one-sentence bedtime story about a unicorn."
)

print(response.output_text)