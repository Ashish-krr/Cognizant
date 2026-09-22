import os

from dotenv import load_dotenv
from google import genai

# Load .env
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY not found in .env file."
    )

client = genai.Client(
    api_key=api_key
)

response = client.models.generate_content(
    model="models/gemini-3.5-flash-lite",
    contents="Write one short example of a vehicle insurance claim."
)

print(response.text)