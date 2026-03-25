import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load .env
load_dotenv()

# Get API key safely
api_key = os.getenv("GEMINI_API_KEY")

# Configure Gemini
genai.configure(api_key=api_key)

# List models
models = genai.list_models()

print("\nAvailable Models:\n")
for m in models:
    print(m.name)