from google import genai
import os

# Ensure your key is set in your environment variables
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

print("Available models for this key:")
for m in client.models.list():
    if "generateContent" in m.supported_actions:
        print(f"- {m.name} ({m.display_name})")