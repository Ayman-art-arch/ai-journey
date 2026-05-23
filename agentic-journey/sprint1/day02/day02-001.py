"""
day02-001 — Hello World with the Gemini API
Objective: First call to Gemini via the google-genai SDK
"""
from dotenv import load_dotenv
from pathlib import Path
import os

load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env")

from google import genai

# Create a Gemini client with the API key
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

# Simple call — Gemini accepts a string directly (no need for messages=[...])
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Explain in 3 sentences why Python is popular for AI.",
)

print("=== Gemini Response ===")
print(response.text)

# Streaming — progressive response, chunk by chunk
print("\n=== Streaming Response ===")
stream = client.models.generate_content_stream(
    model="gemini-2.5-flash",
    contents="Give 5 tips for getting started with machine learning.",
)

for chunk in stream:
    print(chunk.text, end="", flush=True)

print()  # final newline
