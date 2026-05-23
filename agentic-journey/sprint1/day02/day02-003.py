"""
day02-003 — Hello World Ollama (Local)
Objective: First call to a local LLM via the Ollama SDK
"""
from ollama import chat, list as ollama_list

# See which models are installed locally
print("=== Installed Models ===")
models = ollama_list()
for model in models.models:
    size_gb = model.size / (1024**3)
    print(f"  - {model.model} ({size_gb:.1f} GB)")

print()

# Basic call — same message format as Groq/OpenAI
response = chat(
    model="llama3.2",
    messages=[
        {"role": "user", "content": "Explain in 2 sentences what an LLM is."}
    ],
)

print("=== Response (object-style) ===")
print(response.message.content)

# Dict-style access — also works
print("\n=== Response (dict-style) ===")
print(response["message"]["content"])

# Streaming — progressive response
print("\n=== Streaming Response ===")
stream = chat(
    model="llama3.2",
    messages=[
        {"role": "user", "content": "Give 3 advantages of running an LLM locally."}
    ],
    stream=True,
)

for chunk in stream:
    print(chunk["message"]["content"], end="", flush=True)

print()
