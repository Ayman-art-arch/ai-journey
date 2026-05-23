"""
day02-004 — Ollama via REST API
Objective: Call Ollama directly via HTTP, without any SDK
"""
import requests
import json

OLLAMA_BASE_URL = "http://localhost:11434"


def chat_rest(model: str, prompt: str) -> str:
    """Call via the native /api/chat endpoint (no streaming)."""
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        },
    )
    response.raise_for_status()
    return response.json()["message"]["content"]


def chat_rest_stream(model: str, prompt: str) -> None:
    """Call via /api/chat with streaming (NDJSON)."""
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
        },
        stream=True,
    )
    response.raise_for_status()

    # Each line is a separate JSON object (NDJSON)
    for line in response.iter_lines():
        if line:
            chunk = json.loads(line)
            print(chunk["message"]["content"], end="", flush=True)
    print()


def chat_openai_compat(model: str, prompt: str) -> str:
    """Call via the OpenAI-compatible endpoint /v1/chat/completions."""
    response = requests.post(
        f"{OLLAMA_BASE_URL}/v1/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
        },
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


if __name__ == "__main__":
    model = "llama3.2"

    # 1. Simple REST call
    print("=== REST /api/chat (no streaming) ===")
    result = chat_rest(model, "What is a REST API? Answer in 2 sentences.")
    print(result)

    # 2. REST call with streaming
    print("\n=== REST /api/chat (streaming) ===")
    chat_rest_stream(model, "List 3 common network protocols, one line each.")

    # 3. OpenAI-compatible endpoint
    print("\n=== REST /v1/chat/completions (OpenAI compat) ===")
    result = chat_openai_compat(model, "Say 'OpenAI compat OK' and nothing else.")
    print(result)
