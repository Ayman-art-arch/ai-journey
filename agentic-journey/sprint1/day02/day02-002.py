"""
day02-002 — Secure .env Setup
Objective: Load and validate API keys from a .env file
"""
from dotenv import load_dotenv
from pathlib import Path
import os

# Find and load .env from the project root (search upward from this script)
env_path = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(dotenv_path=env_path)


def check_api_keys() -> dict[str, str | None]:
    """Check for API key presence and return their status."""
    keys = {
        "GROQ_API_KEY": os.environ.get("GROQ_API_KEY"),
        "GEMINI_API_KEY": os.environ.get("GEMINI_API_KEY"),
        # Ollama doesn't need a key — it runs locally
    }

    print("=== API Key Check ===\n")

    all_ok = True
    for name, value in keys.items():
        if value:
            # Only show first 8 characters for security
            masked = value[:8] + "..." if len(value) > 8 else value
            print(f"  OK  {name} = {masked}")
        else:
            print(f"  MISSING  {name}")
            all_ok = False

    print(f"\n  INFO  Ollama — no key required (local)")

    if not all_ok:
        print("\n  WARNING: Some keys are missing.")
        print("  Create a .env file at the project root with:")
        print("    GROQ_API_KEY=gsk_...")
        print("    GEMINI_API_KEY=AIza...")
    else:
        print("\n  All keys are configured.")

    return keys


def test_groq(api_key: str) -> None:
    """Quick Groq connection test."""
    from groq import Groq

    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": "Just say 'OK Groq works'."}],
        max_tokens=20,
    )
    print(f"  Groq   -> {response.choices[0].message.content.strip()}")


def test_gemini(api_key: str) -> None:
    """Quick Gemini connection test."""
    from google import genai

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents="Just say 'OK Gemini works'.",
    )
    print(f"  Gemini -> {response.text.strip()}")


def test_ollama() -> None:
    """Quick Ollama connection test."""
    try:
        from ollama import chat

        response = chat(
            model="llama3.2",
            messages=[{"role": "user", "content": "Just say 'OK Ollama works'."}],
        )
        print(f"  Ollama -> {response.message.content.strip()}")
    except Exception as e:
        print(f"  Ollama -> ERROR: {e}")
        print("           Make sure 'ollama serve' is running and llama3.2 is installed.")


if __name__ == "__main__":
    keys = check_api_keys()

    print("\n=== Connection Tests ===\n")

    if keys["GROQ_API_KEY"]:
        test_groq(keys["GROQ_API_KEY"])
    if keys["GEMINI_API_KEY"]:
        test_gemini(keys["GEMINI_API_KEY"])

    test_ollama()
