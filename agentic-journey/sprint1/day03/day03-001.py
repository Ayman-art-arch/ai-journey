"""
day03-001 — Same Prompt on 3 Providers
Objective: Compare Groq, Gemini, and Ollama on identical prompts with latency measurement
"""
from dotenv import load_dotenv
from pathlib import Path
import time

load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env")

# day02-006.py has a hyphen in its name, which Python can't import normally
# importlib.util lets us load a module from any file path
import importlib.util

# Build the absolute path to day02-006.py
_day02_006_path = Path(__file__).resolve().parents[1] / "day02" / "day02-006.py"
# Create a module spec (metadata about the module) from the file path
_spec = importlib.util.spec_from_file_location("day02_006", _day02_006_path)
# Create an empty module object from the spec
_mod = importlib.util.module_from_spec(_spec)
# Execute the module code, populating it with classes/functions
_spec.loader.exec_module(_mod)
# Pull out the class we need
LLMClient = _mod.LLMClient

# The prompt we'll send to all 3 providers — identical for fair comparison
PROMPT = "Explain what a REST API is in exactly 3 sentences."

PROVIDERS = ["groq", "gemini", "ollama"]


def measure_call(provider: str, prompt: str) -> dict:
    """Send a prompt to a provider and measure how long it takes.

    Returns a dict with provider name, model, response text, and latency.
    """
    client = LLMClient(provider=provider)

    start = time.perf_counter()
    response = client.chat(prompt)
    elapsed = time.perf_counter()

    return {
        "provider": provider,
        "model": response.model,
        "content": response.content.strip(),
        "latency_s": round(elapsed, 3),
    }


if __name__ == "__main__":
    results = []

    for provider in PROVIDERS:
        print(f"\n{'='*60}")
        print(f"Calling {provider}...")
        print(f"{'='*60}")

        try:
            result = measure_call(provider, PROMPT)
            results.append(result)

            print(f"Model:   {result['model']}")
            print(f"Latency: {result['latency_s']}s")
            print(f"Response:\n{result['content']}")
        except Exception as e:
            print(f"Error: {e}")

    # Summary table
    if results:
        print(f"\n{'='*60}")
        print("LATENCY SUMMARY")
        print(f"{'='*60}")
        print(f"{'Provider':<12} {'Model':<30} {'Latency':>8}")
        print("-" * 52)
        for r in results:
            print(f"{r['provider']:<12} {r['model']:<30} {r['latency_s']:>7}s")

        # Find the fastest
        fastest = min(results, key=lambda r: r["latency_s"])
        print(f"\nFastest: {fastest['provider']} ({fastest['latency_s']}s)")
