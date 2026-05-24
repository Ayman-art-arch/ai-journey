"""
day03-002 — Benchmark: 5 Task Types
Objective: Compare Groq, Gemini, Ollama across summarization, code, creative, factual, extraction
"""
from dotenv import load_dotenv
from pathlib import Path
import time

load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env")

import importlib.util

_day02_006_path = Path(__file__).resolve().parents[1] / "day02" / "day02-006.py"
_spec = importlib.util.spec_from_file_location("day02_006", _day02_006_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
LLMClient = _mod.LLMClient

# 5 task types with prompts designed to test different capabilities
TASKS = {
    "summarize": (
        "Summarize the following in 2 sentences:\n"
        "Machine learning is a subset of artificial intelligence that enables systems "
        "to learn and improve from experience without being explicitly programmed. "
        "It focuses on developing algorithms that can access data, learn from it, "
        "and make predictions or decisions. Common applications include image recognition, "
        "natural language processing, recommendation systems, and autonomous vehicles."
    ),
    "code": (
        "Write a Python function called `fibonacci(n)` that returns the nth Fibonacci number. "
        "Include a docstring. No explanation, just the code."
    ),
    "creative": (
        "Write a haiku about programming."
    ),
    "factual": (
        "What are the 3 laws of thermodynamics? Answer with one sentence per law."
    ),
    "extraction": (
        "Extract the name, age, and city from this text as JSON:\n"
        "'My name is Alice Johnson, I am 34 years old and I live in Montreal.'"
    ),
}

PROVIDERS = ["groq", "gemini", "ollama"]


def run_benchmark() -> list[dict]:
    """Run all tasks on all providers and collect results."""
    results = []

    for task_name, prompt in TASKS.items():
        print(f"\n{'#'*60}")
        print(f"TASK: {task_name.upper()}")
        print(f"{'#'*60}")

        for provider in PROVIDERS:
            print(f"\n--- {provider} ---")
            try:
                client = LLMClient(provider=provider)

                start = time.perf_counter()
                response = client.chat(prompt)
                elapsed = time.perf_counter() - start

                content = response.content.strip()
                results.append({
                    "task": task_name,
                    "provider": provider,
                    "model": response.model,
                    "latency_s": round(elapsed, 3),
                    "response_len": len(content),
                    "content": content,
                })

                print(f"Latency: {elapsed:.3f}s | Length: {len(content)} chars")
                # Truncate long responses for display
                preview = content[:200] + "..." if len(content) > 200 else content
                print(f"Response: {preview}")

            except Exception as e:
                print(f"Error: {e}")
                results.append({
                    "task": task_name,
                    "provider": provider,
                    "model": "N/A",
                    "latency_s": -1,
                    "response_len": 0,
                    "content": f"ERROR: {e}",
                })

    return results


def print_summary(results: list[dict]) -> None:
    """Print a summary grid of latencies by task and provider."""
    print(f"\n{'='*60}")
    print("BENCHMARK SUMMARY (latency in seconds)")
    print(f"{'='*60}")

    # Header
    print(f"{'Task':<14}", end="")
    for p in PROVIDERS:
        print(f"{p:>12}", end="")
    print()
    print("-" * 50)

    # One row per task
    for task_name in TASKS:
        print(f"{task_name:<14}", end="")
        for provider in PROVIDERS:
            match = [r for r in results if r["task"] == task_name and r["provider"] == provider]
            if match and match[0]["latency_s"] > 0:
                print(f"{match[0]['latency_s']:>11}s", end="")
            else:
                print(f"{'error':>12}", end="")
        print()

    # Average latency per provider
    print("-" * 50)
    print(f"{'AVERAGE':<14}", end="")
    for provider in PROVIDERS:
        latencies = [r["latency_s"] for r in results if r["provider"] == provider and r["latency_s"] > 0]
        if latencies:
            avg = sum(latencies) / len(latencies)
            print(f"{avg:>11.3f}s", end="")
        else:
            print(f"{'N/A':>12}", end="")
    print()


if __name__ == "__main__":
    results = run_benchmark()
    print_summary(results)
