"""
day03-003 — DELIVERABLE: compare_free_llms.py
Objective: Auto-generate a Markdown benchmark report comparing Groq, Gemini, and Ollama
"""
# load_dotenv reads .env file and sets environment variables
from dotenv import load_dotenv
# Path provides OS-independent file path manipulation
from pathlib import Path
# time.perf_counter gives high-resolution timing for benchmarks
import time
# datetime.now() gives current date/time for the report header
from datetime import datetime

# Navigate 2 levels up from this file to find .env at project root
load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env")

# importlib.util lets us import files with hyphens in their names
import importlib.util

# Build the path to day02-006.py which contains our LLMClient class
_day02_006_path = Path(__file__).resolve().parents[1] / "day02" / "day02-006.py"
# spec_from_file_location creates module metadata from a file path
_spec = importlib.util.spec_from_file_location("day02_006", _day02_006_path)
# module_from_spec creates an empty module object we can populate
_mod = importlib.util.module_from_spec(_spec)
# exec_module runs the file's code, filling _mod with its classes/functions
_spec.loader.exec_module(_mod)
# Extract only the class we need from the loaded module
LLMClient = _mod.LLMClient

# Dictionary mapping task type names to their test prompts
# Each prompt tests a different LLM capability
TASKS = {
    # Tests ability to condense information while preserving meaning
    "summarize": (
        "Summarize the following in 2 sentences:\n"
        "Machine learning is a subset of artificial intelligence that enables systems "
        "to learn and improve from experience without being explicitly programmed. "
        "It focuses on developing algorithms that can access data, learn from it, "
        "and make predictions or decisions. Common applications include image recognition, "
        "natural language processing, recommendation systems, and autonomous vehicles."
    ),
    # Tests code generation accuracy and formatting
    "code": (
        "Write a Python function called `fibonacci(n)` that returns the nth Fibonacci number. "
        "Include a docstring. No explanation, just the code."
    ),
    # Tests creativity and adherence to a strict format (5-7-5 syllables)
    "creative": (
        "Write a haiku about programming."
    ),
    # Tests factual accuracy and conciseness
    "factual": (
        "What are the 3 laws of thermodynamics? Answer with one sentence per law."
    ),
    # Tests structured output generation from unstructured text
    "extraction": (
        "Extract the name, age, and city from this text as JSON:\n"
        "'My name is Alice Johnson, I am 34 years old and I live in Montreal.'"
    ),
}

# The 3 free providers we're comparing
PROVIDERS = ["groq", "gemini", "ollama"]


def run_benchmark() -> list[dict]:
    """Run all tasks on all providers. Returns list of result dicts."""
    # Accumulates one dict per (task, provider) combination
    results = []

    # .items() gives us both the key (task name) and value (prompt) each iteration
    for task_name, prompt in TASKS.items():
        # Show progress so the user knows which task is running
        print(f"Running task: {task_name}")
        for provider in PROVIDERS:
            # end=" " keeps cursor on same line, flush=True forces immediate output
            print(f"  {provider}...", end=" ", flush=True)
            try:
                # Create a new client for this provider (uses default model)
                client = LLMClient(provider=provider)
                # Record wall-clock time before the API call
                start = time.perf_counter()
                # Send the prompt and wait for the full response
                response = client.chat(prompt)
                # Subtract start from current time to get duration in seconds
                elapsed = time.perf_counter() - start

                # Store all data we need for report generation
                results.append({
                    "task": task_name,
                    "provider": provider,
                    "model": response.model,
                    # Round to 3 decimal places (millisecond precision)
                    "latency_s": round(elapsed, 3),
                    # Character count helps compare verbosity across providers
                    "response_len": len(response.content.strip()),
                    # .strip() removes leading/trailing whitespace
                    "content": response.content.strip(),
                })
                # :.3f formats the float to 3 decimal places
                print(f"{elapsed:.3f}s")

            except Exception as e:
                # Print error but don't crash — continue to next provider
                print(f"ERROR: {e}")
                # Store a sentinel entry so the report can show "error" instead of missing data
                results.append({
                    "task": task_name,
                    "provider": provider,
                    "model": "N/A",
                    # -1 signals a failed run (filtered out in report generation)
                    "latency_s": -1,
                    "response_len": 0,
                    # Prefix with ERROR so we can detect failures in report generation
                    "content": f"ERROR: {e}",
                })

    return results


def generate_report(results: list[dict]) -> str:
    """Generate a Markdown report from benchmark results."""
    # Build the report as a list of lines, then join at the end
    lines = []
    # Main title of the Markdown document
    lines.append("# Free LLM Benchmark Report")
    # strftime formats datetime as "2026-05-24 14:30"
    lines.append(f"\n*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n")

    # --- Section 1: Models tested ---
    lines.append("## Models Tested\n")
    # Markdown table header row
    lines.append("| Provider | Model |")
    # Markdown table separator (required for rendering)
    lines.append("|----------|-------|")
    # Track which (provider, model) pairs we've already printed
    seen = set()
    for r in results:
        # Tuple of (provider, model) serves as a unique key
        key = (r["provider"], r["model"])
        # Skip duplicates and failed entries
        if key not in seen and r["model"] != "N/A":
            seen.add(key)
            # Backticks around model name renders it as inline code in Markdown
            lines.append(f"| {r['provider']} | `{r['model']}` |")

    # --- Section 2: Latency table ---
    lines.append("\n## Latency Comparison (seconds)\n")
    # ' | '.join() creates "groq | gemini | ollama" for the header
    lines.append(f"| Task | {' | '.join(PROVIDERS)} | Winner |")
    # Generate the right number of --- separators for the columns
    lines.append(f"|------|{'|'.join(['---'] * len(PROVIDERS))}|--------|")

    # One row per task type
    for task_name in TASKS:
        # Start building the row with the task name
        row = f"| {task_name} "
        # Collect latencies for this task to determine the winner
        task_results = {}
        for provider in PROVIDERS:
            # List comprehension filters results to find matching (task, provider) entry
            match = [r for r in results if r["task"] == task_name and r["provider"] == provider]
            # Check both that we found a match and that it didn't fail (-1 = error)
            if match and match[0]["latency_s"] > 0:
                # Store latency for winner calculation below
                task_results[provider] = match[0]["latency_s"]
                # :.3f formats to 3 decimal places in the table cell
                row += f"| {match[0]['latency_s']:.3f} "
            else:
                row += "| error "

        if task_results:
            # min() with key= finds the provider with the lowest latency
            winner = min(task_results, key=task_results.get)
            # **bold** in Markdown highlights the winner
            row += f"| **{winner}** |"
        else:
            row += "| N/A |"
        lines.append(row)

    # Average row — summarizes overall speed per provider
    row = "| **AVERAGE** "
    # Will hold {provider: average_latency} for final winner calculation
    avg_results = {}
    for provider in PROVIDERS:
        # Collect all successful latencies for this provider across all tasks
        latencies = [r["latency_s"] for r in results if r["provider"] == provider and r["latency_s"] > 0]
        if latencies:
            # Simple arithmetic mean
            avg = sum(latencies) / len(latencies)
            avg_results[provider] = avg
            row += f"| {avg:.3f} "
        else:
            row += "| N/A "
    if avg_results:
        # Overall fastest provider across all tasks
        winner = min(avg_results, key=avg_results.get)
        row += f"| **{winner}** |"
    else:
        row += "| N/A |"
    lines.append(row)

    # --- Section 3: Response length ---
    lines.append("\n## Response Length (characters)\n")
    # Same column structure as latency table but without Winner column
    lines.append(f"| Task | {' | '.join(PROVIDERS)} |")
    lines.append(f"|------|{'|'.join(['---'] * len(PROVIDERS))}|")

    for task_name in TASKS:
        row = f"| {task_name} "
        for provider in PROVIDERS:
            match = [r for r in results if r["task"] == task_name and r["provider"] == provider]
            if match and match[0]["response_len"] > 0:
                # Show raw character count — helps identify verbose vs concise models
                row += f"| {match[0]['response_len']} "
            else:
                # Dash indicates missing/failed data
                row += "| - "
        # Close the table row
        row += "|"
        lines.append(row)

    # --- Section 4: Decision table ---
    # Static recommendations based on known provider strengths
    lines.append("\n## Decision Table: Which Provider to Use\n")
    lines.append("| Use Case | Recommended | Reason |")
    lines.append("|----------|-------------|--------|")
    lines.append("| Maximum speed | Groq | Fastest inference via custom LPU hardware |")
    lines.append("| Very long documents (>100k tokens) | Gemini | 1M token context window |")
    lines.append("| Offline / confidential data | Ollama | 100% local, no data leaves your machine |")
    lines.append("| French / multilingual tasks | Ollama (`qwen2.5`) | Strong multilingual performance |")
    lines.append("| Code generation | Groq or Ollama (`mistral`) | Fast and accurate for code |")
    lines.append("| Cost-sensitive prototyping | Ollama | Zero cost, unlimited calls |")
    lines.append("| Production API with SLA | Groq | Reliable cloud API, generous free tier |")

    # --- Section 5: Sample responses ---
    # Include actual LLM outputs so readers can judge quality themselves
    lines.append("\n## Sample Responses\n")
    for task_name in TASKS:
        # .capitalize() uppercases first letter: "summarize" -> "Summarize"
        lines.append(f"### {task_name.capitalize()}\n")
        for provider in PROVIDERS:
            match = [r for r in results if r["task"] == task_name and r["provider"] == provider]
            # Only include successful responses (skip errors)
            if match and match[0]["content"] and not match[0]["content"].startswith("ERROR"):
                # Show which model produced this response
                lines.append(f"**{provider}** (`{match[0]['model']}`):\n")
                # Wrap in code fence so Markdown preserves formatting
                lines.append(f"```\n{match[0]['content']}\n```\n")

    # Join all lines with newlines to create the final Markdown string
    return "\n".join(lines)


if __name__ == "__main__":
    # User-facing progress message
    print("Starting benchmark...\n")
    # Phase 1: collect raw data from all providers
    results = run_benchmark()

    print("\nGenerating report...")
    # Phase 2: transform raw data into formatted Markdown
    report = generate_report(results)

    # Save the report next to this script (sprint1/day03/benchmark_report.md)
    output_path = Path(__file__).parent / "benchmark_report.md"
    # write_text creates or overwrites the file with the report string
    output_path.write_text(report)
    print(f"\nReport saved to: {output_path}")

    # Print a condensed version to terminal for quick review
    print("\n" + "=" * 60)
    print("QUICK SUMMARY")
    print("=" * 60)
    for provider in PROVIDERS:
        # Filter to only successful runs for this provider
        latencies = [r["latency_s"] for r in results if r["provider"] == provider and r["latency_s"] > 0]
        if latencies:
            # Calculate average latency across all task types
            avg = sum(latencies) / len(latencies)
            # :<10 left-aligns provider name in a 10-char wide column
            print(f"  {provider:<10} avg: {avg:.3f}s  (across {len(latencies)} tasks)")
