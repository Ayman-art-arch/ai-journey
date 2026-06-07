"""
day04-005 — DELIVERABLE: eval_prompts.py — A/B Prompt Testing
Objective: Compare two prompt formulations on the same task and auto-evaluate results
"""
from dotenv import load_dotenv
from pathlib import Path
import importlib.util
import time
import re

load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env")

_day02_006_path = Path(__file__).resolve().parents[1] / "day02" / "day02-006.py"
_spec = importlib.util.spec_from_file_location("day02_006", _day02_006_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
LLMClient = _mod.LLMClient


# ============================================================
# Task: Extract key info from a job posting
# ============================================================

JOB_POSTING = """
We're looking for a Senior Backend Engineer to join our platform team in Toronto.
You'll work with Python, FastAPI, and PostgreSQL to build scalable microservices.
Requirements: 5+ years experience, strong SQL skills, experience with Docker and K8s.
Nice to have: experience with Redis, message queues (RabbitMQ/Kafka), and CI/CD pipelines.
Salary range: $130,000 - $165,000 CAD. Remote-friendly with occasional office days.
Benefits include health insurance, 4 weeks PTO, and $2,000 learning budget.
"""

# Prompt A: Naive, minimal instructions
PROMPT_A = f"""Extract the key information from this job posting:

{JOB_POSTING}
"""

# Prompt B: Structured with few-shot, XML tags, and explicit format
PROMPT_B = f"""Extract structured information from a job posting. Return the data in the exact format shown.

<example>
Input: "Hiring a Junior Developer in Vancouver. Python and JavaScript required. 2+ years exp. $70K-$85K. Full benefits."

Output:
- Title: Junior Developer
- Location: Vancouver
- Required skills: Python, JavaScript
- Experience: 2+ years
- Salary: $70,000 - $85,000
- Remote: not specified
- Benefits: Full benefits
</example>

<job_posting>
{JOB_POSTING}
</job_posting>

Extract the information in the exact same format as the example above."""


# ============================================================
# LLM-as-Judge: evaluate which response is better
# ============================================================

JUDGE_PROMPT_TEMPLATE = """You are an impartial judge evaluating two AI responses to the same task.

<task>
Extract structured information from a job posting.
</task>

<response_a>
{response_a}
</response_a>

<response_b>
{response_b}
</response_b>

Evaluate both responses on these criteria (score 1-5 each):
1. Completeness: are all key fields extracted?
2. Format consistency: is the output well-structured and easy to parse?
3. Accuracy: is the extracted information correct?
4. Conciseness: no unnecessary text or explanation?

Respond with exactly this format:
<scores>
Response A: completeness=[1-5] format=[1-5] accuracy=[1-5] conciseness=[1-5] total=[sum]
Response B: completeness=[1-5] format=[1-5] accuracy=[1-5] conciseness=[1-5] total=[sum]
</scores>
<winner>[A or B]</winner>
<reason>[one sentence explaining why]</reason>"""


def run_ab_test(provider: str) -> dict:
    """Run both prompts on a provider and return results."""
    client = LLMClient(provider=provider)

    # Run Prompt A
    start_a = time.perf_counter()
    response_a = client.chat(PROMPT_A)
    latency_a = time.perf_counter() - start_a

    # Run Prompt B
    start_b = time.perf_counter()
    response_b = client.chat(PROMPT_B)
    latency_b = time.perf_counter() - start_b

    return {
        "provider": provider,
        "response_a": response_a.content.strip(),
        "response_b": response_b.content.strip(),
        "latency_a": round(latency_a, 3),
        "latency_b": round(latency_b, 3),
        "len_a": len(response_a.content.strip()),
        "len_b": len(response_b.content.strip()),
    }


def judge_responses(response_a: str, response_b: str, provider: str = "groq") -> str:
    """Use an LLM to judge which response is better."""
    client = LLMClient(provider=provider)
    prompt = JUDGE_PROMPT_TEMPLATE.format(response_a=response_a, response_b=response_b)
    response = client.chat(prompt)
    return response.content.strip()


def extract_winner(judge_output: str) -> str | None:
    """Parse the winner from the judge's response."""
    match = re.search(r"<winner>(.*?)</winner>", judge_output, re.DOTALL)
    return match.group(1).strip() if match else None


if __name__ == "__main__":
    results = []

    # Run A/B test on each provider
    for provider in ["groq", "gemini", "ollama"]:
        print(f"\n{'='*60}")
        print(f"A/B TEST — {provider}")
        print(f"{'='*60}")

        try:
            result = run_ab_test(provider)
            results.append(result)

            print(f"\n--- Prompt A (naive) ---")
            print(f"Latency: {result['latency_a']}s | Length: {result['len_a']} chars")
            print(result["response_a"][:300])

            print(f"\n--- Prompt B (structured) ---")
            print(f"Latency: {result['latency_b']}s | Length: {result['len_b']} chars")
            print(result["response_b"][:300])

        except Exception as e:
            print(f"ERROR: {e}")

    # Use LLM-as-judge on the Groq results
    if results:
        print(f"\n{'#'*60}")
        print("LLM-AS-JUDGE EVALUATION (using Groq)")
        print(f"{'#'*60}")

        groq_result = results[0]
        judge_output = judge_responses(groq_result["response_a"], groq_result["response_b"])
        print(f"\n{judge_output}")

        winner = extract_winner(judge_output)
        if winner:
            print(f"\nVerdict: Prompt {winner} wins!")
            label = "Structured (few-shot + XML)" if winner == "B" else "Naive (minimal)"
            print(f"Strategy: {label}")

    # Summary table
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Provider':<12} {'A latency':>10} {'B latency':>10} {'A len':>8} {'B len':>8}")
    print("-" * 50)
    for r in results:
        print(f"{r['provider']:<12} {r['latency_a']:>9}s {r['latency_b']:>9}s {r['len_a']:>8} {r['len_b']:>8}")

    print("\nKey insight: Prompt B (structured) takes more tokens but produces")
    print("more consistent, parseable output — worth the extra cost.")
