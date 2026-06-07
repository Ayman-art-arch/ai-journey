"""
day04-002 — Chain-of-Thought (CoT) Prompting
Objective: Show how "think step by step" improves accuracy on reasoning tasks
"""
from dotenv import load_dotenv
from pathlib import Path
import importlib.util

load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env")

_day02_006_path = Path(__file__).resolve().parents[1] / "day02" / "day02-006.py"
_spec = importlib.util.spec_from_file_location("day02_006", _day02_006_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
LLMClient = _mod.LLMClient


# --- Problem 1: Math word problem (models often get wrong without CoT) ---

MATH_DIRECT = """A store has 45 apples. They sell 12 in the morning and receive a shipment of 30 in the afternoon. Then they sell 18 more before closing. How many apples do they have at closing?

Answer:"""

MATH_COT = """A store has 45 apples. They sell 12 in the morning and receive a shipment of 30 in the afternoon. Then they sell 18 more before closing. How many apples do they have at closing?

Let's think step by step, then give the final answer."""


# --- Problem 2: Logic puzzle (tricky without explicit reasoning) ---

LOGIC_DIRECT = """If all roses are flowers, and some flowers fade quickly, can we conclude that some roses fade quickly?

Answer (yes or no):"""

LOGIC_COT = """If all roses are flowers, and some flowers fade quickly, can we conclude that some roses fade quickly?

Think through this step by step using logic, then give your final answer (yes or no)."""


# --- Problem 3: Few-shot CoT (providing reasoning examples) ---

FEWSHOT_COT = """Solve the problem by showing your reasoning steps.

Q: There are 15 trees in the park. Workers plant 6 more today and 3 more tomorrow. How many trees will be in the park?
A: Start with 15 trees. Add 6 planted today: 15 + 6 = 21. Add 3 planted tomorrow: 21 + 3 = 24. The answer is 24.

Q: A baker makes 20 loaves in the morning. She sells 8 before lunch and 5 after lunch. She bakes 10 more in the afternoon. How many loaves does she have?
A: Start with 20 loaves. Sell 8 before lunch: 20 - 8 = 12. Sell 5 after lunch: 12 - 5 = 7. Bake 10 more: 7 + 10 = 17. The answer is 17.

Q: A classroom has 32 students. 4 leave early, then 7 new students join for the afternoon session, and 2 more leave. How many students are in the classroom?
A:"""


def run_comparison(label: str, direct: str, cot: str, provider: str = "groq") -> None:
    """Compare direct vs CoT responses for the same problem."""
    client = LLMClient(provider=provider)

    print(f"\n{'='*60}")
    print(f"{label} ({provider})")
    print(f"{'='*60}")

    print("\nDirect (no CoT):")
    response = client.chat(direct)
    print(f"  {response.content.strip()}")

    print("\nWith CoT:")
    response = client.chat(cot)
    print(f"  {response.content.strip()}")


if __name__ == "__main__":
    # Test on Groq (fast) for quick iteration
    run_comparison("MATH PROBLEM (correct answer: 45)", MATH_DIRECT, MATH_COT)
    run_comparison("LOGIC PUZZLE (correct answer: no)", LOGIC_DIRECT, LOGIC_COT)

    # Few-shot CoT example
    print(f"\n{'='*60}")
    print("FEW-SHOT CoT (correct answer: 33)")
    print(f"{'='*60}")

    client = LLMClient(provider="groq")
    response = client.chat(FEWSHOT_COT)
    print(f"\n{response.content.strip()}")

    # Cross-provider comparison on the math problem
    print(f"\n\n{'#'*60}")
    print("CROSS-PROVIDER: Math problem with CoT")
    print(f"{'#'*60}")
    for provider in ["groq", "gemini", "ollama"]:
        try:
            client = LLMClient(provider=provider)
            response = client.chat(MATH_COT)
            # Show just the last line which should contain the final answer
            lines = response.content.strip().split("\n")
            print(f"\n{provider}: {lines[-1]}")
        except Exception as e:
            print(f"\n{provider}: ERROR — {e}")
