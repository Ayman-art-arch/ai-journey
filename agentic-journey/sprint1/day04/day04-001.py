"""
day04-001 — Zero-Shot vs Few-Shot Prompting
Objective: Compare output quality with and without examples in the prompt
"""
from dotenv import load_dotenv
from pathlib import Path
import importlib.util

load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env")

# Import LLMClient from day02-006
_day02_006_path = Path(__file__).resolve().parents[1] / "day02" / "day02-006.py"
_spec = importlib.util.spec_from_file_location("day02_006", _day02_006_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
LLMClient = _mod.LLMClient


# --- Task: Classify customer feedback as positive, negative, or neutral ---

# Zero-shot: just describe what we want, no examples
ZERO_SHOT_PROMPT = """Classify the following customer feedback as "positive", "negative", or "neutral".

Feedback: "The delivery was fast but the packaging was damaged."
Classification:"""

# Few-shot: provide 3 examples so the model learns the exact format and logic
FEW_SHOT_PROMPT = """Classify the following customer feedback as "positive", "negative", or "neutral".

Feedback: "I love this product, it works perfectly!"
Classification: positive

Feedback: "Terrible experience. The item broke on day one."
Classification: negative

Feedback: "It arrived on time. Nothing special."
Classification: neutral

Feedback: "The delivery was fast but the packaging was damaged."
Classification:"""


def compare_prompts(provider: str) -> None:
    """Run both zero-shot and few-shot on the same provider and display results."""
    client = LLMClient(provider=provider)

    print(f"\n{'='*60}")
    print(f"Provider: {provider}")
    print(f"{'='*60}")

    # Zero-shot attempt
    zero_response = client.chat(ZERO_SHOT_PROMPT)
    print(f"Zero-shot: {zero_response.content.strip()}")

    # Few-shot attempt
    few_response = client.chat(FEW_SHOT_PROMPT)
    print(f"Few-shot:  {few_response.content.strip()}")


# --- Second example: structured data extraction ---

ZERO_SHOT_EXTRACTION = """Extract the product and price from this sentence:
"I bought a wireless keyboard for $49.99 at the store yesterday."
"""

FEW_SHOT_EXTRACTION = """Extract the product and price from the sentence. Reply with exactly: product=X, price=Y

Sentence: "She paid $12.50 for a notebook."
product=notebook, price=$12.50

Sentence: "The headphones cost me $89."

Sentence: "I bought a wireless keyboard for $49.99 at the store yesterday."
"""


if __name__ == "__main__":
    # Test classification on all providers
    print("=== CLASSIFICATION TASK ===")
    for provider in ["groq", "gemini", "ollama"]:
        try:
            compare_prompts(provider)
        except Exception as e:
            print(f"\n{provider}: ERROR — {e}")

    # Test extraction to show few-shot controls format precisely
    print(f"\n\n{'#'*60}")
    print("=== EXTRACTION TASK ===")
    print(f"{'#'*60}")

    client = LLMClient(provider="groq")

    print("\nZero-shot extraction:")
    response = client.chat(ZERO_SHOT_EXTRACTION)
    print(f"  {response.content.strip()}")

    print("\nFew-shot extraction:")
    response = client.chat(FEW_SHOT_EXTRACTION)
    print(f"  {response.content.strip()}")

    print("\nKey takeaway: few-shot gives consistent format, zero-shot varies each run.")
