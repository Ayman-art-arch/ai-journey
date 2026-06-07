"""
day04-003 — XML Tags + Structured Output Format
Objective: Use XML tags to separate reasoning from answers and parse structured responses
"""
from dotenv import load_dotenv
from pathlib import Path
import importlib.util
import re

load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env")

_day02_006_path = Path(__file__).resolve().parents[1] / "day02" / "day02-006.py"
_spec = importlib.util.spec_from_file_location("day02_006", _day02_006_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
LLMClient = _mod.LLMClient


def extract_tag(text: str, tag: str) -> str | None:
    """Extract content between <tag> and </tag> using regex.

    re.DOTALL makes . match newlines so multi-line content is captured.
    (.*?) is non-greedy — stops at the first closing tag it finds.
    Returns None if the tag isn't found.
    """
    match = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    return match.group(1).strip() if match else None


# --- Example 1: Separate thinking from answer ---

THINKING_PROMPT = """Answer the following question. Use <thinking> tags for your reasoning and <answer> tags for your final response.

<question>
A company's revenue grew 20% in Q1, dropped 10% in Q2, and grew 15% in Q3.
If they started with $1,000,000, what is their revenue at the end of Q3?
</question>

Format your response exactly like this:
<thinking>
[your step-by-step reasoning here]
</thinking>
<answer>
[your final answer here]
</answer>"""


# --- Example 2: Multi-field extraction with XML ---

EXTRACTION_PROMPT = """Extract information from the text below. Return your response using the exact XML tags shown.

<text>
Dr. Sarah Chen, a cardiologist at Montreal General Hospital, published a study
on heart disease prevention in the Canadian Medical Journal on March 15, 2024.
The study involved 2,500 patients over 3 years.
</text>

<format>
<name>[full name]</name>
<role>[job title]</role>
<institution>[organization]</institution>
<publication>[journal name]</publication>
<date>[publication date]</date>
<patients>[number]</patients>
<duration>[study length]</duration>
</format>"""


# --- Example 3: Structured analysis with sections ---

ANALYSIS_PROMPT = """Analyze the following business decision. Structure your response with XML tags.

<decision>
A small SaaS startup (10 employees) is considering switching from a monolith
architecture to microservices.
</decision>

Respond with:
<pros>
[bullet points of advantages]
</pros>
<cons>
[bullet points of disadvantages]
</cons>
<recommendation>
[one-paragraph recommendation]
</recommendation>"""


if __name__ == "__main__":
    client = LLMClient(provider="groq")

    # --- Test 1: Thinking + Answer separation ---
    print("=" * 60)
    print("TEST 1: Separate reasoning from answer")
    print("=" * 60)

    response = client.chat(THINKING_PROMPT)
    raw = response.content.strip()

    thinking = extract_tag(raw, "thinking")
    answer = extract_tag(raw, "answer")

    print(f"\nFull response length: {len(raw)} chars")
    if thinking:
        print(f"\n[THINKING] (hidden from user):\n  {thinking}")
    if answer:
        print(f"\n[ANSWER] (shown to user):\n  {answer}")
    else:
        print(f"\nRaw response (no tags found):\n  {raw}")

    # --- Test 2: Multi-field extraction ---
    print(f"\n{'='*60}")
    print("TEST 2: Multi-field XML extraction")
    print(f"{'='*60}")

    response = client.chat(EXTRACTION_PROMPT)
    raw = response.content.strip()

    fields = ["name", "role", "institution", "publication", "date", "patients", "duration"]
    print()
    for field in fields:
        value = extract_tag(raw, field)
        print(f"  {field:<15} = {value or 'NOT FOUND'}")

    # --- Test 3: Structured analysis ---
    print(f"\n{'='*60}")
    print("TEST 3: Structured analysis with XML sections")
    print(f"{'='*60}")

    response = client.chat(ANALYSIS_PROMPT)
    raw = response.content.strip()

    for section in ["pros", "cons", "recommendation"]:
        content = extract_tag(raw, section)
        print(f"\n[{section.upper()}]")
        print(f"  {content or 'NOT FOUND'}")

    # --- Cross-provider test ---
    print(f"\n{'='*60}")
    print("CROSS-PROVIDER: XML tag compliance")
    print(f"{'='*60}")

    for provider in ["groq", "gemini", "ollama"]:
        try:
            c = LLMClient(provider=provider)
            r = c.chat(THINKING_PROMPT)
            has_thinking = extract_tag(r.content, "thinking") is not None
            has_answer = extract_tag(r.content, "answer") is not None
            print(f"  {provider:<10} thinking={has_thinking}  answer={has_answer}")
        except Exception as e:
            print(f"  {provider:<10} ERROR: {e}")
