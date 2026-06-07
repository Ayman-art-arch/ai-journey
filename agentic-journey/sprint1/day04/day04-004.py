"""
day04-004 — Reusable Prompt Templates (5 Business Cases)
Objective: Build a library of prompt templates for common freelance/enterprise tasks
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


# ============================================================
# Template 1: Meeting Summary
# ============================================================
MEETING_SUMMARY_TEMPLATE = """You are a professional meeting assistant.

Summarize the following meeting notes into a structured format.

<meeting_notes>
{notes}
</meeting_notes>

Respond with exactly this structure:
## Key Decisions
- [bullet points]

## Action Items
- [who] — [what] — [deadline if mentioned]

## Open Questions
- [bullet points]

Keep it concise. No filler sentences."""


# ============================================================
# Template 2: Data Extraction
# ============================================================
DATA_EXTRACTION_TEMPLATE = """Extract structured data from the document below.

<document>
{document}
</document>

Return a JSON object with these fields (use null if not found):
{fields}

Return ONLY the JSON, no explanation."""


# ============================================================
# Template 3: Email Triage
# ============================================================
EMAIL_TRIAGE_TEMPLATE = """You are an email triage assistant.

Classify this email and suggest an action.

<email>
From: {sender}
Subject: {subject}
Body: {body}
</email>

Respond with exactly:
- Urgency: [critical / high / medium / low]
- Category: [client request / internal / billing / support / spam]
- Suggested action: [one sentence]
- Draft reply needed: [yes / no]"""


# ============================================================
# Template 4: Code Review
# ============================================================
CODE_REVIEW_TEMPLATE = """You are a senior software engineer performing a code review.

Review the following code for bugs, security issues, style problems, and potential improvements.

<language>{language}</language>
<code>
{code}
</code>

Respond with:
## Bugs
- [issues or "None found"]

## Security Concerns
- [issues or "None found"]

## Style & Readability
- [suggestions]

## Suggested Improvements
- [improvements]

Be specific. Reference line numbers or variable names."""


# ============================================================
# Template 5: Report Generation
# ============================================================
REPORT_GENERATION_TEMPLATE = """You are a business analyst writing a professional report.

Transform the raw data below into a well-written narrative report.

<raw_data>
{data}
</raw_data>

<audience>{audience}</audience>
<tone>{tone}</tone>

Write 2-3 paragraphs. Include key metrics. Use professional language appropriate for the audience."""


# --- Sample data to test each template ---

SAMPLE_MEETING_NOTES = """
Project Alpha standup, May 24 2026.
Attendees: Sarah, Mike, Lisa.
- Sarah: finished the auth module, will start testing tomorrow.
- Mike: blocked on the API design, needs input from Lisa.
- Lisa: database migration is 80% done, ETA Friday.
- Decision: launch date pushed to June 15.
- Decision: use JWT tokens instead of sessions.
- Mike will schedule a design review meeting by Wednesday.
- Open: who handles the monitoring setup?
"""

SAMPLE_INVOICE = """
Invoice #INV-2024-0892
Date: March 15, 2024
Bill To: Acme Corp, 123 Main St, Montreal QC H2X 1A1
Item: Cloud Consulting Services - 40 hours @ $150/hr
Subtotal: $6,000.00
Tax (15% QST+GST): $900.00
Total: $6,900.00
Payment terms: Net 30
"""

SAMPLE_EMAIL = {
    "sender": "j.martin@client.com",
    "subject": "URGENT: Production system down",
    "body": "Hi team, our production API has been returning 500 errors since 3am. "
            "Customer complaints are coming in. We need this resolved ASAP. "
            "Can someone look into this immediately? — Jacques",
}

SAMPLE_CODE = """
def process_user(data):
    name = data['name']
    email = data['email']
    query = f"INSERT INTO users (name, email) VALUES ('{name}', '{email}')"
    db.execute(query)
    password = data.get('password', '1234')
    token = md5(password).hexdigest()
    return {'token': token, 'status': 'ok'}
"""


if __name__ == "__main__":
    client = LLMClient(provider="groq")

    # --- Template 1: Meeting Summary ---
    print("=" * 60)
    print("TEMPLATE 1: Meeting Summary")
    print("=" * 60)
    prompt = MEETING_SUMMARY_TEMPLATE.format(notes=SAMPLE_MEETING_NOTES)
    response = client.chat(prompt)
    print(response.content.strip())

    # --- Template 2: Data Extraction ---
    print(f"\n{'='*60}")
    print("TEMPLATE 2: Data Extraction (Invoice)")
    print(f"{'='*60}")
    prompt = DATA_EXTRACTION_TEMPLATE.format(
        document=SAMPLE_INVOICE,
        fields='{"invoice_number", "date", "client", "total", "payment_terms"}',
    )
    response = client.chat(prompt)
    print(response.content.strip())

    # --- Template 3: Email Triage ---
    print(f"\n{'='*60}")
    print("TEMPLATE 3: Email Triage")
    print(f"{'='*60}")
    prompt = EMAIL_TRIAGE_TEMPLATE.format(**SAMPLE_EMAIL)
    response = client.chat(prompt)
    print(response.content.strip())

    # --- Template 4: Code Review ---
    print(f"\n{'='*60}")
    print("TEMPLATE 4: Code Review")
    print(f"{'='*60}")
    prompt = CODE_REVIEW_TEMPLATE.format(language="python", code=SAMPLE_CODE)
    response = client.chat(prompt)
    print(response.content.strip())

    # --- Template 5: Report Generation ---
    print(f"\n{'='*60}")
    print("TEMPLATE 5: Report Generation")
    print(f"{'='*60}")
    raw_data = (
        "Q1 revenue: $2.1M (up 15% YoY)\n"
        "New customers: 340 (target was 300)\n"
        "Churn rate: 4.2% (down from 5.8%)\n"
        "NPS score: 72\n"
        "Top market: Canada (45%), US (35%), Europe (20%)"
    )
    prompt = REPORT_GENERATION_TEMPLATE.format(
        data=raw_data,
        audience="C-suite executives",
        tone="professional and concise",
    )
    response = client.chat(prompt)
    print(response.content.strip())
