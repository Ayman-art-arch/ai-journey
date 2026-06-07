"""
day02-005 — Gemini Long Context (1M Tokens)
Objective: Leverage Gemini's massive context window to analyze a long document
"""
from dotenv import load_dotenv
from pathlib import Path
from pypdf import PdfReader
import sys
import os

load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env")

from google import genai

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

# Simulating a long document (in practice, you'd read a PDF or text file)
# Here we generate a synthetic document for the demo
LONG_DOCUMENT = """
# Quarterly Report Q4 2024 — TechCorp Solutions

## 1. Executive Summary

TechCorp Solutions recorded 23% revenue growth in Q4 2024, reaching 45.2M EUR.
This performance was driven by the Cloud division (+31%) and the new AI Consulting
offering (+67% compared to Q3). Client retention rate remains stable at 94%.

## 2. Performance by Division

### 2.1 Cloud Infrastructure Division
- Revenue: 22.1M EUR (+31% YoY)
- Active clients: 847 (+12%)
- Gross margin: 68% (vs 65% in Q3)
- Flagship product: CloudScale Pro (automated migration)

### 2.2 AI & Data Division
- Revenue: 12.8M EUR (+67% vs Q3)
- Projects delivered: 34 consulting engagements
- New product: DocuMind (AI document analysis)
- Client satisfaction: 4.7/5

### 2.3 Managed Services Division
- Revenue: 10.3M EUR (+5% YoY)
- SLA met: 99.97%
- P1 incidents: 3 (vs 7 in Q3)
- Clients under contract: 234

## 3. Key Concerns

- The Managed Services division shows limited growth (+5%)
- 3 major clients in contract renegotiation for Q1 2025
- Recruiting AI talent remains a challenge (12 positions open for >3 months)
- AWS to multi-cloud migration underway — budget overrun by 15%

## 4. Q1 2025 Outlook

- Launch DocuMind v2 with multi-language support
- Strategic partnership with a hyperscaler (in negotiation)
- Q1 revenue target: 48M EUR (+6.2% vs Q4)
- Opening Munich office for the DACH market

## 5. Board Recommendations

1. Invest 2M EUR in AI recruitment (headhunter budget)
2. Accelerate Managed Services transition to "Managed AI" offering
3. Renegotiate the 3 major contracts before end of February
4. Audit the multi-cloud migration project
"""


def read_pdf(path: str) -> str:
    """Extract all text from a PDF file."""
    reader = PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def analyze_document(document: str, question: str) -> str:
    """Send a full document + question to Gemini."""
    prompt = f"""You are a senior business analyst. Here is a document to analyze.

<document>
{document}
</document>

Question: {question}

Respond in a structured and concise manner."""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )
    return response.text


if __name__ == "__main__":
    # Use a PDF file if provided as argument, otherwise fall back to the demo text
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        print(f"Reading PDF: {pdf_path}")
        document = read_pdf(pdf_path)
    else:
        print("No PDF provided — using built-in demo document.")
        print("Usage: python day02-005.py <path/to/file.pdf>\n")
        document = LONG_DOCUMENT

    # Show document size
    token_estimate = len(document.split()) * 1.3  # rough estimate
    print(f"Document: ~{int(token_estimate)} estimated tokens")
    print(f"Gemini capacity: 1,000,000 tokens")
    print(f"Usage: {token_estimate / 1_000_000:.4%}\n")

    # Question 1: Summary
    print("=== Question 1: Summary ===")
    answer = analyze_document(document, "Summarize the 3 key takeaways from this document as bullet points.")
    print(answer)

    # Question 2: Specific extraction
    print("\n=== Question 2: Extraction ===")
    answer = analyze_document(document, "What are the identified risks and recommended actions?")
    print(answer)

    # Question 3: Critical analysis
    print("\n=== Question 3: Analysis ===")
    answer = analyze_document(
        document,
        "What is the most important insight in this document and why?",
    )
    print(answer)
