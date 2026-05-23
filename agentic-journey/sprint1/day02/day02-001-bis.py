"""
day02-001-bis — Hello World Gemini with LangChain
Objective: Same as day02-001 but using LangChain instead of the raw SDK
"""
from dotenv import load_dotenv
from pathlib import Path
import os

load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env")

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

# Create a Gemini LLM via LangChain
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=os.environ.get("GEMINI_API_KEY"),
)

# --- Simple call ---
print("=== Gemini via LangChain ===")
response = llm.invoke("Explain in 3 sentences why Python is popular for AI.")
print(response.content)

# --- With a prompt template + chain ---
print("\n=== With Prompt Template ===")
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI mentor for beginners."),
    ("user", "{question}"),
])

chain = prompt | llm
response = chain.invoke({"question": "Give 5 tips for getting started with machine learning."})
print(response.content)

# --- Streaming ---
print("\n=== Streaming Response ===")
for chunk in llm.stream("List 3 reasons to use LangChain."):
    print(chunk.content, end="", flush=True)

print()
