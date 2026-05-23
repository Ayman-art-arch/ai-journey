"""
day02-006 — Multi-Provider Abstraction (DELIVERABLE)
Objective: LLMClient class that switches between Groq, Gemini, and Ollama
"""

# load_dotenv reads a .env file and puts its content into environment variables
from dotenv import load_dotenv
# Path helps build file paths that work on any OS (Mac, Windows, Linux)
from pathlib import Path
# os.environ lets us read environment variables (like API keys)
import os
# dataclass is a decorator that auto-generates __init__ for simple classes
from dataclasses import dataclass

# __file__ = the path to THIS script (day02-006.py)
# .resolve() = turn it into an absolute path
# .parents[2] = go up 2 directories (day02 -> sprint1 -> agentic-journey)
# / ".env" = append ".env" to that path
# Result: /Users/kaiji/workspace/ai-journey/agentic-journey/.env
load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env")


# A dictionary that maps provider names to their default model
# If the user doesn't specify a model, we use these
DEFAULT_MODELS = {
    "groq": "llama-3.3-70b-versatile",  # Groq's best free model
    "gemini": "gemini-2.5-flash",        # Google's free model
    "ollama": "llama3.2",                # local model, runs on your machine
}


# @dataclass automatically creates __init__, __repr__, __eq__ for us
# Without it, we'd have to write:
#   def __init__(self, content, provider, model):
#       self.content = content
#       self.provider = provider
#       self.model = model
@dataclass
class LLMResponse:
    """Unified response regardless of provider."""
    content: str   # the text the LLM generated
    provider: str  # "groq", "gemini", or "ollama"
    model: str     # e.g. "llama-3.3-70b-versatile"


class LLMClient:
    """Unified interface for Groq, Gemini, and Ollama."""

    # __init__ is called when you do: client = LLMClient("groq")
    # "self" refers to the object being created
    # provider="groq" means "groq" is the default if nothing is passed
    # model: str | None = None means model is optional (can be a string or None)
    def __init__(self, provider: str = "groq", model: str | None = None):
        # .lower() converts "Groq" or "GROQ" to "groq" for consistency
        self.provider = provider.lower()

        # "or" trick: if model is None (falsy), use the right side instead
        # Example: None or "llama3.2" => "llama3.2"
        # Example: "my-model" or "llama3.2" => "my-model"
        # .get() returns None if the key doesn't exist (unlike [] which crashes)
        self.model = model or DEFAULT_MODELS.get(self.provider)

        # Check if the provider is valid before doing anything else
        if self.provider not in DEFAULT_MODELS:
            # raise = stop execution and throw an error
            # ValueError = built-in error type for invalid arguments
            raise ValueError(
                f"Unknown provider '{provider}'. Options: {list(DEFAULT_MODELS.keys())}"
            )

        # Call _init_client() to create the SDK client for this provider
        # Store it in self._client so we can use it later in chat()
        # _ prefix = convention meaning "private, internal use only"
        self._client = self._init_client()

    def _init_client(self):
        """Create the appropriate SDK client based on self.provider."""

        if self.provider == "groq":
            # Import here (not at top of file) so we only load the SDK we need
            # If you use Ollama, no need to load the Groq SDK
            from groq import Groq

            # os.environ.get() reads the value of GROQ_API_KEY from the system
            # Returns None if the variable doesn't exist (unlike os.environ[] which crashes)
            api_key = os.environ.get("GROQ_API_KEY")
            if not api_key:
                raise ValueError("GROQ_API_KEY missing from environment")
            # Create and return the Groq client object
            return Groq(api_key=api_key)

        elif self.provider == "gemini":
            from google import genai

            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY missing from environment")
            # Create and return the Gemini client object
            return genai.Client(api_key=api_key)

        elif self.provider == "ollama":
            # Ollama runs locally — no API key needed
            # We import the whole module and return it as our "client"
            import ollama
            return ollama

    # -> LLMResponse means this function returns an LLMResponse object
    def chat(self, prompt: str, system: str | None = None) -> LLMResponse:
        """Send a prompt and return a unified response.

        This is the ONLY method callers need to use.
        It routes to the correct private method based on self.provider.
        """
        # Dispatch to the right private method depending on the provider
        if self.provider == "groq":
            return self._chat_groq(prompt, system)
        elif self.provider == "gemini":
            return self._chat_gemini(prompt, system)
        elif self.provider == "ollama":
            return self._chat_ollama(prompt, system)

    def _chat_groq(self, prompt: str, system: str | None) -> LLMResponse:
        # Groq uses the OpenAI message format: a list of dicts
        # Each dict has "role" (who's speaking) and "content" (what they say)
        messages = []  # start with empty list
        if system:
            # System message tells the LLM how to behave (e.g. "be concise")
            messages.append({"role": "system", "content": system})
        # User message is the actual question/instruction
        messages.append({"role": "user", "content": prompt})

        # Send the messages to Groq and get a response
        response = self._client.chat.completions.create(
            model=self.model,      # which model to use
            messages=messages,     # the conversation to send
        )
        # Groq returns: response.choices[0].message.content
        # choices = list of possible responses (usually just 1)
        # [0] = take the first one
        # .message.content = the actual text
        return LLMResponse(
            content=response.choices[0].message.content,
            provider=self.provider,
            model=self.model,
        )

    def _chat_gemini(self, prompt: str, system: str | None) -> LLMResponse:
        # Gemini doesn't support separate system messages
        # Workaround: glue the system message before the prompt
        # f-string: f"{system}\n\n{prompt}" builds a string with variables inside {}
        full_prompt = f"{system}\n\n{prompt}" if system else prompt

        response = self._client.models.generate_content(
            model=self.model,
            contents=full_prompt,  # Gemini takes a simple string, not a list
        )
        # Gemini returns: response.text (much simpler than Groq)
        return LLMResponse(
            content=response.text,
            provider=self.provider,
            model=self.model,
        )

    def _chat_ollama(self, prompt: str, system: str | None) -> LLMResponse:
        # Ollama uses the same message format as Groq/OpenAI
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        # self._client is the ollama module itself (imported in _init_client)
        response = self._client.chat(
            model=self.model,
            messages=messages,
        )
        # Ollama returns: response.message.content (no .choices, no [0])
        return LLMResponse(
            content=response.message.content,
            provider=self.provider,
            model=self.model,
        )


# __name__ == "__main__" is True ONLY when you run: python day02-006.py
# It is False when another file imports this one:
#   from day02_006 import LLMClient  => __name__ is "day02_006", not "__main__"
# This lets us put test/demo code here without it running on import
if __name__ == "__main__":
    prompt = "What is an embedding in AI? Answer in 2 sentences."

    # Loop through all 3 providers with the SAME prompt
    # ["groq", "gemini", "ollama"] is a list, "for" iterates over each item
    for provider in ["groq", "gemini", "ollama"]:
        # f-string with '='*50 prints a line of 50 equal signs as a separator
        print(f"\n{'='*50}")
        print(f"Provider: {provider}")
        print(f"{'='*50}")

        # try/except catches errors so one provider failing doesn't crash everything
        try:
            # Create a client for this provider
            client = LLMClient(provider=provider)
            # Send the prompt and get a unified LLMResponse back
            response = client.chat(prompt)
            # .strip() removes leading/trailing whitespace and newlines
            print(f"Model:    {response.model}")
            print(f"Response: {response.content.strip()}")
        except Exception as e:
            # "e" contains the error message
            print(f"Error:    {e}")

    # Second demo: using a system prompt to control LLM behavior
    print(f"\n{'='*50}")
    print("With system prompt (Groq)")
    print(f"{'='*50}")

    client = LLMClient(provider="groq")
    response = client.chat(
        prompt="What is Python?",
        # system prompt forces the LLM to answer in exactly one sentence
        system="You are an expert who always responds in exactly one sentence.",
    )
    print(f"Response: {response.content.strip()}")
