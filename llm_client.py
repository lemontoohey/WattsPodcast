"""
Shared LLM client wrapper.

Centralises access to the language model used for script generation,
humour, contemplation questions, memory extraction, arc mapping, and
reply episodes -- so the provider/model can be swapped in one place.

Uses Groq's OpenAI-compatible chat completions API. Requires the
GROQ_API_KEY environment variable to be set.
"""

import os

DEFAULT_MODEL = "llama-3.3-70b-versatile"


def get_client():
    """Return a Groq client. Requires GROQ_API_KEY in the environment."""
    from groq import Groq

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY environment variable is not set. "
            "Run: export GROQ_API_KEY=your_key_here"
        )
    return Groq(api_key=api_key)


def complete(client, system, user, max_tokens=1024, model=DEFAULT_MODEL):
    """
    Run a single system+user chat completion and return the response text.
    """
    response = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return response.choices[0].message.content
