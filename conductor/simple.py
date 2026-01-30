"""
Conductor - Simple API

The easiest way to save money on LLM calls.

Usage:
    from conductor import llm

    response = llm("What is 2+2?")
    print(response.text)   # "4"
    print(response.model)  # "gpt-4o-mini" (cheap - simple question)
    print(response.cost)   # $0.0001
    print(response.saved)  # $0.0009 (vs gpt-4o)
"""

import os
from dataclasses import dataclass
from typing import Optional

from conductor.config import get_pricing, get_models

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


@dataclass
class Response:
    """Response from an LLM call."""
    text: str
    model: str
    cost: float
    saved: float
    baseline_cost: float
    prompt_tokens: int
    completion_tokens: int

    def __str__(self):
        return self.text

    def __repr__(self):
        return f"Response(text='{self.text[:50]}...', model='{self.model}', cost=${self.cost:.4f}, saved=${self.saved:.4f})"


# Defaults are loaded via conductor.config
_models_cache = get_models()
CHEAP_MODEL = _models_cache["cheap"]
QUALITY_MODEL = _models_cache["quality"]
BASELINE_MODEL = _models_cache["baseline"]


def _calculate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Calculate cost for a request."""
    pricing = get_pricing()
    rates = pricing.get(model, pricing.get("gpt-4o", {"input": 0.0, "output": 0.0}))
    input_cost = (prompt_tokens / 1_000_000) * rates["input"]
    output_cost = (completion_tokens / 1_000_000) * rates["output"]
    return input_cost + output_cost


def _is_complex(prompt: str) -> bool:
    """Determine if a prompt is complex (needs expensive model).

    Simple heuristics - no external API call needed.
    """
    prompt_lower = prompt.lower()

    # Check simple patterns FIRST (these override complexity)
    simple_patterns = [
        "what is", "who is", "when is", "where is",
        "yes or no", "true or false",
        "list", "name",
        "translate",
        "summarize",
    ]

    for pattern in simple_patterns:
        if pattern in prompt_lower:
            return False

    # Length check - very long prompts often need better models
    if len(prompt) > 2000:
        return True

    # Keywords suggesting complexity
    complex_keywords = [
        "analyze", "explain why", "compare and contrast",
        "write a", "create a", "generate a",
        "code", "debug", "fix", "implement",
        "legal", "contract", "medical", "diagnosis",
        "strategy", "plan", "design",
        "step by step", "detailed",
        "essay", "report",
        "complex", "difficult", "challenging",
    ]

    for keyword in complex_keywords:
        if keyword in prompt_lower:
            return True

    # Default: if short, probably simple
    if len(prompt) < 200:
        return False

    return True


def llm(
    prompt: str,
    quality: str = "auto",
    api_key: Optional[str] = None,
    max_tokens: int = 1000,
) -> Response:
    """Make an LLM call with automatic cost optimization.

    Args:
        prompt: The prompt to send
        quality: "auto" (smart routing), "high" (gpt-4o), or "low" (gpt-4o-mini)
        api_key: OpenAI API key (or set OPENAI_API_KEY env var)
        max_tokens: Maximum tokens in response

    Returns:
        Response with text, model, cost, and savings

    Example:
        >>> response = llm("What is 2+2?")
        >>> print(response.text)
        4
        >>> print(response.cost)
        0.0001
    """
    if not HAS_OPENAI:
        raise ImportError("OpenAI not installed. Run: pip install conductor-llm[openai]")

    # Get API key
    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise ValueError("No API key. Set OPENAI_API_KEY or pass api_key parameter.")

    # Select model
    models = get_models()
    if quality == "high":
        model = models["quality"]
    elif quality == "low":
        model = models["cheap"]
    else:  # auto
        model = models["quality"] if _is_complex(prompt) else models["cheap"]

    # Make request
    client = OpenAI(api_key=key)

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
    )

    # Extract info
    text = response.choices[0].message.content
    prompt_tokens = response.usage.prompt_tokens
    completion_tokens = response.usage.completion_tokens

    # Calculate costs
    cost = _calculate_cost(model, prompt_tokens, completion_tokens)
    baseline_cost = _calculate_cost(models["baseline"], prompt_tokens, completion_tokens)
    saved = baseline_cost - cost

    return Response(
        text=text,
        model=model,
        cost=cost,
        saved=max(0, saved),  # Can't save negative
        baseline_cost=baseline_cost,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )


def llm_dry_run(
    prompt: str,
    quality: str = "auto",
) -> dict:
    """See what model would be chosen without making an API call.

    Args:
        prompt: The prompt to analyze
        quality: "auto", "high", or "low"

    Returns:
        Dict with model choice and reasoning

    Example:
        >>> llm_dry_run("What is 2+2?")
        {'model': 'gpt-4o-mini', 'reason': 'Simple prompt (short, basic question)'}
    """
    if quality == "high":
        return {"model": get_models()["quality"], "reason": "Quality set to high"}
    elif quality == "low":
        return {"model": get_models()["cheap"], "reason": "Quality set to low"}

    is_complex = _is_complex(prompt)

    if is_complex:
        return {
            "model": get_models()["quality"],
            "reason": "Complex prompt (length or keywords suggest complexity)"
        }
    else:
        return {
            "model": get_models()["cheap"],
            "reason": "Simple prompt (short, basic question)"
        }
