"""
Smart Classifier - LLM-based complexity detection.

Uses a quick, cheap LLM call to understand prompt complexity
instead of relying on regex rules.

Cost: ~$0.00002 per classification
"""

import hashlib
from dataclasses import dataclass
from typing import Optional, Dict
from collections import OrderedDict

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


CLASSIFICATION_PROMPT = """Rate this prompt's complexity for an AI assistant (1-10):

1-2: Trivial (basic facts, simple math, yes/no)
3-4: Simple (straightforward tasks, common knowledge)
5-6: Moderate (some reasoning, clear instructions)
7-8: Complex (multi-step reasoning, expertise needed)
9-10: Very complex (deep analysis, specialized domain, ambiguity)

Prompt: "{prompt}"

Reply with ONLY a number 1-10, nothing else."""


@dataclass
class SmartClassification:
    """Result of smart classification."""
    complexity_score: float  # 0.0 to 1.0
    raw_score: int  # 1 to 10
    reasoning: str
    from_cache: bool = False


class SmartClassifier:
    """LLM-based prompt complexity classifier.

    Uses a quick LLM call to semantically understand prompt complexity.
    Much better than regex rules at understanding context.

    Example:
        classifier = SmartClassifier()
        result = classifier.classify("What is 2+2?")
        print(result.complexity_score)  # 0.1 (very simple)

        result = classifier.classify("Analyze the economic implications...")
        print(result.complexity_score)  # 0.8 (complex)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        cache_enabled: bool = True,
        max_cache_size: int = 1000,
        timeout_ms: int = 2000,
    ):
        """Initialize the classifier.

        Args:
            api_key: OpenAI API key (or uses OPENAI_API_KEY env var)
            model: Model to use for classification (default: gpt-4o-mini)
            cache_enabled: Whether to cache results
            max_cache_size: Maximum cache entries
            timeout_ms: Timeout for classification calls
        """
        self.api_key = api_key
        self.model = model
        self.cache_enabled = cache_enabled
        self.max_cache_size = max_cache_size
        self.timeout_ms = timeout_ms

        self._cache: OrderedDict[str, SmartClassification] = OrderedDict()
        self._client: Optional[OpenAI] = None

    def _get_client(self) -> OpenAI:
        """Get or create OpenAI client."""
        if self._client is None:
            if not HAS_OPENAI:
                raise ImportError("OpenAI not installed. Run: pip install openai")
            self._client = OpenAI(api_key=self.api_key)
        return self._client

    def _cache_key(self, prompt: str) -> str:
        """Generate cache key for a prompt."""
        return hashlib.md5(prompt.encode()).hexdigest()[:16]

    def _parse_score(self, response: str) -> int:
        """Parse score from LLM response."""
        try:
            # Extract just the number
            cleaned = ''.join(c for c in response if c.isdigit())
            if cleaned:
                score = int(cleaned[:2])  # Take first 2 digits max
                return max(1, min(10, score))  # Clamp to 1-10
        except (ValueError, IndexError):
            pass
        return 5  # Default to medium if parsing fails

    def _get_reasoning(self, score: int) -> str:
        """Get human-readable reasoning for a score."""
        if score <= 2:
            return "Trivial task - basic facts or simple operations"
        elif score <= 4:
            return "Simple task - straightforward with clear answer"
        elif score <= 6:
            return "Moderate task - requires some reasoning"
        elif score <= 8:
            return "Complex task - multi-step reasoning needed"
        else:
            return "Very complex - deep analysis or specialized knowledge"

    def classify(self, prompt: str) -> SmartClassification:
        """Classify a prompt's complexity.

        Args:
            prompt: The prompt to classify

        Returns:
            SmartClassification with complexity score (0-1), raw score (1-10),
            and reasoning.
        """
        # Check cache
        if self.cache_enabled:
            cache_key = self._cache_key(prompt)
            if cache_key in self._cache:
                cached = self._cache[cache_key]
                # Move to end (LRU)
                self._cache.move_to_end(cache_key)
                return SmartClassification(
                    complexity_score=cached.complexity_score,
                    raw_score=cached.raw_score,
                    reasoning=cached.reasoning,
                    from_cache=True,
                )

        # Try LLM classification
        try:
            client = self._get_client()

            response = client.chat.completions.create(
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": CLASSIFICATION_PROMPT.format(
                        prompt=prompt[:500]  # Limit prompt length
                    )
                }],
                max_tokens=10,
                temperature=0,
            )

            raw_score = self._parse_score(response.choices[0].message.content)

        except Exception:
            # Fallback to heuristic
            raw_score = self._fallback_score(prompt)

        # Convert to 0-1 scale
        complexity_score = (raw_score - 1) / 9.0
        reasoning = self._get_reasoning(raw_score)

        result = SmartClassification(
            complexity_score=complexity_score,
            raw_score=raw_score,
            reasoning=reasoning,
            from_cache=False,
        )

        # Cache result
        if self.cache_enabled:
            self._cache[cache_key] = result
            # Evict oldest if over limit
            while len(self._cache) > self.max_cache_size:
                self._cache.popitem(last=False)

        return result

    def _fallback_score(self, prompt: str) -> int:
        """Fallback heuristic when LLM call fails."""
        prompt_lower = prompt.lower()

        # Very simple patterns
        simple_patterns = ["what is", "who is", "yes or no", "true or false"]
        for pattern in simple_patterns:
            if prompt_lower.startswith(pattern):
                return 2

        # Complex keywords
        complex_keywords = ["analyze", "explain why", "compare", "debug", "implement"]
        for keyword in complex_keywords:
            if keyword in prompt_lower:
                return 7

        # Length-based
        if len(prompt) < 50:
            return 3
        elif len(prompt) < 200:
            return 5
        else:
            return 6

    def clear_cache(self):
        """Clear the classification cache."""
        self._cache.clear()


def smart_classify(prompt: str, api_key: Optional[str] = None) -> SmartClassification:
    """Convenience function to classify a single prompt.

    Args:
        prompt: The prompt to classify
        api_key: Optional OpenAI API key

    Returns:
        SmartClassification result
    """
    classifier = SmartClassifier(api_key=api_key)
    return classifier.classify(prompt)
