"""
Smart Classifier for Conductor.

Uses a lightweight LLM call to semantically understand prompt complexity,
rather than relying on regex patterns.
"""

import re
from typing import Optional
from dataclasses import dataclass


@dataclass
class SmartClassification:
    """Result of smart classification."""
    complexity_score: float  # 0.0 to 1.0
    reasoning: str  # Brief explanation
    raw_score: int  # Original 1-10 score
    from_cache: bool = False


# Classification prompt - kept minimal for speed/cost
CLASSIFICATION_PROMPT = """Rate this prompt's complexity for an AI assistant (1-10):

1-2: Trivial (basic facts, simple math, yes/no)
3-4: Simple (straightforward tasks, common knowledge)
5-6: Moderate (some reasoning, clear instructions)
7-8: Complex (multi-step reasoning, expertise needed)
9-10: Very complex (deep analysis, specialized domain, ambiguity)

Consider:
- Reasoning depth required
- Domain expertise needed
- Ambiguity/interpretation needed
- Number of steps/subtasks

Prompt: "{prompt}"

Reply with ONLY a number 1-10, nothing else."""


class SmartClassifier:
    """
    Uses LLM to classify prompt complexity semantically.

    Much more accurate than regex-based classification because it
    actually understands what the prompt is asking for.
    """

    def __init__(
        self,
        provider=None,
        cache_enabled: bool = True,
        max_cache_size: int = 1000,
        timeout_ms: int = 2000,
    ):
        """
        Initialize smart classifier.

        Args:
            provider: LLM provider for classification calls.
                     If None, will use OpenAI with OPENAI_API_KEY.
            cache_enabled: Cache classifications for identical prompts.
            max_cache_size: Maximum cache entries.
            timeout_ms: Timeout for classification call.
        """
        self.provider = provider
        self.cache_enabled = cache_enabled
        self.max_cache_size = max_cache_size
        self.timeout_ms = timeout_ms
        self._cache: dict[str, SmartClassification] = {}

    def classify(self, prompt: str) -> SmartClassification:
        """
        Classify prompt complexity using LLM.

        Args:
            prompt: The user's prompt to classify.

        Returns:
            SmartClassification with complexity score.
        """
        # Check cache first
        cache_key = self._get_cache_key(prompt)
        if self.cache_enabled and cache_key in self._cache:
            result = self._cache[cache_key]
            return SmartClassification(
                complexity_score=result.complexity_score,
                reasoning=result.reasoning,
                raw_score=result.raw_score,
                from_cache=True,
            )

        # Make LLM call
        raw_score = self._call_llm(prompt)

        # Convert 1-10 to 0.0-1.0
        complexity_score = (raw_score - 1) / 9.0

        # Generate reasoning based on score
        reasoning = self._get_reasoning(raw_score)

        result = SmartClassification(
            complexity_score=complexity_score,
            reasoning=reasoning,
            raw_score=raw_score,
            from_cache=False,
        )

        # Cache result
        if self.cache_enabled:
            self._add_to_cache(cache_key, result)

        return result

    def classify_sync(self, prompt: str) -> SmartClassification:
        """Synchronous version of classify."""
        return self.classify(prompt)

    def _call_llm(self, prompt: str) -> int:
        """
        Make the actual LLM call for classification.

        Returns score 1-10.
        """
        classification_prompt = CLASSIFICATION_PROMPT.format(
            prompt=prompt[:2000]  # Truncate very long prompts
        )

        if self.provider is None:
            # Try to use OpenAI
            return self._call_openai(classification_prompt)
        else:
            # Use provided provider
            return self._call_provider(classification_prompt)

    def _call_openai(self, classification_prompt: str) -> int:
        """Call OpenAI API for classification."""
        try:
            import openai
            import os

            client = openai.OpenAI(
                api_key=os.environ.get("OPENAI_API_KEY"),
                timeout=self.timeout_ms / 1000,
            )

            response = client.chat.completions.create(
                model="gpt-4o-mini",  # Cheapest, fastest
                messages=[
                    {"role": "user", "content": classification_prompt}
                ],
                max_tokens=5,  # Just need a number
                temperature=0,  # Deterministic
            )

            return self._parse_score(response.choices[0].message.content)

        except ImportError:
            # OpenAI not installed, fall back to heuristic
            return self._fallback_score(classification_prompt)
        except Exception:
            # API error, fall back to heuristic
            return self._fallback_score(classification_prompt)

    def _call_provider(self, classification_prompt: str) -> int:
        """Call custom provider for classification."""
        try:
            # Provider should have a simple complete method
            response = self.provider.complete(
                prompt=classification_prompt,
                max_tokens=5,
                temperature=0,
            )
            return self._parse_score(response)
        except Exception:
            return self._fallback_score(classification_prompt)

    def _parse_score(self, response: str) -> int:
        """Parse LLM response to extract score."""
        if response is None:
            return 5  # Default to medium

        # Extract first number from response
        response = response.strip()
        match = re.search(r'\d+', response)

        if match:
            score = int(match.group())
            return max(1, min(10, score))  # Clamp to 1-10

        return 5  # Default to medium if can't parse

    def _fallback_score(self, prompt: str) -> int:
        """
        Fallback heuristic when LLM call fails.

        Uses simple rules as backup.
        """
        score = 5  # Start at medium

        prompt_lower = prompt.lower()

        # Simple indicators (reduce score)
        simple_patterns = [
            r'\bwhat is\b', r'\bwho is\b', r'\bwhen\b',
            r'\byes or no\b', r'\btrue or false\b',
            r'\bdefine\b', r'\blist\b',
        ]
        for pattern in simple_patterns:
            if re.search(pattern, prompt_lower):
                score -= 1
                break

        # Complex indicators (increase score)
        complex_patterns = [
            r'\bexplain why\b', r'\banalyze\b', r'\bcompare and contrast\b',
            r'\bstep by step\b', r'\bprove\b', r'\bderive\b',
            r'\bdesign\b', r'\barchitect\b', r'\bimplement\b',
            r'\bdebug\b', r'\boptimize\b', r'\brefactor\b',
        ]
        for pattern in complex_patterns:
            if re.search(pattern, prompt_lower):
                score += 1

        # Length factor
        if len(prompt) > 2000:
            score += 1
        elif len(prompt) < 50:
            score -= 1

        # Code indicators
        if re.search(r'```|def |class |function |import ', prompt):
            score += 1

        return max(1, min(10, score))

    def _get_reasoning(self, score: int) -> str:
        """Get human-readable reasoning for score."""
        if score <= 2:
            return "Trivial task - basic facts or simple operations"
        elif score <= 4:
            return "Simple task - straightforward, common knowledge"
        elif score <= 6:
            return "Moderate task - requires some reasoning"
        elif score <= 8:
            return "Complex task - multi-step reasoning or expertise needed"
        else:
            return "Very complex - deep analysis, specialized domain"

    def _get_cache_key(self, prompt: str) -> str:
        """Generate cache key from prompt."""
        # Use first 500 chars + length as key
        # This handles most cases while keeping key size reasonable
        return f"{prompt[:500]}_{len(prompt)}"

    def _add_to_cache(self, key: str, result: SmartClassification):
        """Add result to cache, evicting old entries if needed."""
        if len(self._cache) >= self.max_cache_size:
            # Simple eviction: remove first entry
            first_key = next(iter(self._cache))
            del self._cache[first_key]

        self._cache[key] = result

    def clear_cache(self):
        """Clear the classification cache."""
        self._cache.clear()

    def get_cache_stats(self) -> dict:
        """Get cache statistics."""
        return {
            "size": len(self._cache),
            "max_size": self.max_cache_size,
            "enabled": self.cache_enabled,
        }


# Singleton instance for convenience
_default_classifier: Optional[SmartClassifier] = None


def get_smart_classifier() -> SmartClassifier:
    """Get or create the default smart classifier."""
    global _default_classifier
    if _default_classifier is None:
        _default_classifier = SmartClassifier()
    return _default_classifier


def smart_classify(prompt: str) -> SmartClassification:
    """
    Convenience function to classify a prompt.

    Example:
        result = smart_classify("What is 2+2?")
        print(result.complexity_score)  # 0.1 (low)
        print(result.reasoning)  # "Trivial task - basic facts"
    """
    return get_smart_classifier().classify(prompt)
