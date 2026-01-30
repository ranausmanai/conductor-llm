"""
Input validation for Conductor.

Validates user inputs to prevent errors and API waste.
"""

from typing import Optional


class ValidationError(ValueError):
    """Raised when input validation fails."""
    pass


MAX_PROMPT_LENGTH = 1_000_000  # 1M characters (~250K tokens)
MAX_LATENCY_MS = 600_000  # 10 minutes
MIN_QUALITY_SCORE = 0.0
MAX_QUALITY_SCORE = 1.0
MAX_COST_USD = 100.0  # Sanity check: $100 per request


def validate_prompt(prompt: str) -> None:
    """
    Validate prompt input.

    Args:
        prompt: User prompt

    Raises:
        ValidationError: If prompt is invalid
    """
    if not isinstance(prompt, str):
        raise ValidationError(f"Prompt must be a string, got {type(prompt).__name__}")

    if not prompt or not prompt.strip():
        raise ValidationError("Prompt cannot be empty or whitespace-only")

    if len(prompt) > MAX_PROMPT_LENGTH:
        raise ValidationError(
            f"Prompt too long: {len(prompt):,} characters "
            f"(max: {MAX_PROMPT_LENGTH:,})"
        )


def validate_latency(max_latency_ms: int) -> None:
    """
    Validate latency constraint.

    Args:
        max_latency_ms: Maximum latency in milliseconds

    Raises:
        ValidationError: If latency is invalid
    """
    if not isinstance(max_latency_ms, int):
        raise ValidationError(
            f"max_latency_ms must be an integer, got {type(max_latency_ms).__name__}"
        )

    if max_latency_ms <= 0:
        raise ValidationError(f"max_latency_ms must be positive, got {max_latency_ms}")

    if max_latency_ms > MAX_LATENCY_MS:
        raise ValidationError(
            f"max_latency_ms too large: {max_latency_ms:,}ms "
            f"(max: {MAX_LATENCY_MS:,}ms)"
        )


def validate_quality(min_quality: float) -> None:
    """
    Validate quality threshold.

    Args:
        min_quality: Minimum quality score (0.0 to 1.0)

    Raises:
        ValidationError: If quality is invalid
    """
    if not isinstance(min_quality, (int, float)):
        raise ValidationError(
            f"min_quality must be a number, got {type(min_quality).__name__}"
        )

    if min_quality < MIN_QUALITY_SCORE or min_quality > MAX_QUALITY_SCORE:
        raise ValidationError(
            f"min_quality must be between {MIN_QUALITY_SCORE} and {MAX_QUALITY_SCORE}, "
            f"got {min_quality}"
        )


def validate_cost(max_cost_usd: Optional[float]) -> None:
    """
    Validate cost budget.

    Args:
        max_cost_usd: Maximum cost in USD (optional)

    Raises:
        ValidationError: If cost is invalid
    """
    if max_cost_usd is None:
        return

    if not isinstance(max_cost_usd, (int, float)):
        raise ValidationError(
            f"max_cost_usd must be a number, got {type(max_cost_usd).__name__}"
        )

    if max_cost_usd <= 0:
        raise ValidationError(f"max_cost_usd must be positive, got {max_cost_usd}")

    if max_cost_usd > MAX_COST_USD:
        raise ValidationError(
            f"max_cost_usd too large: ${max_cost_usd:.2f} "
            f"(max: ${MAX_COST_USD:.2f} per request)"
        )


def validate_output_tokens(expected_output_tokens: Optional[int]) -> None:
    """
    Validate expected output tokens.

    Args:
        expected_output_tokens: Expected output length hint (optional)

    Raises:
        ValidationError: If value is invalid
    """
    if expected_output_tokens is None:
        return

    if not isinstance(expected_output_tokens, int):
        raise ValidationError(
            f"expected_output_tokens must be an integer, "
            f"got {type(expected_output_tokens).__name__}"
        )

    if expected_output_tokens < 0:
        raise ValidationError(
            f"expected_output_tokens cannot be negative, got {expected_output_tokens}"
        )

    if expected_output_tokens > 128000:  # Claude/GPT-4 max context
        raise ValidationError(
            f"expected_output_tokens too large: {expected_output_tokens:,} "
            f"(max: 128,000)"
        )


def validate_request(
    prompt: str,
    max_latency_ms: int,
    min_quality: float,
    max_cost_usd: Optional[float],
    expected_output_tokens: Optional[int],
) -> None:
    """
    Validate all request parameters.

    Args:
        prompt: User prompt
        max_latency_ms: Maximum latency
        min_quality: Minimum quality score
        max_cost_usd: Maximum cost (optional)
        expected_output_tokens: Expected output length (optional)

    Raises:
        ValidationError: If any parameter is invalid
    """
    validate_prompt(prompt)
    validate_latency(max_latency_ms)
    validate_quality(min_quality)
    validate_cost(max_cost_usd)
    validate_output_tokens(expected_output_tokens)
