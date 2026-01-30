"""
Basic usage examples for Conductor.

Demonstrates common use cases and best practices.
"""

from conductor import Conductor, TaskType, ValidationError


def example_basic():
    """Basic request routing."""
    print("=" * 60)
    print("Example 1: Basic Usage")
    print("=" * 60)

    # Create conductor in dry-run mode (no API calls)
    conductor = Conductor(dry_run=True)

    # Simple classification task
    response = conductor.complete(
        prompt="Is this email spam? Subject: Win a free iPhone!",
        task_type=TaskType.CLASSIFY,
    )

    print(f"Model used: {response.model_used}")
    print(f"Cost: ${response.cost:.6f}")
    print(f"Latency: {response.latency_ms}ms")
    print(f"Why: {response.why}")
    print()


def example_with_constraints():
    """Using SLA constraints."""
    print("=" * 60)
    print("Example 2: With SLA Constraints")
    print("=" * 60)

    conductor = Conductor(dry_run=True)

    # High-quality extraction with tight latency SLA
    response = conductor.complete(
        prompt="Extract all person names, companies, and dates from: ...",
        task_type=TaskType.EXTRACT_JSON,
        max_latency_ms=2000,  # Must respond in <2s
        min_quality=0.95,  # High quality required
    )

    print(f"Strategy: {response.decision.strategy_id}")
    print(f"Model: {response.model_used}")
    print(f"Predicted P95 latency: {response.decision.predicted_latency_p95_ms}ms")
    print(f"Predicted quality risk: {response.decision.predicted_quality_risk:.1%}")
    print()


def example_cost_optimization():
    """Optimizing for cost."""
    print("=" * 60)
    print("Example 3: Cost Optimization")
    print("=" * 60)

    conductor = Conductor(dry_run=True)

    # Allow more latency to use cheaper models
    response = conductor.complete(
        prompt="Summarize this 10-page document: ...",
        task_type=TaskType.SUMMARIZE,
        max_latency_ms=10000,  # Relaxed latency allows batching
        min_quality=0.75,  # Acceptable quality
        allow_batching=True,  # Enable batching for cost savings
    )

    print(f"Strategy: {response.decision.strategy_id}")
    print(f"Cost: ${response.cost:.6f}")
    print(f"Batching delay: {response.decision.batching_delay_ms}ms")
    print()


def example_error_handling():
    """Handling validation errors."""
    print("=" * 60)
    print("Example 4: Error Handling")
    print("=" * 60)

    conductor = Conductor(dry_run=True)

    # Invalid inputs are caught early
    try:
        response = conductor.complete(
            prompt="",  # Empty prompt!
            task_type=TaskType.CLASSIFY,
        )
    except ValidationError as e:
        print(f"Validation error caught: {e}")

    try:
        response = conductor.complete(
            prompt="Test",
            task_type=TaskType.CLASSIFY,
            min_quality=1.5,  # Invalid: >1.0
        )
    except ValidationError as e:
        print(f"Validation error caught: {e}")

    print()


def example_with_real_api():
    """Using real OpenAI API."""
    print("=" * 60)
    print("Example 5: Real API Usage")
    print("=" * 60)

    # Requires OPENAI_API_KEY environment variable
    conductor = Conductor.with_openai()

    response = conductor.complete(
        prompt="What is the capital of France?",
        task_type=TaskType.CLASSIFY,
    )

    print(f"Response: {response.text}")
    print(f"Model: {response.model_used}")
    print(f"Actual cost: ${response.cost:.6f}")
    print(f"Actual latency: {response.latency_ms}ms")
    print()


def example_logging_and_analytics():
    """Logging requests for analysis."""
    print("=" * 60)
    print("Example 6: Logging and Analytics")
    print("=" * 60)

    from pathlib import Path

    conductor = Conductor(dry_run=True)

    # Make several requests
    for i in range(5):
        response = conductor.complete(
            prompt=f"Request {i}: sample prompt",
            task_type=TaskType.CLASSIFY if i % 2 == 0 else TaskType.SUMMARIZE,
            client_id=f"client_{i % 2}",  # Track per client
        )

    # Get aggregated logs
    logs = conductor.get_logs()
    print(f"Total requests logged: {len(logs)}")

    # Group by strategy
    by_strategy = {}
    for log in logs:
        strategy = log.decision.strategy_id
        by_strategy[strategy] = by_strategy.get(strategy, 0) + 1

    print("\nRequests by strategy:")
    for strategy, count in by_strategy.items():
        print(f"  {strategy}: {count}")

    print()


if __name__ == "__main__":
    # Run all examples
    example_basic()
    example_with_constraints()
    example_cost_optimization()
    example_error_handling()

    # Uncomment to test with real API (requires API key)
    # example_with_real_api()

    example_logging_and_analytics()

    print("All examples completed!")
