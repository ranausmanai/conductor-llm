"""
ROI Calculator: Demonstrate cost savings from using Conductor.

Shows the potential savings compared to always using expensive models.
"""

from conductor import Conductor, TaskType
import random


def simulate_traffic(num_requests: int = 1000):
    """
    Simulate realistic traffic patterns.

    Returns:
        List of (prompt, task_type, requirements) tuples
    """
    traffic = []

    # Realistic distribution:
    # 40% simple classification
    # 30% summarization
    # 20% JSON extraction
    # 10% long-form generation

    for i in range(num_requests):
        rand = random.random()

        if rand < 0.4:
            # Simple classification - can use cheap model
            task = TaskType.CLASSIFY
            prompt = f"Classify this short text {i}"
            min_quality = 0.75
        elif rand < 0.7:
            # Summarization - medium complexity
            task = TaskType.SUMMARIZE
            prompt = f"Summarize this document {i}: " + ("x" * 500)
            min_quality = 0.8
        elif rand < 0.9:
            # JSON extraction - needs accuracy
            task = TaskType.EXTRACT_JSON
            prompt = f"Extract entities from {i}: " + ("x" * 300)
            min_quality = 0.9
        else:
            # Long-form generation - complex
            task = TaskType.GENERATE_LONG
            prompt = f"Generate a detailed article about {i}"
            min_quality = 0.85

        traffic.append((prompt, task, min_quality))

    return traffic


def calculate_roi():
    """Calculate ROI from using Conductor vs always using expensive model."""
    print("=" * 70)
    print("CONDUCTOR ROI CALCULATOR")
    print("=" * 70)
    print()

    num_requests = 10000
    print(f"Simulating {num_requests:,} requests...")
    print()

    # Generate realistic traffic
    traffic = simulate_traffic(num_requests)

    # Scenario 1: Always use GPT-4o (expensive but safe)
    print("Scenario 1: Always use GPT-4o")
    print("-" * 70)

    total_cost_always_expensive = 0.0
    for prompt, task, min_quality in traffic:
        # Rough estimate: average 300 input tokens, 150 output tokens
        input_tokens = len(prompt) // 4
        output_tokens = 150
        cost = (input_tokens / 1000 * 0.0025) + (output_tokens / 1000 * 0.01)
        total_cost_always_expensive += cost

    print(f"Total cost: ${total_cost_always_expensive:.2f}")
    print(f"Cost per request: ${total_cost_always_expensive / num_requests:.4f}")
    print()

    # Scenario 2: Use Conductor to route intelligently
    print("Scenario 2: Use Conductor (intelligent routing)")
    print("-" * 70)

    conductor = Conductor(dry_run=True)

    total_cost_conductor = 0.0
    strategy_distribution = {}

    for prompt, task, min_quality in traffic:
        response = conductor.complete(
            prompt=prompt,
            task_type=task,
            min_quality=min_quality,
        )

        total_cost_conductor += response.cost

        # Track strategy usage
        strategy = response.decision.strategy_id
        strategy_distribution[strategy] = strategy_distribution.get(strategy, 0) + 1

    print(f"Total cost: ${total_cost_conductor:.2f}")
    print(f"Cost per request: ${total_cost_conductor / num_requests:.4f}")
    print()

    # Calculate savings
    print("ROI ANALYSIS")
    print("=" * 70)

    savings = total_cost_always_expensive - total_cost_conductor
    savings_pct = (savings / total_cost_always_expensive) * 100

    print(f"Total savings: ${savings:.2f} ({savings_pct:.1f}%)")
    print(f"Savings per request: ${savings / num_requests:.4f}")
    print()

    # Monthly/annual projections
    monthly_requests = num_requests * 30
    annual_requests = num_requests * 365

    monthly_savings = (savings / num_requests) * monthly_requests
    annual_savings = (savings / num_requests) * annual_requests

    print("PROJECTED SAVINGS")
    print("-" * 70)
    print(f"At {num_requests:,} requests/day:")
    print(f"  Monthly:  ${monthly_savings:,.2f}")
    print(f"  Annually: ${annual_savings:,.2f}")
    print()

    # Strategy distribution
    print("STRATEGY DISTRIBUTION")
    print("-" * 70)
    for strategy, count in sorted(
        strategy_distribution.items(), key=lambda x: x[1], reverse=True
    ):
        pct = (count / num_requests) * 100
        print(f"  {strategy:20s}: {count:5d} ({pct:5.1f}%)")
    print()

    # Break-even analysis
    print("BREAK-EVEN ANALYSIS")
    print("-" * 70)
    print("If Conductor costs $100/month to operate:")
    break_even_requests = 100 / (savings / num_requests) if savings > 0 else float("inf")
    print(f"  Break-even at: {break_even_requests:,.0f} requests/month")
    print(
        f"  Daily break-even: {break_even_requests / 30:,.0f} requests/day"
    )
    print()

    # Recommendation
    print("RECOMMENDATION")
    print("=" * 70)
    if annual_savings > 1000:
        print("✓ Conductor is HIGHLY RECOMMENDED")
        print(f"  You would save ${annual_savings:,.2f}/year")
        print(f"  ROI: {annual_savings * 10:.0f}x (assuming $100/month operational cost)")
    elif annual_savings > 500:
        print("✓ Conductor is RECOMMENDED")
        print(f"  You would save ${annual_savings:,.2f}/year")
    else:
        print("⚠ Conductor may not be worth the complexity")
        print(f"  Projected savings: ${annual_savings:,.2f}/year")
        print("  Consider if manual model selection is simpler")
    print()


if __name__ == "__main__":
    random.seed(42)  # Reproducible results
    calculate_roi()
