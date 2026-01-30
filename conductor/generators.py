"""
Test data generators for Conductor.

Generates synthetic requests for testing and evaluation.
"""

import random
from typing import Iterator, Optional

from conductor.schemas import (
    Request,
    TaskType,
    LatencyPercentile,
)


# =============================================================================
# SAMPLE PROMPTS BY TASK TYPE
# =============================================================================

CLASSIFY_PROMPTS = [
    "Is this email spam or not spam? Email: 'Congratulations! You've won $1000!'",
    "Classify the sentiment: 'I love this product, it's amazing!'",
    "Is this review positive, negative, or neutral? 'The food was okay.'",
    "Categorize this support ticket: 'My order hasn't arrived yet.'",
    "Is this text about sports, politics, or technology? 'The new iPhone was announced today.'",
    "Classify intent: 'I want to cancel my subscription'",
    "Is this a question or a statement? 'What time does the store close'",
    "Determine if this is urgent: 'Server is down, customers can't checkout'",
]

EXTRACT_JSON_PROMPTS = [
    "Extract the name and email from: 'Contact John Smith at john@example.com'",
    "Parse this address: '123 Main St, New York, NY 10001'",
    "Extract order details: 'Order #12345, 2 items, total $59.99, shipped to California'",
    "Get the date and time: 'Meeting scheduled for January 15, 2024 at 3pm EST'",
    "Extract product info: 'iPhone 15 Pro, 256GB, Space Black, $999'",
    "Parse contact: 'Jane Doe, CEO, Acme Corp, jane@acme.com, 555-123-4567'",
    "Extract metrics: 'Revenue: $1.2M, Users: 50,000, Growth: 25%'",
    "Get event details: 'Tech Conference 2024, March 1-3, San Francisco, $500'",
]

SUMMARIZE_PROMPTS = [
    """Summarize this article: The Federal Reserve announced today that it would maintain 
    interest rates at their current levels, citing ongoing concerns about inflation while 
    acknowledging recent improvements in the labor market. Fed Chair Powell emphasized 
    that future decisions would be data-dependent and that the committee remains committed 
    to its 2% inflation target. Markets reacted positively to the news, with major indices 
    rising by approximately 1% following the announcement.""",
    
    """Summarize: Machine learning is transforming how businesses operate. Companies are 
    using ML for everything from customer service chatbots to fraud detection systems. 
    The technology works by training algorithms on large datasets, allowing them to 
    identify patterns and make predictions. While implementation can be challenging, 
    the potential benefits in efficiency and accuracy are significant.""",
    
    """Provide a brief summary: The quarterly earnings report showed revenue of $5.2 billion, 
    up 15% year-over-year. Net income was $800 million, exceeding analyst expectations. 
    The company attributed growth to strong performance in its cloud division, which grew 
    30% compared to last year. Management provided guidance of $5.5-5.7 billion for next quarter.""",
]

REWRITE_PROMPTS = [
    "Rewrite this more professionally: 'Hey, just wanted to check if you got my email from yesterday?'",
    "Make this more concise: 'In my opinion, I think that we should probably consider the possibility of maybe postponing the meeting.'",
    "Rewrite for clarity: 'The implementation of the new system will be done by the team in the next few weeks hopefully.'",
    "Make this friendlier: 'Your request has been denied due to policy violations.'",
    "Simplify this: 'The utilization of synergistic methodologies will facilitate the optimization of operational efficiency.'",
]

GENERATE_LONG_PROMPTS = [
    "Write a detailed product description for a smart home security camera.",
    "Create a comprehensive guide on how to start a vegetable garden.",
    "Write an engaging blog post about the future of remote work.",
    "Generate a detailed FAQ section for a subscription service.",
    "Write a thorough explanation of how blockchain technology works.",
]

CHAT_PROMPTS = [
    "What's the best way to learn Python programming?",
    "Can you explain the difference between REST and GraphQL?",
    "What are some healthy breakfast options?",
    "How do I improve my public speaking skills?",
    "What should I consider when buying a laptop?",
]


# =============================================================================
# GENERATORS
# =============================================================================

def generate_request(
    task_type: Optional[TaskType] = None,
    max_latency_ms: Optional[int] = None,
    min_quality: Optional[float] = None,
) -> Request:
    """
    Generate a single random request.
    
    Args:
        task_type: Specific task type, or random if None.
        max_latency_ms: Specific latency SLA, or random if None.
        min_quality: Specific quality threshold, or random if None.
    
    Returns:
        Generated Request.
    """
    if task_type is None:
        task_type = random.choice(list(TaskType))
    
    # Select prompt based on task type
    prompts = {
        TaskType.CLASSIFY: CLASSIFY_PROMPTS,
        TaskType.EXTRACT_JSON: EXTRACT_JSON_PROMPTS,
        TaskType.SUMMARIZE: SUMMARIZE_PROMPTS,
        TaskType.REWRITE: REWRITE_PROMPTS,
        TaskType.GENERATE_LONG: GENERATE_LONG_PROMPTS,
        TaskType.CHAT: CHAT_PROMPTS,
    }
    
    prompt = random.choice(prompts.get(task_type, CHAT_PROMPTS))
    
    # Random latency SLA if not specified
    if max_latency_ms is None:
        max_latency_ms = random.choice([500, 1000, 2000, 3000, 5000, 10000])
    
    # Random quality threshold if not specified
    if min_quality is None:
        min_quality = random.choice([0.7, 0.8, 0.85, 0.9, 0.95])
    
    # Random latency percentile
    latency_percentile = random.choice(list(LatencyPercentile))
    
    return Request(
        prompt=prompt,
        task_type=task_type,
        max_latency_ms=max_latency_ms,
        latency_percentile=latency_percentile,
        min_quality_score=min_quality,
        allow_batching=random.random() > 0.3,  # 70% allow batching
    )


def generate_steady_traffic(
    count: int,
    task_distribution: Optional[dict[TaskType, float]] = None,
) -> list[Request]:
    """
    Generate steady traffic with uniform characteristics.
    
    Args:
        count: Number of requests to generate.
        task_distribution: Optional dict of task_type -> probability.
    
    Returns:
        List of generated requests.
    """
    if task_distribution is None:
        # Default: uniform distribution
        task_types = list(TaskType)
    else:
        task_types = []
        for task_type, prob in task_distribution.items():
            task_types.extend([task_type] * int(prob * 100))
    
    requests = []
    for _ in range(count):
        task_type = random.choice(task_types)
        request = generate_request(
            task_type=task_type,
            max_latency_ms=2000,  # Consistent SLA
            min_quality=0.85,     # Consistent quality bar
        )
        requests.append(request)
    
    return requests


def generate_bursty_traffic(
    base_count: int,
    burst_count: int,
    burst_ratio: float = 0.2,
) -> list[Request]:
    """
    Generate bursty traffic pattern.
    
    Args:
        base_count: Number of baseline requests.
        burst_count: Number of burst requests.
        burst_ratio: Fraction of requests that are "urgent" during burst.
    
    Returns:
        List of requests (base + burst interleaved).
    """
    requests = []
    
    # Base requests: relaxed SLAs
    for _ in range(base_count):
        request = generate_request(
            max_latency_ms=5000,
            min_quality=0.8,
        )
        requests.append(request)
    
    # Burst requests: tight SLAs
    for _ in range(burst_count):
        if random.random() < burst_ratio:
            # Urgent request
            request = generate_request(
                max_latency_ms=500,
                min_quality=0.9,
            )
        else:
            # Normal burst request
            request = generate_request(
                max_latency_ms=1000,
                min_quality=0.85,
            )
        requests.append(request)
    
    # Shuffle to simulate interleaved arrival
    random.shuffle(requests)
    return requests


def generate_mixed_prompt_sizes(
    count: int,
    short_ratio: float = 0.3,
    medium_ratio: float = 0.5,
    long_ratio: float = 0.2,
) -> list[Request]:
    """
    Generate requests with varying prompt sizes.
    
    Args:
        count: Total number of requests.
        short_ratio: Fraction of short prompts (<500 chars).
        medium_ratio: Fraction of medium prompts (500-2000 chars).
        long_ratio: Fraction of long prompts (>2000 chars).
    
    Returns:
        List of requests with varied sizes.
    """
    requests = []
    
    for _ in range(count):
        r = random.random()
        
        if r < short_ratio:
            # Short prompt
            task_type = random.choice([TaskType.CLASSIFY, TaskType.CHAT])
            request = generate_request(task_type=task_type)
            
        elif r < short_ratio + medium_ratio:
            # Medium prompt
            task_type = random.choice([TaskType.EXTRACT_JSON, TaskType.REWRITE])
            request = generate_request(task_type=task_type)
            
        else:
            # Long prompt
            task_type = random.choice([TaskType.SUMMARIZE, TaskType.GENERATE_LONG])
            base_request = generate_request(task_type=task_type)
            
            # Pad prompt to make it longer
            padding = "\n\nAdditional context: " + " ".join(
                ["This is additional context that makes the prompt longer."] * 20
            )
            
            request = Request(
                prompt=base_request.prompt + padding,
                task_type=base_request.task_type,
                max_latency_ms=base_request.max_latency_ms,
                latency_percentile=base_request.latency_percentile,
                min_quality_score=base_request.min_quality_score,
                allow_batching=base_request.allow_batching,
            )
        
        requests.append(request)
    
    return requests


def generate_by_task_type(
    task_type: TaskType,
    count: int,
    sla_profile: str = "normal",
) -> list[Request]:
    """
    Generate requests for a specific task type.
    
    Args:
        task_type: The task type to generate.
        count: Number of requests.
        sla_profile: "relaxed", "normal", or "strict"
    
    Returns:
        List of requests.
    """
    sla_configs = {
        "relaxed": {"max_latency_ms": 10000, "min_quality": 0.7},
        "normal": {"max_latency_ms": 3000, "min_quality": 0.85},
        "strict": {"max_latency_ms": 1000, "min_quality": 0.95},
    }
    
    config = sla_configs.get(sla_profile, sla_configs["normal"])
    
    return [
        generate_request(
            task_type=task_type,
            max_latency_ms=config["max_latency_ms"],
            min_quality=config["min_quality"],
        )
        for _ in range(count)
    ]
