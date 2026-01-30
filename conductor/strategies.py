"""
Strategy definitions for Conductor.

Each strategy is a specific configuration of model + parameters
optimized for certain use cases.
"""

from conductor.schemas import Strategy, TaskType, Provider


# =============================================================================
# DEFAULT STRATEGIES
# =============================================================================

STRATEGIES: dict[str, Strategy] = {
    
    # -------------------------------------------------------------------------
    # TURBO-CHEAP: Fast and cheap for simple tasks
    # -------------------------------------------------------------------------
    "turbo-cheap": Strategy(
        strategy_id="turbo-cheap",
        model_id="gpt-4o-mini",
        temperature=0.0,
        max_tokens=256,
        batching_delay_ms=0,
        verifier_enabled=False,
        tools_enabled=False,
        min_input_tokens=0,
        max_input_tokens=4000,
        supported_task_types=[
            TaskType.CLASSIFY,
            TaskType.EXTRACT_JSON,
            TaskType.REWRITE,
            TaskType.CHAT,
        ],
        provider=Provider.OPENAI,
        timeout_ms=10000,
    ),
    
    # -------------------------------------------------------------------------
    # TURBO-QUALITY: Fast with better quality
    # -------------------------------------------------------------------------
    "turbo-quality": Strategy(
        strategy_id="turbo-quality",
        model_id="gpt-4o",
        temperature=0.0,
        max_tokens=512,
        batching_delay_ms=0,
        verifier_enabled=False,
        tools_enabled=False,
        min_input_tokens=0,
        max_input_tokens=16000,
        supported_task_types=[
            TaskType.CLASSIFY,
            TaskType.EXTRACT_JSON,
            TaskType.SUMMARIZE,
            TaskType.REWRITE,
            TaskType.CHAT,
        ],
        provider=Provider.OPENAI,
        timeout_ms=15000,
    ),
    
    # -------------------------------------------------------------------------
    # BATCH-CHEAP: Batched for high throughput, relaxed latency
    # -------------------------------------------------------------------------
    "batch-cheap": Strategy(
        strategy_id="batch-cheap",
        model_id="gpt-4o-mini",
        temperature=0.0,
        max_tokens=512,
        batching_delay_ms=500,
        verifier_enabled=False,
        tools_enabled=False,
        min_input_tokens=0,
        max_input_tokens=4000,
        supported_task_types=[
            TaskType.CLASSIFY,
            TaskType.SUMMARIZE,
            TaskType.REWRITE,
        ],
        provider=Provider.OPENAI,
        timeout_ms=20000,
    ),
    
    # -------------------------------------------------------------------------
    # QUALITY-VERIFIED: High stakes, double-checked
    # -------------------------------------------------------------------------
    "quality-verified": Strategy(
        strategy_id="quality-verified",
        model_id="gpt-4o",
        temperature=0.2,
        max_tokens=1024,
        batching_delay_ms=0,
        verifier_enabled=True,
        tools_enabled=False,
        min_input_tokens=0,
        max_input_tokens=32000,
        supported_task_types=[
            TaskType.CLASSIFY,
            TaskType.EXTRACT_JSON,
            TaskType.SUMMARIZE,
            TaskType.GENERATE_LONG,
        ],
        provider=Provider.OPENAI,
        timeout_ms=30000,
    ),
    
    # -------------------------------------------------------------------------
    # TOOL-AUGMENTED: Function calling for structured extraction
    # -------------------------------------------------------------------------
    "tool-augmented": Strategy(
        strategy_id="tool-augmented",
        model_id="gpt-4o",
        temperature=0.0,
        max_tokens=512,
        batching_delay_ms=0,
        verifier_enabled=False,
        tools_enabled=True,
        min_input_tokens=0,
        max_input_tokens=16000,
        supported_task_types=[
            TaskType.EXTRACT_JSON,
        ],
        provider=Provider.OPENAI,
        timeout_ms=20000,
    ),
    
    # -------------------------------------------------------------------------
    # LONG-FORM: Long content generation
    # -------------------------------------------------------------------------
    "long-form": Strategy(
        strategy_id="long-form",
        model_id="gpt-4o",
        temperature=0.7,
        max_tokens=4096,
        batching_delay_ms=0,
        verifier_enabled=False,
        tools_enabled=False,
        min_input_tokens=100,
        max_input_tokens=64000,
        supported_task_types=[
            TaskType.SUMMARIZE,
            TaskType.GENERATE_LONG,
        ],
        provider=Provider.OPENAI,
        timeout_ms=60000,
    ),
}


# =============================================================================
# ANTHROPIC STRATEGIES (Optional, add if using Claude)
# =============================================================================

ANTHROPIC_STRATEGIES: dict[str, Strategy] = {
    
    "claude-fast": Strategy(
        strategy_id="claude-fast",
        model_id="claude-3-haiku-20240307",
        temperature=0.0,
        max_tokens=256,
        batching_delay_ms=0,
        verifier_enabled=False,
        tools_enabled=False,
        min_input_tokens=0,
        max_input_tokens=4000,
        supported_task_types=[
            TaskType.CLASSIFY,
            TaskType.EXTRACT_JSON,
            TaskType.REWRITE,
            TaskType.CHAT,
        ],
        provider=Provider.ANTHROPIC,
        timeout_ms=10000,
    ),
    
    "claude-quality": Strategy(
        strategy_id="claude-quality",
        model_id="claude-3-5-sonnet-20241022",
        temperature=0.0,
        max_tokens=1024,
        batching_delay_ms=0,
        verifier_enabled=False,
        tools_enabled=False,
        min_input_tokens=0,
        max_input_tokens=32000,
        supported_task_types=[
            TaskType.CLASSIFY,
            TaskType.EXTRACT_JSON,
            TaskType.SUMMARIZE,
            TaskType.REWRITE,
            TaskType.GENERATE_LONG,
            TaskType.CHAT,
        ],
        provider=Provider.ANTHROPIC,
        timeout_ms=30000,
    ),
}


def get_strategy(strategy_id: str) -> Strategy:
    """Get a strategy by ID."""
    all_strategies = {**STRATEGIES, **ANTHROPIC_STRATEGIES}
    if strategy_id not in all_strategies:
        raise ValueError(f"Unknown strategy: {strategy_id}")
    return all_strategies[strategy_id]


def get_all_strategies(include_anthropic: bool = False) -> dict[str, Strategy]:
    """Get all available strategies."""
    if include_anthropic:
        return {**STRATEGIES, **ANTHROPIC_STRATEGIES}
    return STRATEGIES.copy()


def add_custom_strategy(strategy: Strategy) -> None:
    """Register a custom strategy."""
    STRATEGIES[strategy.strategy_id] = strategy
