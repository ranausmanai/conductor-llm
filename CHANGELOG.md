# Changelog

All notable changes to Conductor will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.3.0] - 2026-01-25

### Added

#### Cost Control for Teams - "Never get surprised by an LLM bill again"

- **Per-user budgets** - Set hourly/daily/weekly/monthly limits per user
  - Hard limits that block requests when exceeded
  - Soft limits that warn but allow
  - Automatic enforcement on every request

- **Real-time alerts** - Get notified before you blow your budget
  - Configurable thresholds and periods
  - Callback support (Slack, email, PagerDuty, etc.)
  - Cooldown to prevent alert spam

- **Cost attribution reports** - See who's spending what
  - Group by user, model, task type, or feature
  - Top spenders list
  - Export to CSV

- **Anomaly detection** - Catch unusual spending patterns
  - Automatic spike detection
  - Configurable sensitivity
  - Per-user tracking

```python
# Set user budgets
conductor.set_user_budget("user_123", daily_usd=5.00, monthly_usd=50.00)

# Add alerts
conductor.add_cost_alert(threshold_usd=100, callback=send_slack)

# Get reports
report = conductor.get_cost_report(group_by="user", period=Period.DAILY)

# Detect anomalies
conductor.enable_anomaly_detection(sensitivity="medium", callback=alert_fn)
```

#### New Exports
- `CostController` - Full cost control functionality
- `CostReport` - Report data structure
- `UserBudget` - Budget configuration
- `Period` - Time periods (HOURLY, DAILY, WEEKLY, MONTHLY)
- `BudgetExceededError` - Raised when budget exceeded
- `AnomalyAlert` - Alert for spending anomalies

---

## [1.2.0] - 2026-01-25

### Added

#### Sessions & Workflows
- **Session management** - Group related LLM calls for tracking and budget management
  - Track total cost and latency across multiple calls
  - Budget-aware routing that adapts as budget depletes
  - Three budget strategies: adaptive, strict, warn

- **Savings visibility** - See exactly how much you saved
  - `session.total_savings` - Amount saved vs GPT-4o baseline
  - `session.savings_pct` - Percentage saved
  - `session.get_savings_report()` - Human-readable report
  - Automatic baseline cost calculation for comparison

- **Parallel execution** - Run multiple prompts concurrently
  - `session.parallel()` for concurrent requests
  - `session.map()` for applying a template to multiple items
  - `session.reduce()` for map-reduce patterns

- **Reusable workflows** - Define workflows once, run many times
  - `WorkflowBuilder` for declarative workflow definition
  - Automatic output passing between steps
  - Per-run budget constraints

```python
# Session with budget
with conductor.session("process-doc", budget_usd=0.05) as session:
    r1 = session.complete(prompt="Summarize...", task_type=TaskType.SUMMARIZE)
    r2 = session.complete(prompt=f"Extract: {r1.text}", task_type=TaskType.EXTRACT_JSON)
print(session.total_cost)  # Combined cost

# Reusable workflow
workflow = conductor.workflow_builder("pipeline")
    .add_step("summarize", TaskType.SUMMARIZE, "Summarize: {input}")
    .add_step("extract", TaskType.EXTRACT_JSON, "Extract: {summarize}")
    .build()
result = workflow.run(input=document)
```

#### New Exports
- `Session` - Session context manager
- `Workflow` - Reusable workflow runner
- `WorkflowBuilder` - Workflow definition builder
- `BudgetExceededError` - Raised when budget exceeded in strict mode

---

## [1.1.0] - 2026-01-25

### Added

#### Smart Complexity Detection
- **LLM-based complexity classification** - Uses a quick, cheap LLM call to semantically understand prompt complexity instead of regex rules
- Understands that "What's 2+2?" is simpler than "Explain quantum physics" even though both are short
- Cost: ~$0.00002 per classification (pays for itself with better routing)
- Caching: Identical prompts are cached to avoid repeat API calls
- Fallback: If API fails, falls back to rule-based heuristics
- Optional: Disable with `smart_classification=False` for faster, rule-based only routing

```python
# Smart classification enabled by default
conductor = Conductor.with_openai()

# Disable for speed (rule-based only)
conductor = Conductor.with_openai(smart_classification=False)
```

#### New Exports
- `SmartClassifier` - Direct access to complexity classifier
- `smart_classify()` - Convenience function for one-off classification

---

## [1.0.0] - 2026-01-25

### 🎉 Production Release

This is the first production-ready release of Conductor!

### Added

#### Core Features
- **Input Validation** - Comprehensive validation for all user inputs
  - Empty prompt detection
  - Parameter range checking (latency, quality, cost)
  - Maximum prompt length limits (1M characters)
  - Clear error messages with `ValidationError`

- **Rate Limiting** - Per-client rate limiting with cost tracking
  - Configurable requests per minute/hour limits
  - Cost-based limits (max USD per hour)
  - Thread-safe implementation
  - Real-time usage statistics

- **Circuit Breaker** - Protection against cascading failures
  - Automatic provider failure detection
  - Configurable failure thresholds
  - Half-open state for testing recovery
  - Per-provider circuit tracking

- **Metrics & Observability** - Comprehensive monitoring
  - Structured logging
  - Metrics collection (counters, gauges, histograms)
  - JSONL metrics export
  - Aggregated statistics (P50/P95/P99)

#### Deployment & Operations
- **Production Deployment Guide** - Complete guide for production deployment
  - Docker and Kubernetes examples
  - Environment variable configuration
  - Security best practices
  - Health check implementations

- **Examples & Case Studies**
  - Basic usage examples
  - ROI calculator showing cost savings
  - Real-world e-commerce case study (59% cost reduction)
  - Integration patterns

### Changed

#### Performance Improvements
- **Optimized Default Coefficients** - Reduced fallback rate from 65-80% to <10%
  - Reduced base latency estimates (gpt-4o-mini: 150ms → 80ms)
  - Reduced P95 multipliers (2.0x → 1.5x average)
  - Reduced quality risk estimates (0.15 → 0.08 for gpt-4o-mini)
  - More realistic latency-per-token coefficients

#### Error Handling
- **Enhanced Retry Logic** - Improved retry mechanism with exponential backoff
  - Configurable max retries (default: 3)
  - Exponential backoff delays (100ms → 10s)
  - Retry statistics tracking

- **Better Error Messages** - More informative error messages
  - Specific validation failures
  - Rate limit details with remaining capacity
  - Circuit breaker state information

### Fixed

- **Datetime Deprecation Warnings** - Replaced deprecated `datetime.utcnow()` with `datetime.now(UTC)`
  - Fixed in all modules (control_plane, router, executor, calibrator, evaluator, cli, schemas, coefficients)
  - 3,538 deprecation warnings eliminated

- **Empty Prompt Bug** - Now properly validates and rejects empty prompts
- **Missing Error Handling** - Added validation for edge cases

### Removed

- **conductor_final.py** - Removed redundant implementation file
  - Kept main conductor package as canonical implementation
  - Simplified project structure

## [0.1.0] - 2026-01-24

### Initial Release

- Core routing engine
- Strategy definitions
- Cost/latency/quality predictions
- CLI interface
- A/B testing framework
- Calibration system
- Basic logging

---

## Migration Guide

### Upgrading from 0.1.0 to 1.0.0

#### Breaking Changes

None! Version 1.0.0 is fully backward compatible with 0.1.0.

#### New Features You Should Use

1. **Add Input Validation Error Handling**

```python
from conductor import Conductor, ValidationError

try:
    response = conductor.complete(...)
except ValidationError as e:
    # Handle validation errors
    logger.error(f"Invalid input: {e}")
```

2. **Add Rate Limiting for Production**

```python
from conductor.rate_limiter import RateLimiter, RateLimitConfig

limiter = RateLimiter(
    config=RateLimitConfig(
        requests_per_minute=100,
        max_cost_per_hour_usd=50.0
    )
)

# Check before each request
limiter.check_and_record(client_id="user_123", estimated_cost_usd=0.001)
```

3. **Enable Metrics Collection**

```python
from conductor.metrics import MetricsCollector
from pathlib import Path

metrics = MetricsCollector(
    metrics_file=Path("/var/log/conductor/metrics.jsonl"),
    enable_logging=True
)

# Metrics are automatically collected
response = conductor.complete(...)
stats = metrics.get_stats()
```

#### Recommended Actions

1. **Update Coefficients** - The new defaults are much better, but you may want to calibrate:

```bash
conductor calibrate --log-file production.jsonl --output calibrated.json
```

2. **Test Fallback Rate** - Run experiments to verify low fallback rate:

```bash
conductor experiment --pattern steady --count 1000
# Should show <10% fallback rate
```

3. **Set Up Monitoring** - Add health checks and metrics collection for production deployments

## Support

For questions or issues:
- GitHub Issues: https://github.com/ranausmans/conductor/issues
- Documentation: https://github.com/ranausmans/conductor#readme
