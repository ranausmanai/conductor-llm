# Conductor 🎭

**Never get surprised by an LLM bill again.**

Conductor = Smart routing + Cost control for teams.

```python
pip install conductor-llm
```

## Why Conductor?

| Feature | LiteLLM | Helicone | RouteLLM | **Conductor** |
|---------|---------|----------|----------|---------------|
| Smart routing | ❌ | ❌ | ✅ | ✅ |
| Per-user budgets | ❌ | ✅ | ❌ | ✅ |
| Real-time alerts | ❌ | ✅ | ❌ | ✅ |
| Anomaly detection | ❌ | ❌ | ❌ | ✅ |
| Savings visibility | ❌ | ❌ | ❌ | ✅ |
| Sessions/workflows | ❌ | ❌ | ❌ | ✅ |
| Self-hosted | ✅ | ❌ | ✅ | ✅ |
| Free & open source | ✅ | Paid | ✅ | ✅ |

**Conductor is like having a CFO for your LLM costs.**

## 30-Second Quick Start

```python
from conductor import Conductor, TaskType

# Set up (uses OPENAI_API_KEY from environment)
conductor = Conductor.with_openai()

# Use it like normal - Conductor picks the best model automatically
response = conductor.complete(
    prompt="What is the capital of France?",
    task_type=TaskType.CLASSIFY,
)

print(response.text)        # "Paris"
print(response.model_used)  # "gpt-4o-mini" (cheap model - simple question)
print(response.cost)        # $0.0001
```

That's it. Conductor figures out the rest.

---

## Cost Control for Teams

### Set User Budgets
```python
# Set budget limits per user
conductor.set_user_budget(
    "user_123",
    daily_usd=5.00,
    monthly_usd=50.00
)

# Now this user's requests are automatically limited
response = conductor.complete(
    prompt="...",
    task_type=TaskType.CLASSIFY,
    client_id="user_123"  # Budget enforced automatically
)
# Raises BudgetExceededError if over budget
```

### Get Alerts Before It's Too Late
```python
def send_slack_alert(info):
    requests.post(SLACK_WEBHOOK, json={
        "text": f"LLM spend hit ${info['total']:.2f}!"
    })

conductor.add_cost_alert(
    threshold_usd=100.00,
    period=Period.DAILY,
    callback=send_slack_alert
)
```

### See Who's Spending What
```python
report = conductor.get_cost_report(group_by="user", period=Period.DAILY)

print(f"Total: ${report.total_cost_usd:.2f}")
for user, cost in report.top_spenders:
    print(f"  {user}: ${cost:.4f}")
```

### Detect Anomalies
```python
conductor.enable_anomaly_detection(
    sensitivity="medium",
    callback=lambda alert: send_pagerduty(alert.message)
)
# "User X spending 500% above normal!"
```

---

## How It Saves Money

| Your Request | What Conductor Does | Cost |
|--------------|---------------------|------|
| "Is this spam?" | Uses cheap model (simple) | $0.0001 |
| "Summarize this doc" | Uses cheap model (straightforward) | $0.0003 |
| "Extract all entities as JSON" | Uses quality model (needs accuracy) | $0.002 |
| "Write a 2000-word article" | Uses quality model (complex) | $0.01 |

**Result:** 60-70% of requests use the cheap model. Same quality. Lower cost.

---

## Smart Complexity Detection

Conductor **actually understands** your prompts. It uses a quick LLM call to assess complexity - not just regex rules.

| Prompt | Complexity | Conductor's Understanding |
|--------|------------|---------------------------|
| "What's 2+2?" | Low | Basic math, trivial |
| "Summarize this email" | Low | Straightforward task |
| "Explain quantum entanglement simply" | Medium | Needs clarity, analogy |
| "Debug this code and explain the fix" | High | Reasoning + code analysis |
| "Write a legal contract for X" | High | Domain expertise, precision |

This costs ~$0.00002 per classification - pays for itself with better routing.

Want rule-based only (faster, no extra API call)?
```python
conductor = Conductor.with_openai(smart_classification=False)
```

---

## Task Types

Tell Conductor what you're doing:

```python
TaskType.CLASSIFY      # Yes/no, categories, simple answers
TaskType.SUMMARIZE     # Summarize text
TaskType.EXTRACT_JSON  # Extract structured data
TaskType.REWRITE       # Rewrite/edit text
TaskType.GENERATE_LONG # Write long content
TaskType.CHAT          # Conversation
```

---

## Set Quality Requirements

Need higher accuracy? Just ask:

```python
response = conductor.complete(
    prompt="Extract all person names from this legal document...",
    task_type=TaskType.EXTRACT_JSON,
    min_quality=0.95,  # High accuracy needed
)
# Conductor will use a better model
```

Need fast responses?

```python
response = conductor.complete(
    prompt="Is this urgent?",
    task_type=TaskType.CLASSIFY,
    max_latency_ms=500,  # Must respond in 500ms
)
# Conductor will pick the fastest option
```

---

## Dry Run (Test Without API Calls)

```python
conductor = Conductor(dry_run=True)

response = conductor.complete(
    prompt="Test request",
    task_type=TaskType.CLASSIFY,
)

print(response.model_used)  # See what model WOULD be chosen
print(response.cost)        # See predicted cost
print(response.why)         # See why it chose that model
```

---

## Real Results

From production deployments:

- **59% cost reduction** ($45K → $18.5K/month)
- **40% faster** responses
- **Same quality** scores

See full [case study](examples/CASE_STUDY.md).

---

---

## Multi-Step Workflows

For workflows with multiple prompts, use sessions:

```python
# Track costs across related calls
with conductor.session("process-invoice") as session:
    summary = session.complete(prompt="Summarize: " + doc, task_type=TaskType.SUMMARIZE)
    entities = session.complete(prompt=f"Extract names: {summary.text}", task_type=TaskType.EXTRACT_JSON)
    report = session.complete(prompt=f"Format report: {entities.text}", task_type=TaskType.REWRITE)

print(session.total_cost)       # Total cost of all 3 calls
print(session.total_latency_ms) # Total time
print(session.steps)            # Details of each step
```

### See Your Savings

Every session shows exactly how much you saved:

```python
with conductor.session("process-invoice") as session:
    # ... your workflow ...

# See the savings
print(session.total_cost)      # $0.0023 (what you paid)
print(session.baseline_cost)   # $0.0089 (GPT-4o for everything)
print(session.total_savings)   # $0.0066 (you saved this much)
print(session.savings_pct)     # 74% savings

# Or get a full report
print(session.get_savings_report())
```

Output:
```
===================================
   Conductor Savings Report
===================================
Session: process-invoice
Steps: 3

Cost Breakdown:
  Actual cost:    $0.0023
  Baseline cost:  $0.0089 (GPT-4o for all)
  You saved:      $0.0066 (74%)

Models Used:
  gpt-4o-mini: 2 calls
  gpt-4o: 1 call
===================================
```

### Budget-Aware Routing

Set a budget and Conductor adapts as it depletes:

```python
with conductor.session("user-request", budget_usd=0.05) as session:
    r1 = session.complete(...)  # Full budget, uses optimal model
    r2 = session.complete(...)  # Budget getting low, may use cheaper
    r3 = session.complete(...)  # Adapts to stay within budget
```

### Parallel Execution

Process multiple items concurrently:

```python
with conductor.session("batch-analysis") as session:
    results = session.parallel([
        {"prompt": "Analyze doc 1", "task_type": TaskType.CLASSIFY},
        {"prompt": "Analyze doc 2", "task_type": TaskType.CLASSIFY},
        {"prompt": "Analyze doc 3", "task_type": TaskType.CLASSIFY},
    ])

# Or use map for cleaner syntax:
results = session.map(
    items=documents,
    prompt_template="Summarize: {item}",
    task_type=TaskType.SUMMARIZE,
)
```

### Reusable Workflows

Define workflows once, run many times:

```python
# Define
doc_processor = (
    conductor.workflow_builder("doc-processor")
    .add_step("summarize", TaskType.SUMMARIZE, "Summarize: {input}")
    .add_step("extract", TaskType.EXTRACT_JSON, "Extract entities: {summarize}")
    .add_step("format", TaskType.REWRITE, "Format as report: {extract}")
    .build()
)

# Run on different documents
result1 = doc_processor.run(input=doc1, budget_usd=0.05)
result2 = doc_processor.run(input=doc2, budget_usd=0.05)

print(result1["result"])  # Final output
print(result1["stats"])   # Cost, latency, models used
```

---

## That's the Basics!

For most users, that's all you need. Conductor handles the rest.

---

## Want More Control?

<details>
<summary>📊 Analytics & Monitoring</summary>

```python
from conductor.metrics import MetricsCollector

metrics = MetricsCollector()

# After running requests...
stats = metrics.get_stats()
print(f"Total cost: ${stats['cost']['total_usd']:.2f}")
print(f"Avg latency: {stats['latency']['avg_ms']}ms")
print(f"Requests: {stats['counters']['requests_total']}")
```

</details>

<details>
<summary>🛡️ Rate Limiting</summary>

```python
from conductor.rate_limiter import RateLimiter, RateLimitConfig

limiter = RateLimiter(
    config=RateLimitConfig(
        requests_per_minute=100,
        max_cost_per_hour_usd=50.0
    )
)

# Check before each request
limiter.check_and_record(client_id="user_123")
```

</details>

<details>
<summary>🔧 Custom Strategies</summary>

```python
from conductor.strategies import Strategy, add_custom_strategy

custom = Strategy(
    strategy_id="my-strategy",
    model_id="gpt-4o",
    temperature=0.3,
    max_tokens=2000,
    # ... other settings
)

add_custom_strategy(custom)
```

</details>

<details>
<summary>📈 A/B Testing Policies</summary>

```bash
conductor compare --name-a baseline --name-b new --count 1000
```

</details>

<details>
<summary>🚀 Production Deployment</summary>

See [DEPLOYMENT.md](DEPLOYMENT.md) for Docker, Kubernetes, and security setup.

</details>

---

## CLI Commands

```bash
# See what model would be chosen
conductor predict "Your prompt here" --task-type classify

# Run experiments
conductor experiment --count 1000

# Compare policies
conductor compare --name-a current --name-b new --count 500
```

---

## Providers

```python
# OpenAI (default)
conductor = Conductor.with_openai()

# Anthropic
conductor = Conductor.with_anthropic()

# Both (set OPENAI_API_KEY and ANTHROPIC_API_KEY)
conductor = Conductor.with_openai()  # Can route to either
```

---

## Installation

```bash
# Basic
pip install conductor-llm

# With OpenAI
pip install conductor-llm[openai]

# With Anthropic
pip install conductor-llm[anthropic]

# Everything
pip install conductor-llm[all]
```

---

## Links

- [Examples](examples/) - Code samples
- [Deployment Guide](DEPLOYMENT.md) - Production setup
- [Case Study](examples/CASE_STUDY.md) - Real-world results
- [Changelog](CHANGELOG.md) - Version history

---

## License

MIT
