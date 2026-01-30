# Conductor

**Save money on LLM calls. Automatically.**

```bash
pip install conductor-llm[openai]
```

```python
from conductor import llm

response = llm("What is 2+2?")

print(response.text)   # "4"
print(response.model)  # "gpt-4o-mini" (auto-selected)
print(response.cost)   # $0.0001
print(response.saved)  # $0.0009 (vs always using gpt-4o)
```

---

## Features

### 1. Auto Model Selection

Conductor picks cheap or expensive models automatically:

| Your Prompt | Model Chosen | Why |
|-------------|--------------|-----|
| "What is 2+2?" | gpt-4o-mini | Simple → cheap |
| "Summarize this" | gpt-4o-mini | Straightforward → cheap |
| "Analyze this contract" | gpt-4o | Complex → quality |
| "Debug this code" | gpt-4o | Needs reasoning → quality |

```python
from conductor import llm

# Auto mode (default)
response = llm("What is the capital of France?")

# Force cheap
response = llm("Quick question", quality="low")

# Force quality
response = llm("Complex analysis...", quality="high")
```

---

### 2. Sessions (Multi-step Tracking)

Track costs across multiple related calls:

```python
from conductor import session

with session(budget_usd=1.00) as s:
    r1 = s.llm("Summarize this document")
    r2 = s.llm(f"Extract entities from: {r1.text}")
    r3 = s.llm(f"Format as report: {r2.text}")

print(s.total_cost)    # $0.0034
print(s.total_saved)   # $0.0089
print(s.savings_pct)   # 72%
print(s.get_report())  # Full breakdown
```

Budget strategies:
- `"warn"` - Log warning when budget exceeded (default)
- `"strict"` - Raise error when budget exceeded
- `"adaptive"` - Switch to cheaper models as budget depletes

---

### 3. Smart Classification (LLM-based)

Use an LLM to understand prompt complexity (~$0.00002 per call):

```python
from conductor import smart_classify

result = smart_classify("What is 2+2?")
print(result.complexity_score)  # 0.1 (simple)
print(result.reasoning)         # "Trivial task"

result = smart_classify("Analyze the economic implications...")
print(result.complexity_score)  # 0.8 (complex)
print(result.reasoning)         # "Complex task - multi-step reasoning"
```

---

### 4. Cost Control (Per-user Budgets)

Track and limit spending per user:

```python
from conductor import CostController, Period

controller = CostController()

# Set budget
controller.set_user_budget("user_123", daily_usd=5.00, monthly_usd=50.00)

# Check before request
controller.check_budget("user_123", estimated_cost=0.01)
# Raises BudgetExceededError if over limit

# Record after request
controller.record_cost("user_123", cost_usd=0.008, model="gpt-4o-mini", task_type="chat")

# Get reports
report = controller.get_cost_report(group_by="user", period=Period.DAILY)
print(report.total_cost_usd)
print(report.top_spenders)
```

---
### 5. Persistent Storage (Team-ready)

Use SQLite to persist budgets and cost records across processes:

```python
from conductor import CostController, SQLiteStorage

storage = SQLiteStorage(db_path="conductor.db")
controller = CostController(storage=storage)

controller.set_user_budget("user_123", daily_usd=5.00)
controller.record_cost("user_123", 0.008, "gpt-4o-mini", "chat")
```

---

## API Reference

### `llm(prompt, quality="auto")`

Make an LLM call with automatic model selection.

**Parameters:**
- `prompt` - Your prompt
- `quality` - `"auto"`, `"high"`, or `"low"`
- `api_key` - Optional OpenAI API key
- `max_tokens` - Max response tokens (default: 1000)

**Returns:** `Response` with `.text`, `.model`, `.cost`, `.saved`

---

### `llm_dry_run(prompt, quality="auto")`

See what model would be chosen without making an API call.

```python
from conductor import llm_dry_run

result = llm_dry_run("What is 2+2?")
# {'model': 'gpt-4o-mini', 'reason': 'Simple prompt'}
```

---

### `session(budget_usd=None, budget_strategy="warn")`

Create a session to track multiple calls.

```python
from conductor import session

with session(budget_usd=1.00, budget_strategy="strict") as s:
    s.llm("First call")
    s.llm("Second call")

print(s.total_cost)
print(s.get_report())
```

---

### `smart_classify(prompt)`

Classify prompt complexity using LLM.

```python
from conductor import smart_classify

result = smart_classify("Complex analysis...")
print(result.complexity_score)  # 0.0 to 1.0
print(result.raw_score)         # 1 to 10
print(result.reasoning)         # Human explanation
```

---

### `CostController`

Manage per-user budgets and tracking.

```python
from conductor import CostController, Period

controller = CostController()
controller.set_user_budget("user_id", daily_usd=10.00)
controller.check_budget("user_id")
controller.record_cost("user_id", 0.01, "gpt-4o-mini", "chat")
report = controller.get_cost_report(group_by="user", period=Period.DAILY)
```

---
### `SQLiteStorage(db_path="conductor.db")`

SQLite-backed storage for budgets and cost records.

```python
from conductor import CostController, SQLiteStorage

storage = SQLiteStorage("conductor.db")
controller = CostController(storage=storage)
```

---

## Configuration

Override pricing and model defaults programmatically:

```python
from conductor import set_pricing, set_models

set_pricing({
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
})

set_models(cheap="gpt-4o-mini", quality="gpt-4o", baseline="gpt-4o")
```

Or via environment variables:

```bash
export CONDUCTOR_PRICING_JSON='{"gpt-4o-mini":{"input":0.15,"output":0.60},"gpt-4o":{"input":2.50,"output":10.00}}'
export CONDUCTOR_MODELS_JSON='{"cheap":"gpt-4o-mini","quality":"gpt-4o","baseline":"gpt-4o"}'
```

---

## Setup

```bash
pip install conductor-llm[openai]
export OPENAI_API_KEY=sk-...
```

Or pass the key directly:

```python
response = llm("Hello", api_key="sk-...")
```

---
## Optional API Server

Run a self-hosted API server with SQLite (default):

```bash
pip install conductor-llm[openai,api]
export OPENAI_API_KEY=sk-...
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Optional env vars:

```bash
export CONDUCTOR_DB_PATH="conductor.db"
export CONDUCTOR_API_KEY="your-api-key"  # if set, require X-API-Key header
```

Example request:

```bash
curl -X POST http://localhost:8000/llm \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{"prompt":"Summarize this note","quality":"auto"}'
```

---

## Pricing

| Model | Input (per 1M) | Output (per 1M) |
|-------|----------------|-----------------|
| gpt-4o-mini | $0.15 | $0.60 |
| gpt-4o | $2.50 | $10.00 |

**gpt-4o-mini is ~17x cheaper.** Conductor uses it for simple tasks.

---

## License

MIT
