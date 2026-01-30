# Production Deployment Guide

This guide covers deploying Conductor in production environments.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Docker Deployment](#docker-deployment)
- [Environment Variables](#environment-variables)
- [Rate Limiting](#rate-limiting)
- [Monitoring](#monitoring)
- [Security](#security)
- [Scaling](#scaling)
- [Troubleshooting](#troubleshooting)

## Prerequisites

- Python 3.10 or higher
- OpenAI API key and/or Anthropic API key
- (Optional) Redis for distributed rate limiting
- (Optional) PostgreSQL for persistent logging

## Installation

### Basic Installation

```bash
pip install conductor-llm
```

### With Optional Dependencies

```bash
# For OpenAI support
pip install conductor-llm[openai]

# For Anthropic support
pip install conductor-llm[anthropic]

# For all providers
pip install conductor-llm[all]

# For development
pip install conductor-llm[dev]
```

### From Source

```bash
git clone https://github.com/your-org/conductor
cd conductor
pip install -e .
```

## Configuration

### Basic Configuration

Create a configuration file `conductor_config.json`:

```json
{
  "policy_version": "v1.0.0",
  "rate_limits": {
    "requests_per_minute": 100,
    "requests_per_hour": 5000,
    "max_cost_per_hour_usd": 50.0
  },
  "retry_config": {
    "max_retries": 3,
    "retry_base_delay_ms": 100,
    "retry_max_delay_ms": 10000
  },
  "circuit_breaker": {
    "failure_threshold": 5,
    "success_threshold": 2,
    "timeout_seconds": 60.0
  }
}
```

### Using Custom Coefficients

```python
from conductor import Conductor, Coefficients
from pathlib import Path

# Load custom coefficients
coefficients = Coefficients.from_dict({
    "version": "production-v1",
    "models": {
        "gpt-4o-mini": {
            "cost_per_input_1k": 0.00015,
            "cost_per_output_1k": 0.0006,
            "base_latency_ms": 80,
            # ... other parameters
        }
    }
})

conductor = Conductor(coefficients=coefficients)
```

## Docker Deployment

### Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Install conductor
RUN pip install -e .

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "from conductor import Conductor; c = Conductor(dry_run=True)"

# Run application
CMD ["python", "-m", "your_app"]
```

### Docker Compose

```yaml
version: '3.8'

services:
  conductor:
    build: .
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - LOG_LEVEL=INFO
    volumes:
      - ./logs:/app/logs
      - ./config:/app/config
    ports:
      - "8000:8000"
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

## Environment Variables

Required:
- `OPENAI_API_KEY` - OpenAI API key (if using OpenAI)
- `ANTHROPIC_API_KEY` - Anthropic API key (if using Anthropic)

Optional:
- `CONDUCTOR_LOG_LEVEL` - Logging level (default: INFO)
- `CONDUCTOR_LOG_FILE` - Path to log file
- `CONDUCTOR_METRICS_FILE` - Path to metrics file
- `CONDUCTOR_DRY_RUN` - Set to "true" for dry-run mode
- `CONDUCTOR_POLICY_VERSION` - Policy version identifier

## Rate Limiting

### Per-Client Rate Limiting

```python
from conductor import Conductor
from conductor.rate_limiter import RateLimiter, RateLimitConfig

# Configure rate limits
rate_limiter = RateLimiter(
    config=RateLimitConfig(
        requests_per_minute=60,
        requests_per_hour=1000,
        max_cost_per_hour_usd=10.0
    )
)

conductor = Conductor()

# Before making request
try:
    rate_limiter.check_and_record(
        client_id="user_123",
        estimated_cost_usd=0.001
    )
    response = conductor.complete(...)
except RateLimitError as e:
    print(f"Rate limit exceeded: {e}")
```

### Get Rate Limit Stats

```python
stats = rate_limiter.get_stats("user_123")
print(f"Remaining this hour: {stats['hour_remaining']}")
print(f"Cost remaining: ${stats['cost_remaining_usd']:.2f}")
```

## Monitoring

### Metrics Collection

```python
from conductor import Conductor
from conductor.metrics import MetricsCollector
from pathlib import Path

# Set up metrics
metrics = MetricsCollector(
    metrics_file=Path("/var/log/conductor/metrics.jsonl"),
    enable_logging=True
)

conductor = Conductor()

# Metrics are automatically collected
response = conductor.complete(...)

# Get stats
stats = metrics.get_stats()
print(f"Total requests: {stats['counters']['requests_total']}")
print(f"Average cost: ${stats['cost']['avg_usd']:.4f}")
print(f"P95 latency: {stats['latency']['p95_ms']}ms")
```

### Health Checks

```python
from conductor import Conductor, TaskType

def health_check():
    """Health check endpoint for load balancers."""
    try:
        conductor = Conductor(dry_run=True)
        response = conductor.complete(
            prompt="health check",
            task_type=TaskType.CLASSIFY
        )
        return {"status": "healthy", "version": "1.0.0"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}
```

## Security

### API Key Management

**Never commit API keys to source control!**

Use environment variables or a secrets manager:

```python
import os
from conductor import Conductor

# From environment
conductor = Conductor.with_openai(
    api_key=os.environ["OPENAI_API_KEY"]
)

# Or from secrets manager (AWS example)
import boto3

def get_secret(secret_name):
    client = boto3.client('secretsmanager')
    response = client.get_secret_value(SecretId=secret_name)
    return response['SecretString']

conductor = Conductor.with_openai(
    api_key=get_secret("production/openai/api-key")
)
```

### Input Validation

Conductor includes built-in input validation:

```python
from conductor import Conductor, ValidationError

conductor = Conductor()

try:
    response = conductor.complete(
        prompt="",  # Empty prompt
        task_type=TaskType.CLASSIFY
    )
except ValidationError as e:
    print(f"Invalid input: {e}")
```

## Scaling

### Horizontal Scaling

Conductor is stateless and can be scaled horizontally:

```yaml
# Kubernetes deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: conductor
spec:
  replicas: 3
  selector:
    matchLabels:
      app: conductor
  template:
    metadata:
      labels:
        app: conductor
    spec:
      containers:
      - name: conductor
        image: your-registry/conductor:latest
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: conductor-secrets
              key: openai-api-key
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
```

### Load Balancing

Use a load balancer with health checks:

```bash
# HAProxy example
backend conductor
    balance roundrobin
    option httpchk GET /health
    server conductor1 10.0.0.1:8000 check
    server conductor2 10.0.0.2:8000 check
    server conductor3 10.0.0.3:8000 check
```

## Troubleshooting

### High Fallback Rate

If you see high fallback rates (>10%):

1. Check your default SLAs - they might be too strict
2. Calibrate coefficients from production logs:

```bash
conductor calibrate --log-file production.jsonl --output calibrated.json
```

3. Tune quality thresholds - start with 0.7-0.8 instead of 0.9+

### High Latency

If requests are slow:

1. Check P95 predictions in logs
2. Reduce `max_latency_ms` to force faster models
3. Enable batching for non-realtime requests
4. Use circuit breakers to detect slow providers

### Cost Overruns

Monitor and control costs:

```python
# Set strict cost limits
response = conductor.complete(
    prompt="...",
    task_type=TaskType.SUMMARIZE,
    max_cost_usd=0.01  # Hard limit
)

# Use rate limiting with cost tracking
from conductor.rate_limiter import RateLimiter, RateLimitConfig

limiter = RateLimiter(
    config=RateLimitConfig(
        max_cost_per_hour_usd=100.0
    )
)
```

### Circuit Breaker Tripped

If providers are failing:

```python
from conductor.circuit_breaker import CircuitBreaker

# Check circuit state
breaker = CircuitBreaker("openai")
stats = breaker.get_stats()
print(f"State: {stats['state']}")
print(f"Time until retry: {stats['time_until_retry']}s")

# Manual reset if needed
breaker.reset()
```

## Performance Tuning

### Coefficient Calibration

Regularly calibrate from production data:

```bash
# Export logs
conductor export-logs --format csv --output logs.csv

# Calibrate
conductor calibrate --log-file logs.jsonl --min-entries 1000

# Test new coefficients
conductor compare \
  --name-a current \
  --name-b calibrated \
  --coefficients-b calibrated.json \
  --count 1000
```

### Caching

Implement caching for repeated queries:

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_complete(prompt_hash, task_type):
    return conductor.complete(
        prompt=original_prompt,
        task_type=task_type
    )
```

## Production Checklist

Before going to production:

- [ ] API keys stored in secrets manager
- [ ] Rate limiting configured
- [ ] Monitoring and alerts set up
- [ ] Health checks implemented
- [ ] Logging configured
- [ ] Error handling tested
- [ ] Load testing completed
- [ ] Backup/failover plan ready
- [ ] Cost alerts configured
- [ ] Documentation updated

## Support

For issues or questions:
- GitHub Issues: https://github.com/your-org/conductor/issues
- Documentation: https://github.com/your-org/conductor#readme
