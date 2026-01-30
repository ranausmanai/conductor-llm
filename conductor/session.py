"""
Session management for Conductor.

Sessions group related LLM calls together for:
- Cost tracking across a workflow
- Budget-aware routing (adapts as budget depletes)
- Latency tracking
- Parallel execution
"""

from dataclasses import dataclass, field
from datetime import datetime, UTC
from typing import Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import uuid
import threading


@dataclass
class StepResult:
    """Result of a single step in a session."""
    step_id: str
    step_number: int
    prompt: str
    task_type: str
    text: str
    model_used: str
    cost_usd: float
    latency_ms: int
    timestamp: datetime
    baseline_cost_usd: float = 0.0  # What GPT-4o would have cost
    savings_usd: float = 0.0  # baseline - actual
    metadata: dict = field(default_factory=dict)


@dataclass
class SessionStats:
    """Aggregated statistics for a session."""
    session_id: str
    total_steps: int
    total_cost_usd: float
    total_latency_ms: int
    avg_cost_per_step: float
    avg_latency_per_step: float
    models_used: dict[str, int]  # model -> count
    budget_usd: Optional[float]
    budget_remaining_usd: Optional[float]
    budget_utilization_pct: Optional[float]
    # Savings tracking
    baseline_cost_usd: float = 0.0  # What it would cost with GPT-4o only
    total_savings_usd: float = 0.0  # baseline - actual
    savings_pct: float = 0.0  # Percentage saved


class Session:
    """
    A session groups related LLM calls for tracking and budget management.

    Features:
    - Tracks total cost and latency across all calls
    - Optional budget constraint with adaptive routing
    - Parallel execution support
    - Works as a context manager

    Example:
        ```python
        with conductor.session("process-doc", budget_usd=0.05) as session:
            r1 = session.complete(prompt="Summarize...", task_type=TaskType.SUMMARIZE)
            r2 = session.complete(prompt=f"Extract from {r1.text}", task_type=TaskType.EXTRACT_JSON)

        print(session.total_cost)  # $0.0034
        print(session.steps)       # List of StepResult
        ```
    """

    def __init__(
        self,
        conductor,
        session_id: Optional[str] = None,
        budget_usd: Optional[float] = None,
        budget_strategy: str = "adaptive",
        max_parallel: int = 10,
        metadata: Optional[dict] = None,
    ):
        """
        Initialize a session.

        Args:
            conductor: The Conductor instance to use.
            session_id: Unique identifier for this session. Auto-generated if not provided.
            budget_usd: Optional total budget for all calls in this session.
            budget_strategy: How to handle budget constraints:
                - "adaptive": Adjust model selection as budget depletes (default)
                - "strict": Fail if a call would exceed budget
                - "warn": Log warning but allow exceeding budget
            max_parallel: Maximum concurrent calls for parallel execution.
            metadata: Optional metadata to attach to the session.
        """
        self._conductor = conductor
        self.session_id = session_id or f"session-{uuid.uuid4().hex[:8]}"
        self.budget_usd = budget_usd
        self.budget_strategy = budget_strategy
        self.max_parallel = max_parallel
        self.metadata = metadata or {}

        self._steps: list[StepResult] = []
        self._step_counter = 0
        self._lock = threading.Lock()
        self._started_at: Optional[datetime] = None
        self._ended_at: Optional[datetime] = None
        self._is_active = False

    def __enter__(self):
        """Start the session."""
        self._started_at = datetime.now(UTC)
        self._is_active = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """End the session."""
        self._ended_at = datetime.now(UTC)
        self._is_active = False
        return False  # Don't suppress exceptions

    def complete(
        self,
        prompt: str,
        task_type,
        **kwargs,
    ):
        """
        Complete a prompt within this session.

        Tracks the call and applies budget-aware routing if configured.

        Args:
            prompt: The prompt to complete.
            task_type: The type of task.
            **kwargs: Additional arguments passed to conductor.complete()

        Returns:
            ConductorResponse with the completion result.
        """
        if not self._is_active:
            raise RuntimeError("Session is not active. Use 'with session:' context manager.")

        # Apply budget-aware adjustments
        kwargs = self._apply_budget_strategy(kwargs)

        # Make the call
        response = self._conductor.complete(prompt=prompt, task_type=task_type, **kwargs)

        # Calculate baseline cost (what GPT-4o would have cost)
        # GPT-4o pricing: $2.50/1M input, $10/1M output (as of 2024)
        baseline_cost = self._estimate_baseline_cost(prompt, response.text)
        actual_cost = response.cost
        savings = max(0, baseline_cost - actual_cost)

        # Record the step
        with self._lock:
            self._step_counter += 1
            step = StepResult(
                step_id=f"{self.session_id}-step-{self._step_counter}",
                step_number=self._step_counter,
                prompt=prompt[:200] + "..." if len(prompt) > 200 else prompt,
                task_type=task_type.value if hasattr(task_type, 'value') else str(task_type),
                text=response.text,
                model_used=response.model_used,
                cost_usd=actual_cost,
                latency_ms=response.latency_ms,
                timestamp=datetime.now(UTC),
                baseline_cost_usd=baseline_cost,
                savings_usd=savings,
                metadata=kwargs.get('metadata', {}),
            )
            self._steps.append(step)

        return response

    def _estimate_baseline_cost(self, prompt: str, response_text: str) -> float:
        """
        Estimate what GPT-4o would have cost for this request.

        Uses GPT-4o pricing as the baseline for comparison:
        - Input: $2.50 per 1M tokens
        - Output: $10.00 per 1M tokens
        """
        # Rough token estimation: ~4 chars per token
        input_tokens = len(prompt) / 4
        output_tokens = len(response_text) / 4 if response_text else 50

        # GPT-4o pricing
        input_cost = (input_tokens / 1_000_000) * 2.50
        output_cost = (output_tokens / 1_000_000) * 10.00

        return input_cost + output_cost

    def parallel(
        self,
        requests: list[dict],
        fail_fast: bool = False,
    ) -> list:
        """
        Execute multiple requests in parallel.

        Args:
            requests: List of request dicts, each with 'prompt' and 'task_type' keys.
            fail_fast: If True, cancel remaining requests on first failure.

        Returns:
            List of ConductorResponse objects in the same order as requests.

        Example:
            ```python
            results = session.parallel([
                {"prompt": "Analyze doc 1", "task_type": TaskType.CLASSIFY},
                {"prompt": "Analyze doc 2", "task_type": TaskType.CLASSIFY},
                {"prompt": "Analyze doc 3", "task_type": TaskType.CLASSIFY},
            ])
            ```
        """
        if not self._is_active:
            raise RuntimeError("Session is not active. Use 'with session:' context manager.")

        results = [None] * len(requests)
        errors = []

        with ThreadPoolExecutor(max_workers=min(self.max_parallel, len(requests))) as executor:
            # Submit all tasks
            future_to_index = {}
            for i, req in enumerate(requests):
                future = executor.submit(
                    self.complete,
                    prompt=req['prompt'],
                    task_type=req['task_type'],
                    **{k: v for k, v in req.items() if k not in ('prompt', 'task_type')}
                )
                future_to_index[future] = i

            # Collect results
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    results[index] = future.result()
                except Exception as e:
                    errors.append((index, e))
                    if fail_fast:
                        executor.shutdown(wait=False, cancel_futures=True)
                        raise

        if errors and not fail_fast:
            # Return partial results, errors are None
            for index, error in errors:
                results[index] = None

        return results

    def map(
        self,
        items: list,
        prompt_template: str,
        task_type,
        **kwargs,
    ) -> list:
        """
        Map a prompt template over a list of items.

        Args:
            items: List of items to process.
            prompt_template: Template string with {item} placeholder.
            task_type: The type of task.
            **kwargs: Additional arguments passed to each complete() call.

        Returns:
            List of ConductorResponse objects.

        Example:
            ```python
            summaries = session.map(
                items=documents,
                prompt_template="Summarize this document: {item}",
                task_type=TaskType.SUMMARIZE,
            )
            ```
        """
        requests = [
            {"prompt": prompt_template.format(item=item), "task_type": task_type, **kwargs}
            for item in items
        ]
        return self.parallel(requests)

    def reduce(
        self,
        items: list[str],
        reduce_prompt: str,
        task_type,
        batch_size: int = 5,
        **kwargs,
    ):
        """
        Reduce multiple items into a single result.

        Processes items in batches, then combines batch results.

        Args:
            items: List of text items to reduce.
            reduce_prompt: Template with {items} placeholder.
            task_type: The type of task.
            batch_size: Number of items to process per batch.
            **kwargs: Additional arguments.

        Returns:
            ConductorResponse with the final reduced result.

        Example:
            ```python
            result = session.reduce(
                items=chapter_summaries,
                reduce_prompt="Combine these summaries into one: {items}",
                task_type=TaskType.SUMMARIZE,
            )
            ```
        """
        if len(items) <= batch_size:
            # Single batch, just combine
            combined = "\n\n".join(f"- {item}" for item in items)
            return self.complete(
                prompt=reduce_prompt.format(items=combined),
                task_type=task_type,
                **kwargs,
            )

        # Process in batches
        batches = [items[i:i+batch_size] for i in range(0, len(items), batch_size)]
        batch_results = []

        for batch in batches:
            combined = "\n\n".join(f"- {item}" for item in batch)
            result = self.complete(
                prompt=reduce_prompt.format(items=combined),
                task_type=task_type,
                **kwargs,
            )
            batch_results.append(result.text)

        # Recursively reduce if still too many
        if len(batch_results) > batch_size:
            return self.reduce(batch_results, reduce_prompt, task_type, batch_size, **kwargs)

        # Final reduction
        combined = "\n\n".join(f"- {item}" for item in batch_results)
        return self.complete(
            prompt=reduce_prompt.format(items=combined),
            task_type=task_type,
            **kwargs,
        )

    def _apply_budget_strategy(self, kwargs: dict) -> dict:
        """Apply budget-aware adjustments to the request."""
        if self.budget_usd is None:
            return kwargs

        remaining = self.budget_remaining

        if remaining <= 0:
            if self.budget_strategy == "strict":
                raise BudgetExceededError(
                    f"Session budget exhausted. Used ${self.total_cost:.4f} of ${self.budget_usd:.4f}"
                )
            elif self.budget_strategy == "warn":
                import warnings
                warnings.warn(f"Session budget exceeded: ${self.total_cost:.4f} of ${self.budget_usd:.4f}")

        if self.budget_strategy == "adaptive":
            # Adjust quality requirements based on remaining budget
            budget_pct = remaining / self.budget_usd if self.budget_usd > 0 else 0

            if budget_pct < 0.2:
                # Less than 20% budget remaining - prefer cheaper models
                # Lower the quality requirement slightly to allow cheaper routing
                current_quality = kwargs.get('min_quality', 0.8)
                kwargs['min_quality'] = max(0.7, current_quality - 0.1)

                # Also set a cost ceiling based on remaining budget
                if 'max_cost_usd' not in kwargs:
                    kwargs['max_cost_usd'] = remaining * 0.5  # Don't spend more than half remaining

            elif budget_pct < 0.5:
                # Less than 50% - be more conservative
                if 'max_cost_usd' not in kwargs:
                    kwargs['max_cost_usd'] = remaining * 0.3

        return kwargs

    @property
    def steps(self) -> list[StepResult]:
        """Get all steps in this session."""
        return self._steps.copy()

    @property
    def total_cost(self) -> float:
        """Total cost of all calls in this session."""
        return sum(step.cost_usd for step in self._steps)

    @property
    def total_latency_ms(self) -> int:
        """Total latency of all calls (sequential sum)."""
        return sum(step.latency_ms for step in self._steps)

    @property
    def budget_remaining(self) -> Optional[float]:
        """Remaining budget, or None if no budget set."""
        if self.budget_usd is None:
            return None
        return max(0, self.budget_usd - self.total_cost)

    @property
    def budget_utilization(self) -> Optional[float]:
        """Budget utilization as percentage (0-100), or None if no budget."""
        if self.budget_usd is None or self.budget_usd == 0:
            return None
        return min(100, (self.total_cost / self.budget_usd) * 100)

    @property
    def step_count(self) -> int:
        """Number of steps completed."""
        return len(self._steps)

    @property
    def baseline_cost(self) -> float:
        """What this session would have cost using GPT-4o for everything."""
        return sum(step.baseline_cost_usd for step in self._steps)

    @property
    def total_savings(self) -> float:
        """Total amount saved compared to using GPT-4o for everything."""
        return sum(step.savings_usd for step in self._steps)

    @property
    def savings_pct(self) -> float:
        """Percentage saved compared to GPT-4o baseline."""
        if self.baseline_cost == 0:
            return 0.0
        return (self.total_savings / self.baseline_cost) * 100

    def get_savings_report(self) -> str:
        """
        Get a human-readable savings report.

        Example:
            ```
            === Conductor Savings Report ===
            Session: process-invoice
            Steps: 3

            Cost Breakdown:
              Actual cost:    $0.0023
              Baseline cost:  $0.0089 (GPT-4o for all)
              You saved:      $0.0066 (74%)

            Models Used:
              gpt-4o-mini: 2 calls
              gpt-4o: 1 call
            ```
        """
        lines = [
            "=" * 35,
            "   Conductor Savings Report",
            "=" * 35,
            f"Session: {self.session_id}",
            f"Steps: {self.step_count}",
            "",
            "Cost Breakdown:",
            f"  Actual cost:    ${self.total_cost:.4f}",
            f"  Baseline cost:  ${self.baseline_cost:.4f} (GPT-4o for all)",
            f"  You saved:      ${self.total_savings:.4f} ({self.savings_pct:.0f}%)",
            "",
            "Models Used:",
        ]

        models_used: dict[str, int] = {}
        for step in self._steps:
            models_used[step.model_used] = models_used.get(step.model_used, 0) + 1

        for model, count in sorted(models_used.items()):
            lines.append(f"  {model}: {count} call{'s' if count > 1 else ''}")

        lines.append("=" * 35)
        return "\n".join(lines)

    def get_stats(self) -> SessionStats:
        """Get aggregated statistics for this session."""
        models_used: dict[str, int] = {}
        for step in self._steps:
            models_used[step.model_used] = models_used.get(step.model_used, 0) + 1

        return SessionStats(
            session_id=self.session_id,
            total_steps=len(self._steps),
            total_cost_usd=self.total_cost,
            total_latency_ms=self.total_latency_ms,
            avg_cost_per_step=self.total_cost / len(self._steps) if self._steps else 0,
            avg_latency_per_step=self.total_latency_ms / len(self._steps) if self._steps else 0,
            models_used=models_used,
            budget_usd=self.budget_usd,
            budget_remaining_usd=self.budget_remaining,
            budget_utilization_pct=self.budget_utilization,
            # Savings tracking
            baseline_cost_usd=self.baseline_cost,
            total_savings_usd=self.total_savings,
            savings_pct=self.savings_pct,
        )

    def __repr__(self) -> str:
        status = "active" if self._is_active else "inactive"
        budget_info = f", budget=${self.budget_usd:.2f}" if self.budget_usd else ""
        savings_info = f", saved={self.savings_pct:.0f}%" if self._steps else ""
        return f"Session(id='{self.session_id}', steps={len(self._steps)}, cost=${self.total_cost:.4f}{savings_info}{budget_info}, {status})"


class BudgetExceededError(Exception):
    """Raised when a session's budget has been exceeded (in strict mode)."""
    pass


class WorkflowBuilder:
    """
    Builder for defining reusable workflows.

    Example:
        ```python
        # Define a reusable workflow
        summarize_and_extract = (
            conductor.workflow_builder("doc-processor")
            .add_step("summarize", TaskType.SUMMARIZE, "Summarize: {input}")
            .add_step("extract", TaskType.EXTRACT_JSON, "Extract entities from: {summarize}")
            .add_step("format", TaskType.REWRITE, "Format as report: {extract}")
            .build()
        )

        # Run it multiple times
        result1 = summarize_and_extract.run(input=doc1)
        result2 = summarize_and_extract.run(input=doc2)
        ```
    """

    def __init__(self, conductor, workflow_id: str):
        self._conductor = conductor
        self.workflow_id = workflow_id
        self._steps: list[dict] = []

    def add_step(
        self,
        name: str,
        task_type,
        prompt_template: str,
        **kwargs,
    ) -> "WorkflowBuilder":
        """
        Add a step to the workflow.

        Args:
            name: Unique name for this step (used in templates).
            task_type: The type of task.
            prompt_template: Template with placeholders like {input}, {step_name}.
            **kwargs: Additional arguments for the step.

        Returns:
            Self for chaining.
        """
        self._steps.append({
            "name": name,
            "task_type": task_type,
            "prompt_template": prompt_template,
            **kwargs,
        })
        return self

    def build(self) -> "Workflow":
        """Build the workflow."""
        return Workflow(self._conductor, self.workflow_id, self._steps)


class Workflow:
    """
    A reusable workflow definition.

    Created via WorkflowBuilder.
    """

    def __init__(self, conductor, workflow_id: str, steps: list[dict]):
        self._conductor = conductor
        self.workflow_id = workflow_id
        self._steps = steps

    def run(
        self,
        budget_usd: Optional[float] = None,
        **inputs,
    ) -> dict:
        """
        Run the workflow with given inputs.

        Args:
            budget_usd: Optional budget for this run.
            **inputs: Input values for the workflow (e.g., input=document).

        Returns:
            Dict with 'result' (final output), 'steps' (all step results), 'stats' (session stats).
        """
        results = dict(inputs)  # Start with inputs

        with self._conductor.session(
            f"{self.workflow_id}-{uuid.uuid4().hex[:6]}",
            budget_usd=budget_usd,
        ) as session:
            for step in self._steps:
                # Format prompt with available values
                prompt = step["prompt_template"].format(**results)

                # Execute step
                response = session.complete(
                    prompt=prompt,
                    task_type=step["task_type"],
                    **{k: v for k, v in step.items() if k not in ("name", "task_type", "prompt_template")}
                )

                # Store result for next steps
                results[step["name"]] = response.text

            return {
                "result": response.text,  # Final step output
                "steps": {step["name"]: results[step["name"]] for step in self._steps},
                "stats": session.get_stats(),
                "session": session,
            }
