"""
Conductor: The main LLM Control Plane.

This is the primary entry point for using Conductor.
"""

from dataclasses import dataclass
from datetime import datetime, UTC
from pathlib import Path
from typing import Optional

from conductor.schemas import (
    Request,
    RequestFeatures,
    Decision,
    Outcome,
    LogEntry,
    TaskType,
    LatencyPercentile,
)
from conductor.classifier import Classifier
from conductor.predictor import Predictor
from conductor.router import Router
from conductor.executor import Executor, LLMProvider, MockProvider, OpenAIProvider, AnthropicProvider
from conductor.logger import Logger, LogStore, InMemoryLogStore, JSONLLogStore
from conductor.coefficients import Coefficients, load_coefficients
from conductor.strategies import Strategy, get_all_strategies, STRATEGIES
from conductor.validation import validate_request, ValidationError
from conductor.smart_classifier import SmartClassifier, SmartClassification
from conductor.session import Session, WorkflowBuilder, BudgetExceededError as SessionBudgetExceededError
from conductor.cost_control import (
    CostController,
    UserBudget,
    CostReport,
    Period,
    BudgetExceededError,
    AnomalyAlert,
)


@dataclass
class ConductorResponse:
    """
    Response from Conductor.
    
    Contains the LLM response plus routing metadata.
    """
    # The actual response
    text: str
    
    # Routing decision
    decision: Decision
    
    # Execution outcome
    outcome: Outcome
    
    # Request features
    features: RequestFeatures
    
    @property
    def model_used(self) -> str:
        """Which model handled this request."""
        return self.decision.model_id
    
    @property
    def cost(self) -> float:
        """Actual cost in USD."""
        return self.outcome.actual_cost_usd
    
    @property
    def latency_ms(self) -> int:
        """Actual latency in milliseconds."""
        return self.outcome.actual_latency_ms
    
    @property
    def why(self) -> str:
        """Explanation for the routing decision."""
        return self.decision.why


class Conductor:
    """
    LLM Control Plane for intelligent model routing.
    
    Automatically routes requests to the optimal model/strategy
    to minimize cost while meeting latency SLAs and quality thresholds.
    
    Example:
        ```python
        from conductor import Conductor, TaskType
        
        # Create conductor
        conductor = Conductor()
        
        # Make a request
        response = conductor.complete(
            prompt="What is 2 + 2?",
            task_type=TaskType.CLASSIFY,
            max_latency_ms=2000,
            min_quality=0.8,
        )
        
        print(response.text)
        print(f"Used: {response.model_used}")
        print(f"Cost: ${response.cost:.4f}")
        print(f"Why: {response.why}")
        ```
    """
    
    def __init__(
        self,
        coefficients: Optional[Coefficients] = None,
        strategies: Optional[dict[str, Strategy]] = None,
        providers: Optional[dict] = None,
        log_store: Optional[LogStore] = None,
        policy_version: str = "v1.0.0",
        dry_run: bool = False,
        smart_classification: bool = True,
    ):
        """
        Initialize Conductor.

        Args:
            coefficients: Prediction coefficients. Uses defaults if not provided.
            strategies: Available strategies. Uses defaults if not provided.
            providers: LLM providers keyed by Provider enum.
            log_store: Where to persist logs. Uses in-memory if not provided.
            policy_version: Version string for this policy configuration.
            dry_run: If True, only predict without executing.
            smart_classification: If True (default), use LLM to understand prompt
                complexity. If False, use rule-based heuristics (faster but less accurate).
        """
        self.coefficients = coefficients or Coefficients()
        self.strategies = strategies or get_all_strategies()
        self.policy_version = policy_version
        self.dry_run = dry_run
        self.smart_classification = smart_classification

        # Initialize components
        self.classifier = Classifier(self.coefficients)
        self.predictor = Predictor(self.coefficients)
        self.router = Router(
            coefficients=self.coefficients,
            strategies=self.strategies,
            policy_version=policy_version,
        )
        self.executor = Executor(providers=providers or {})
        self.logger = Logger(log_store or InMemoryLogStore())

        # Smart classifier for semantic complexity understanding
        self.smart_classifier = SmartClassifier() if smart_classification else None

        # Cost controller for budget management and alerts
        self.cost_controller = CostController()
    
    def complete(
        self,
        prompt: str,
        task_type: TaskType | str,
        system_prompt: Optional[str] = None,
        max_latency_ms: int = 5000,
        latency_percentile: LatencyPercentile | str = LatencyPercentile.P95,
        min_quality: float = 0.8,
        max_cost_usd: Optional[float] = None,
        expected_output_tokens: Optional[int] = None,
        allow_batching: bool = True,
        require_tools: bool = False,
        require_verifier: bool = False,
        client_id: str = "default",
        trace_id: Optional[str] = None,
    ) -> ConductorResponse:
        """
        Complete a prompt using the optimal routing strategy.

        Args:
            prompt: The user prompt.
            task_type: Type of task (classify, extract_json, summarize, etc.)
            system_prompt: Optional system prompt.
            max_latency_ms: Maximum acceptable latency in milliseconds.
            latency_percentile: Which percentile the latency SLA applies to.
            min_quality: Minimum acceptable quality score (0.0-1.0).
            max_cost_usd: Optional maximum cost budget.
            expected_output_tokens: Hint for expected output length.
            allow_batching: Whether batching is allowed.
            require_tools: Whether function calling is required.
            require_verifier: Whether verification is required.
            client_id: Identifier for the calling client.
            trace_id: Distributed tracing correlation ID.

        Returns:
            ConductorResponse with text, decision, outcome, and metadata.

        Raises:
            ValidationError: If input parameters are invalid.
            ValueError: If task_type or latency_percentile are invalid enum values.
        """
        # Validate inputs
        validate_request(
            prompt=prompt,
            max_latency_ms=max_latency_ms,
            min_quality=min_quality,
            max_cost_usd=max_cost_usd,
            expected_output_tokens=expected_output_tokens,
        )

        # Check user budget before making request (raises BudgetExceededError if over)
        budget_check = self.cost_controller.check_budget(client_id, estimated_cost=0.01)
        for warning in budget_check.get("warnings", []):
            import warnings
            warnings.warn(f"Budget warning for {client_id}: {warning}")

        # Normalize enums
        if isinstance(task_type, str):
            task_type = TaskType(task_type)
        if isinstance(latency_percentile, str):
            latency_percentile = LatencyPercentile(latency_percentile)

        # Build request
        request = Request(
            prompt=prompt,
            task_type=task_type,
            system_prompt=system_prompt,
            max_latency_ms=max_latency_ms,
            latency_percentile=latency_percentile,
            min_quality_score=min_quality,
            max_cost_usd=max_cost_usd,
            expected_output_tokens=expected_output_tokens,
            allow_batching=allow_batching,
            require_tools=require_tools,
            require_verifier=require_verifier,
            client_id=client_id,
            trace_id=trace_id,
        )

        # Execute and record cost
        response = self.route_and_execute(request)

        # Record cost for tracking and alerts
        self.cost_controller.record_cost(
            user_id=client_id,
            cost_usd=response.cost,
            model=response.model_used,
            task_type=task_type.value if hasattr(task_type, 'value') else str(task_type),
        )

        return response
    
    def route_and_execute(self, request: Request) -> ConductorResponse:
        """
        Route and execute a request.

        This is the main processing pipeline:
        1. Classify request to extract features
        2. Use smart classification for complexity (if enabled)
        3. Route to optimal strategy
        4. Execute (unless dry_run)
        5. Log the outcome

        Args:
            request: The request to process.

        Returns:
            ConductorResponse.
        """
        # Step 1: Classify (basic features)
        features = self.classifier.classify(request)

        # Step 2: Smart classification for complexity (if enabled)
        if self.smart_classifier is not None:
            smart_result = self.smart_classifier.classify(request.prompt)
            # Override rule-based complexity with LLM-understood complexity
            features = RequestFeatures(
                request_id=features.request_id,
                task_type=features.task_type,
                input_token_count=features.input_token_count,
                estimated_output_tokens=features.estimated_output_tokens,
                complexity_score=smart_result.complexity_score,  # Smart score
                urgency_tier=features.urgency_tier,
                has_structured_output=features.has_structured_output,
                max_latency_ms=features.max_latency_ms,
                latency_percentile=features.latency_percentile,
                min_quality_score=features.min_quality_score,
                max_cost_usd=features.max_cost_usd,
                allow_batching=features.allow_batching,
                require_tools=features.require_tools,
                require_verifier=features.require_verifier,
            )

        # Step 3: Route
        decision = self.router.route(features)
        
        # Step 3: Execute
        if self.dry_run:
            # Create mock outcome for dry run
            outcome = self._mock_outcome(decision)
            text = "[DRY RUN - No actual execution]"
        else:
            outcome = self.executor.execute(request, decision)
            text = outcome.response_text or ""
        
        # Step 4: Log
        self.logger.log(request, features, decision, outcome)
        
        return ConductorResponse(
            text=text,
            decision=decision,
            outcome=outcome,
            features=features,
        )
    
    def predict_only(self, request: Request) -> Decision:
        """
        Get routing decision without executing.
        
        Useful for:
        - Understanding what Conductor would do
        - Cost estimation
        - Debugging routing behavior
        
        Args:
            request: The request to analyze.
        
        Returns:
            Decision with predictions and explanation.
        """
        features = self.classifier.classify(request)
        return self.router.route(features)
    
    def get_logs(self) -> list[LogEntry]:
        """Get all logged entries."""
        return self.logger.get_entries()
    
    def export_logs(self, path: Path, format: str = "csv") -> int:
        """
        Export logs to file.
        
        Args:
            path: Output file path.
            format: Output format ("csv" or "jsonl").
        
        Returns:
            Number of entries exported.
        """
        if format == "csv":
            return self.logger.export_csv(path)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _mock_outcome(self, decision: Decision) -> Outcome:
        """Create a mock outcome for dry run mode."""
        from conductor.schemas import FinishReason
        import uuid
        
        return Outcome(
            outcome_id=str(uuid.uuid4()),
            decision_id=decision.decision_id,
            request_id=decision.request_id,
            timestamp_utc=datetime.now(UTC),
            actual_cost_usd=decision.predicted_cost_usd,
            actual_latency_ms=decision.predicted_latency_p50_ms,
            actual_input_tokens=0,
            actual_output_tokens=0,
            finish_reason=FinishReason.STOP,
            http_status=200,
            retry_count=0,
            verifier_passed=None,
            format_valid=True,
            quality_proxy_score=1.0 - decision.predicted_quality_risk,
            sla_met=True,
            cost_error_pct=0.0,
            latency_error_pct=0.0,
        )
    
    @classmethod
    def from_config(
        cls,
        coefficients_path: Optional[Path] = None,
        log_path: Optional[Path] = None,
        **kwargs,
    ) -> "Conductor":
        """
        Create Conductor from configuration files.
        
        Args:
            coefficients_path: Path to coefficients JSON file.
            log_path: Path to JSONL log file.
            **kwargs: Additional arguments passed to constructor.
        
        Returns:
            Configured Conductor instance.
        """
        coefficients = None
        if coefficients_path:
            coefficients = load_coefficients(coefficients_path)
        
        log_store = None
        if log_path:
            log_store = JSONLLogStore(log_path)
        
        return cls(
            coefficients=coefficients,
            log_store=log_store,
            **kwargs,
        )
    
    @classmethod
    def with_openai(
        cls,
        api_key: Optional[str] = None,
        **kwargs,
    ) -> "Conductor":
        """
        Create Conductor configured for OpenAI.
        
        Args:
            api_key: OpenAI API key. Uses OPENAI_API_KEY env var if not provided.
            **kwargs: Additional arguments passed to constructor.
        
        Returns:
            Conductor configured with OpenAI provider.
        """
        from conductor.schemas import Provider
        
        providers = {
            Provider.OPENAI: OpenAIProvider(api_key),
        }
        
        return cls(providers=providers, **kwargs)
    
    @classmethod
    def with_anthropic(
        cls,
        api_key: Optional[str] = None,
        **kwargs,
    ) -> "Conductor":
        """
        Create Conductor configured for Anthropic.
        
        Args:
            api_key: Anthropic API key. Uses ANTHROPIC_API_KEY env var if not provided.
            **kwargs: Additional arguments passed to constructor.
        
        Returns:
            Conductor configured with Anthropic provider.
        """
        from conductor.schemas import Provider
        from conductor.strategies import ANTHROPIC_STRATEGIES
        
        providers = {
            Provider.ANTHROPIC: AnthropicProvider(api_key),
        }
        
        # Include Anthropic strategies
        strategies = {**STRATEGIES, **ANTHROPIC_STRATEGIES}
        
        return cls(providers=providers, strategies=strategies, **kwargs)

    def session(
        self,
        session_id: Optional[str] = None,
        budget_usd: Optional[float] = None,
        budget_strategy: str = "adaptive",
        max_parallel: int = 10,
        metadata: Optional[dict] = None,
    ) -> Session:
        """
        Create a session for grouping related LLM calls.

        Sessions provide:
        - Cost tracking across multiple calls
        - Budget-aware routing (adapts as budget depletes)
        - Parallel execution support
        - Workflow analytics

        Args:
            session_id: Unique identifier for this session. Auto-generated if not provided.
            budget_usd: Optional total budget for all calls in this session.
            budget_strategy: How to handle budget:
                - "adaptive": Adjust routing as budget depletes (default)
                - "strict": Fail if a call would exceed budget
                - "warn": Log warning but allow exceeding
            max_parallel: Maximum concurrent calls for parallel execution.
            metadata: Optional metadata to attach to the session.

        Returns:
            Session context manager.

        Example:
            ```python
            with conductor.session("process-doc", budget_usd=0.05) as session:
                r1 = session.complete(prompt="Summarize...", task_type=TaskType.SUMMARIZE)
                r2 = session.complete(prompt=f"Extract: {r1.text}", task_type=TaskType.EXTRACT_JSON)

            print(session.total_cost)  # Total cost of all calls
            print(session.steps)       # List of all step results
            ```
        """
        return Session(
            conductor=self,
            session_id=session_id,
            budget_usd=budget_usd,
            budget_strategy=budget_strategy,
            max_parallel=max_parallel,
            metadata=metadata,
        )

    def workflow_builder(self, workflow_id: str) -> WorkflowBuilder:
        """
        Create a workflow builder for defining reusable workflows.

        Args:
            workflow_id: Unique identifier for this workflow type.

        Returns:
            WorkflowBuilder for chaining step definitions.

        Example:
            ```python
            # Define a reusable workflow
            doc_processor = (
                conductor.workflow_builder("doc-processor")
                .add_step("summarize", TaskType.SUMMARIZE, "Summarize: {input}")
                .add_step("extract", TaskType.EXTRACT_JSON, "Extract from: {summarize}")
                .add_step("format", TaskType.REWRITE, "Format: {extract}")
                .build()
            )

            # Run it multiple times
            result1 = doc_processor.run(input=doc1)
            result2 = doc_processor.run(input=doc2)

            print(result1["result"])  # Final output
            print(result1["stats"])   # Session statistics
            ```
        """
        return WorkflowBuilder(conductor=self, workflow_id=workflow_id)

    # =========================================================================
    # Cost Control - "Never get surprised by an LLM bill again"
    # =========================================================================

    def set_user_budget(
        self,
        user_id: str,
        hourly_usd: Optional[float] = None,
        daily_usd: Optional[float] = None,
        weekly_usd: Optional[float] = None,
        monthly_usd: Optional[float] = None,
        hard_limit: bool = True,
    ) -> UserBudget:
        """
        Set budget limits for a user.

        Args:
            user_id: Unique identifier for the user.
            hourly_usd: Maximum spend per hour.
            daily_usd: Maximum spend per day.
            weekly_usd: Maximum spend per week.
            monthly_usd: Maximum spend per month.
            hard_limit: If True, block requests over budget. If False, warn only.

        Returns:
            The created UserBudget.

        Example:
            ```python
            conductor.set_user_budget(
                "user_123",
                daily_usd=5.00,
                monthly_usd=50.00
            )

            # Now requests from this user will be blocked if over budget
            response = conductor.complete(
                prompt="...",
                task_type=TaskType.CLASSIFY,
                client_id="user_123"  # This user's budget is checked
            )
            ```
        """
        return self.cost_controller.set_user_budget(
            user_id=user_id,
            hourly_usd=hourly_usd,
            daily_usd=daily_usd,
            weekly_usd=weekly_usd,
            monthly_usd=monthly_usd,
            hard_limit=hard_limit,
        )

    def add_cost_alert(
        self,
        threshold_usd: float,
        period: Period = Period.DAILY,
        callback: Optional[callable] = None,
        cooldown_minutes: int = 60,
    ) -> str:
        """
        Add an alert that triggers when spend exceeds threshold.

        Args:
            threshold_usd: Trigger when total spend exceeds this amount.
            period: Time period (HOURLY, DAILY, WEEKLY, MONTHLY).
            callback: Function to call when triggered. Receives dict with info.
            cooldown_minutes: Don't re-trigger within this period.

        Returns:
            Alert ID for later management.

        Example:
            ```python
            def send_slack_alert(info):
                requests.post(SLACK_WEBHOOK, json={
                    "text": f"LLM Spend Alert: ${info['total']:.2f} ({info['period']})"
                })

            conductor.add_cost_alert(
                threshold_usd=100.00,
                period=Period.DAILY,
                callback=send_slack_alert
            )
            ```
        """
        return self.cost_controller.add_alert(
            threshold_usd=threshold_usd,
            period=period,
            callback=callback,
            cooldown_minutes=cooldown_minutes,
        )

    def get_cost_report(
        self,
        group_by: str = "user",
        period: Period = Period.DAILY,
        top_n: int = 10,
    ) -> CostReport:
        """
        Generate a cost attribution report.

        Args:
            group_by: "user", "model", "task_type", or "feature"
            period: Time period (HOURLY, DAILY, WEEKLY, MONTHLY)
            top_n: Number of top spenders to include

        Returns:
            CostReport with breakdown and statistics.

        Example:
            ```python
            report = conductor.get_cost_report(group_by="user", period=Period.DAILY)

            print(f"Total spend: ${report.total_cost_usd:.2f}")
            print(f"Requests: {report.total_requests}")
            print("Top spenders:")
            for name, cost in report.top_spenders:
                print(f"  {name}: ${cost:.4f}")
            ```
        """
        return self.cost_controller.get_cost_report(
            group_by=group_by,
            period=period,
            top_n=top_n,
        )

    def get_user_cost_report(self, user_id: str) -> dict:
        """
        Get detailed cost report for a specific user.

        Args:
            user_id: The user to get report for.

        Returns:
            Dict with user's spending details.

        Example:
            ```python
            report = conductor.get_user_cost_report("user_123")

            print(f"Total cost: ${report['total_cost_usd']:.2f}")
            print(f"Today's spend: ${report['spend_by_period']['daily']:.4f}")
            print(f"Budget remaining: ${report['budget_status']['daily']['limit'] - report['budget_status']['daily']['spent']:.2f}")
            ```
        """
        return self.cost_controller.get_user_report(user_id)

    def enable_anomaly_detection(
        self,
        sensitivity: str = "medium",
        callback: Optional[callable] = None,
    ):
        """
        Enable anomaly detection for unusual spending patterns.

        Args:
            sensitivity: "low", "medium", or "high"
            callback: Function called when anomaly detected.

        Example:
            ```python
            def alert_on_anomaly(alert):
                print(f"ANOMALY: {alert.message}")
                send_pagerduty(alert.message)

            conductor.enable_anomaly_detection(
                sensitivity="medium",
                callback=alert_on_anomaly
            )
            # Now you'll be alerted if any user's spending spikes unexpectedly
            ```
        """
        self.cost_controller.enable_anomaly_detection(
            sensitivity=sensitivity,
            callback=callback,
        )

    def get_cost_summary(self) -> dict:
        """
        Get overall cost control summary.

        Returns:
            Dict with spend totals and status.
        """
        return self.cost_controller.get_summary()
