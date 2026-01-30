"""
Evaluator for Conductor.

Compares different routing policies using offline replay.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, UTC
from typing import Optional, Callable
import statistics

from conductor.schemas import (
    Request,
    RequestFeatures,
    Decision,
    LogEntry,
    TaskType,
)
from conductor.classifier import Classifier
from conductor.router import Router
from conductor.coefficients import Coefficients, load_coefficients
from conductor.strategies import Strategy, get_all_strategies


@dataclass
class PolicyMetrics:
    """Metrics for a single policy."""
    policy_name: str
    
    # Request counts
    total_requests: int = 0
    
    # Cost metrics
    total_predicted_cost: float = 0.0
    avg_predicted_cost: float = 0.0
    
    # Latency metrics
    sla_pass_rate_p95: float = 0.0
    sla_pass_rate_p99: float = 0.0
    avg_predicted_latency_p50: float = 0.0
    avg_predicted_latency_p95: float = 0.0
    
    # Quality metrics
    avg_predicted_quality_risk: float = 0.0
    fallback_rate: float = 0.0
    
    # Strategy distribution
    strategy_distribution: dict[str, int] = field(default_factory=dict)
    
    # Top routing reasons
    top_reasons: list[tuple[str, int]] = field(default_factory=list)


@dataclass
class ComparisonResult:
    """Result of comparing two policies."""
    policy_a: PolicyMetrics
    policy_b: PolicyMetrics
    
    # Relative differences
    cost_diff_pct: float = 0.0  # (B - A) / A * 100, negative = B is cheaper
    latency_diff_pct: float = 0.0
    quality_risk_diff_pct: float = 0.0
    
    # Samples where policies differed
    different_decisions: int = 0
    same_decisions: int = 0
    
    # Per-task-type breakdown
    by_task_type: dict[str, dict] = field(default_factory=dict)


class Policy:
    """
    A routing policy that can be evaluated.
    
    Wraps a Router with specific configuration.
    """
    
    def __init__(
        self,
        name: str,
        coefficients: Optional[Coefficients] = None,
        strategies: Optional[dict[str, Strategy]] = None,
        policy_version: str = "v1.0.0",
    ):
        self.name = name
        self.coefficients = coefficients or Coefficients()
        self.strategies = strategies or get_all_strategies()
        self.router = Router(
            coefficients=self.coefficients,
            strategies=self.strategies,
            policy_version=policy_version,
        )
        self.classifier = Classifier(self.coefficients)
    
    def decide(self, request: Request) -> tuple[RequestFeatures, Decision]:
        """Make a routing decision for a request."""
        features = self.classifier.classify(request)
        decision = self.router.route(features)
        return features, decision


class Evaluator:
    """
    Evaluates and compares routing policies.
    
    Supports:
    1. Single policy evaluation on a request set
    2. A/B comparison of two policies
    3. Replay of logged requests with different policies
    """
    
    def __init__(self):
        pass
    
    def evaluate_policy(
        self,
        policy: Policy,
        requests: list[Request],
    ) -> PolicyMetrics:
        """
        Evaluate a policy on a set of requests.
        
        Args:
            policy: The policy to evaluate.
            requests: Requests to route.
        
        Returns:
            Aggregated metrics for the policy.
        """
        metrics = PolicyMetrics(policy_name=policy.name)
        metrics.total_requests = len(requests)
        
        total_cost = 0.0
        latencies_p50 = []
        latencies_p95 = []
        latencies_p99 = []
        quality_risks = []
        fallback_count = 0
        strategy_counts = defaultdict(int)
        reason_counts = defaultdict(int)
        
        sla_pass_p95 = 0
        sla_pass_p99 = 0
        
        for request in requests:
            features, decision = policy.decide(request)
            
            total_cost += decision.predicted_cost_usd
            latencies_p50.append(decision.predicted_latency_p50_ms)
            latencies_p95.append(decision.predicted_latency_p95_ms)
            latencies_p99.append(decision.predicted_latency_p99_ms)
            quality_risks.append(decision.predicted_quality_risk)
            
            if decision.fallback_used:
                fallback_count += 1
            
            strategy_counts[decision.strategy_id] += 1
            
            # Extract reason category
            reason = self._categorize_reason(decision.why)
            reason_counts[reason] += 1
            
            # Check SLA pass (using predicted latency)
            if decision.predicted_latency_p95_ms <= request.max_latency_ms:
                sla_pass_p95 += 1
            if decision.predicted_latency_p99_ms <= request.max_latency_ms:
                sla_pass_p99 += 1
        
        # Compute aggregates
        metrics.total_predicted_cost = total_cost
        metrics.avg_predicted_cost = total_cost / len(requests) if requests else 0
        metrics.avg_predicted_latency_p50 = statistics.mean(latencies_p50) if latencies_p50 else 0
        metrics.avg_predicted_latency_p95 = statistics.mean(latencies_p95) if latencies_p95 else 0
        metrics.avg_predicted_quality_risk = statistics.mean(quality_risks) if quality_risks else 0
        metrics.fallback_rate = fallback_count / len(requests) if requests else 0
        metrics.sla_pass_rate_p95 = sla_pass_p95 / len(requests) if requests else 0
        metrics.sla_pass_rate_p99 = sla_pass_p99 / len(requests) if requests else 0
        
        metrics.strategy_distribution = dict(strategy_counts)
        metrics.top_reasons = sorted(
            reason_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        return metrics
    
    def compare_policies(
        self,
        policy_a: Policy,
        policy_b: Policy,
        requests: list[Request],
    ) -> ComparisonResult:
        """
        Compare two policies on the same request set.
        
        Args:
            policy_a: First policy (baseline).
            policy_b: Second policy (candidate).
            requests: Requests to evaluate.
        
        Returns:
            Comparison result with relative metrics.
        """
        metrics_a = self.evaluate_policy(policy_a, requests)
        metrics_b = self.evaluate_policy(policy_b, requests)
        
        result = ComparisonResult(
            policy_a=metrics_a,
            policy_b=metrics_b,
        )
        
        # Compute relative differences
        if metrics_a.total_predicted_cost > 0:
            result.cost_diff_pct = (
                (metrics_b.total_predicted_cost - metrics_a.total_predicted_cost) /
                metrics_a.total_predicted_cost * 100
            )
        
        if metrics_a.avg_predicted_latency_p95 > 0:
            result.latency_diff_pct = (
                (metrics_b.avg_predicted_latency_p95 - metrics_a.avg_predicted_latency_p95) /
                metrics_a.avg_predicted_latency_p95 * 100
            )
        
        if metrics_a.avg_predicted_quality_risk > 0:
            result.quality_risk_diff_pct = (
                (metrics_b.avg_predicted_quality_risk - metrics_a.avg_predicted_quality_risk) /
                metrics_a.avg_predicted_quality_risk * 100
            )
        
        # Count different decisions
        same = 0
        different = 0
        
        for request in requests:
            _, dec_a = policy_a.decide(request)
            _, dec_b = policy_b.decide(request)
            
            if dec_a.strategy_id == dec_b.strategy_id:
                same += 1
            else:
                different += 1
        
        result.same_decisions = same
        result.different_decisions = different
        
        # Breakdown by task type
        by_task: dict[str, dict] = {}
        requests_by_task = defaultdict(list)
        
        for request in requests:
            requests_by_task[request.task_type.value].append(request)
        
        for task_type, task_requests in requests_by_task.items():
            task_metrics_a = self.evaluate_policy(policy_a, task_requests)
            task_metrics_b = self.evaluate_policy(policy_b, task_requests)
            
            cost_diff = 0
            if task_metrics_a.total_predicted_cost > 0:
                cost_diff = (
                    (task_metrics_b.total_predicted_cost - task_metrics_a.total_predicted_cost) /
                    task_metrics_a.total_predicted_cost * 100
                )
            
            by_task[task_type] = {
                "count": len(task_requests),
                "cost_a": task_metrics_a.total_predicted_cost,
                "cost_b": task_metrics_b.total_predicted_cost,
                "cost_diff_pct": cost_diff,
                "sla_pass_rate_a": task_metrics_a.sla_pass_rate_p95,
                "sla_pass_rate_b": task_metrics_b.sla_pass_rate_p95,
            }
        
        result.by_task_type = by_task
        
        return result
    
    def replay_logs(
        self,
        entries: list[LogEntry],
        policy: Policy,
    ) -> tuple[PolicyMetrics, dict]:
        """
        Replay logged requests with a new policy.
        
        Args:
            entries: Historical log entries.
            policy: Policy to evaluate.
        
        Returns:
            Tuple of (metrics, comparison_to_original)
        """
        # Extract requests from logs
        requests = [entry.request for entry in entries]
        
        # Evaluate new policy
        new_metrics = self.evaluate_policy(policy, requests)
        
        # Compare to original decisions
        original_cost = sum(e.decision.predicted_cost_usd for e in entries)
        original_fallbacks = sum(1 for e in entries if e.decision.fallback_used)
        
        comparison = {
            "original_cost": original_cost,
            "new_cost": new_metrics.total_predicted_cost,
            "cost_savings_pct": (
                (original_cost - new_metrics.total_predicted_cost) /
                original_cost * 100
                if original_cost > 0 else 0
            ),
            "original_fallback_rate": original_fallbacks / len(entries) if entries else 0,
            "new_fallback_rate": new_metrics.fallback_rate,
        }
        
        return new_metrics, comparison
    
    def _categorize_reason(self, why: str) -> str:
        """Categorize a routing reason into a short label."""
        why_lower = why.lower()
        
        if "fallback" in why_lower:
            return "FALLBACK"
        elif "lowest cost" in why_lower:
            return "LOWEST_COST"
        elif "only strategy" in why_lower:
            return "ONLY_OPTION"
        elif "quality" in why_lower:
            return "QUALITY_REQUIREMENT"
        else:
            return "OTHER"
    
    def print_comparison_report(self, result: ComparisonResult) -> str:
        """Generate a human-readable comparison report."""
        lines = []
        lines.append("=" * 80)
        lines.append("POLICY COMPARISON REPORT")
        lines.append(f"Generated: {datetime.now(UTC).isoformat()}")
        lines.append("=" * 80)
        lines.append("")
        
        # Summary
        lines.append("SUMMARY")
        lines.append("-" * 40)
        lines.append(f"Policy A: {result.policy_a.policy_name}")
        lines.append(f"Policy B: {result.policy_b.policy_name}")
        lines.append(f"Total requests: {result.policy_a.total_requests}")
        lines.append(f"Different decisions: {result.different_decisions} ({result.different_decisions/result.policy_a.total_requests*100:.1f}%)")
        lines.append("")
        
        # Cost comparison
        lines.append("COST")
        lines.append("-" * 40)
        lines.append(f"Policy A total: ${result.policy_a.total_predicted_cost:.4f}")
        lines.append(f"Policy B total: ${result.policy_b.total_predicted_cost:.4f}")
        lines.append(f"Difference: {result.cost_diff_pct:+.1f}%")
        if result.cost_diff_pct < 0:
            lines.append(f"  → Policy B saves ${result.policy_a.total_predicted_cost - result.policy_b.total_predicted_cost:.4f}")
        lines.append("")
        
        # Latency comparison
        lines.append("LATENCY (P95)")
        lines.append("-" * 40)
        lines.append(f"Policy A avg: {result.policy_a.avg_predicted_latency_p95:.0f}ms")
        lines.append(f"Policy B avg: {result.policy_b.avg_predicted_latency_p95:.0f}ms")
        lines.append(f"Difference: {result.latency_diff_pct:+.1f}%")
        lines.append("")
        
        # SLA pass rates
        lines.append("SLA PASS RATE")
        lines.append("-" * 40)
        lines.append(f"Policy A P95: {result.policy_a.sla_pass_rate_p95:.1%}")
        lines.append(f"Policy B P95: {result.policy_b.sla_pass_rate_p95:.1%}")
        lines.append(f"Policy A P99: {result.policy_a.sla_pass_rate_p99:.1%}")
        lines.append(f"Policy B P99: {result.policy_b.sla_pass_rate_p99:.1%}")
        lines.append("")
        
        # Quality
        lines.append("QUALITY")
        lines.append("-" * 40)
        lines.append(f"Policy A avg risk: {result.policy_a.avg_predicted_quality_risk:.2%}")
        lines.append(f"Policy B avg risk: {result.policy_b.avg_predicted_quality_risk:.2%}")
        lines.append(f"Policy A fallback rate: {result.policy_a.fallback_rate:.1%}")
        lines.append(f"Policy B fallback rate: {result.policy_b.fallback_rate:.1%}")
        lines.append("")
        
        # Strategy distribution
        lines.append("STRATEGY DISTRIBUTION")
        lines.append("-" * 40)
        lines.append("Policy A:")
        for strategy, count in sorted(result.policy_a.strategy_distribution.items(), key=lambda x: -x[1]):
            pct = count / result.policy_a.total_requests * 100
            lines.append(f"  {strategy}: {count} ({pct:.1f}%)")
        lines.append("")
        lines.append("Policy B:")
        for strategy, count in sorted(result.policy_b.strategy_distribution.items(), key=lambda x: -x[1]):
            pct = count / result.policy_b.total_requests * 100
            lines.append(f"  {strategy}: {count} ({pct:.1f}%)")
        lines.append("")
        
        # By task type
        lines.append("BREAKDOWN BY TASK TYPE")
        lines.append("-" * 40)
        for task_type, data in result.by_task_type.items():
            lines.append(f"{task_type}:")
            lines.append(f"  Requests: {data['count']}")
            lines.append(f"  Cost A: ${data['cost_a']:.4f}, Cost B: ${data['cost_b']:.4f} ({data['cost_diff_pct']:+.1f}%)")
            lines.append(f"  SLA pass A: {data['sla_pass_rate_a']:.1%}, B: {data['sla_pass_rate_b']:.1%}")
        lines.append("")
        
        # Top reasons
        lines.append("TOP ROUTING REASONS")
        lines.append("-" * 40)
        lines.append("Policy A:")
        for reason, count in result.policy_a.top_reasons[:5]:
            lines.append(f"  {reason}: {count}")
        lines.append("")
        lines.append("Policy B:")
        for reason, count in result.policy_b.top_reasons[:5]:
            lines.append(f"  {reason}: {count}")
        
        lines.append("")
        lines.append("=" * 80)
        
        return "\n".join(lines)
