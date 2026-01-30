"""
Command-line interface for Conductor.

Provides commands for:
- Running experiments
- Comparing policies
- Calibrating coefficients
- Generating reports
"""

import argparse
import json
import sys
from datetime import datetime, UTC
from pathlib import Path
from typing import Optional

from conductor import Conductor, TaskType
from conductor.coefficients import Coefficients, load_coefficients
from conductor.evaluator import Evaluator, Policy
from conductor.calibrator import Calibrator
from conductor.logger import JSONLLogStore
from conductor.generators import (
    generate_steady_traffic,
    generate_bursty_traffic,
    generate_mixed_prompt_sizes,
    generate_by_task_type,
)


def cmd_predict(args):
    """Run prediction for a single prompt."""
    conductor = Conductor(dry_run=True)
    
    task_type = TaskType(args.task_type)
    
    response = conductor.complete(
        prompt=args.prompt,
        task_type=task_type,
        max_latency_ms=args.max_latency,
        min_quality=args.min_quality,
    )
    
    print("\n" + "=" * 60)
    print("CONDUCTOR PREDICTION")
    print("=" * 60)
    print(f"\nPrompt: {args.prompt[:100]}...")
    print(f"Task Type: {task_type.value}")
    print(f"Max Latency: {args.max_latency}ms")
    print(f"Min Quality: {args.min_quality}")
    print()
    print("-" * 60)
    print("DECISION")
    print("-" * 60)
    print(f"Strategy: {response.decision.strategy_id}")
    print(f"Model: {response.decision.model_id}")
    print(f"Predicted Cost: ${response.decision.predicted_cost_usd:.6f}")
    print(f"Predicted Latency P50: {response.decision.predicted_latency_p50_ms}ms")
    print(f"Predicted Latency P95: {response.decision.predicted_latency_p95_ms}ms")
    print(f"Predicted Quality Risk: {response.decision.predicted_quality_risk:.2%}")
    print()
    print(f"WHY: {response.decision.why}")
    print()
    
    if response.decision.filter_reasons:
        print("-" * 60)
        print("REJECTED STRATEGIES")
        print("-" * 60)
        for strategy_id, reason in response.decision.filter_reasons.items():
            print(f"  {strategy_id}: {reason}")
    
    print("=" * 60)


def cmd_experiment(args):
    """Run a synthetic traffic experiment."""
    print("\n" + "=" * 60)
    print("CONDUCTOR EXPERIMENT")
    print("=" * 60)
    print(f"Traffic Pattern: {args.pattern}")
    print(f"Request Count: {args.count}")
    print()
    
    # Generate traffic based on pattern
    if args.pattern == "steady":
        requests = generate_steady_traffic(count=args.count)
    elif args.pattern == "bursty":
        requests = generate_bursty_traffic(
            base_count=args.count // 2,
            burst_count=args.count // 2,
        )
    elif args.pattern == "mixed":
        requests = generate_mixed_prompt_sizes(count=args.count)
    else:
        print(f"Unknown pattern: {args.pattern}")
        sys.exit(1)
    
    # Create policy
    coefficients = None
    if args.coefficients:
        coefficients = load_coefficients(Path(args.coefficients))
    
    policy = Policy(name=args.policy_name, coefficients=coefficients)
    
    # Run evaluation
    evaluator = Evaluator()
    metrics = evaluator.evaluate_policy(policy, requests)
    
    # Print results
    print("-" * 60)
    print("RESULTS")
    print("-" * 60)
    print(f"Total Requests: {metrics.total_requests}")
    print(f"Total Predicted Cost: ${metrics.total_predicted_cost:.4f}")
    print(f"Avg Predicted Cost: ${metrics.avg_predicted_cost:.6f}")
    print(f"Avg Latency P50: {metrics.avg_predicted_latency_p50:.0f}ms")
    print(f"Avg Latency P95: {metrics.avg_predicted_latency_p95:.0f}ms")
    print(f"Avg Quality Risk: {metrics.avg_predicted_quality_risk:.2%}")
    print(f"SLA Pass Rate (P95): {metrics.sla_pass_rate_p95:.1%}")
    print(f"Fallback Rate: {metrics.fallback_rate:.1%}")
    print()
    
    print("-" * 60)
    print("STRATEGY DISTRIBUTION")
    print("-" * 60)
    for strategy, count in sorted(metrics.strategy_distribution.items(), key=lambda x: -x[1]):
        pct = count / metrics.total_requests * 100
        bar = "█" * int(pct / 2)
        print(f"  {strategy:20} {count:5} ({pct:5.1f}%) {bar}")
    print()
    
    print("-" * 60)
    print("TOP ROUTING REASONS")
    print("-" * 60)
    for reason, count in metrics.top_reasons[:5]:
        print(f"  {reason}: {count}")
    
    print("=" * 60)
    
    # Save results if output path specified
    if args.output:
        output_path = Path(args.output)
        results = {
            "timestamp": datetime.utcnow().isoformat(),
            "pattern": args.pattern,
            "count": args.count,
            "policy_name": args.policy_name,
            "metrics": {
                "total_requests": metrics.total_requests,
                "total_predicted_cost": metrics.total_predicted_cost,
                "avg_predicted_cost": metrics.avg_predicted_cost,
                "avg_predicted_latency_p50": metrics.avg_predicted_latency_p50,
                "avg_predicted_latency_p95": metrics.avg_predicted_latency_p95,
                "avg_predicted_quality_risk": metrics.avg_predicted_quality_risk,
                "sla_pass_rate_p95": metrics.sla_pass_rate_p95,
                "fallback_rate": metrics.fallback_rate,
                "strategy_distribution": metrics.strategy_distribution,
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {output_path}")


def cmd_compare(args):
    """Compare two policies."""
    print("\n" + "=" * 60)
    print("CONDUCTOR POLICY COMPARISON")
    print("=" * 60)
    
    # Load coefficients
    coeffs_a = None
    coeffs_b = None
    
    if args.coefficients_a:
        coeffs_a = load_coefficients(Path(args.coefficients_a))
    if args.coefficients_b:
        coeffs_b = load_coefficients(Path(args.coefficients_b))
    
    # Create policies
    policy_a = Policy(name=args.name_a, coefficients=coeffs_a)
    policy_b = Policy(name=args.name_b, coefficients=coeffs_b)
    
    # Generate traffic
    if args.pattern == "steady":
        requests = generate_steady_traffic(count=args.count)
    elif args.pattern == "bursty":
        requests = generate_bursty_traffic(
            base_count=args.count // 2,
            burst_count=args.count // 2,
        )
    elif args.pattern == "mixed":
        requests = generate_mixed_prompt_sizes(count=args.count)
    else:
        requests = generate_steady_traffic(count=args.count)
    
    # Run comparison
    evaluator = Evaluator()
    result = evaluator.compare_policies(policy_a, policy_b, requests)
    
    # Print report
    report = evaluator.print_comparison_report(result)
    print(report)
    
    # Save if output specified
    if args.output:
        output_path = Path(args.output)
        with open(output_path, 'w') as f:
            f.write(report)
        print(f"\nReport saved to: {output_path}")


def cmd_calibrate(args):
    """Calibrate coefficients from logs."""
    print("\n" + "=" * 60)
    print("CONDUCTOR CALIBRATION")
    print("=" * 60)
    
    log_path = Path(args.log_file)
    if not log_path.exists():
        print(f"Error: Log file not found: {log_path}")
        sys.exit(1)
    
    # Load logs
    log_store = JSONLLogStore(log_path)
    entries = list(log_store.read_all())
    print(f"Loaded {len(entries)} log entries")
    
    # Load or create coefficients
    if args.coefficients:
        coefficients = load_coefficients(Path(args.coefficients))
    else:
        coefficients = Coefficients()
    
    # Run calibration
    calibrator = Calibrator(
        coefficients=coefficients,
        min_entries=args.min_entries,
    )
    
    results = calibrator.calibrate(entries)
    
    # Print report
    report = calibrator.print_report(results)
    print(report)
    
    # Save updated coefficients
    if args.output:
        output_path = Path(args.output)
        calibrator.save_coefficients(output_path)
        print(f"\nUpdated coefficients saved to: {output_path}")


def cmd_generate_test_logs(args):
    """Generate synthetic logs for testing."""
    print("\n" + "=" * 60)
    print("GENERATING TEST LOGS")
    print("=" * 60)
    
    output_path = Path(args.output)
    
    # Create conductor with file logging
    conductor = Conductor(
        log_store=JSONLLogStore(output_path),
        dry_run=True,
    )
    
    # Generate and execute requests
    requests = generate_steady_traffic(count=args.count)
    
    print(f"Generating {len(requests)} log entries...")
    
    for i, request in enumerate(requests):
        conductor.route_and_execute(request)
        if (i + 1) % 50 == 0:
            print(f"  Generated {i + 1}/{len(requests)} entries")
    
    print(f"\nLogs saved to: {output_path}")
    print("=" * 60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Conductor: LLM Control Plane CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Get prediction for a prompt
  conductor predict "What is the capital of France?" --task-type classify
  
  # Run an experiment with 1000 requests
  conductor experiment --pattern steady --count 1000
  
  # Compare two policies
  conductor compare --name-a baseline --name-b candidate --count 500
  
  # Calibrate from logs
  conductor calibrate --log-file logs.jsonl --output coefficients.json
  
  # Generate test logs
  conductor generate-logs --count 200 --output test_logs.jsonl
""",
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Predict routing for a prompt")
    predict_parser.add_argument("prompt", help="The prompt to analyze")
    predict_parser.add_argument("--task-type", "-t", default="chat",
                                choices=["classify", "extract_json", "summarize", 
                                        "rewrite", "generate_long", "chat"],
                                help="Type of task")
    predict_parser.add_argument("--max-latency", "-l", type=int, default=2000,
                                help="Max latency SLA in ms")
    predict_parser.add_argument("--min-quality", "-q", type=float, default=0.8,
                                help="Min quality threshold (0-1)")
    
    # Experiment command
    exp_parser = subparsers.add_parser("experiment", help="Run a traffic experiment")
    exp_parser.add_argument("--pattern", "-p", default="steady",
                           choices=["steady", "bursty", "mixed"],
                           help="Traffic pattern to simulate")
    exp_parser.add_argument("--count", "-n", type=int, default=100,
                           help="Number of requests to generate")
    exp_parser.add_argument("--policy-name", default="default",
                           help="Name for the policy")
    exp_parser.add_argument("--coefficients", "-c",
                           help="Path to coefficients JSON file")
    exp_parser.add_argument("--output", "-o",
                           help="Path to save results JSON")
    
    # Compare command
    cmp_parser = subparsers.add_parser("compare", help="Compare two policies")
    cmp_parser.add_argument("--name-a", default="policy_a", help="Name for policy A")
    cmp_parser.add_argument("--name-b", default="policy_b", help="Name for policy B")
    cmp_parser.add_argument("--coefficients-a", help="Coefficients file for policy A")
    cmp_parser.add_argument("--coefficients-b", help="Coefficients file for policy B")
    cmp_parser.add_argument("--pattern", "-p", default="steady",
                           choices=["steady", "bursty", "mixed"])
    cmp_parser.add_argument("--count", "-n", type=int, default=100)
    cmp_parser.add_argument("--output", "-o", help="Path to save report")
    
    # Calibrate command
    cal_parser = subparsers.add_parser("calibrate", help="Calibrate from logs")
    cal_parser.add_argument("--log-file", "-l", required=True,
                           help="Path to JSONL log file")
    cal_parser.add_argument("--coefficients", "-c",
                           help="Path to existing coefficients to update")
    cal_parser.add_argument("--min-entries", type=int, default=100,
                           help="Minimum entries required per model")
    cal_parser.add_argument("--output", "-o",
                           help="Path to save updated coefficients")
    
    # Generate logs command
    gen_parser = subparsers.add_parser("generate-logs", help="Generate test logs")
    gen_parser.add_argument("--count", "-n", type=int, default=100,
                           help="Number of log entries to generate")
    gen_parser.add_argument("--output", "-o", required=True,
                           help="Path to output JSONL file")
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    # Dispatch to command handler
    commands = {
        "predict": cmd_predict,
        "experiment": cmd_experiment,
        "compare": cmd_compare,
        "calibrate": cmd_calibrate,
        "generate-logs": cmd_generate_test_logs,
    }
    
    handler = commands.get(args.command)
    if handler:
        handler(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
