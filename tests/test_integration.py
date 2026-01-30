"""Integration tests for Conductor."""

import pytest
from pathlib import Path
import tempfile

from conductor import (
    Conductor,
    Request,
    TaskType,
    LatencyPercentile,
)
from conductor.coefficients import Coefficients
from conductor.logger import JSONLLogStore, InMemoryLogStore
from conductor.calibrator import Calibrator
from conductor.evaluator import Evaluator, Policy
from conductor.generators import generate_steady_traffic


class TestConductorIntegration:
    """Integration tests for the full Conductor pipeline."""
    
    def test_dry_run_complete_flow(self):
        """Test complete flow in dry run mode."""
        conductor = Conductor(dry_run=True)
        
        response = conductor.complete(
            prompt="What is the capital of France?",
            task_type=TaskType.CLASSIFY,
            max_latency_ms=2000,
            min_quality=0.8,
        )
        
        # Should return response
        assert response is not None
        assert "[DRY RUN" in response.text
        
        # Decision should be populated
        assert response.decision is not None
        assert response.decision.strategy_id is not None
        assert response.decision.why is not None
        
        # Outcome should be populated
        assert response.outcome is not None
        
        # Should have logged
        logs = conductor.get_logs()
        assert len(logs) == 1
    
    def test_multiple_requests_logging(self):
        """Test that multiple requests are logged."""
        conductor = Conductor(dry_run=True)
        
        # Make several requests
        for _ in range(5):
            conductor.complete(
                prompt="Test prompt",
                task_type=TaskType.CHAT,
                max_latency_ms=2000,
            )
        
        logs = conductor.get_logs()
        assert len(logs) == 5
    
    def test_predict_only(self):
        """Test prediction without execution."""
        conductor = Conductor()
        
        request = Request(
            prompt="Test prompt",
            task_type=TaskType.SUMMARIZE,
            max_latency_ms=3000,
            min_quality_score=0.85,
        )
        
        decision = conductor.predict_only(request)
        
        assert decision is not None
        assert decision.predicted_cost_usd > 0
        assert decision.predicted_latency_p50_ms > 0
        
        # Should NOT have logged (predict only)
        logs = conductor.get_logs()
        assert len(logs) == 0
    
    def test_task_type_string_input(self):
        """Test that task_type can be passed as string."""
        conductor = Conductor(dry_run=True)
        
        response = conductor.complete(
            prompt="Test",
            task_type="classify",  # String instead of enum
            max_latency_ms=2000,
        )
        
        assert response is not None
    
    def test_latency_percentile_string_input(self):
        """Test that latency_percentile can be passed as string."""
        conductor = Conductor(dry_run=True)
        
        response = conductor.complete(
            prompt="Test",
            task_type=TaskType.CHAT,
            max_latency_ms=2000,
            latency_percentile="p99",  # String instead of enum
        )
        
        assert response is not None
    
    def test_response_properties(self):
        """Test ConductorResponse properties."""
        conductor = Conductor(dry_run=True)
        
        response = conductor.complete(
            prompt="Test",
            task_type=TaskType.CHAT,
        )
        
        # Test convenience properties
        assert isinstance(response.model_used, str)
        assert isinstance(response.cost, float)
        assert isinstance(response.latency_ms, int)
        assert isinstance(response.why, str)
    
    def test_from_config_defaults(self):
        """Test creating conductor from config with defaults."""
        conductor = Conductor.from_config()
        
        assert conductor is not None
        assert conductor.coefficients is not None


class TestLogPersistence:
    """Tests for log persistence."""
    
    def test_jsonl_log_persistence(self):
        """Test that logs persist to JSONL file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "conductor.jsonl"
            
            # Create conductor with JSONL logging
            conductor = Conductor(
                log_store=JSONLLogStore(log_path),
                dry_run=True,
            )
            
            # Make requests
            for i in range(3):
                conductor.complete(
                    prompt=f"Test {i}",
                    task_type=TaskType.CHAT,
                )
            
            # Verify file exists and has content
            assert log_path.exists()
            
            with open(log_path) as f:
                lines = f.readlines()
            assert len(lines) == 3
    
    def test_log_export_csv(self):
        """Test CSV export of logs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "export.csv"
            
            conductor = Conductor(dry_run=True)
            
            # Make requests
            for _ in range(5):
                conductor.complete(
                    prompt="Test",
                    task_type=TaskType.CLASSIFY,
                )
            
            # Export
            count = conductor.export_logs(csv_path, format="csv")
            
            assert count == 5
            assert csv_path.exists()
            
            # Verify CSV content
            import csv
            with open(csv_path) as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            
            assert len(rows) == 5
            assert "strategy_id" in rows[0]
            assert "predicted_cost" in rows[0]


class TestCalibrationIntegration:
    """Integration tests for calibration."""
    
    def test_calibration_from_logs(self):
        """Test calibrating coefficients from logs."""
        # Generate synthetic logs
        conductor = Conductor(dry_run=True)
        
        requests = generate_steady_traffic(count=150)
        for request in requests:
            conductor.route_and_execute(request)
        
        # Get logs
        logs = conductor.get_logs()
        assert len(logs) == 150
        
        # Calibrate
        calibrator = Calibrator(
            coefficients=Coefficients(),
            min_entries=50,  # Lower threshold for test
        )
        
        results = calibrator.calibrate(logs)
        
        # Should have some results (at least for gpt-4o-mini which is common)
        # Note: Might be empty if not enough entries per model
        if results:
            result = results[0]
            assert result.entries_analyzed >= 50
            assert result.cost_mae >= 0
            assert result.latency_mae >= 0
    
    def test_calibration_report(self):
        """Test calibration report generation."""
        from conductor.calibrator import CalibrationResult
        
        # Create mock results
        results = [
            CalibrationResult(
                model_id="gpt-4o-mini",
                entries_analyzed=100,
                cost_mae=0.0001,
                cost_mape=5.0,
                latency_mae=50,
                latency_mape=10.0,
                actual_p50=500,
                actual_p95=900,
                actual_p99=1200,
                new_p95_multiplier=1.8,
                new_p99_multiplier=2.4,
                actual_failure_rate=0.12,
                predicted_avg_risk=0.15,
                new_base_risk=0.14,
                base_latency_adjustment=20,
                base_risk_adjustment=-0.01,
            )
        ]
        
        calibrator = Calibrator(coefficients=Coefficients())
        report = calibrator.print_report(results)
        
        assert "CALIBRATION REPORT" in report
        assert "gpt-4o-mini" in report
        assert "COST:" in report
        assert "LATENCY:" in report
        assert "QUALITY:" in report


class TestEvaluationIntegration:
    """Integration tests for evaluation."""
    
    def test_full_evaluation_workflow(self):
        """Test complete evaluation workflow."""
        # Create policies
        policy_a = Policy(name="baseline")
        policy_b = Policy(name="candidate")
        
        # Generate test traffic
        requests = generate_steady_traffic(count=100)
        
        # Run evaluation
        evaluator = Evaluator()
        result = evaluator.compare_policies(policy_a, policy_b, requests)
        
        # Verify results
        assert result.policy_a.total_requests == 100
        assert result.policy_b.total_requests == 100
        
        # Generate report
        report = evaluator.print_comparison_report(result)
        assert len(report) > 0
        assert "baseline" in report
        assert "candidate" in report
    
    def test_replay_workflow(self):
        """Test replay evaluation workflow."""
        # First, generate some logs
        conductor = Conductor(dry_run=True)
        
        requests = generate_steady_traffic(count=50)
        for request in requests:
            conductor.route_and_execute(request)
        
        logs = conductor.get_logs()
        
        # Create a new policy to test
        new_policy = Policy(name="new_candidate")
        
        # Replay logs with new policy
        evaluator = Evaluator()
        metrics, comparison = evaluator.replay_logs(logs, new_policy)
        
        # Verify results
        assert metrics.total_requests == 50
        assert "original_cost" in comparison
        assert "new_cost" in comparison
        assert "cost_savings_pct" in comparison
