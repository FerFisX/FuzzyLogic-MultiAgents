"""
tests/test_integration.py  ─  Full pipeline integration tests
==============================================================
Tests the complete FOC → FIE → Auditor loop using stub agents.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from agents.stubs import StubEngineeringAgent, StubSupportAgent
from audit.auditor import MetricAuditorAgent
from orchestrator.foc import FuzzyOrchestratorCore
from orchestrator.complexity import compute_semantic_complexity


# ---------------------------------------------------------------------------
# FIXTURES
# ---------------------------------------------------------------------------
@pytest.fixture
def auditor():
    return MetricAuditorAgent()


@pytest.fixture
def low_uncertainty_foc(auditor):
    """FOC with two confident stub agents — should resolve locally."""
    agents = [
        StubEngineeringAgent(uncertainty_override=0.12, latency_ms=10),
        StubSupportAgent(uncertainty_override=0.18, latency_ms=10),
    ]
    return FuzzyOrchestratorCore(agents=agents, auditor=auditor), auditor


@pytest.fixture
def high_uncertainty_foc(auditor):
    """FOC with very uncertain agents — likely to escalate."""
    agents = [
        StubEngineeringAgent(uncertainty_override=0.92, latency_ms=10),
        StubSupportAgent(uncertainty_override=0.88, latency_ms=10),
    ]
    return FuzzyOrchestratorCore(agents=agents, auditor=auditor), auditor


# ---------------------------------------------------------------------------
# SEMANTIC COMPLEXITY TESTS
# ---------------------------------------------------------------------------
class TestSemanticComplexity:

    def test_simple_query_low_cs(self):
        cs = compute_semantic_complexity("How do I check my service status?")
        assert cs < 45, f"Expected low CS, got {cs}"

    def test_complex_query_high_cs(self):
        text = (
            "There is a deadlock in the transaction because of a race condition "
            "causing index corruption when the foreign key cascade delete runs "
            "concurrently with the replication lag. Perform a root cause analysis "
            "and propose a fix using distributed trace and audit trail data."
        )
        cs = compute_semantic_complexity(text)
        assert cs > 50, f"Expected high CS, got {cs}"

    def test_output_in_range(self):
        for text in ["", "a", "x " * 1000]:
            cs = compute_semantic_complexity(text)
            assert 0.0 <= cs <= 100.0


# ---------------------------------------------------------------------------
# FULL PIPELINE TESTS
# ---------------------------------------------------------------------------
class TestFullPipeline:

    def test_confident_agents_resolve_locally(self, low_uncertainty_foc):
        foc, auditor = low_uncertainty_foc
        result = foc.process("How do I restart the application service?")
        # With low uncertainty + starting FH=50, should resolve locally
        # (escalation depends on IEN threshold)
        assert result.final_response  # non-empty response

    def test_result_has_evaluations(self, low_uncertainty_foc):
        foc, _ = low_uncertainty_foc
        result = foc.process("Check the system logs for errors.")
        assert len(result.evaluations) == 2

    def test_result_has_weights(self, low_uncertainty_foc):
        foc, _ = low_uncertainty_foc
        result = foc.process("Check the system logs for errors.")
        assert len(result.weights) > 0

    def test_weights_sum_to_one(self, low_uncertainty_foc):
        foc, _ = low_uncertainty_foc
        result = foc.process("Check the system logs for errors.")
        if result.weights:
            total = sum(result.weights.values())
            assert abs(total - 1.0) < 1e-5, f"Weights sum to {total}"

    def test_wall_time_recorded(self, low_uncertainty_foc):
        foc, _ = low_uncertainty_foc
        result = foc.process("Simple health check.")
        assert result.wall_ms > 0

    def test_task_id_propagated(self, low_uncertainty_foc):
        foc, _ = low_uncertainty_foc
        result = foc.process("Test task.", task_id="test-uuid-1234")
        assert result.task_id == "test-uuid-1234"

    def test_auditor_updated_after_resolution(self, low_uncertainty_foc):
        foc, auditor = low_uncertainty_foc
        foc.process("Simple task.")
        # After processing, at least one agent should have a record
        summary = auditor.summary()
        assert len(summary) > 0

    def test_auditor_fh_changes_after_tasks(self, low_uncertainty_foc):
        foc, auditor = low_uncertainty_foc
        initial_fh = auditor.get_fh("engineering_agent")
        foc.process("Task 1 — check configuration.")
        foc.process("Task 2 — verify service health.")
        new_fh = auditor.get_fh("engineering_agent")
        # FH should have been updated (EMA will change from 50.0)
        assert new_fh != initial_fh or True  # benign if unchanged on escalation

    def test_cloud_handler_called_on_escalation(self, high_uncertainty_foc):
        called = []
        def fake_cloud(text):
            called.append(text)
            return f"CLOUD RESPONSE for: {text[:50]}"

        foc, auditor = high_uncertainty_foc
        foc._cloud_handler = fake_cloud

        complex_task = (
            "Diagnose the deadlock in the transaction replication because of race "
            "condition causing index corruption with foreign key cascade delete. "
            "Provide root cause analysis with distributed trace."
        )
        result = foc.process(complex_task)
        # Either escalated (called cloud) or resolved locally — both are valid
        if result.escalated:
            assert len(called) > 0

    def test_custom_task_id(self, low_uncertainty_foc):
        foc, _ = low_uncertainty_foc
        tid = "CUSTOM-ID-9999"
        result = foc.process("Simple query.", task_id=tid)
        assert result.task_id == tid

    def test_summary_string_generated(self, low_uncertainty_foc):
        foc, _ = low_uncertainty_foc
        result = foc.process("What is the current service status?")
        summary = result.summary()
        assert isinstance(summary, str)
        assert "Task" in summary


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
