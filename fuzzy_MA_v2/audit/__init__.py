"""
audit  ─  Metric Auditor Agent (MA) package
============================================
Exports:
    MetricAuditorAgent  (FH via exponential moving average)
"""

from .auditor import AgentRecord, MetricAuditorAgent, TaskAuditEntry

__all__ = ["AgentRecord", "MetricAuditorAgent", "TaskAuditEntry"]
