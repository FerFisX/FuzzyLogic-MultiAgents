"""
audit/auditor.py  ─  Metric Auditor Agent (MA)
================================================
Records per-agent performance and computes the Historical Reliability
score (FH) used as the third FIE input variable.

Design
------
FH is computed as an Exponential Decay Moving Average (EDMA) over
binary success/failure outcomes, allowing recent performance to
modulate historical weight continuously (Section 3 of the paper).

    FH_new = α · success_indicator + (1 − α) · FH_old    (scaled to [0,100])

The auditor also logs:
  - Per-task latency
  - Cloud escalation events
  - Consensus resolutions vs. cloud fallbacks
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# EMA smoothing factor — higher = faster adaptation to recent performance
_EMA_ALPHA: float = 0.15
# Starting reliability for a new/unknown agent
_INITIAL_FH: float = 50.0


@dataclass
class AgentRecord:
    """Runtime reliability state for a single agent."""
    agent_id: str
    fh: float = _INITIAL_FH          # Historical Reliability ∈ [0, 100]
    total_tasks: int = 0
    successes: int = 0
    failures: int = 0
    cloud_escalations: int = 0
    total_latency_ms: float = 0.0
    last_updated: float = field(default_factory=time.time)

    @property
    def success_rate(self) -> float:
        if self.total_tasks == 0:
            return 0.5
        return self.successes / self.total_tasks

    @property
    def mean_latency_ms(self) -> float:
        if self.total_tasks == 0:
            return 0.0
        return self.total_latency_ms / self.total_tasks


@dataclass
class TaskAuditEntry:
    """Immutable audit record for a single completed task."""
    task_id: str
    agent_id: str
    cs: float
    ir: float
    fh_at_eval: float
    nc: float
    ien: float
    escalated: bool
    success: bool
    latency_ms: float
    timestamp: float = field(default_factory=time.time)

    def as_dict(self) -> dict:
        return asdict(self)


class MetricAuditorAgent:
    """
    Passive monitor that maintains FH scores and audit logs.

    Thread safety: not guaranteed — use external locking for concurrent
    multi-agent scenarios (the FOC is the single writer in this design).
    """

    def __init__(self, log_path: Optional[Path] = None, ema_alpha: float = _EMA_ALPHA):
        self._records: Dict[str, AgentRecord] = {}
        self._log: List[TaskAuditEntry] = []
        self._ema_alpha = ema_alpha
        self._log_path = log_path

        if log_path:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info("Audit log → %s", log_path)

    # ------------------------------------------------------------------
    # QUERY
    # ------------------------------------------------------------------
    def get_fh(self, agent_id: str) -> float:
        """
        Return the current Historical Reliability score for an agent.
        Unknown agents start at the neutral midpoint (50.0).
        """
        return self._records.get(agent_id, AgentRecord(agent_id=agent_id)).fh

    def get_record(self, agent_id: str) -> AgentRecord:
        return self._records.get(agent_id, AgentRecord(agent_id=agent_id))

    def summary(self) -> Dict[str, dict]:
        return {aid: asdict(rec) for aid, rec in self._records.items()}

    # ------------------------------------------------------------------
    # UPDATE
    # ------------------------------------------------------------------
    def record_outcome(
        self,
        task_id: str,
        agent_id: str,
        *,
        cs: float,
        ir: float,
        nc: float,
        ien: float,
        escalated: bool,
        success: bool,
        latency_ms: float,
    ) -> float:
        """
        Update FH for the given agent after task completion and
        append an immutable audit entry.

        Returns the new FH value.
        """
        if agent_id not in self._records:
            self._records[agent_id] = AgentRecord(agent_id=agent_id)

        rec = self._records[agent_id]
        fh_before = rec.fh

        # EMA update on [0, 100] scale
        outcome_scaled = 100.0 if success else 0.0
        rec.fh = self._ema_alpha * outcome_scaled + (1.0 - self._ema_alpha) * rec.fh

        # Counters
        rec.total_tasks += 1
        if success:
            rec.successes += 1
        else:
            rec.failures += 1
        if escalated:
            rec.cloud_escalations += 1
        rec.total_latency_ms += latency_ms
        rec.last_updated = time.time()

        # Audit log entry
        entry = TaskAuditEntry(
            task_id=task_id,
            agent_id=agent_id,
            cs=cs,
            ir=ir,
            fh_at_eval=fh_before,
            nc=nc,
            ien=ien,
            escalated=escalated,
            success=success,
            latency_ms=latency_ms,
        )
        self._log.append(entry)

        logger.debug(
            "Audit | agent=%s task=%s success=%s FH: %.1f → %.1f",
            agent_id, task_id, success, fh_before, rec.fh,
        )

        if self._log_path:
            self._append_to_file(entry)

        return rec.fh

    # ------------------------------------------------------------------
    # PERSISTENCE
    # ------------------------------------------------------------------
    def _append_to_file(self, entry: TaskAuditEntry) -> None:
        try:
            with open(self._log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry.as_dict()) + "\n")
        except OSError as exc:
            logger.warning("Could not write audit log: %s", exc)

    def export_json(self, path: Path) -> None:
        """Dump full audit log + agent records to a JSON file."""
        data = {
            "agents": self.summary(),
            "log": [e.as_dict() for e in self._log],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info("Exported audit data → %s", path)
