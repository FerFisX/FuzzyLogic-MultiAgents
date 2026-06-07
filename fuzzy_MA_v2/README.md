# fuzzy_mas_v2 — Fuzzy Inference Engine + Multi-Agent Orchestrator

**MICAI 2026 — Track B**  
*A Fuzzy Inference Engine for Trust Evaluation and Adaptive Task Routing in Asynchronous Multi-Agent LLM Architectures*

---

## Project structure

```
fuzzy_mas_v2/
│
├── fie/                        ← Fuzzy Inference Engine (FIE)
│   ├── engine.py               ← Core Mamdani math (pure NumPy, zero extra deps)
│   ├── visualisation.py        ← Membership function plots + Pareto figure
│   └── __init__.py
│
├── orchestrator/               ← Fuzzy Orchestrator Core (FOC)
│   ├── foc.py                  ← Algorithm 1 from the paper
│   ├── complexity.py           ← Semantic Complexity (CS) estimator
│   └── __init__.py
│
├── agents/                     ← Agent base class + stub agents
│   ├── base.py                 ← BaseAgent, AgentProposal, TaskRequest
│   ├── stubs.py                ← Stub EA + SA (no LLM needed for tests)
│   └── __init__.py
│
├── audit/                      ← Metric Auditor Agent (MA)
│   ├── auditor.py              ← EMA-based Historical Reliability (FH)
│   └── __init__.py
│
├── tests/
│   ├── test_fie.py             ← Unit tests: MF math, fuzzify, evaluate
│   └── test_integration.py     ← Integration: full FOC pipeline
│
├── demo.py                     ← End-to-end demo runner
├── requirements.txt
└── README.md
```

---

## Relation to v1 files

| v1 file | Status | Notes |
|---|---|---|
| `motor_difuso.py` | **Superseded** | Used `skfuzzy` high-level API — math is hidden. `evaluar_propuesta()` compatibility alias preserved in `fie/engine.py` |
| `fuzzy_inference_engine.py` | **Promoted to core** | Pure-NumPy Mamdani engine is the basis for `fie/engine.py`. Refactored into a proper package with `FIEResult` dataclass, input validation, and timing |
| `generar_arquitectura.py` | **Unchanged** | Still works as-is to regenerate the architecture PNG |

---

## Quick start

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
python -m pytest tests/ -v

# Run demo (with plots)
python demo.py

# Run demo without matplotlib windows
python demo.py --no-plots
```

---

## FIE public API

```python
from fie import evaluate, evaluar_propuesta, FIEResult

# Full result with diagnostics
result: FIEResult = evaluate(cs=70.0, ir=0.6, fh=40.0)
print(result.nc)             # Confidence Level ∈ [0, 1]
print(result.ien)            # Escalation Index ∈ [0, 100]
print(result.should_escalate) # True when IEN ≥ 75
print(result.eval_ms)        # wall-clock time in ms

# Compatibility alias (matches both v1 files)
nc, ien = evaluar_propuesta(cs=70.0, ir=0.6, fh=40.0)
```

---

## Full orchestrator pipeline

```python
from agents.stubs import StubEngineeringAgent, StubSupportAgent
from audit.auditor import MetricAuditorAgent
from orchestrator.foc import FuzzyOrchestratorCore

auditor = MetricAuditorAgent()
agents  = [StubEngineeringAgent(), StubSupportAgent()]
foc     = FuzzyOrchestratorCore(agents=agents, auditor=auditor)

result = foc.process("Diagnose the replication lag issue.")
print(result.summary())
```

---

## Wiring real LLM agents

Subclass `agents.base.BaseAgent`, implement `handle()`, and pass your agent to the `FuzzyOrchestratorCore`. The key thing your agent must return is `response_uncertainty` (IR) — derive it from average Shannon entropy over token log-probabilities from your LLM.

```python
class OllamaEngineeringAgent(BaseAgent):
    def handle(self, request: TaskRequest) -> AgentProposal:
        response, logprobs = call_ollama("llama3.2:7b", request.content)
        ir = compute_shannon_entropy(logprobs)  # your IR computation
        return self._make_proposal(request, response, ir)
```

For cloud escalation, pass a `cloud_handler` callable to `FuzzyOrchestratorCore`:

```python
import anthropic

def my_cloud_handler(task_text: str) -> str:
    client = anthropic.Anthropic()
    msg = client.messages.create(model="claude-opus-4-6", ...)
    return msg.content[0].text

foc = FuzzyOrchestratorCore(agents=agents, auditor=auditor,
                             cloud_handler=my_cloud_handler)
```

---

## Paper alignment

| Paper section | Code location |
|---|---|
| Sec 4.1 — Input MFs (Eqs 1–9) | `fie/engine.py` `_MF` dict + `_mu_tri` / `_mu_trap` |
| Sec 4.3 — 27-rule base (Table 1) | `fie/engine.py` `_RULES` list |
| Eqs 3–5 — Mamdani inference + defuzz | `fie/engine.py` `_infer()` + `_defuzz()` |
| Sec 3.2 — Task Lifecycle | `orchestrator/foc.py` `FuzzyOrchestratorCore.process()` |
| Algorithm 1 — Consensus & Routing | `orchestrator/foc.py` |
| Eqs 6–8 — Consensus weights + fusion | `orchestrator/foc.py` `_compute_weights()` + `_fuse()` |
| Sec 3.1 — MA / FH with EMA | `audit/auditor.py` `MetricAuditorAgent.record_outcome()` |
| Fig 1 — Pareto frontier | `fie/visualisation.py` `plot_pareto()` |
