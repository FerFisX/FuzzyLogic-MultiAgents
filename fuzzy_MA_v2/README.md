# fuzzy_mas_v2 — Fuzzy Inference Engine + Multi-Agent Orchestrator

**MICAI 2026 — Track B**
*A Fuzzy Inference Engine for Trust Evaluation and Adaptive Task Routing in Asynchronous Multi-Agent LLM Architectures*

---

## Project structure

```
fuzzy_mas_v2/
│
├── fie/                        ← Fuzzy Inference Engine (FIE)
│   ├── engine.py               ← Core Mamdani math (pure NumPy)
│   ├── visualisation.py        ← Membership function plots + Pareto figure
│   └── __init__.py
│
├── orchestrator/               ← Fuzzy Orchestrator Core (FOC)
│   ├── foc.py                  ← Algorithm 1 from the paper
│   ├── complexity.py           ← Semantic Complexity (CS) estimator
│   ├── similarity.py           ← Eq. 23: cosine similarity via Ollama embeddings
│   └── __init__.py
│
├── agents/                     ← Technical Specialty Agents (TSA)
│   ├── base.py                 ← BaseAgent, AgentProposal, TaskRequest
│   ├── stubs.py                ← Stub EA + SA (no LLM needed for tests)
│   ├── ollama_agent.py         ← REAL agents: llama3.2 (EA) + gemma3:12b (SA)
│   │                              IR = Shannon entropy over token logprobs
│   └── __init__.py
│
├── cloud/                      ← Cloud escalation
│   ├── bedrock.py              ← AWS Bedrock handler (converse API,
│   │                              model fallback chain, cost tracking)
│   └── __init__.py
│
├── audit/                      ← Metric Auditor Agent (MA)
│   ├── auditor.py              ← EMA-based Historical Reliability (FH)
│   └── __init__.py
│
├── tests/
│   ├── test_fie.py             ← Unit tests: MF math, fuzzify, evaluate
│   └── test_integration.py     ← Integration: full FOC pipeline (stubs)
│
├── smoke_test.py               ← End-to-end check with REAL Ollama + Bedrock
├── benchmark.py                ← Three-config benchmark → Table 2 of the paper
├── demo.py                     ← Demo runner with stub agents + plots
├── pipeline_milenstone.py      ← (superseded by benchmark.py)
├── requirements.txt
└── README.md
```

---

## Prerequisites

1. **Ollama** running locally with these models pulled:
   ```bash
   ollama pull llama3.2          # EA — Engineering Agent
   ollama pull gemma3:12b        # SA — Support Agent
   ollama pull nomic-embed-text  # E(R) — embeddings for Eq. 23
   ```
   IR requires Ollama ≥ 0.12 (per-token logprobs on the OpenAI-compatible
   endpoint). Verified working on 0.30.7.

2. **AWS CLI** configured with Bedrock model access in `us-east-1`
   (Anthropic Claude and/or Amazon Nova). The handler tries a fallback
   chain automatically: Claude Haiku 4.5 → Claude 3.5 Haiku → Nova Pro.

3. Python deps:
   ```bash
   pip install -r requirements.txt
   ```

---

## How to run (in order)

```bash
cd fuzzy_MA_v2

# 1. Unit + integration tests (no LLMs needed — uses stubs)
python -m pytest tests/ -v

# 2. Demo with stub agents + membership/Pareto figures
python demo.py --no-plots

# 3. End-to-end smoke test with REAL Ollama agents + Bedrock escalation
#    (≈ 6 Ollama calls + 1-2 Bedrock calls, costs < $0.01)
python smoke_test.py
python smoke_test.py --skip-cloud     # local-only variant

# 4. Full benchmark — generates the real data for Table 2 of the paper
#    (150 tasks × 3 configs; pure-cloud + judge cost ≈ $0.30-0.60 total)
python benchmark.py
#    Quick variants:
python benchmark.py --tasks 30 --configs fuzzy-hybrid          # hybrid only
python benchmark.py --tasks 30 --no-judge                      # skip accuracy judge
```

> On Windows, set `$env:PYTHONIOENCODING='utf-8'` first to avoid
> cp1252 console errors when printing model responses.

Benchmark output → `outputs/benchmark_results.json` with:
- `table2_metrics`: one row per configuration (latency, cost/1k, accuracy,
  escalation rate, false-positive escalations, high-complexity detection)
- `runs`: per-task records (CS, IR, NC, IEN, weights, routing, responses)
  for full reproducibility.

---

## FIE public API

```python
from fie import evaluate, evaluar_propuesta, FIEResult

result: FIEResult = evaluate(cs=70.0, ir=0.6, fh=40.0)
print(result.nc)              # Confidence Level ∈ [0, 1]
print(result.ien)             # Escalation Index ∈ [0, 100]
print(result.should_escalate) # True when IEN ≥ 75
```

---

## Full real pipeline

```python
from agents.ollama_agent import OllamaEngineeringAgent, OllamaSupportAgent
from audit.auditor import MetricAuditorAgent
from cloud.bedrock import BedrockCloudHandler
from orchestrator.foc import FuzzyOrchestratorCore
from orchestrator.similarity import OllamaEmbeddingSimilarity

foc = FuzzyOrchestratorCore(
    agents=[OllamaEngineeringAgent(),   # llama3.2  — low-level engineering
            OllamaSupportAgent()],      # gemma3:12b — semantic support
    auditor=MetricAuditorAgent(),
    cloud_handler=BedrockCloudHandler(),            # AWS Bedrock escalation
    similarity_fn=OllamaEmbeddingSimilarity(),      # Eq. 23 (nomic-embed-text)
)

result = foc.process("Diagnose the replication lag issue.")
print(result.summary())
```

---

## How IR (Response Uncertainty) is computed — paper Section 4.1

For every token the local LLM generates, Ollama returns the top-k
log-probabilities. The agent renormalises them, computes the Shannon
entropy, normalises by `log(k)`, and averages over all tokens:

```
p_i = exp(logprob_i) / Σ exp(logprob_j)
H_t = -Σ p_i·log(p_i)        →  Ĥ_t = H_t / log(k)
IR  = mean(Ĥ_t)  ∈ [0, 1]
```

Implementation: `agents/ollama_agent.py :: shannon_entropy_ir()`.

---

## Paper alignment

| Paper section | Code location |
|---|---|
| Sec 4.1 — Input MFs (Eqs 11–19) | `fie/engine.py` `_MF` dict |
| Sec 4.1 — IR via Shannon entropy | `agents/ollama_agent.py` `shannon_entropy_ir()` |
| Sec 4.1 — CS analytic estimator | `orchestrator/complexity.py` |
| Sec 4.1 — FH via EMA window | `audit/auditor.py` `record_outcome()` |
| Sec 4.3 — 27-rule base (Table 1) | `fie/engine.py` `_RULES` |
| Eqs 6–10 — Mamdani + centroid | `fie/engine.py` `_infer()` + `_defuzz()` |
| Sec 3.2 — Task lifecycle | `orchestrator/foc.py` `process()` |
| Eq. 23 — embedding cosine similarity | `orchestrator/similarity.py` |
| Eqs 24–25 — consensus support + weights | `orchestrator/foc.py` `_compute_weights()` |
| Eq. 26 — fusion operator G | `orchestrator/foc.py` `_fuse()` |
| Sec 3.2 step 5 — cloud escalation | `cloud/bedrock.py` `BedrockCloudHandler` |
| Sec 6 — experimental methodology | `benchmark.py` |
| Table 2 — KPI comparison | `outputs/benchmark_results.json` `table2_metrics` |
| Pareto figure | `fie/visualisation.py` `plot_pareto()` |
