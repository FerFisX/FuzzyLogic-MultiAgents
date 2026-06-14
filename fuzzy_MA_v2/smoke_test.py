"""Smoke test: real Ollama agents (IR via Shannon entropy) + embedding similarity + Bedrock."""
import argparse
import logging

logging.basicConfig(level=logging.WARNING)

from agents.base import TaskRequest
from agents.ollama_agent import OllamaEngineeringAgent, OllamaSupportAgent
from orchestrator.similarity import OllamaEmbeddingSimilarity


def _sa_model() -> str:
    """Prefer gemma3:4b (fits in GPU); fall back to whatever gemma3 exists."""
    import requests as _rq
    try:
        tags = _rq.get("http://localhost:11434/api/tags", timeout=5).json()
        names = [m["name"] for m in tags.get("models", [])]
        if "gemma3:4b" in names:
            return "gemma3:4b"
    except Exception:
        pass
    return "gemma3:12b"


def test_ollama_agents():
    print("=== 1. Agentes Ollama reales (IR = entropia de Shannon) ===")
    ea = OllamaEngineeringAgent()
    sa = OllamaSupportAgent(model=_sa_model())

    hard = TaskRequest(content=(
        "Deadlock detectado en transacciones concurrentes con index corruption. "
        "Diagnostica la causa raiz con el distributed trace."
    ))
    easy = TaskRequest(content="Como verifico el estado del servicio local?")

    for label, agent, req in [("EA/dificil", ea, hard), ("EA/facil", ea, easy),
                              ("SA/dificil", sa, hard)]:
        p = agent.handle(req)
        print(f"  {label:<12} IR={p.response_uncertainty:.3f} "
              f"src={p.metadata['ir_source']} lat={p.latency_ms:.0f}ms "
              f"modelo={p.metadata['model']}")
        print(f"     resp: {p.response_text[:120]!r}")


def test_embeddings():
    print("\n=== 2. Similitud coseno por embeddings (Ec. 23) ===")
    sim = OllamaEmbeddingSimilarity()
    texts = [
        "reiniciar el servicio de base de datos para liberar conexiones",
        "hacer restart del servicio de la BD y liberar las conexiones",
        "el clima esta soleado hoy en la ciudad",
    ]
    m = sim(texts)
    print(f"  s(0,1) [parecidos]  = {m[0,1]:.3f}")
    print(f"  s(0,2) [distintos]  = {m[0,2]:.3f}")
    assert m[0, 1] > m[0, 2], "embeddings deberian distinguir similitud semantica"
    print("  OK: la matriz distingue semantica correctamente")


def test_full_pipeline():
    print("\n=== 3. Pipeline completo FOC: Ollama + FIE + Bedrock ===")
    from audit.auditor import MetricAuditorAgent
    from cloud.bedrock import BedrockCloudHandler
    from orchestrator.foc import FuzzyOrchestratorCore

    auditor = MetricAuditorAgent()
    handler = BedrockCloudHandler()
    foc = FuzzyOrchestratorCore(
        agents=[OllamaEngineeringAgent(), OllamaSupportAgent(model=_sa_model())],
        auditor=auditor,
        cloud_handler=handler,
        similarity_fn=OllamaEmbeddingSimilarity(),
        timeout_s=300.0,
    )

    tasks = [
        "Como verifico el estado del servicio local?",
        ("Deadlock critico en transacciones concurrentes con index corruption y "
         "replication lag de 5000ms violando eventual consistency. Root cause "
         "analysis completo con two-phase commit y plan de rollback idempotente."),
    ]
    for t in tasks:
        r = foc.process(t)
        route = "NUBE" if r.escalated else "LOCAL"
        print(f"  [{route}] wall={r.wall_ms:.0f}ms razon={r.escalation_reason}")
        for ev in r.evaluations:
            print(f"     {ev.proposal.agent_id}: CS={ev.cs:.1f} "
                  f"IR={ev.proposal.response_uncertainty:.3f} NC={ev.nc:.3f} IEN={ev.ien:.1f}")
        print(f"     resp: {r.final_response[:140]!r}")
    print(f"  Bedrock stats: {handler.stats()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-cloud", action="store_true")
    args = parser.parse_args()
    test_ollama_agents()
    test_embeddings()
    if not args.skip_cloud:
        test_full_pipeline()
    print("\nSmoke test completo.")
