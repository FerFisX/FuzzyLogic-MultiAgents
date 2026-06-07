from diagrams import Diagram, Cluster, Edge
from diagrams.onprem.client import User
from diagrams.onprem.queue import ActiveMQ
from diagrams.programming.language import Python
from diagrams.onprem.compute import Server
from diagrams.onprem.network import Internet
from diagrams.onprem.database import PostgreSQL

# Configuramos la generación del diagrama
with Diagram("Arquitectura Híbrida Multi-Agente Asíncrona", show=False, filename="arquitectura_agentes", outformat="png"):
    
    # Origen de la incidencia
    usuario = User("Petición / Webhook")
    
    # Entorno Local (ej. WSL2 / Contenedor)
    with Cluster("Entorno Local Híbrido"):
        foc = Python("Núcleo Orquestador Difuso\n(FIE + Scikit-Fuzzy)")
        bus = ActiveMQ("Bus de Mensajes\n(Broadcast Asíncrono)")
        auditor = PostgreSQL("Agente Auditor\n(Fiabilidad Histórica)")
        
        with Cluster("Enjambre de Agentes (Ollama)"):
            agente_ing = Server("Agente de Ingeniería\n(Llama 3 local)")
            agente_soporte = Server("Agente de Soporte\n(Mistral local)")
            
    # Infraestructura Externa
    nube = Internet("Modelos en la Nube\n(Escalamiento Crítico)")
    
    # ---- Lógica de Enrutamiento (Edges) ----
    
    # 1. Ingreso
    usuario >> Edge(label="1. Incidencia de entrada") >> foc
    
    # 2. Difusión asíncrona a los agentes
    foc >> Edge(label="2. Envío a Cola") >> bus
    bus >> agente_ing
    bus >> agente_soporte
    
    # 3. Retorno de respuestas con su incertidumbre local
    agente_ing >> Edge(label="3. Respuesta + Incertidumbre") >> foc
    agente_soporte >> Edge(label="3. Respuesta + Incertidumbre") >> foc
    
    # 4. Auditoría
    foc >> Edge(label="Logs de fiabilidad") >> auditor
    auditor >> Edge(label="Consulta de reputación") >> foc
    
    # 5. Toma de decisión difusa: Escalamiento vs Local
    foc >> Edge(label="IEN > 75 (Escalamiento)", color="red", style="dashed") >> nube
    nube >> Edge(label="Solución Cloud", color="red") >> foc
    
    # 6. Salida final
    foc >> Edge(label="Respuesta de Consenso") >> usuario

print("Diagrama generado exitosamente como 'arquitectura_agentes.png'")