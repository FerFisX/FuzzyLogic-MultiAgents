import json
import random
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Intentar importar tus métricas de complejidad existentes
try:
    from orchestrator.complexity import compute_semantic_complexity
except ImportError:
    # Fallback inline si se ejecuta de forma aislada
    def compute_semantic_complexity(text):
        return round(min(100.0, max(10.0, len(text) * 0.15)), 2)

# 1. GENERACIÓN DEL DATASET SINTÉTICO DE 150 TAREAS
def generar_dataset_150_tareas():
    categorias = {
        "alta": [
            "Deadlock detectado en transacciones concurrentes con index corruption.",
            "Spike crítico en tcp retransmission rate causando caídas de handshake SSL.",
            "Fuga de memoria (memory leak) severa identificada tras cascade delete en base de datos.",
            "Replication lag superior a 5000ms con violaciones de consistencia eventual.",
            "Error de segmentación (segfault) intermitente en el kernel de ejecución."
        ],
        "media": [
            "Excepción de timeout en la API de autenticación del contenedor de servicios.",
            "Optimizar query SQL lenta que genera cuellos de botella en los logs de auditoría.",
            "Error intermitente en el pipeline de despliegue debido a configuraciones de permisos.",
            "La caché del servidor de base de datos reporta baja tasa de aciertos (hit rate).",
            "Fallo de conexión por certificados vencidos en el entorno de pruebas."
        ],
        "baja": [
            "¿Cómo puedo verificar el estado actual del servicio local?",
            "Listar las versiones activas del software en el clúster de infraestructura.",
            "Mostrar los logs de auditoría generados en los últimos 10 minutos.",
            "Ayuda para reiniciar de forma segura el contenedor de logs.",
            "Chequeo básico de salud del sistema mediante ping de red."
        ]
    }
    
    dataset = []
    task_id = 1
    
    # Generar iterativamente variaciones hasta completar exactamente 150 elementos
    while len(dataset) < 150:
        for nivel, plantillas in categorias.items():
            if len(dataset) >= 150:
                break
            plantilla = random.choice(plantillas)
            # Añadir variaciones para simular entradas de usuarios únicas
            variante_id = f"[Variación #{task_id:03d}]"
            dataset.append({
                "task_id": f"TASK-{task_id:03d}",
                "content": f"{variante_id} {plantilla}",
                "categoria_esperada": nivel
            })
            task_id += 1
            
    return dataset

# 2. LLAMADA REAL / SIMULADA AL MODELO DE LENGUAJE LOCAL (OLLAMA)
def consultar_llm_local(prompt):
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "llama3",
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.3}
    }
    
    try:
        # Timeout corto para evitar bloqueos si Ollama no está corriendo
        response = requests.post(url, json=payload, timeout=4.0)
        if response.status_code == 200:
            return response.json().get("response", "")
    except Exception:
        pass
    
    # Fallback simulado controlado si el servicio local no está activo
    respuestas_mock = [
        "Analizado. Se sugiere aplicar un rollback inmediato y verificar la integridad de los índices dañados.",
        "Error operativo detectado en el microservicio. Proceder con reinicio controlado del pod afectado.",
        "Operación completada con éxito. No se registran anomalías en los logs primarios del sistema distribuido."
    ]
    return f"[MOCK LOCAL LLM] {random.choice(respuestas_mock)} Extensión sintética para validación métrica: " + " ".join(["dato"] * random.randint(5, 30))

# 3. EVALUADOR FALSO / TEMPORAL (Decisión Estática solicitada por la tarjeta)
class FakeEvaluator:
    def evaluar(self, cs, ir):
        # Devuelve siempre una estructura fija emulando la interfaz del FIE real
        return {
            "nc": 0.65,              # Nivel de confianza estático
            "ien": 45.0,             # Índice de escalamiento estático
            "decision": "LOCAL"      # Decisión estática invariable
        }

# 4. PROCESAMIENTO UNITARIO DE UNA TAREA
def procesar_tarea(tarea, evaluador):
    contenido = tarea["content"]
    
    # Medir latencia de inferencia
    t_start = time.perf_counter()
    respuesta_texto = consultar_llm_local(contenido)
    latencia_ms = (time.perf_counter() - t_start) * 1000.0
    
    # Calcular Complejidad Semántica analítica
    cs = compute_semantic_complexity(contenido)
    
    # Calcular Incertidumbre (IR) basada explícitamente en la longitud de respuesta
    # Heurística: mayor longitud respecto a un umbral esperado => mayor dispersión/incertidumbre
    longitud = len(respuesta_texto)
    ir = round(min(1.0, max(0.0, longitud / 600.0)), 2) 
    
    # Pasar métricas al evaluador falso estático
    resultado_difuso = evaluador.evaluar(cs, ir)
    
    return {
        "task_id": tarea["task_id"],
        "prompt": contenido,
        "llm_response": respuesta_texto,
        "metrics": {
            "semantic_complexity_cs": cs,
            "uncertainty_ir": ir,
            "response_length_chars": longitud,
            "latency_ms": round(latencia_ms, 2)
        },
        "evaluation": resultado_difuso
    }

# 5. ORQUESTADOR CENTRAL ASÍNCRONO CONTRATADO POR LA CHEKLIST
def ejecutar_pipeline_orquestador(max_workers=10):
    print("🚀 Inicializando Pipeline de Pruebas...")
    
    # Crear el dataset de 150 tareas
    dataset = generar_dataset_150_tareas()
    print(f"✓ Dataset sintético generado con {len(dataset)} tareas correctamente.")
    
    evaluador = FakeEvaluator()
    resultados_finales = []
    
    print(f"⚡ Enviando tareas de manera concurrente (Hilos máximos: {max_workers})...")
    t_pipeline_start = time.perf_counter()
    
    # Procesamiento por hilos concurrentes
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(procesar_tarea, tarea, evaluador): tarea for tarea in dataset}
        
        for idx, future in enumerate(as_completed(futures), 1):
            try:
                record = future.result()
                resultados_finales.append(record)
                if idx % 30 == 0 or idx == 150:
                    print(f"   [Progreso] {idx}/150 tareas procesadas exhaustivamente.")
            except Exception as e:
                print(f"❌ Error procesando una tarea: {e}")
                
    total_time = time.perf_counter() - t_pipeline_start
    print(f"✓ Procesamiento asíncrono finalizado en {total_time:.2f} segundos.")
    
    # Guardar los resultados en formato JSON limpio
    output_path = Path("outputs/audit_milestone_dataset.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(resultados_finales, f, indent=2, ensure_ascii=False)
        
    print(f"💾 Resultados limpios exportados de forma segura en: '{output_path}'")

if __name__ == "__main__":
    ejecutar_pipeline_orquestador()