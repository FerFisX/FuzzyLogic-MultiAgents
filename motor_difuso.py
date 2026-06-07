import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

def crear_motor_difuso():
    # 1. DEFINICIÓN DE LOS UNIVERSOS DE DISCURSO (RANGOS)
    # Entradas
    complejidad = ctrl.Antecedent(np.arange(0, 101, 1), 'complejidad')
    incertidumbre = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'incertidumbre')
    fiabilidad = ctrl.Antecedent(np.arange(0, 101, 1), 'fiabilidad')
    
    # Salidas
    confianza = ctrl.Consequent(np.arange(0, 1.01, 0.01), 'confianza')
    escalamiento = ctrl.Consequent(np.arange(0, 101, 1), 'escalamiento')

    # 2. FUNCIONES DE PERTENENCIA (Fuzzificación basada en el paper)
    
    # Complejidad Semántica (CS)
    complejidad['baja'] = fuzz.trapmf(complejidad.universe, [0, 0, 20, 45])
    complejidad['media'] = fuzz.trimf(complejidad.universe, [30, 50, 70])
    complejidad['alta'] = fuzz.trapmf(complejidad.universe, [55, 80, 100, 100])

    # Incertidumbre de la Respuesta (IR)
    incertidumbre['minima'] = fuzz.trapmf(incertidumbre.universe, [0, 0, 0.15, 0.35])
    incertidumbre['moderada'] = fuzz.trimf(incertidumbre.universe, [0.25, 0.50, 0.75])
    incertidumbre['elevada'] = fuzz.trapmf(incertidumbre.universe, [0.65, 0.85, 1.0, 1.0])

    # Fiabilidad Histórica (FH)
    fiabilidad['deficiente'] = fuzz.trapmf(fiabilidad.universe, [0, 0, 40, 60])
    fiabilidad['aceptable'] = fuzz.trimf(fiabilidad.universe, [50, 75, 90])
    fiabilidad['excelente'] = fuzz.trapmf(fiabilidad.universe, [80, 95, 100, 100])

    # Nivel de Confianza (NC) - SALIDA 1
    confianza['bajo'] = fuzz.trapmf(confianza.universe, [0, 0, 0.25, 0.50])
    confianza['medio'] = fuzz.trimf(confianza.universe, [0.35, 0.60, 0.85])
    confianza['alto'] = fuzz.trapmf(confianza.universe, [0.70, 0.90, 1.0, 1.0])

    # Índice de Escalamiento a la Nube (IEN) - SALIDA 2
    escalamiento['innecesario'] = fuzz.trapmf(escalamiento.universe, [0, 0, 25, 45])
    escalamiento['condicional'] = fuzz.trimf(escalamiento.universe, [35, 55, 75])
    escalamiento['critico'] = fuzz.trapmf(escalamiento.universe, [65, 85, 100, 100])

    # 3. BASE DE REGLAS (Las 27 reglas de la Tabla 1 del paper)
    reglas = [
        # Complejidad Baja
        ctrl.Rule(complejidad['baja'] & incertidumbre['minima'] & fiabilidad['excelente'], (confianza['alto'], escalamiento['innecesario'])),
        ctrl.Rule(complejidad['baja'] & incertidumbre['minima'] & fiabilidad['aceptable'], (confianza['alto'], escalamiento['innecesario'])),
        ctrl.Rule(complejidad['baja'] & incertidumbre['minima'] & fiabilidad['deficiente'], (confianza['medio'], escalamiento['condicional'])),
        ctrl.Rule(complejidad['baja'] & incertidumbre['moderada'] & fiabilidad['excelente'], (confianza['alto'], escalamiento['innecesario'])),
        ctrl.Rule(complejidad['baja'] & incertidumbre['moderada'] & fiabilidad['aceptable'], (confianza['medio'], escalamiento['innecesario'])),
        ctrl.Rule(complejidad['baja'] & incertidumbre['moderada'] & fiabilidad['deficiente'], (confianza['bajo'], escalamiento['condicional'])),
        ctrl.Rule(complejidad['baja'] & incertidumbre['elevada'] & fiabilidad['excelente'], (confianza['medio'], escalamiento['condicional'])),
        ctrl.Rule(complejidad['baja'] & incertidumbre['elevada'] & fiabilidad['aceptable'], (confianza['bajo'], escalamiento['condicional'])),
        ctrl.Rule(complejidad['baja'] & incertidumbre['elevada'] & fiabilidad['deficiente'], (confianza['bajo'], escalamiento['critico'])),

        # Complejidad Media
        ctrl.Rule(complejidad['media'] & incertidumbre['minima'] & fiabilidad['excelente'], (confianza['alto'], escalamiento['innecesario'])),
        ctrl.Rule(complejidad['media'] & incertidumbre['minima'] & fiabilidad['aceptable'], (confianza['alto'], escalamiento['innecesario'])),
        ctrl.Rule(complejidad['media'] & incertidumbre['minima'] & fiabilidad['deficiente'], (confianza['medio'], escalamiento['condicional'])),
        ctrl.Rule(complejidad['media'] & incertidumbre['moderada'] & fiabilidad['excelente'], (confianza['alto'], escalamiento['innecesario'])),
        ctrl.Rule(complejidad['media'] & incertidumbre['moderada'] & fiabilidad['aceptable'], (confianza['medio'], escalamiento['condicional'])),
        ctrl.Rule(complejidad['media'] & incertidumbre['moderada'] & fiabilidad['deficiente'], (confianza['bajo'], escalamiento['critico'])),
        ctrl.Rule(complejidad['media'] & incertidumbre['elevada'] & fiabilidad['excelente'], (confianza['medio'], escalamiento['condicional'])),
        ctrl.Rule(complejidad['media'] & incertidumbre['elevada'] & fiabilidad['aceptable'], (confianza['bajo'], escalamiento['critico'])),
        ctrl.Rule(complejidad['media'] & incertidumbre['elevada'] & fiabilidad['deficiente'], (confianza['bajo'], escalamiento['critico'])),

        # Complejidad Alta
        ctrl.Rule(complejidad['alta'] & incertidumbre['minima'] & fiabilidad['excelente'], (confianza['alto'], escalamiento['innecesario'])),
        ctrl.Rule(complejidad['alta'] & incertidumbre['minima'] & fiabilidad['aceptable'], (confianza['medio'], escalamiento['condicional'])),
        ctrl.Rule(complejidad['alta'] & incertidumbre['minima'] & fiabilidad['deficiente'], (confianza['bajo'], escalamiento['critico'])),
        ctrl.Rule(complejidad['alta'] & incertidumbre['moderada'] & fiabilidad['excelente'], (confianza['medio'], escalamiento['condicional'])),
        ctrl.Rule(complejidad['alta'] & incertidumbre['moderada'] & fiabilidad['aceptable'], (confianza['medio'], escalamiento['critico'])),
        ctrl.Rule(complejidad['alta'] & incertidumbre['moderada'] & fiabilidad['deficiente'], (confianza['bajo'], escalamiento['critico'])),
        ctrl.Rule(complejidad['alta'] & incertidumbre['elevada'] & fiabilidad['excelente'], (confianza['bajo'], escalamiento['critico'])),
        ctrl.Rule(complejidad['alta'] & incertidumbre['elevada'] & fiabilidad['aceptable'], (confianza['bajo'], escalamiento['critico'])),
        ctrl.Rule(complejidad['alta'] & incertidumbre['elevada'] & fiabilidad['deficiente'], (confianza['bajo'], escalamiento['critico']))
    ]

    # 4. CREACIÓN DEL SISTEMA DE CONTROL
    sistema_control = ctrl.ControlSystem(reglas)
    simulador = ctrl.ControlSystemSimulation(sistema_control)
    
    return simulador

# Wrapper fácil de usar para evaluar propuestas en tiempo real
def evaluar_propuesta(cs, ir, fh):
    """
    cs: Complejidad Semántica (0-100)
    ir: Incertidumbre (0.0-1.0)
    fh: Fiabilidad Histórica (0-100)
    """
    simulador = crear_motor_difuso()
    
    # Inyectar los valores reales
    simulador.input['complejidad'] = cs
    simulador.input['incertidumbre'] = ir
    simulador.input['fiabilidad'] = fh
    
    # Ejecutar la defuzzificación (Centroide matemático)
    simulador.compute()
    
    nc = simulador.output['confianza']
    ien = simulador.output['escalamiento']
    
    return nc, ien

# Bloque de prueba
if __name__ == "__main__":
    print("Iniciando pruebas del Motor de Inferencia Difusa (FIE)...")
    
    # Caso 1: Tarea sencilla, agente seguro de sí mismo y con buena reputación
    nc1, ien1 = evaluar_propuesta(cs=10, ir=0.1, fh=95)
    print(f"\nCaso 1 (Fácil/Seguro) -> Confianza: {nc1:.2f} | Índice Escalamiento: {ien1:.2f} (Esperado: Confianza Alta, Escalamiento Bajo)")
    
    # Caso 2: Tarea muy compleja (discrepancia de base de datos), el agente duda un poco, reputación aceptable
    nc2, ien2 = evaluar_propuesta(cs=85, ir=0.6, fh=70)
    print(f"Caso 2 (Difícil/Dudoso) -> Confianza: {nc2:.2f} | Índice Escalamiento: {ien2:.2f} (Esperado: Confianza Baja/Media, Escalamiento CRÍTICO)")