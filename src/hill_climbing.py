import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import time
import random
import sys
import os

# ==============================
# SOPORTE Y CARGA
# ==============================

def cargar_serie(nombre_archivo):
    """Carga la serie desde un .txt con formato [ p1 p2 ... pk ]"""
    try:
        with open(nombre_archivo, 'r') as f:
            contenido = f.read().strip().replace('[', '').replace(']', '')
            return np.fromstring(contenido, sep=' ')
    except Exception as e:
        print(f"Error al cargar archivo: {e}")
        return None

def calcular_mse_segmento(x, y):
    """Calcula el MSE de una regresión lineal de un segmento"""
    if len(x) < 2:
        return 1e9
    model = LinearRegression().fit(x.reshape(-1, 1), y)
    predicciones = model.predict(x.reshape(-1, 1))
    return np.mean((y - predicciones) ** 2)

def evaluar_solucion(cortes, serie):
    """Calcula la media de los errores cuadráticos medios"""
    N = len(serie)
    puntos = [0] + sorted(cortes) + [N]
    errores = []

    for i in range(len(puntos) - 1):
        inicio, fin = puntos[i], puntos[i+1]
        y_seg = serie[inicio:fin]
        x_seg = np.arange(inicio, fin)
        errores.append(calcular_mse_segmento(x_seg, y_seg))

    return np.mean(errores)

# ==============================
# HILL CLIMBING SIMPLE
# ==============================

def hill_climbing_simple(serie, k, iteraciones):
    N = len(serie)
    actual = sorted(random.sample(range(1, N), k - 1))
    error_actual = evaluar_solucion(actual, serie)
    historial_errores = []
    start_time = time.time()
    rango = max(1, int(N * 0.01))

    for _ in range(iteraciones):
        vecino = list(actual)
        idx = random.randint(0, len(vecino) - 1)
        vecino[idx] += random.randint(-rango, rango)

        if 0 < vecino[idx] < N: 
            error_vecino = evaluar_solucion(vecino, serie)
            if error_vecino < error_actual: 
                actual = sorted(vecino)
                error_actual = error_vecino

        historial_errores.append(error_actual)

    media_mse = np.mean(historial_errores)
    tiempo = time.time() - start_time
    return actual, error_actual, tiempo, media_mse
    

# ==============================
# HILL CLIMBING ESTOCÁSTICO
# ==============================

def hill_climbing_estocastico(serie, k, iteraciones):
    N = len(serie)
    actual = sorted(random.sample(range(1, N), k - 1))
    error_actual = evaluar_solucion(actual, serie)
    historial_errores = []
    start_time = time.time()
    rango = max(1, int(N * 0.01))

    evals = 0
    while evals < iteraciones:
        candidatos = []
        # Evaluamos una pequeña muestra de 10 vecinos para elegir uno que mejore
        for _ in range(10):
            if evals >= iteraciones: break
            v = list(actual)
            idx = random.randint(0, len(v) - 1)
            v[idx] += random.randint(-rango, rango)
            
            if 0 < v[idx] < N:
                err_v = evaluar_solucion(v, serie)
                evals += 1
                if err_v < error_actual:
                    candidatos.append((v, err_v))
                historial_errores.append(error_actual)

        if candidatos:
            actual, error_actual = random.choice(candidatos)
            actual = sorted(actual)
        elif evals < iteraciones:
            # Si en la muestra de 10 ninguno mejoró, seguimos registrando el error actual
            historial_errores.append(error_actual)
            evals += 1

    media_mse = np.mean(historial_errores)
    return actual, error_actual, time.time() - start_time, media_mse


# ==============================
# HILL CLIMBING MÁXIMA PENDIENTE
# ==============================

def hill_climbing_maxima_pendiente(serie, k, iteraciones):
    N = len(serie)
    actual = sorted(random.sample(range(1, N), k - 1))
    error_actual = evaluar_solucion(actual, serie)
    historial_errores = []
    start_time = time.time()
    rango = max(1, int(N * 0.01))
    evaluaciones = 0
    
    while evaluaciones < iteraciones:
        mejor_vecino = None
        mejor_error = error_actual
        
        # Exploración de vecindad (20 vecinos por paso)
        for _ in range(20): 
            if evaluaciones >= iteraciones: break
            
            v_tmp = list(actual)
            idx = random.randint(0, len(v_tmp) - 1)
            v_tmp[idx] += random.randint(-rango, rango)

            if 0 < v_tmp[idx] < N:
                error_vecino = evaluar_solucion(v_tmp, serie)
                evaluaciones += 1
                if error_vecino < mejor_error:
                    mejor_error = error_vecino
                    mejor_vecino = sorted(v_tmp)
            
            historial_errores.append(error_actual)

        if mejor_vecino:
            actual = mejor_vecino
            error_actual = mejor_error
        else:
            # Si no hay mejora en la vecindad, rellenamos el historial hasta max_iter
            while evaluaciones < iteraciones:
                historial_errores.append(error_actual)
                evaluaciones += 1
            break

    duracion = time.time() - start_time
    media_mse = np.mean(historial_errores)
    return actual, error_actual, duracion, media_mse

# ==============================
# BLOQUE PRINCIPAL (MAIN)
# ==============================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python hill_climbing.py <archivo_serie>")
        sys.exit(1)

    ruta_archivo = sys.argv[1]
    serie = cargar_serie(ruta_archivo)
    if serie is None:
        sys.exit(1)

    nombre_base = os.path.basename(ruta_archivo).upper()
    k_map = {"TS1": 9, "TS2": 10, "TS3": 20, "TS4": 50}
    k_objetivo = next((v for n, v in k_map.items() if n in nombre_base), 10)
    ITERACIONES = 1000

    print(f"\nArchivo: {ruta_archivo} | K: {k_objetivo} | Iteraciones: {ITERACIONES}")
    print("Elige la variante de Hill Climbing:")
    print("1 - Simple\n2 - Estocástico\n3 - Máxima Pendiente\n4 - Todas")
    opcion = input("Opción: ")

    resultados = []
    
    if opcion in ["1", "4"]:
        c, e, t, m = hill_climbing_simple(serie, k_objetivo, ITERACIONES)
        resultados.append(("Simple", c, e, t, m))
    if opcion in ["2", "4"]:
        c, e, t, m = hill_climbing_estocastico(serie, k_objetivo, ITERACIONES)
        resultados.append(("Estocástico", c, e, t, m))
    if opcion in ["3", "4"]:
        c, e, t, m = hill_climbing_maxima_pendiente(serie, k_objetivo, ITERACIONES)
        resultados.append(("Máxima Pendiente", c, e, t, m))

    for nombre, cortes, error, tiempo, media in resultados:
        print(f"\n--- {nombre} ---")
        print(f"MSE Final: {error:.6f}")
        print(f"Media MSE: {media:.6f}")
        print(f"Tiempo: {tiempo:.4f}s")

        # Visualización
        plt.figure(figsize=(10, 5))
        plt.plot(serie, color='silver', label='Serie Original', alpha=0.6)
        puntos = [0] + cortes + [len(serie)]
        for i in range(len(puntos)-1):
            idx = np.arange(puntos[i], puntos[i+1])
            reg = LinearRegression().fit(idx.reshape(-1,1), serie[idx])
            plt.plot(idx, reg.predict(idx.reshape(-1,1)), linewidth=2)
        plt.title(f"{nombre} - {nombre_base} (MSE: {error:.4f})")
        plt.legend()
        plt.show()