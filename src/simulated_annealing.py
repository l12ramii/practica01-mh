# ==========================================================
# PRACTICA 1 - METAHEURISTICAS
# Segmentación de series temporales desde archivos .txt
# ==========================================================

import numpy as np
import matplotlib.pyplot as plt
import time
import math
import os


# ==========================================================
# 1️⃣ CARGAR SERIE DESDE ARCHIVO .txt
# ==========================================================

def load_series(filepath):
    """
    Carga una serie temporal desde un archivo .txt
    Soporta:
    - Lista tipo Python: [1, 2, 3]
    - Valores separados por comas
    - Valores uno por línea
    """
    with open(filepath, "r") as f:
        content = f.read().strip()

    # Eliminar corchetes si existen
    content = content.replace("[", "").replace("]", "")

    # Reemplazar comas por espacios
    content = content.replace(",", " ")

    # Convertir a array numpy
    series = np.array([float(x) for x in content.split()])

    return series


# ==========================================================
# 2️⃣ ERROR DE UN SEGMENTO (REGRESION LINEAL + MSE)
# ==========================================================

def segment_error(x, y):
    """
    Calcula el MSE de una regresión lineal
    usando fórmula cerrada (mucho más rápido que sklearn)
    """
    if len(x) < 2:
        return 0  # segmento muy pequeño
    
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean) ** 2)
    
    if denominator == 0:
        return 0
    
    b = numerator / denominator
    a = y_mean - b * x_mean
    
    y_pred = a + b * x
    mse = np.mean((y - y_pred) ** 2)
    
    return mse

def fit_line(x, y):
    """
    Calcula los parámetros de la recta ajustada
    usando regresión lineal cerrada.
    Devuelve a, b (intercepto y pendiente)
    """
    if len(x) < 2:
        return 0, 0
    
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean) ** 2)
    
    if denominator == 0:
        return y_mean, 0
    
    b = numerator / denominator
    a = y_mean - b * x_mean
    
    return a, b
# ==========================================================
# 3️⃣ FUNCION OBJETIVO
# Media de los MSE de los segmentos
# ==========================================================

def objective_function(series, cuts):
    prev = 0
    errors = []

    for c in cuts:
        x = np.arange(prev, c)
        y = series[prev:c]
        errors.append(segment_error(x, y))
        prev = c

    # Último segmento
    x = np.arange(prev, len(series))
    y = series[prev:]
    errors.append(segment_error(x, y))

    return np.mean(errors)


# ==========================================================
# 4️⃣ SOLUCION ALEATORIA INICIAL
# ==========================================================

def random_solution(N, k):
    return sorted(np.random.choice(range(1, N-1), k-1, replace=False))


# ==========================================================
# 5️⃣ GENERADOR DE VECINO
# ==========================================================

def generate_neighbor(cuts, N):
    new_cuts = cuts.copy()
    idx = np.random.randint(len(cuts))
    move = np.random.choice([-1, 1])

    new_cuts[idx] += move
    new_cuts[idx] = max(1, min(N-2, new_cuts[idx]))

    return sorted(new_cuts)


# ==========================================================
# 6️⃣ RANDOM SEARCH
# ==========================================================

def random_search(series, k, max_iter=1000):
    N = len(series)
    best_error = float("inf")
    best_cuts = None
    history = []

    for _ in range(max_iter):
        cuts = random_solution(N, k)
        error = objective_function(series, cuts)

        if error < best_error:
            best_error = error
            best_cuts = cuts

        history.append(best_error)

    return best_cuts, best_error, history


# ==========================================================
# 7️⃣ HILL CLIMBING
# ==========================================================

def hill_climbing(series, k, max_iter=1000):
    N = len(series)
    cuts = random_solution(N, k)
    current_error = objective_function(series, cuts)

    best_cuts = cuts.copy()
    best_error = current_error
    history = []

    for _ in range(max_iter):
        new_cuts = generate_neighbor(cuts, N)
        new_error = objective_function(series, new_cuts)

        if new_error < current_error:
            cuts = new_cuts
            current_error = new_error

        if current_error < best_error:
            best_cuts = cuts.copy()
            best_error = current_error

        history.append(best_error)

    return best_cuts, best_error, history


# ==========================================================
# 8️⃣ SIMULATED ANNEALING
# ==========================================================

def simulated_annealing(series, k, T0=100, alpha=0.95, max_iter=1000):
    N = len(series)
    cuts = random_solution(N, k)
    current_error = objective_function(series, cuts)

    best_cuts = cuts.copy()
    best_error = current_error

    T = T0
    history = []

    for _ in range(max_iter):
        new_cuts = generate_neighbor(cuts, N)
        new_error = objective_function(series, new_cuts)

        if new_error < current_error:
            cuts = new_cuts
            current_error = new_error
        else:
            prob = math.exp(-(new_error - current_error) / T)
            if np.random.rand() < prob:
                cuts = new_cuts
                current_error = new_error

        if current_error < best_error:
            best_cuts = cuts.copy()
            best_error = current_error

        history.append(best_error)
        T = alpha * T

    return best_cuts, best_error, history


# ==========================================================
# 9️⃣ EVALUACION ESTADISTICA
# ==========================================================

def evaluate_method(method, series, k, runs=10):
    results = []
    times = []

    for _ in range(runs):
        start = time.time()
        _, best_error, _ = method(series, k)
        end = time.time()

        results.append(best_error)
        times.append(end - start)

    return np.mean(results), np.std(results), np.mean(times), results


# ==========================================================
# 🔟 EJECUCION SOBRE TS1-TS4
# ==========================================================

if __name__ == "__main__":

    k_values = {
        "TS1": 9,
        "TS2": 10,
        "TS3": 20,
        "TS4": 50
    }

    folder = "../series"

    for name, k in k_values.items():

        print(f"\n================ {name} =================")

        filepath = os.path.join(folder, name + ".txt")
        series = load_series(filepath)

        # Ejecutar múltiples veces SA para estadísticas
        mean_error, std_error, mean_time, all_results = evaluate_method(
            simulated_annealing, series, k, runs=10
        )

        print("Simulated Annealing")
        print("Media del error:", mean_error)
        print("Desviación típica:", std_error)
        print("Tiempo medio:", mean_time)

        # Ejecutar una vez más para obtener la mejor segmentación
        best_cuts, best_error, history = simulated_annealing(series, k)

        # -------------------------------------------------
        # 1️⃣ GRAFICA DE CONVERGENCIA
        # -------------------------------------------------
        plt.figure()
        plt.plot(history)
        plt.xlabel("Iteración")
        plt.ylabel("Mejor error encontrado")
        plt.title(f"Convergencia SA - {name}")
        plt.grid(True)
        plt.show()

    # -------------------------------------------------
    # 2️⃣ GRAFICA AVANZADA: SEGMENTACION + RECTAS
    # -------------------------------------------------
    
        plt.figure(figsize=(10,6))

        colors = plt.cm.tab20(np.linspace(0,1,k))

        prev = 0
        for i, cut in enumerate(best_cuts + [len(series)]):
        
            x = np.arange(prev, cut)
            y = series[prev:cut]

            # Pintar puntos del segmento
            plt.plot(x, y, color=colors[i], alpha=0.6)

            # Calcular recta ajustada
            a, b = fit_line(x, y)
            y_pred = a + b * x

            # Dibujar recta ajustada
            plt.plot(x, y_pred, color=colors[i], linewidth=2)

            prev = cut

        plt.title(f"Segmentación + Ajuste Lineal SA - {name}")
        plt.xlabel("Tiempo")
        plt.ylabel("Valor")
        plt.grid(True)
        plt.show()