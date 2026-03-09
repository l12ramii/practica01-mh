import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import time
import random
import sys
import os
import argparse


def seleccionar_series_menu():
    """Menu interactivo para elegir serie(s) TS1-TS4."""
    print("\nSelecciona la serie de datos:")
    print("1) TS1")
    print("2) TS2")
    print("3) TS3")
    print("4) TS4")
    print("5) ALL (todas)")

    opciones = {
        "1": ["TS1"],
        "2": ["TS2"],
        "3": ["TS3"],
        "4": ["TS4"],
        "5": ["TS1", "TS2", "TS3", "TS4"],
    }

    while True:
        eleccion = input("Opcion [1-5]: ").strip()
        if eleccion in opciones:
            return opciones[eleccion]
        print("Opcion invalida. Introduce un numero entre 1 y 5.")

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
    """Calcula el MSE de una regresión lineal en un segmento"""
    if len(x) < 2: return 1e9 
    model = LinearRegression().fit(x.reshape(-1, 1), y)
    predicciones = model.predict(x.reshape(-1, 1))
    return np.mean((y - predicciones) ** 2)

def evaluar_solucion(cortes, serie):
    """Calcula el MSE global de toda la serie (ponderado por tamaño de segmento)."""
    N = len(serie)
    puntos = [0] + sorted(cortes) + [N]

    sse_total = 0.0  # Sum of Squared Errors total

    for i in range(len(puntos) - 1):
        inicio, fin = puntos[i], puntos[i + 1]
        y_seg = serie[inicio:fin]
        x_seg = np.arange(inicio, fin)

        mse_seg = calcular_mse_segmento(x_seg, y_seg)
        sse_total += mse_seg * len(y_seg)

    return sse_total / N

def random_search(serie, k, iteraciones):
    """Algoritmo de búsqueda aleatoria"""
    N = len(serie)
    mejor_error = float('inf')
    mejor_cortes = None
    start_time = time.time()
    
    for _ in range(iteraciones):
        cortes_actuales = random.sample(range(1, N), k - 1)
        error_actual = evaluar_solucion(cortes_actuales, serie)
        if error_actual < mejor_error:
            mejor_error = error_actual
            mejor_cortes = sorted(cortes_actuales)
            
    duracion = time.time() - start_time
    return mejor_cortes, mejor_error, duracion


def ejecutar_runs_random_search(serie, k, iteraciones, runs):
    """Ejecuta Random Search varias veces y devuelve la mejor ejecucion."""
    mejor_global_error = float("inf")
    mejor_global_cortes = None
    mejor_global_tiempo = None

    for run_idx in range(1, runs + 1):
        cortes, error, duracion = random_search(serie, k, iteraciones)
        print(
            f"Run {run_idx}/{runs} | MSE={error:.6f} | Tiempo={duracion:.4f}s",
            flush=True,
        )
        if error < mejor_global_error:
            mejor_global_error = error
            mejor_global_cortes = cortes
            mejor_global_tiempo = duracion

    return mejor_global_cortes, mejor_global_error, mejor_global_tiempo

if __name__ == "__main__":
    # Mapeo de valores de K según el enunciado 
    config_k = {"TS1": 9, "TS2": 10, "TS3": 20, "TS4": 50}

    base_dir = os.path.dirname(os.path.abspath(__file__))
    series_dir = os.path.join(base_dir, "..", "series")
    series_seleccionadas = seleccionar_series_menu()

    parser = argparse.ArgumentParser(description="Random Search para segmentacion")
    parser.add_argument(
        "--iteraciones",
        type=int,
        default=1000,
        help="Numero de iteraciones internas por cada run",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Numero de ejecuciones completas del algoritmo",
    )
    args = parser.parse_args()

    ITERACIONES = args.iteraciones
    RUNS = max(1, args.runs)
    print(f"\nSeries seleccionadas: {', '.join(series_seleccionadas)}")

    for nombre_serie in series_seleccionadas:
        ruta_archivo = os.path.join(series_dir, f"{nombre_serie}.txt")
        k_objetivo = config_k[nombre_serie]

        serie = cargar_serie(ruta_archivo)
        if serie is None:
            print(f"No se pudo cargar la serie: {ruta_archivo}")
            continue

        print(f"\n--- Iniciando Random Search ({nombre_serie}) ---")
        print(
            f"Archivo: {ruta_archivo} | K: {k_objetivo} | "
            f"Iteraciones: {ITERACIONES} | Runs: {RUNS}"
        )

        cortes, error, duracion = ejecutar_runs_random_search(
            serie,
            k_objetivo,
            ITERACIONES,
            RUNS,
        )

        print(f"\nRESULTADOS:")
        print(f"- Mejor MSE Global (entre {RUNS} runs): {error:.6f}")
        print(f"- Tiempo de Cómputo del mejor run: {duracion:.4f} segundos")

        # Generar grafica con el mismo estilo visual que SA/HC.
        plt.figure(figsize=(12, 6))
        x_full = np.arange(len(serie))
        plt.plot(x_full, serie, color="gray", alpha=0.45, linewidth=1.2, label="Serie original")
        puntos_completos = [0] + cortes + [len(serie)]
        colors = plt.cm.tab20(np.linspace(0, 1, max(1, len(puntos_completos) - 1)))
        for i in range(len(puntos_completos)-1):
            idx = np.arange(puntos_completos[i], puntos_completos[i+1])
            reg = LinearRegression().fit(idx.reshape(-1, 1), serie[idx])
            plt.plot(idx, reg.predict(idx.reshape(-1, 1)), color=colors[i], linewidth=2.5)
            if puntos_completos[i + 1] < len(serie):
                plt.axvline(puntos_completos[i + 1], color=colors[i], linestyle="--", linewidth=1.2, alpha=0.8)

        plt.title(f"Random Search: {nombre_serie} (k={k_objetivo}) | MSE global: {error:.6f}")
        plt.legend()

        output_dir = os.path.join(base_dir, "..", "graficas", "analisis", "RS")
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"resultado_{nombre_serie}.png")
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Gráfica guardada como: {output_file}")