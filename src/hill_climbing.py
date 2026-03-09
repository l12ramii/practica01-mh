import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import time
import random
import sys
import os
import argparse

MIN_SEG_LEN = 2
# Cambia solo esta constante para activar/desactivar el arranque aleatorio por defecto.
HC_INICIO_ALEATORIO_DEFAULT = True


def construir_estadisticas_serie(serie):
    """Precalcula sumas acumuladas para evaluar segmentos en O(1)."""
    n = len(serie)
    x = np.arange(n, dtype=np.float64)
    y = serie.astype(np.float64)

    pref_y = np.zeros(n + 1, dtype=np.float64)
    pref_y2 = np.zeros(n + 1, dtype=np.float64)
    pref_xy = np.zeros(n + 1, dtype=np.float64)

    pref_y[1:] = np.cumsum(y)
    pref_y2[1:] = np.cumsum(y * y)
    pref_xy[1:] = np.cumsum(x * y)

    return {
        "pref_y": pref_y,
        "pref_y2": pref_y2,
        "pref_xy": pref_xy,
    }


def suma_cuadrados_hasta(m):
    """Devuelve 1^2 + ... + m^2 (si m<=0, devuelve 0)."""
    if m <= 0:
        return 0.0
    m = float(m)
    return m * (m + 1.0) * (2.0 * m + 1.0) / 6.0


def mse_segmento_analitico(inicio, fin, stats):
    """Calcula MSE de regresion lineal en [inicio, fin) sin entrenar modelos."""
    n = fin - inicio
    if n < 2:
        return float("inf")

    pref_y = stats["pref_y"]
    pref_y2 = stats["pref_y2"]
    pref_xy = stats["pref_xy"]

    sy = pref_y[fin] - pref_y[inicio]
    syy = pref_y2[fin] - pref_y2[inicio]
    sxy = pref_xy[fin] - pref_xy[inicio]

    # Sumas de x y x^2 sobre indices absolutos [inicio, fin).
    sx = 0.5 * (inicio + (fin - 1)) * n
    sxx = suma_cuadrados_hasta(fin - 1) - suma_cuadrados_hasta(inicio - 1)

    denom = n * sxx - sx * sx
    if abs(denom) < 1e-12:
        media = sy / n
        return (syy - 2.0 * media * sy + n * media * media) / n

    b = (n * sxy - sx * sy) / denom
    a = (sy - b * sx) / n

    # SSE = y^T y - beta^T X^T y, con beta=[a,b], X^T y=[sy,sxy].
    sse = syy - (a * sy + b * sxy)
    if sse < 0 and sse > -1e-9:
        sse = 0.0
    return sse / n

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


def es_solucion_valida(cortes, n_puntos, min_len=MIN_SEG_LEN):
    """Comprueba que los cortes generen segmentos validos y no vacios."""
    if not cortes:
        return n_puntos >= min_len

    cortes_ordenados = sorted(cortes)

    # Sin repetidos y dentro de rango.
    if len(set(cortes_ordenados)) != len(cortes_ordenados):
        return False
    if any(c <= 0 or c >= n_puntos for c in cortes_ordenados):
        return False

    # Longitud minima por segmento.
    prev = 0
    for c in cortes_ordenados:
        if c - prev < min_len:
            return False
        prev = c
    if n_puntos - prev < min_len:
        return False

    return True


def generar_solucion_inicial(
    n_puntos,
    k,
    min_len=MIN_SEG_LEN,
    max_intentos=2000,
    inicio_aleatorio=HC_INICIO_ALEATORIO_DEFAULT,
):
    """Genera una solucion inicial valida.

    - inicio_aleatorio=False: intenta primero una particion casi uniforme (determinista).
    - inicio_aleatorio=True: fuerza arranque aleatorio para favorecer variabilidad entre runs.
    """
    if k * min_len > n_puntos:
        raise ValueError("No hay suficientes puntos para generar segmentos validos")

    if k <= 1:
        return []

    if not inicio_aleatorio:
        # Arranque estable: cortes cerca de i * N / k, respetando longitud minima.
        cortes_uniformes = []
        prev = 0
        for i in range(1, k):
            objetivo = int(round(i * n_puntos / k))
            seg_restantes = k - i
            c_min = prev + min_len
            c_max = n_puntos - seg_restantes * min_len
            corte = max(c_min, min(objetivo, c_max))
            cortes_uniformes.append(corte)
            prev = corte

        if es_solucion_valida(cortes_uniformes, n_puntos, min_len=min_len):
            return cortes_uniformes

    for _ in range(max_intentos):
        cortes = sorted(random.sample(range(1, n_puntos), k - 1))
        if es_solucion_valida(cortes, n_puntos, min_len=min_len):
            return cortes

    raise RuntimeError("No se pudo generar una solucion inicial valida")

def evaluar_solucion(cortes, serie, stats=None):
    """Calcula el MSE global ponderado por tamano de segmento."""
    N = len(serie)

    if not es_solucion_valida(cortes, N, min_len=MIN_SEG_LEN):
        return float("inf")

    if stats is None:
        stats = construir_estadisticas_serie(serie)

    puntos = [0] + sorted(cortes) + [N]
    sse_total = 0.0

    for i in range(len(puntos) - 1):
        inicio, fin = puntos[i], puntos[i+1]
        mse_seg = mse_segmento_analitico(inicio, fin, stats)
        sse_total += mse_seg * (fin - inicio)

    return sse_total / N

# ==============================
# HILL CLIMBING SIMPLE
# ==============================

def hill_climbing_simple(serie, k, iteraciones, inicio_aleatorio=HC_INICIO_ALEATORIO_DEFAULT):
    N = len(serie)
    stats = construir_estadisticas_serie(serie)
    evaluar = lambda cortes: evaluar_solucion(cortes, serie, stats)

    actual = generar_solucion_inicial(N, k, inicio_aleatorio=inicio_aleatorio)
    error_actual = evaluar(actual)
    historial_errores = []
    start_time = time.time()
    rango = max(1, int(N * 0.01))

    # k=1 implica un unico segmento (sin cortes), no hay vecindad que explorar.
    if len(actual) == 0:
        historial_errores = [error_actual] * max(1, iteraciones)
        tiempo_total = time.time() - start_time
        estadisticas = {
            "mse_min": float(np.min(historial_errores)),
            "mse_max": float(np.max(historial_errores)),
            "mse_mean": float(np.mean(historial_errores)),
            "time_min": float(tiempo_total),
            "time_max": float(tiempo_total),
            "time_mean": float(tiempo_total),
            "mse_std": float(np.std(historial_errores)),
        }
        return actual, error_actual, tiempo_total, estadisticas

    for _ in range(iteraciones):
        mejora_encontrada = False
        idxs = list(range(len(actual)))
        deltas = [d for d in range(-rango, rango + 1) if d != 0]
        random.shuffle(idxs)
        random.shuffle(deltas)

        # First-improvement: recorre vecinos y se mueve al primero que mejora.
        for idx in idxs:
            for delta in deltas:
                vecino = list(actual)
                vecino[idx] += delta
                vecino = sorted(vecino)

                if not es_solucion_valida(vecino, N, min_len=MIN_SEG_LEN):
                    continue

                error_vecino = evaluar(vecino)
                if error_vecino < error_actual:
                    actual = vecino
                    error_actual = error_vecino
                    mejora_encontrada = True
                    break
            if mejora_encontrada:
                break

        historial_errores.append(error_actual)
        if not mejora_encontrada:
            break

    tiempo_total = time.time() - start_time
    historial_mse = historial_errores if historial_errores else [error_actual]
    estadisticas = {
        "mse_min": float(np.min(historial_mse)),
        "mse_max": float(np.max(historial_mse)),
        "mse_mean": float(np.mean(historial_mse)),
        "time_min": float(tiempo_total),
        "time_max": float(tiempo_total),
        "time_mean": float(tiempo_total),
        "mse_std": float(np.std(historial_mse)),
    }
    return actual, error_actual, tiempo_total, estadisticas
    

# ==============================
# HILL CLIMBING ESTOCÁSTICO
# ==============================

def hill_climbing_estocastico(serie, k, iteraciones, inicio_aleatorio=HC_INICIO_ALEATORIO_DEFAULT):
    N = len(serie)
    stats = construir_estadisticas_serie(serie)
    evaluar = lambda cortes: evaluar_solucion(cortes, serie, stats)

    actual = generar_solucion_inicial(N, k, inicio_aleatorio=inicio_aleatorio)
    error_actual = evaluar(actual)
    historial_errores = []
    start_time = time.time()
    rango = max(1, int(N * 0.01))

    # k=1 implica un unico segmento (sin cortes), no hay vecindad que explorar.
    if len(actual) == 0:
        historial_errores = [error_actual] * max(1, iteraciones)
        tiempo_total = time.time() - start_time
        estadisticas = {
            "mse_min": float(np.min(historial_errores)),
            "mse_max": float(np.max(historial_errores)),
            "mse_mean": float(np.mean(historial_errores)),
            "time_min": float(tiempo_total),
            "time_max": float(tiempo_total),
            "time_mean": float(tiempo_total),
            "mse_std": float(np.std(historial_errores)),
        }
        return actual, error_actual, tiempo_total, estadisticas

    for _ in range(iteraciones):
        # Estocastico: selecciona un unico vecino aleatorio y decide si moverse.
        v = list(actual)
        idx = random.randint(0, len(v) - 1)
        delta = random.randint(-rango, rango)

        if delta == 0:
            historial_errores.append(error_actual)
            continue

        v[idx] += delta
        v = sorted(v)

        if es_solucion_valida(v, N, min_len=MIN_SEG_LEN):
            err_v = evaluar(v)
            if err_v < error_actual:
                actual = v
                error_actual = err_v

        # Registramos el estado tras la posible actualizacion para no sesgar la media.
        historial_errores.append(error_actual)

    tiempo_total = time.time() - start_time
    historial_mse = historial_errores if historial_errores else [error_actual]
    estadisticas = {
        "mse_min": float(np.min(historial_mse)),
        "mse_max": float(np.max(historial_mse)),
        "mse_mean": float(np.mean(historial_mse)),
        "time_min": float(tiempo_total),
        "time_max": float(tiempo_total),
        "time_mean": float(tiempo_total),
        "mse_std": float(np.std(historial_mse)),
    }
    return actual, error_actual, tiempo_total, estadisticas


# ==============================
# HILL CLIMBING MÁXIMA PENDIENTE
# ==============================

def hill_climbing_maxima_pendiente(serie, k, iteraciones, inicio_aleatorio=HC_INICIO_ALEATORIO_DEFAULT):
    N = len(serie)
    stats = construir_estadisticas_serie(serie)
    evaluar = lambda cortes: evaluar_solucion(cortes, serie, stats)

    actual = generar_solucion_inicial(N, k, inicio_aleatorio=inicio_aleatorio)
    error_actual = evaluar(actual)
    historial_errores = []
    start_time = time.time()
    rango = max(1, int(N * 0.01))

    # k=1 implica un unico segmento (sin cortes), no hay vecindad que explorar.
    if len(actual) == 0:
        historial_errores = [error_actual] * max(1, iteraciones)
        tiempo_total = time.time() - start_time
        estadisticas = {
            "mse_min": float(np.min(historial_errores)),
            "mse_max": float(np.max(historial_errores)),
            "mse_mean": float(np.mean(historial_errores)),
            "time_min": float(tiempo_total),
            "time_max": float(tiempo_total),
            "time_mean": float(tiempo_total),
            "mse_std": float(np.std(historial_errores)),
        }
        return actual, error_actual, tiempo_total, estadisticas
    
    for _ in range(iteraciones):
        mejor_vecino = None
        mejor_error = error_actual
        
        # Maxima pendiente estricta sobre la vecindad definida:
        # mover un unico corte en el rango [-rango, rango], excluyendo 0.
        vecinos_visitados = set()
        for idx in range(len(actual)):
            for delta in range(-rango, rango + 1):
                if delta == 0:
                    continue

                v_tmp = list(actual)
                v_tmp[idx] += delta
                v_tmp = sorted(v_tmp)

                clave = tuple(v_tmp)
                if clave in vecinos_visitados:
                    continue
                vecinos_visitados.add(clave)

                if not es_solucion_valida(v_tmp, N, min_len=MIN_SEG_LEN):
                    continue

                error_vecino = evaluar(v_tmp)
                if error_vecino < mejor_error:
                    mejor_error = error_vecino
                    mejor_vecino = v_tmp

        if mejor_vecino:
            actual = mejor_vecino
            error_actual = mejor_error
        else:
            historial_errores.append(error_actual)
            break

        # Registramos el estado tras aplicar (o no) la mejor mejora del paso.
        historial_errores.append(error_actual)

    tiempo_total = time.time() - start_time
    historial_mse = historial_errores if historial_errores else [error_actual]
    estadisticas = {
        "mse_min": float(np.min(historial_mse)),
        "mse_max": float(np.max(historial_mse)),
        "mse_mean": float(np.mean(historial_mse)),
        "time_min": float(tiempo_total),
        "time_max": float(tiempo_total),
        "time_mean": float(tiempo_total),
        "mse_std": float(np.std(historial_mse)),
    }
    return actual, error_actual, tiempo_total, estadisticas


def ejecutar_runs_hc(fn_hc, serie, k, iteraciones, runs, inicio_aleatorio=HC_INICIO_ALEATORIO_DEFAULT):
    """Ejecuta una variante HC varias veces y devuelve la mejor ejecucion."""
    mejor_cortes = None
    mejor_error = float("inf")
    mejor_tiempo = None
    mejor_stats = None

    for run_idx in range(1, runs + 1):
        c, e, t, stats = fn_hc(serie, k, iteraciones, inicio_aleatorio=inicio_aleatorio)
        print(
            f"Run {run_idx}/{runs} | MSE={e:.6f} | Tiempo={t:.4f}s",
            flush=True,
        )
        if e < mejor_error:
            mejor_cortes = c
            mejor_error = e
            mejor_tiempo = t
            mejor_stats = stats

    return mejor_cortes, mejor_error, mejor_tiempo, mejor_stats

# ==============================
# BLOQUE PRINCIPAL (MAIN)
# ==============================

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    series_dir = os.path.join(base_dir, "..", "series")
    series_map = {
        "TS1": os.path.join(series_dir, "TS1.txt"),
        "TS2": os.path.join(series_dir, "TS2.txt"),
        "TS3": os.path.join(series_dir, "TS3.txt"),
        "TS4": os.path.join(series_dir, "TS4.txt"),
    }
    k_map = {"TS1": 9, "TS2": 10, "TS3": 20, "TS4": 50}
    parser = argparse.ArgumentParser(description="Hill Climbing para segmentacion")
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
        help="Numero de ejecuciones completas por cada variante",
    )
    args = parser.parse_args()

    ITERACIONES = args.iteraciones
    RUNS = max(1, args.runs)

    print(f"\nIteraciones: {ITERACIONES}")
    print("Elige la variante de Hill Climbing:")
    print("1 - Simple\n2 - Estocástico\n3 - Máxima Pendiente\n4 - Todas")
    opcion = input("Opción: ")

    if opcion not in ["1", "2", "3", "4"]:
        print("Opción de Hill Climbing no válida")
        sys.exit(1)

    print("\nElige la serie de datos:")
    print("1 - TS1\n2 - TS2\n3 - TS3\n4 - TS4\n5 - Todas")
    opcion_serie = input("Opción: ")

    if opcion_serie == "1":
        series_a_procesar = ["TS1"]
    elif opcion_serie == "2":
        series_a_procesar = ["TS2"]
    elif opcion_serie == "3":
        series_a_procesar = ["TS3"]
    elif opcion_serie == "4":
        series_a_procesar = ["TS4"]
    elif opcion_serie == "5":
        series_a_procesar = ["TS1", "TS2", "TS3", "TS4"]
    else:
        print("Opción de serie no válida")
        sys.exit(1)

    for nombre_serie in series_a_procesar:
        ruta_archivo = series_map[nombre_serie]
        serie = cargar_serie(ruta_archivo)
        if serie is None:
            print(f"No se pudo cargar {ruta_archivo}")
            continue

        k_objetivo = k_map[nombre_serie]
        print(
            f"\nArchivo: {ruta_archivo} | K: {k_objetivo} | "
            f"Iteraciones: {ITERACIONES} | Runs: {RUNS}"
        )

        resultados = []
        if opcion in ["1", "4"]:
            print("\n[Simple]")
            c, e, t, stats = ejecutar_runs_hc(
                hill_climbing_simple,
                serie,
                k_objetivo,
                ITERACIONES,
                RUNS,
            )
            resultados.append(("Simple", c, e, t, stats))
        if opcion in ["2", "4"]:
            print("\n[Estocástico]")
            c, e, t, stats = ejecutar_runs_hc(
                hill_climbing_estocastico,
                serie,
                k_objetivo,
                ITERACIONES,
                RUNS,
            )
            resultados.append(("Estocástico", c, e, t, stats))
        if opcion in ["3", "4"]:
            print("\n[Máxima Pendiente]")
            c, e, t, stats = ejecutar_runs_hc(
                hill_climbing_maxima_pendiente,
                serie,
                k_objetivo,
                ITERACIONES,
                RUNS,
            )
            resultados.append(("Máxima Pendiente", c, e, t, stats))

        for nombre, cortes, error, tiempo, stats in resultados:
            print(f"\n--- {nombre_serie} | {nombre} ---")
            print(f"Mejor MSE global (entre {RUNS} runs): {error:.6f}")
            print(f"Tiempo de ejecucion del mejor run: {tiempo:.6f} s")

            # Visualización
            plt.figure(figsize=(10, 5))
            x_full = np.arange(len(serie))
            plt.plot(x_full, serie, color="gray", alpha=0.45, linewidth=1.2, label="Serie original")
            puntos = [0] + cortes + [len(serie)]
            colors = plt.cm.tab20(np.linspace(0, 1, max(1, len(puntos) - 1)))
            for i in range(len(puntos)-1):
                idx = np.arange(puntos[i], puntos[i+1])
                reg = LinearRegression().fit(idx.reshape(-1,1), serie[idx])
                plt.plot(idx, reg.predict(idx.reshape(-1,1)), color=colors[i], linewidth=2.5)
                if puntos[i+1] < len(serie):
                    plt.axvline(puntos[i+1], color=colors[i], linestyle="--", linewidth=1.2, alpha=0.8)
            plt.title(f"{nombre} - {nombre_serie} (MSE: {error:.4f})")
            plt.legend()

            # Guardar grafica en ../graficas/analisis/HC/<tipo_hc>/<serie>.png
            out_dir = os.path.join(base_dir, "..", "graficas", "analisis", "HC", nombre)
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"{nombre_serie}.png")
            plt.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"Gráfica guardada en: {out_path}")