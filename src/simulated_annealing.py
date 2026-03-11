# ==========================================================
# PRACTICA 1 - METAHEURISTICAS
# Segmentación de series temporales desde archivos .txt
# ==========================================================

import numpy as np
import matplotlib.pyplot as plt
import time
import math
import os
import argparse


# ==========================================================
# CARGAR SERIE DESDE ARCHIVO .txt
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
# ERROR DE UN SEGMENTO (REGRESION LINEAL + MSE)
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


def is_valid_cuts(cuts, N, min_len=2):
    """
    Comprueba si los cortes generan segmentos validos.
    Todos los segmentos deben tener al menos min_len puntos.
    """
    if any(c <= 0 or c >= N for c in cuts):
        return False

    if cuts != sorted(cuts):
        return False

    prev = 0
    for c in cuts:
        if c - prev < min_len:
            return False
        prev = c

    if N - prev < min_len:
        return False

    return True

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
# FUNCION OBJETIVO
# MSE global ponderado por tamano de segmento
# ==========================================================

def objective_function(series, cuts, min_len=2):
    N = len(series)
    if not is_valid_cuts(cuts, N, min_len=min_len):
        return float("inf")

    prev = 0
    total_sse = 0.0
    total_points = 0

    for c in cuts:
        x = np.arange(prev, c)
        y = series[prev:c]
        n = len(y)
        if n < min_len:
            return float("inf")
        mse = segment_error(x, y)
        total_sse += mse * n
        total_points += n
        prev = c

    # Último segmento
    x = np.arange(prev, len(series))
    y = series[prev:]
    n = len(y)
    if n < min_len:
        return float("inf")
    mse = segment_error(x, y)
    total_sse += mse * n
    total_points += n

    return total_sse / total_points if total_points > 0 else float("inf")


# ==========================================================
# SOLUCION ALEATORIA INICIAL
# ==========================================================

def random_solution(N, k, min_len=2, max_tries=1000):
    if k * min_len > N:
        raise ValueError("No hay suficientes puntos para crear segmentos validos")

    for _ in range(max_tries):
        cuts = sorted(np.random.choice(range(1, N - 1), k - 1, replace=False))
        if is_valid_cuts(cuts, N, min_len=min_len):
            return cuts

    raise RuntimeError("No se pudo generar una solucion inicial valida")


# ==========================================================
# GENERADOR DE VECINO
# ==========================================================

def generate_neighbor(cuts, N, min_len=2, max_tries=30):
    if len(cuts) == 0:
        return cuts.copy()

    # Vecindario tipo HC simple: mover un corte aleatorio dentro de un rango +/- 1% de N.
    range_size = max(1, int(N * 0.01))

    for _ in range(max_tries):
        new_cuts = cuts.copy()
        idx = np.random.randint(len(new_cuts))
        move = np.random.randint(-range_size, range_size + 1)

        new_cuts[idx] += move
        new_cuts[idx] = max(1, min(N - 2, new_cuts[idx]))
        new_cuts = sorted(new_cuts)

        if is_valid_cuts(new_cuts, N, min_len=min_len):
            return new_cuts

    return cuts.copy()

# ==========================================================
# Modos de enfriamiento para Simulated Annealing
# ==========================================================

def get_cooling_schedule_name(cooling_type):
    names = {
        "linear": "Lineal",
        "geometric": "Geometrica/Exponencial",
        "logarithmic": "Logaritmica/Boltzmann",
        "cauchy": "Cauchy",
        "cauchy_modified": "Cauchy modificado",
    }
    return names.get(cooling_type, cooling_type)


def select_cooling_menu():
    """
    Menu interactivo para elegir el tipo de enfriamiento.
    """
    options = {
        "1": "linear",
        "2": "geometric",
        "3": "logarithmic",
        "4": "cauchy",
        "5": "cauchy_modified",
        "6": "ALL",
    }

    print("\nSelecciona el tipo de enfriamiento:")
    print("1) Lineal")
    print("2) Geometrica o exponencial")
    print("3) Logaritmica o de Boltzmann")
    print("4) Cauchy")
    print("5) Cauchy modificado")
    print("6) ALL (todos)")

    while True:
        choice = input("Opcion [1-6]: ").strip()
        if choice in options:
            return options[choice]
        print("Opcion invalida. Introduce un numero entre 1 y 6.")


def select_series_menu(k_values):
    """
    Menu interactivo para elegir la serie a ejecutar.
    """
    print("\nSelecciona la serie de datos:")
    print("1) TS1")
    print("2) TS2")
    print("3) TS3")
    print("4) TS4")
    print("5) ALL (todas)")

    options = {
        "1": "TS1",
        "2": "TS2",
        "3": "TS3",
        "4": "TS4",
        "5": "ALL",
    }

    while True:
        choice = input("Opcion [1-5]: ").strip()
        if choice in options:
            selected = options[choice]
            if selected == "ALL":
                return list(k_values.keys())
            return [selected]
        print("Opcion invalida. Introduce un numero entre 1 y 5.")


def get_cooling_parameters(cooling_type, T0, Tf, max_iter, alpha=None):
    """
    Calcula parametros efectivos por esquema para que el enfriamiento
    quede calibrado respecto a (T0, Tf, max_iter).
    """
    max_iter_eff = max(1, max_iter)

    if cooling_type == "linear":
        beta_linear = (T0 - Tf) / max_iter_eff
        return {"beta_linear": beta_linear}

    if cooling_type == "geometric":
        alpha_eff = alpha if alpha is not None else (Tf / T0) ** (1.0 / max_iter_eff)
        return {"alpha_eff": alpha_eff}

    if cooling_type == "logarithmic":
        # T_i = T0 / (1 + c * log(1 + i)), ajustando c para que T_M ~= Tf.
        denom = max(math.log(1.0 + max_iter_eff), 1e-12)
        c_log = (T0 / Tf - 1.0) / denom
        return {"c_log": c_log}

    if cooling_type == "cauchy":
        # T_i = T0 / (1 + gamma * i), ajustando gamma para que T_M ~= Tf.
        gamma_cauchy = (T0 / Tf - 1.0) / max_iter_eff
        return {"gamma_cauchy": gamma_cauchy}

    if cooling_type == "cauchy_modified":
        beta_mod = (T0 - Tf) / (max_iter_eff * T0 * Tf)
        return {"beta_mod": beta_mod}

    raise ValueError(f"Tipo de enfriamiento no soportado: {cooling_type}")


def update_temperature(cooling_type, T, T0, iteration, max_iter, cooling_params=None, Tf=1.0):
    """
    Actualiza la temperatura segun el esquema de enfriamiento seleccionado.
    iteration empieza en 1.
    """
    params = cooling_params if cooling_params is not None else {}

    if cooling_type == "linear":
        # T = T0 - i * beta, calibrado para llegar a Tf al final.
        beta_linear = params.get("beta_linear", (T0 - Tf) / max(1, max_iter))
        T = T0 - iteration * beta_linear
        return max(Tf, T)

    if cooling_type == "geometric":
        # T_i = alpha^i * T0. Si alpha no se pasa, se calibra para llegar a Tf.
        alpha_eff = params.get("alpha_eff", (Tf / T0) ** (1.0 / max(1, max_iter)))
        return max(Tf, (alpha_eff ** iteration) * T0)

    if cooling_type == "logarithmic":
        # T = T0 / (1 + c * log(1 + i)); i >= 1
        c_log = params.get("c_log", 1.0)
        return max(Tf, T0 / (1.0 + c_log * math.log(1.0 + iteration)))

    if cooling_type == "cauchy":
        # T = T0 / (1 + gamma * i)
        gamma_cauchy = params.get("gamma_cauchy", 1.0)
        return max(Tf, T0 / (1.0 + gamma_cauchy * iteration))

    if cooling_type == "cauchy_modified":
        # T = T / (1 + beta * T), beta = (T0 - Tf)/(M * T0 * Tf)
        beta_mod = params.get("beta_mod", (T0 - Tf) / (max(1, max_iter) * T0 * Tf))
        return T / (1.0 + beta_mod * T)

    raise ValueError(f"Tipo de enfriamiento no soportado: {cooling_type}")


# ==========================================================
# SIMULATED ANNEALING
# ==========================================================

def simulated_annealing(
    series,
    k,
    T0=100,
    alpha=None,
    Tf=0.01,
    max_iter=3000,
    cooling_type="geometric",
):
    if T0 <= 0:
        raise ValueError("T0 debe ser mayor que 0")
    if Tf <= 0 or Tf >= T0:
        raise ValueError("Tf debe cumplir 0 < Tf < T0")
    if cooling_type == "geometric" and alpha is not None and not (0 < alpha < 1):
        raise ValueError("Para enfriamiento geometrico, alpha debe cumplir 0 < alpha < 1")

    cooling_params = get_cooling_parameters(
        cooling_type=cooling_type,
        T0=T0,
        Tf=Tf,
        max_iter=max_iter,
        alpha=alpha,
    )

    N = len(series)
    cuts = random_solution(N, k)
    current_error = objective_function(series, cuts)

    best_cuts = cuts.copy()
    best_error = current_error

    T = T0
    history = []
    min_temperature = 1e-12

    for i in range(1, max_iter + 1):
        new_cuts = generate_neighbor(cuts, N)
        new_error = objective_function(series, new_cuts)

        if new_error < current_error:
            cuts = new_cuts
            current_error = new_error
        else:
            if T > min_temperature:
                prob = math.exp(-(new_error - current_error) / T)
                if np.random.rand() < prob:
                    cuts = new_cuts
                    current_error = new_error

        if current_error < best_error:
            best_cuts = cuts.copy()
            best_error = current_error

        history.append(best_error)
        T = update_temperature(
            cooling_type=cooling_type,
            T=T,
            T0=T0,
            iteration=i,
            max_iter=max_iter,
            cooling_params=cooling_params,
            Tf=Tf,
        )
        if T <= min_temperature:
            T = min_temperature

    return best_cuts, best_error, history


# ==========================================================
# EVALUACION ESTADISTICA
# ==========================================================

def evaluate_method(method, series, k, runs=1, **method_kwargs):
    mse_results = []
    times = []
    best_overall_mse = float("inf")
    best_overall_cuts = None
    best_overall_history = []
    best_overall_time = None

    for run_idx in range(1, runs + 1):
        start = time.time()
        cuts, best_mse, history = method(series, k, **method_kwargs)
        end = time.time()
        elapsed = end - start

        mse_results.append(best_mse)
        times.append(elapsed)

        print(
            f"Run {run_idx}/{runs} | MSE={best_mse:.6f} | Tiempo={elapsed:.4f}s",
            flush=True,
        )

        if best_mse < best_overall_mse:
            best_overall_mse = best_mse
            best_overall_cuts = cuts.copy()
            best_overall_history = history.copy()
            best_overall_time = elapsed

    return (
        mse_results,
        times,
        best_overall_cuts,
        best_overall_mse,
        best_overall_history,
        best_overall_time,
    )


# ==========================================================
# EJECUCION SOBRE TS1-TS4
# ==========================================================

if __name__ == "__main__":

    available_cooling_types = ["linear", "geometric", "logarithmic", "cauchy", "cauchy_modified"]

    k_values = {
        "TS1": 9,
        "TS2": 10,
        "TS3": 20,
        "TS4": 50
    }

    folder = "../series"
    output_dir = os.path.join("..", "graficas", "analisis", "SA")
    os.makedirs(output_dir, exist_ok=True)

    parser = argparse.ArgumentParser(
        description="Ejecuta Simulated Annealing con menus de enfriamiento y serie"
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Numero de ejecuciones para estadísticas (fijado a 1 en este script)"
    )
    parser.add_argument(
        "--t0",
        type=float,
        default=100.0,
        help="Temperatura inicial T0"
    )
    parser.add_argument(
        "--tf",
        type=float,
        default=0.01,
        help="Temperatura final objetivo Tf (0 < Tf < T0)"
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=3000,
        help="Numero maximo de iteraciones de SA"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Alpha para enfriamiento geometrico (si no se pasa, se calibra automaticamente)"
    )
    parser.add_argument(
        "--cooling",
        choices=["linear", "geometric", "logarithmic", "cauchy", "cauchy_modified", "all"],
        default=None,
        help="Tipo de enfriamiento o 'all' para ejecutar todos (si no se indica, se pide por menu)"
    )
    args = parser.parse_args()

    cooling_selection = args.cooling if args.cooling else select_cooling_menu()
    if cooling_selection in ["all", "ALL"]:
        selected_cooling_types = available_cooling_types
    else:
        selected_cooling_types = [cooling_selection]

    cooling_names = [get_cooling_schedule_name(c) for c in selected_cooling_types]
    print(f"\nEnfriamiento(s) seleccionado(s): {', '.join(cooling_names)}")

    if args.tf <= 0 or args.tf >= args.t0:
        raise ValueError("Parametros invalidos: debe cumplirse 0 < --tf < --t0")

    selected_series = select_series_menu(k_values)
    print(f"Series seleccionadas: {', '.join(selected_series)}")

    for name in selected_series:
        k = k_values[name]

        print(f"\n================ {name} =================")

        filepath = os.path.join(folder, name + ".txt")
        series = load_series(filepath)

        for cooling_type in selected_cooling_types:
            cooling_output_dir = os.path.join(output_dir, cooling_type)
            os.makedirs(cooling_output_dir, exist_ok=True)

            print(f"\n--- Enfriamiento: {get_cooling_schedule_name(cooling_type)} ---")
            cooling_params = get_cooling_parameters(
                cooling_type=cooling_type,
                T0=args.t0,
                Tf=args.tf,
                max_iter=args.max_iter,
                alpha=args.alpha,
            )

            if cooling_type == "geometric":
                print(f"Alpha geometrico usado: {cooling_params['alpha_eff']:.6f}")
            elif cooling_type == "linear":
                print(f"Beta lineal usado: {cooling_params['beta_linear']:.6f}")
            elif cooling_type == "logarithmic":
                print(f"c logaritmico usado: {cooling_params['c_log']:.6f}")
            elif cooling_type == "cauchy":
                print(f"Gamma Cauchy usado: {cooling_params['gamma_cauchy']:.6f}")
            elif cooling_type == "cauchy_modified":
                print(f"Beta Cauchy modificado usado: {cooling_params['beta_mod']:.6f}")

            # Ejecutar múltiples veces SA para estadísticas
            all_mse, all_times, best_cuts, best_error, history, best_time = evaluate_method(
                simulated_annealing,
                series,
                k,
                runs=max(1, args.runs),
                cooling_type=cooling_type,
                T0=args.t0,
                Tf=args.tf,
                alpha=args.alpha,
                max_iter=args.max_iter,
            )

            mse_global = float(best_error)
            tiempo_ejecucion = float(best_time)

            print("Simulated Annealing")
            print(f"Mejor MSE global (entre {max(1, args.runs)} runs): {mse_global:.6f}")
            print(f"Tiempo de ejecucion del mejor run: {tiempo_ejecucion:.6f} s")

            # -------------------------------------------------
            # GRAFICA DE CONVERGENCIA
            # -------------------------------------------------
            plt.figure()
            plt.plot(history)
            plt.xlabel("Iteración")
            plt.ylabel("Mejor MSE encontrado")
            plt.title(f"Convergencia SA - {name} ({get_cooling_schedule_name(cooling_type)})")
            plt.grid(True)
            convergence_file = os.path.join(cooling_output_dir, f"convergencia_{name}_{cooling_type}.png")
            plt.savefig(convergence_file)
            print(f"Gráfica de convergencia guardada como: {convergence_file}")
            plt.show()

            # -------------------------------------------------
            # GRAFICA AVANZADA: SEGMENTACION + RECTAS
            # -------------------------------------------------

            plt.figure(figsize=(10,6))

            colors = plt.cm.tab20(np.linspace(0,1,k))

            # Serie original en gris para separar claramente datos y aproximaciones.
            x_full = np.arange(len(series))
            plt.plot(x_full, series, color="gray", alpha=0.45, linewidth=1.2, label="Serie original")

            prev = 0
            for i, cut in enumerate(best_cuts + [len(series)]):

                x = np.arange(prev, cut)
                y = series[prev:cut]

                # Calcular recta ajustada
                a, b = fit_line(x, y)
                y_pred = a + b * x

                # Dibujar recta ajustada
                plt.plot(x, y_pred, color=colors[i], linewidth=2.5)

                # Marcar cortes con color por segmento para distinguirlos mejor.
                if cut < len(series):
                    plt.axvline(cut, color=colors[i], linestyle="--", linewidth=1.2, alpha=0.8)

                prev = cut

            plt.title(
                f"Segmentación + Ajuste Lineal SA - {name} ({get_cooling_schedule_name(cooling_type)}) | MSE={best_error:.6f}"
            )

            output_file = os.path.join(cooling_output_dir, f"resultado_{name}_{cooling_type}.png")
            plt.savefig(output_file)
            print(f"Gráfica guardada como: {output_file}")
            plt.xlabel("Tiempo")
            plt.ylabel("Valor")
            plt.grid(True)
            plt.show()