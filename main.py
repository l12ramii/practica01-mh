import os
import sys
import time
from datetime import datetime

import matplotlib
import numpy as np


# Backend no interactivo para guardar figuras sin bloquear por GUI.
matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")
SERIES_DIR = os.path.join(BASE_DIR, "series")
PERF_DIR = os.path.join(BASE_DIR, "graficas", "rendimiento")

if SRC_DIR not in sys.path:
	sys.path.insert(0, SRC_DIR)

from random_search import random_search as run_random_search  # noqa: E402
from hill_climbing import (  # noqa: E402
	hill_climbing_estocastico,
	hill_climbing_maxima_pendiente,
	hill_climbing_simple,
)
from simulated_annealing import simulated_annealing  # noqa: E402


K_MAP = {
	"TS1": 9,
	"TS2": 10,
	"TS3": 20,
	"TS4": 50,
}

HC_VARIANTS = {
	"simple": hill_climbing_simple,
	"estocastico": hill_climbing_estocastico,
	"maxima_pendiente": hill_climbing_maxima_pendiente,
}

SA_COOLINGS = ["linear", "geometric", "logarithmic", "cauchy", "cauchy_modified"]


def cargar_serie(nombre_serie):
	file_path = os.path.join(SERIES_DIR, f"{nombre_serie}.txt")
	with open(file_path, "r", encoding="utf-8") as f:
		contenido = f.read().strip().replace("[", "").replace("]", "")
	serie = np.fromstring(contenido.replace(",", " "), sep=" ")
	if serie.size == 0:
		raise ValueError(f"La serie {nombre_serie} esta vacia o no tiene formato valido")
	return serie


def pedir_opcion(titulo, opciones):
	print(f"\n{titulo}")
	for key, label in opciones:
		print(f"{key}) {label}")

	validas = {k for k, _ in opciones}
	while True:
		eleccion = input("Opcion: ").strip()
		if eleccion in validas:
			return eleccion
		print("Opcion invalida, prueba de nuevo.")


def pedir_entero_positivo(msg):
	while True:
		valor = input(msg).strip()
		if valor.isdigit() and int(valor) > 0:
			return int(valor)
		print("Introduce un entero positivo.")


def std_acumulada(valores):
	acumulada = []
	for i in range(1, len(valores) + 1):
		acumulada.append(float(np.std(valores[:i], ddof=0)))
	return acumulada


def ejecutar_random_search(serie, k, iteraciones=1000):
	_, mse, duracion = run_random_search(serie, k, iteraciones)
	return float(mse), float(duracion)


def ejecutar_hc(serie, k, variante, iteraciones=1000):
	fn = HC_VARIANTS[variante]
	_, mse, duracion, _ = fn(serie, k, iteraciones)
	return float(mse), float(duracion)


def ejecutar_sa(serie, k, cooling_type, max_iter=1000, t0=100.0, tf=0.01, alpha=None):
	start = time.time()
	_, mse, _ = simulated_annealing(
		serie,
		k,
		T0=t0,
		Tf=tf,
		alpha=alpha,
		max_iter=max_iter,
		cooling_type=cooling_type,
	)
	duracion = time.time() - start
	return float(mse), float(duracion)


def construir_configs(algoritmo, selector):
	if algoritmo == "random_search":
		return [("random_search", {"algo": "random_search"})]

	if algoritmo == "hill_climbing":
		if selector == "all":
			return [
				(f"hc_{v}", {"algo": "hill_climbing", "variant": v})
				for v in HC_VARIANTS
			]
		return [(f"hc_{selector}", {"algo": "hill_climbing", "variant": selector})]

	if algoritmo == "simulated_annealing":
		if selector == "all":
			return [
				(f"sa_{c}", {"algo": "simulated_annealing", "cooling": c})
				for c in SA_COOLINGS
			]
		return [(f"sa_{selector}", {"algo": "simulated_annealing", "cooling": selector})]

	if algoritmo == "all_algorithms":
		hc_variant = selector["hc"]
		sa_cooling = selector["sa"]
		return [
			("random_search", {"algo": "random_search"}),
			("hill_climbing", {"algo": "hill_climbing", "variant": hc_variant}),
			("simulated_annealing", {"algo": "simulated_annealing", "cooling": sa_cooling}),
		]

	raise ValueError("Algoritmo no soportado")


def ejecutar_runs(serie, k, configs, runs, serie_nombre):
	resultados = {}

	for config_name, config_selector in configs:
		mse_runs = []
		time_runs = []

		for idx in range(1, runs + 1):
			algo = config_selector["algo"]
			print(f"[{serie_nombre}] {config_name} - run {idx}/{runs}...", flush=True)

			if algo == "random_search":
				mse, duracion = ejecutar_random_search(serie, k)
			elif algo == "hill_climbing":
				mse, duracion = ejecutar_hc(serie, k, config_selector["variant"])
			elif algo == "simulated_annealing":
				mse, duracion = ejecutar_sa(serie, k, config_selector["cooling"])
			else:
				raise ValueError("Algoritmo no soportado")

			mse_runs.append(mse)
			time_runs.append(duracion)
			print(
				f"[{serie_nombre}] {config_name} - run {idx}/{runs} completado | "
				f"MSE={mse:.6f} | tiempo={duracion:.4f}s",
				flush=True,
			)

		resultados[config_name] = {
			"mse_runs": mse_runs,
			"time_runs": time_runs,
			"std_runs": std_acumulada(mse_runs),
			"mse_mean": float(np.mean(mse_runs)),
			"time_mean": float(np.mean(time_runs)),
			"mse_std": float(np.std(mse_runs, ddof=0)),
		}

	return resultados


def asegurar_directorios_rendimiento():
	dirs = {
		"MSE": os.path.join(PERF_DIR, "MSE"),
		"tiempo": os.path.join(PERF_DIR, "tiempo"),
		"desviacion_tipica": os.path.join(PERF_DIR, "desviacion_tipica"),
	}
	for d in dirs.values():
		os.makedirs(d, exist_ok=True)
	return dirs


def guardar_graficas(resultados, serie_nombre, runs, contexto):
	dirs = asegurar_directorios_rendimiento()
	x = np.arange(1, runs + 1)
	stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

	metricas = [
		("mse_runs", "MSE por run", "MSE", dirs["MSE"]),
		("time_runs", "Tiempo por run", "Tiempo (s)", dirs["tiempo"]),
		(
			"std_runs",
			"Desviacion tipica acumulada del error",
			"Desviacion tipica de MSE",
			dirs["desviacion_tipica"],
		),
	]

	for clave, titulo, ylabel, output_dir in metricas:
		plt.figure(figsize=(11, 5.5))
		for config_name, data in resultados.items():
			plt.plot(x, data[clave], marker="o", linewidth=1.8, label=config_name)

		plt.title(f"{titulo} - {serie_nombre} ({contexto})")
		plt.xlabel("Numero de ejecucion (run)")
		plt.ylabel(ylabel)
		plt.grid(True, alpha=0.25)
		plt.legend()

		out_file = os.path.join(output_dir, f"{serie_nombre}_{contexto}_{stamp}.png")
		plt.savefig(out_file, dpi=150, bbox_inches="tight")
		plt.close()
		print(f"Grafica guardada: {out_file}")


def imprimir_resumen(resultados, serie_nombre):
	print(f"\nResumen estadistico para {serie_nombre}:")
	ranking = []

	for config_name, data in resultados.items():
		print(f"- {config_name}")
		print(f"  media MSE: {data['mse_mean']:.6f}")
		print(f"  media tiempo: {data['time_mean']:.6f} s")
		print(f"  desviacion tipica del error (MSE): {data['mse_std']:.6f}")
		ranking.append((data["mse_mean"], config_name))

	ranking.sort(key=lambda x: x[0])
	mejor = ranking[0][1]
	print(f"Mejor configuracion por media MSE en {serie_nombre}: {mejor}")


def seleccionar_series():
	opcion = pedir_opcion(
		"1- Elige la serie de datos",
		[
			("1", "TS1"),
			("2", "TS2"),
			("3", "TS3"),
			("4", "TS4"),
			("5", "ALL (todas)"),
		],
	)

	if opcion == "1":
		return ["TS1"]
	if opcion == "2":
		return ["TS2"]
	if opcion == "3":
		return ["TS3"]
	if opcion == "4":
		return ["TS4"]
	return ["TS1", "TS2", "TS3", "TS4"]


def seleccionar_algoritmo():
	opcion = pedir_opcion(
		"2- Elige el programa/metaheuristica",
		[
			("1", "random_search"),
			("2", "hill_climbing"),
			("3", "simulated_annealing"),
			("4", "ALL (comparar los tres algoritmos)"),
		],
	)

	if opcion == "1":
		return "random_search", None, "random_search"

	if opcion == "2":
		hc_op = pedir_opcion(
			"3- Elige tipo de Hill Climbing",
			[
				("1", "simple"),
				("2", "estocastico"),
				("3", "maxima_pendiente"),
				("4", "ALL (comparar los tres)"),
			],
		)
		mapa = {
			"1": "simple",
			"2": "estocastico",
			"3": "maxima_pendiente",
			"4": "all",
		}
		selector = mapa[hc_op]
		contexto = "hc_tipos" if selector == "all" else f"hc_{selector}"
		return "hill_climbing", selector, contexto

	if opcion == "3":
		sa_op = pedir_opcion(
			"3- Elige enfriamiento de Simulated Annealing",
			[
				("1", "linear"),
				("2", "geometric"),
				("3", "logarithmic"),
				("4", "cauchy"),
				("5", "cauchy_modified"),
				("6", "ALL (comparar enfriamientos)"),
			],
		)

		mapa_sa = {
			"1": "linear",
			"2": "geometric",
			"3": "logarithmic",
			"4": "cauchy",
			"5": "cauchy_modified",
			"6": "all",
		}
		selector = mapa_sa[sa_op]
		contexto = "sa_enfriamientos" if selector == "all" else f"sa_{selector}"
		return "simulated_annealing", selector, contexto

	# Opcion 4: comparar los tres algoritmos en las mismas graficas.
	hc_op = pedir_opcion(
		"3- Elige tipo de Hill Climbing para la comparativa de algoritmos",
		[
			("1", "simple"),
			("2", "estocastico"),
			("3", "maxima_pendiente"),
		],
	)
	sa_op = pedir_opcion(
		"3- Elige enfriamiento de Simulated Annealing",
		[
			("1", "linear"),
			("2", "geometric"),
			("3", "logarithmic"),
			("4", "cauchy"),
			("5", "cauchy_modified"),
		],
	)

	mapa_hc = {
		"1": "simple",
		"2": "estocastico",
		"3": "maxima_pendiente",
	}
	mapa_sa = {
		"1": "linear",
		"2": "geometric",
		"3": "logarithmic",
		"4": "cauchy",
		"5": "cauchy_modified",
	}
	selector = {
		"hc": mapa_hc[hc_op],
		"sa": mapa_sa[sa_op],
	}
	contexto = f"all_algorithms_hc_{selector['hc']}_sa_{selector['sa']}"
	return "all_algorithms", selector, contexto


def main():
	print("Script de automatizacion de rendimiento")
	series_objetivo = seleccionar_series()
	algoritmo, selector, contexto = seleccionar_algoritmo()
	runs = pedir_entero_positivo("4) Numero de runs: ")

	configs = construir_configs(algoritmo, selector)

	print("\nConfiguracion seleccionada:")
	print(f"- Series: {', '.join(series_objetivo)}")
	print(f"- Algoritmo: {algoritmo}")
	if selector is not None:
		if isinstance(selector, dict):
			print(f"- Tipo HC: {selector['hc']}")
			print(f"- Enfriamiento SA: {selector['sa']}")
		else:
			print(f"- Tipo: {selector}")
	print(f"- Runs: {runs}")

	for serie_nombre in series_objetivo:
		serie = cargar_serie(serie_nombre)
		k = K_MAP[serie_nombre]
		print(f"\n=== Ejecutando {serie_nombre} (k={k}) ===")

		resultados = ejecutar_runs(serie, k, configs, runs, serie_nombre)
		imprimir_resumen(resultados, serie_nombre)
		guardar_graficas(resultados, serie_nombre, runs, contexto)

	print("\nProceso completado. Graficas guardadas en graficas/rendimiento/.")


if __name__ == "__main__":
	main()
