import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import time
import random
import sys
import os

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

def generar_vecinos(cortes, N):
    """Genera vecinos moviendo cada corte ±1 posición"""
    vecinos = []
    cortes_ordenados = sorted(cortes)

    for i in range(len(cortes_ordenados)):
        for delta in [-1, 1]:
            nuevo_corte = cortes_ordenados[i] + delta
            if 1 <= nuevo_corte <= N:
                nuevo = cortes_ordenados.copy()
                nuevo[i] = nuevo_corte
                if(len(set(nuevo))) == len(nuevo):
                    vecinos.append(sorted(nuevo))

    return vecinos


# ==============================
# HILL CLIMBING SIMPLE
# ==============================

def hill_climbing_simple(serie, k, iteraciones):
    N = len(serie)


    mean = 0
    mejor_corte = None
    mejor_error = 1000

    
    start_time = time.time()

    for _ in range(iteraciones):
        cortes_actual = sorted(random.sample(range(1, N), k - 1))
        error_actual = evaluar_solucion(cortes_actual, serie) 

        while True :
            vecinos = generar_vecinos(cortes_actual, N)
            mejora = False

            for vecino in vecinos:
                error_vecino = evaluar_solucion(vecino, serie)
                if error_vecino < error_actual:
                    cortes_actual = vecino
                    error_actual = error_vecino
                    mejora = True
                    break
            if not mejora:
                break

        if error_actual < mejor_error:
            mejor_error = error_actual
            mejor_corte = cortes_actual
        mean += error_actual

    mean /= iteraciones
    duracion = time.time() - start_time
    return mejor_corte, mejor_error, duracion, mean
    

# ==============================
# HILL CLIMBING ESTOCÁSTICO
# ==============================

def hill_climbing_estocastico(serie, k, iteraciones):
    N = len(serie)

    mean= 0
    mejor_corte = None
    mejor_error = 1000
    start_time = time.time()

    for _ in range(iteraciones):
        cortes_actual = sorted(random.sample(range(1, N), k - 1))
        error_actual = evaluar_solucion(cortes_actual, serie)

        while True: 

            vecinos = generar_vecinos(cortes_actual, N)
            mejores_vecinos = []

            for vecino in vecinos:
                error_vecino = evaluar_solucion(vecino, serie)
                if error_vecino < error_actual:
                    mejores_vecinos.append((vecino, error_vecino))
            
            if not mejores_vecinos:
                break

            cortes_actual, error_actual = random.choice(mejores_vecinos)
        if error_actual < mejor_error:
            mejor_error = error_actual
            mejor_corte = cortes_actual

        mean += error_actual

    mean /= iteraciones
    duracion = time.time() - start_time
    return mejor_corte, mejor_error, duracion, mean


# ==============================
# HILL CLIMBING MÁXIMA PENDIENTE
# ==============================

def hill_climbing_maxima_pendiente(serie, k, iteraciones):
    N=len(serie)

    mean = 0
    mejor_corte = None
    mejor_error1 = 1000
    start_time=time.time()



    for _ in range(iteraciones):
        
        cortes_actual = sorted(random.sample(range(1, N), k - 1))
        error_actual = evaluar_solucion(cortes_actual, serie)

        while True: 
            vecinos = generar_vecinos(cortes_actual, N)
            mejor_vecino = None
            mejor_error = error_actual

            for vecino in vecinos:
                error_vecino = evaluar_solucion(vecino, serie)
                if error_vecino < mejor_error:
                    mejor_vecino = vecino
                    mejor_error = error_vecino
            
            if mejor_vecino is None:
                break
            
            cortes_actual = mejor_vecino 
            error_actual = mejor_error

        if error_actual < mejor_error1:
            mejor_error1 = error_actual
            mejor_corte = cortes_actual
        mean += error_actual
    
    mean /= iteraciones

    
    duracion = time.time() - start_time
    return mejor_corte, mejor_error1, duracion, mean


# ==============================
# MAIN
# ==============================

if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Uso: python hill_climbing.py <ruta_archivo_ts>")
        sys.exit(1)

    ruta_archivo = sys.argv[1]
    nombre_base = os.path.basename(ruta_archivo).upper()

    config_k = {"TS1": 9, "TS2": 10, "TS3": 20, "TS4": 50}
    k_objetivo = next((v for k, v in config_k.items() if k in nombre_base), None)

    if k_objetivo is None:
        k_objetivo = int(input("No se reconoció la serie. Introduce el valor de k: "))
    
    serie = cargar_serie(ruta_archivo)

    if serie is not None:
        ITERACIONES = 500

        print(f"Archivo: {ruta_archivo} | K: {k_objetivo} | Iteraciones: {ITERACIONES}")

        # -------- ELECCIÓN DEL ALGORITMO --------
        print("\nElige la variante de Hill Climbing a ejecutar:")
        print("1 - Simple")
        print("2 - Estocástico")
        print("3 - Máxima Pendiente")
        print("4 - Todas para comparar")
        opcion = input("Opción: ").strip()

        resultados = []

        # -------- SIMPLE --------
        if opcion == "1" or opcion == "4":
            print("\n--- Hill Climbing Simple ---")
            cortes_s, error_s, tiempo_s, mean_s = hill_climbing_simple(serie, k_objetivo, ITERACIONES)
            print(f"MSE: {error_s:.6f}")
            print(f"Tiempo: {tiempo_s:.4f} segundos")
            print(f"Media MSE durante iteraciones: {mean_s:.6f}")
            print(f"RMSE: {np.sqrt(error_s):.6f}")
            resultados.append(("Simple", cortes_s, error_s, tiempo_s))

        # -------- ESTOCÁSTICO --------
        if opcion == "2" or opcion == "4":
            print("\n--- Hill Climbing Estocástico ---")
            cortes_e, error_e, tiempo_e, mean_e = hill_climbing_estocastico(serie, k_objetivo, ITERACIONES)
            print(f"MSE: {error_e:.6f}")
            print(f"Tiempo: {tiempo_e:.4f} segundos")
            print(f"Media MSE durante iteraciones: {mean_e:.6f}")
            print(f"RMSE: {np.sqrt(error_e):.6f}")
            resultados.append(("Estocástico", cortes_e, error_e, tiempo_e))

        # -------- MÁXIMA PENDIENTE --------
        if opcion == "3" or opcion == "4":
            print("\n--- Hill Climbing Máxima Pendiente ---")
            cortes_m, error_m, tiempo_m, mean_m = hill_climbing_maxima_pendiente(serie, k_objetivo, ITERACIONES)
            print(f"MSE: {error_m:.6f}")
            print(f"Tiempo: {tiempo_m:.4f} segundos")
            print(f"Media MSE durante iteraciones: {mean_m:.6f}")
            print(f"RMSE: {np.sqrt(error_m):.6f}")
            resultados.append(("Máxima Pendiente", cortes_m, error_m, tiempo_m))

        for nombre_alg, cortes, _, _ in resultados:
            plt.figure(figsize=(12, 6))
            plt.plot(serie, color='lightgray', label='Serie Original')

            puntos = [0] + cortes + [len(serie)]
            for i in range(len(puntos)-1):
                idx = range(puntos[i], puntos[i+1])
                reg = LinearRegression().fit(np.array(idx).reshape(-1, 1), serie[idx])
                plt.plot(idx, reg.predict(np.array(idx).reshape(-1, 1)), linewidth=2)

            plt.title(f"Hill Climbing ({nombre_alg}): {nombre_base} (k={k_objetivo})")
            plt.legend([nombre_alg])
            nombre_img = f"resultado_HC_{nombre_base.split('.')[0]}_{nombre_alg.replace(' ', '_')}.png"
            plt.savefig(nombre_img)
            print(f"Gráfica guardada como: {nombre_img}")
            plt.close()  # Cierra la figura para no mezclarla con la siguiente