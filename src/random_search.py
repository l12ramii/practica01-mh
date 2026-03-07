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
    """Calcula el MSE de una regresión lineal en un segmento"""
    if len(x) < 2: return 1e9 
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

if __name__ == "__main__":
    # Verificación de argumentos de entrada
    if len(sys.argv) < 3:
        print("Uso: python random_search.py <ruta_archivo_ts> <numero_iteraciones")
        sys.exit(1)

    ruta_archivo = sys.argv[1]
    nombre_base = os.path.basename(ruta_archivo).upper()

    # Mapeo de valores de K según el enunciado 
    config_k = {"TS1": 9, "TS2": 10, "TS3": 20, "TS4": 50}
    
    # Intentar detectar K automáticamente o pedirlo por consola
    k_objetivo = next((v for k, v in config_k.items() if k in nombre_base), None)
    
    if k_objetivo is None:
        k_objetivo = int(input(f"No se reconoció la serie. Introduce el valor de k: "))

    serie = cargar_serie(ruta_archivo)
    ITERACIONES = int(sys.argv[2])


    if serie is not None:
            print(f"--- Iniciando Random Search ---")
            print(f"Archivo: {ruta_archivo} | K: {k_objetivo} | Iteraciones: {ITERACIONES}")
        
            cortes, error, duracion = random_search(serie, k_objetivo, ITERACIONES)
        
            print(f"\nRESULTADOS:")
            print(f"- MSE Promedio (Exactitud): {error:.6f}")
            print(f"- Tiempo de Cómputo: {duracion:.4f} segundos")

            # Guardar datos en fichero .txt
            with open("random_search.txt", "a") as file:
                file.write(f"\n{nombre_base}, {error:.6f}, {duracion:.4f}, {ITERACIONES}")
        
        # Generar gráfica 
    plt.figure(figsize=(12, 6))
    plt.plot(serie, color='lightgray', label='Serie Original')
    puntos_completos = [0] + cortes + [len(serie)]
    for i in range(len(puntos_completos)-1):
        idx = range(puntos_completos[i], puntos_completos[i+1])
        reg = LinearRegression().fit(np.array(idx).reshape(-1, 1), serie[idx])
        plt.plot(idx, reg.predict(np.array(idx).reshape(-1, 1)), linewidth=2)
        
    plt.title(f"Random Search: {nombre_base} (k={k_objetivo}) MSE={error}")
    plt.savefig(f"resultado_{nombre_base.split('.')[0]}.png")
    print(f"Gráfica guardada como: resultado_{nombre_base.split('.')[0]}.png")
    plt.show()