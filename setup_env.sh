#!/bin/bash

# Nombre del entorno
ENV_NAME="env_metaheuristica_p1"

echo "--- Configurando entorno para Metaheurísticas (Curso 2025/2026) ---"

# Localizar el binario de Python 3
PYTHON_BIN=$(which python3)

if [ -z "$PYTHON_BIN" ]; then
    echo "Error: Python3 no encontrado. Por favor instálalo primero."
    exit 1
fi

# 1. Crear el entorno virtual
$PYTHON_BIN -m venv $ENV_NAME

# 2. Activar el entorno
source $ENV_NAME/bin/activate

# 3. Actualizar pip
pip install --upgrade pip

# 4. Instalar dependencias
pip install numpy scikit-learn matplotlib pandas gnuplotlib

echo "Para activar el entorno virtual, escribe: source $ENV_NAME/bin/activate"