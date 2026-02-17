#!/bin/bash

# Nombre del entorno definido anteriormente
ENV_NAME="env_metaheuristica_p1"

echo "--- Eliminando el entorno virtual: $ENV_NAME ---"

# 1. Intentar desactivar si está activo en la sesión actual
if [[ "$VIRTUAL_ENV" != "" ]]; then
    deactivate
    echo "Entorno desactivado."
fi

# 2. Borrar la carpeta del entorno
if [ -d "$ENV_NAME" ]; then
    rm -rf "$ENV_NAME"
    echo "Carpeta '$ENV_NAME' eliminada con éxito."
else
    echo "La carpeta '$ENV_NAME' no existe."
fi

echo "--- Proceso finalizado ---"