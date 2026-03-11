Para crear el entorno virtual:

Dale permisos de ejecución al archivo:
chmod +x setup_env.sh

Ejecuta el script:
./setup_env.sh

Activar el entorno virtual:
source ./env_metaheuristica_p1/bin/activate 

Para seleccionar el entorno virtual en VS Code: 
shift+ctrl+p -> Python: seleccionar intérprete -> env_metaheuristica_p1

Para borrar el entorno virtual: 
chmod +x cleanup_env.sh

Ejecuta el script: 
./cleanup_env.sh

Dentro de src, están los programas de cada heuristica y metaheuristca:
Se ejecutan con python nombre_del_archivo.py
Como argumento adicional está --runs nº para darle varias runs y que guarde la mejor, por defecto o si no se pone se hace solo 1 run.

En la carpeta raíz está el main.py que permite generar las graficas del analisis de rendimiento de cada una de las heuristicas vistas, de tal forma que puedan verse de forma más sencilla sus diferencias.