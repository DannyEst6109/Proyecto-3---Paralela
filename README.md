# Proyecto 3 de Computación Paralela

La parte 1 adapta el laboratorio ASL a una preparación independiente en CPU y a una ejecución de referencia de la CNN base en CPU. Los datos, el modelo y la configuración quedan listos para que la persona 2 implemente CUDA y la persona 3 repita las mediciones y compile el informe.

## Entorno

Probado con Python 3.13 en Windows x64. Cree un entorno aislado desde la raíz del repositorio:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
```

`requirements.txt` fija las dependencias directas; `requirements-lock.txt` registra todas las versiones del entorno local validado. Para reproducir ese entorno use el archivo lock en lugar del primero. El entrenamiento GPU requiere que la persona 2 configure TensorFlow/CUDA en Linux o WSL2; Windows nativo se utiliza aquí solo para CPU.

## Datos

Consulte [origen y estructura del dataset](docs/DATASET.md). La copia local ya está descargada y extraída. Git excluye `data/`, el entorno virtual y los pesos generados. Se necesitan las 29 clases completas y los índices 1..3000 por clase.

## Preparación independiente del entrenamiento

Modo secuencial de lectura y transformación:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m asl_cpu.prepare --workers 1
```

Modo con cuatro procesos CPU, en una carpeta distinta para conservar ambos resultados:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m asl_cpu.prepare --workers 4 --output-dir data/processed/parallel4
.\.venv\Scripts\python.exe -X utf8 -m asl_cpu.compare_preparations data/processed/baseline data/processed/parallel4
```

Cada salida contiene `asl_64x64.npz`, inventario y particiones en Parquet, `particion.csv`, `metadata.json` y `resources.csv`. Las imágenes exportadas son RGB uint8; la normalización CPU se realiza y mide una sola vez al iniciar cada entrenamiento. No se importa TensorFlow en la preparación. El modo multiproceso limita OpenCV a un hilo por trabajador y conserva el orden de entrada.

Las carpetas de salida deben estar vacías. Para repetir una medición use un nombre nuevo; los comandos no sobrescriben evidencia.

## Entrenamiento CPU

Prueba pequeña de integración, no utilizable como benchmark:

```powershell
.\.venv\Scripts\python.exe -m asl_cpu.train --smoke --output-dir results/mi_smoke
```

Referencia completa con las 17,400 imágenes y 20 épocas:

```powershell
.\.venv\Scripts\python.exe -m asl_cpu.train --threads 8 --output-dir results/mi_cpu_01
```

El programa oculta las GPU antes de importar TensorFlow y verifica que los pesos estén en `/CPU:0`. Usa Adam con tasa 0.001, lote 64, precisión float32 y semilla 42. Las épocas son fijas: no hay parada temprana ni aumentación. `tf.data` mezcla el conjunto de entrenamiento de forma determinista; validación y prueba conservan su orden.

Esta es una referencia de entrenamiento en CPU con bibliotecas que pueden usar varios hilos. `--threads 1` permite limitar la aritmética de TensorFlow a un hilo si el equipo necesita esa variante; no debe mezclarse con resultados de ocho hilos sin etiquetarla.

Cada ejecución genera:

- `metrics.json`: configuración, entorno, hashes, dispositivo, tiempos y exactitudes finales.
- `train.log`, `epochs.csv` y `progress.json`: evidencia del entrenamiento y avance por época.
- `resources.csv`: CPU por procesador lógico del sistema y CPU/RSS del proceso Python y sus hijos.
- `model.json`, `initial.weights.h5`, `final.weights.h5` y `predictions.npz`: arquitectura, pesos y resultados de evaluación.

La prueba pequeña usa 16/4/4 imágenes por clase y una época; `run_kind=smoke` la distingue de una referencia completa. Los mensajes nativos de TensorFlow anteriores a su inicialización pueden aparecer solo en la consola; el log conserva las verificaciones explícitas del dispositivo y el entrenamiento.

## Evidencia y documentación

```powershell
.\.venv\Scripts\python.exe -m asl_cpu.plot_resources results/cpu_baseline_01/resources.csv results/cpu_baseline_01/resources.png
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

La figura de recursos grafica el CSV real y no sustituye una captura de pantalla. La ejecución local incluye además una captura real del Administrador de tareas en `results/cpu_baseline_01/evidence/`.

- [Análisis de la parte 1 y tablas del enunciado](docs/PARTE_1.md).
- [Contrato de integración para las personas 2 y 3](docs/INTEGRACION.md).
- [Procedencia del notebook original](reference/README.md).

Los resultados locales son una comprobación funcional y una medición inicial. Las repeticiones comparables CPU/GPU se deben realizar preferentemente en el mismo equipo NVIDIA. No se ha calculado un speedup CUDA a partir de máquinas distintas.
