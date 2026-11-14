# Proyecto 3 de Computación Paralela

La parte 1 adapta el laboratorio ASL a una preparación independiente en CPU y a una ejecución de referencia de la CNN base en CPU. La parte 2 entrena la misma CNN en una GPU NVIDIA con CUDA (TensorFlow → cuDNN/cuBLAS), manteniendo archivos, preparación y pipeline de entrada en CPU. La persona 3 repite las mediciones y compila el informe.

## Entorno

Probado con Python 3.13 en Windows x64. Cree un entorno aislado desde la raíz del repositorio:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
```

`requirements.txt` fija las dependencias directas; `requirements-lock.txt` registra todas las versiones del entorno CPU validado en Windows y `requirements-gpu-lock.txt` las del entorno CUDA validado en Linux. Para reproducir ese entorno use el archivo lock en lugar del primero. TensorFlow 2.21 no usa GPU en Windows nativo; el entrenamiento CUDA se ejecuta en Linux o WSL2 (ver la sección «Entrenamiento GPU con CUDA»).

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

## Entrenamiento GPU con CUDA

Probado en Ubuntu (kernel 6.8) con Python 3.12, NVIDIA GeForce RTX 3060 Laptop (6 GB, capacidad 8.6) y driver 550.163.01. Solo se necesita el driver NVIDIA del sistema: `tensorflow[and-cuda]` instala CUDA, cuDNN y cuBLAS como paquetes pip.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-gpu.txt
.venv/bin/python -m asl_cpu.gpu_check --output results/gpu_environment.json
```

`gpu_check` registra GPU, driver, versiones de CUDA/cuDNN y ejecuta un matmul y una convolución en `/GPU:0`. Los scripts cargan `libcusolver` antes de importar TensorFlow porque el wheel 2.21 no la encuentra por sí solo (síntoma: «Cannot dlopen some GPU libraries» y lista de GPU vacía).

Prueba pequeña y entrenamiento completo con la misma configuración que la referencia CPU:

```bash
.venv/bin/python -m asl_cpu.train --smoke --device gpu --output-dir results/mi_smoke_gpu
.venv/bin/python -m asl_cpu.train --device gpu --output-dir results/mi_gpu_01
.venv/bin/python -m asl_cpu.train --device cpu --threads 8 --output-dir results/mi_cpu_01
.venv/bin/python -m asl_cpu.compare_runs results/mi_cpu_01 results/mi_gpu_01
```

`--device gpu` falla si TensorFlow no ve una GPU CUDA y verifica que pesos y variables de Adam estén en `/GPU:0`. Los datos, la normalización y `tf.data` siguen en CPU. TensorFloat-32 queda desactivado para conservar float32 estricto; `--allow-tf32` lo habilita como experimento separado (`run_kind` termina en `_tf32`). `--initial-weights` permite partir de un `initial.weights.h5` de otra corrida.

Además de los archivos de la versión CPU, cada corrida guarda `gpu.csv` (nvidia-smi cada 500 ms), `evidence/nvidia_smi_epoch1.txt` (procesos con contexto CUDA durante la primera época) y en `metrics.json` los dispositivos, transferencias estimadas y memoria GPU.

```bash
.venv/bin/python -m asl_cpu.plot_gpu results/mi_gpu_01
.venv/bin/python -m asl_cpu.profile_kernels --output-dir results/mi_kernels
```

`profile_kernels` perfila pasos de entrenamiento con CUPTI y guarda grid, block, registros y ocupación de cada kernel CUDA que lanzan cuDNN, cuBLAS y TensorFlow.

### Kernels propios en CUDA Python (Numba)

`asl_cpu/cuda_kernels.py` contiene los kernels diseñados por el equipo, cada uno con su distribución de hilos, bloques y grid:

- Adam (un hilo por parámetro);
- convolución 3 × 3 directa (v1) y con teselas en memoria compartida (v2);
- MaxPooling, multiplicación de matrices con teselas 16 × 16 y softmax.

```bash
.venv/bin/python -m asl_cpu.train --device gpu --optimizer numba --output-dir results/mi_gpu_numba_01
.venv/bin/python -m asl_cpu.benchmark_kernels --weights results/mi_gpu_01/final.weights.h5 --output-dir results/mi_kernels_numba
```

- `--optimizer numba`: TensorFlow calcula los gradientes y el kernel propio actualiza los pesos. Los tensores pasan entre ambos por DLPack sin salir de la GPU.
- `benchmark_kernels`: ejecuta el forward completo de la CNN entrenada solo con los kernels propios, verifica que clasifica igual que TensorFlow y mide cada capa en NumPy (CPU), Numba (GPU) y cuDNN.

En este equipo (driver 12.4) se usa el target CUDA incluido en `numba` junto con `pynvjitlink`; `cuda_kernels.setup_numba_cuda()` configura NVVM y libcudart de los paquetes pip. No instalar `numba-cuda`, que exige un driver de la serie 12.5 o superior.

### Evidencia con capturas de pantalla

Abrir una terminal con `watch -n 0.5 nvidia-smi` y el Monitor del sistema (pestaña Recursos), y en otra terminal:

```bash
bash scripts/evidencia_parte2.sh
```

El script corre en orden la versión CPU, la GPU con Adam de Keras y la GPU con el kernel Numba. Antes de cada una pausa para que se tomen las capturas durante el entrenamiento. Al terminar genera gráficos y comparaciones en `results/capturas_<fecha>/`, donde también deben guardarse las imágenes.

Diseño, decisiones y resultados en [docs/PARTE_2.md](docs/PARTE_2.md).

## Evidencia y documentación

```powershell
.\.venv\Scripts\python.exe -m asl_cpu.plot_resources results/cpu_baseline_01/resources.csv results/cpu_baseline_01/resources.png
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

La figura de recursos grafica el CSV real y no sustituye una captura de pantalla. La ejecución local incluye además una captura real del Administrador de tareas en `results/cpu_baseline_01/evidence/`.

- [Análisis de la parte 1 y tablas del enunciado](docs/PARTE_1.md).
- [Parte 2: entrenamiento con CUDA, diseño paralelo y resultados GPU](docs/PARTE_2.md).
- [Contrato de integración para las personas 2 y 3](docs/INTEGRACION.md).
- [Procedencia del notebook original](reference/README.md).

Los resultados locales son una comprobación funcional y una medición inicial. Las corridas `cpu_rtx_01` y `gpu_baseline_01` se hicieron en el mismo equipo NVIDIA; son una pareja de comprobación, no los promedios de cinco repeticiones de la parte 3. No se calcula speedup CUDA a partir de máquinas distintas.
