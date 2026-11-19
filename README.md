# Proyecto 3 – Mejorando algoritmos con CUDA

Aceleración con CUDA del entrenamiento de la red neuronal convolucional del Laboratorio 3 de Data Science, que clasifica fotografías del alfabeto del Lenguaje de Señas Americano, separando la preparación de imágenes en el CPU con varios procesos del entrenamiento de la red en una GPU NVIDIA, con mediciones repetidas y cálculo del speedup por fase.

## Integrantes

• Sebastián Estrada Tuch – 21405  
• Carlos Daniel Estrada Vega – 20853  
• André Emilio Pivaral López – 23574  

**Universidad del Valle de Guatemala**  
Facultad de Ingeniería  
Departamento de Ciencias de la Computación  
Computación Paralela y Distribuida  
  
**Catedrática:** Kimberly Marisol Barrera Álvarez  
**Sección:** 10

## Descripción

El proyecto parte del notebook original del Laboratorio 3, que entrena en el CPU una red convolucional sobre el conjunto ASL Alphabet de 29 clases, y lo reorganiza en un paquete de Python con dos comandos independientes. El primero prepara los datos en el CPU, es decir, recorre las 87,000 imágenes, selecciona 17,400 con la misma partición por bloques del laboratorio, las decodifica, las reduce a 64 por 64 píxeles en RGB y las exporta, con uno o con varios procesos. El segundo entrena la red con la misma configuración en el CPU o en la GPU, mientras los archivos, la normalización y el armado de lotes permanecen en el CPU.

La primera parte analiza el programa secuencial, construye las tablas de operaciones y de operaciones repetitivas, separa la preparación del entrenamiento y deja una referencia reproducible en el CPU. La segunda parte entrena la misma red en una GPU NVIDIA con CUDA a través de TensorFlow, cuDNN y cuBLAS, sustituye el optimizador por un kernel Adam propio escrito en CUDA Python con Numba y diseña kernels propios de convolución, MaxPooling, producto de matrices y softmax con su distribución de hilos, bloques y grids, que reproducen la inferencia de la red con la misma exactitud que TensorFlow. La tercera parte repite cinco veces cada configuración con calentamiento descartado, orden rotado y verificación automática de la carga de trabajo, calcula promedios, dispersión y speedup por fase, y replica el experimento en Google Colab como segundo entorno.

Todas las versiones comparten las mismas imágenes, particiones, arquitectura de 1,145,693 parámetros, optimizador Adam con tasa de 0.001, lotes de 64, veinte épocas, precisión float32 estricta y pesos iniciales. Las tablas, el speedup, las capturas de pantalla y la discusión técnica se encuentran en el informe en PDF; el detalle de cada parte está en la carpeta `docs`.

## Estructura del proyecto

    Proyecto-3---Paralela/
    ├── .venv/                                   entorno virtual, excluido de Git
    ├── asl_cpu/
    │   ├── analyze_benchmark.py                 promedios, speedup por fase, gráficos y reporte de la parte 3
    │   ├── benchmark.py                         orquestador de las repeticiones de la parte 3
    │   ├── benchmark_kernels.py                 forward de la red con kernels propios frente a NumPy y cuDNN
    │   ├── common.py                            rutas, configuración, huellas y registro del entorno
    │   ├── compare_preparations.py              equivalencia entre dos preparaciones
    │   ├── compare_runs.py                      verificación de carga idéntica entre dos corridas
    │   ├── cuda_kernels.py                      kernels propios en CUDA Python con Numba
    │   ├── data.py                              inventario, partición y lectura de imágenes con varios procesos
    │   ├── gpu_check.py                         registro del entorno CUDA y prueba de cómputo en la GPU
    │   ├── model.py                             red convolucional base del laboratorio
    │   ├── numba_adam.py                        optimizador Adam con el kernel propio
    │   ├── plot_gpu.py                          gráfico del uso de la GPU por fase
    │   ├── plot_resources.py                    gráfico del uso del CPU por núcleo
    │   ├── prepare.py                           comando de preparación en el CPU
    │   ├── profile_kernels.py                   perfil de los kernels de biblioteca con CUPTI
    │   ├── resources.py                         monitores de CPU y GPU durante cada corrida
    │   └── train.py                             comando de entrenamiento en el CPU o en la GPU
    ├── config/
    │   └── baseline.json                        configuración común de todas las corridas
    ├── data/                                    excluida de Git
    │   ├── downloads/                           archivo original del dataset
    │   ├── processed/baseline/                  imágenes preparadas en NPZ, particiones y metadatos
    │   └── raw/                                 imágenes originales del dataset
    ├── docs/
    │   ├── DATASET.md                           origen, licencia y validación del dataset
    │   ├── INTEGRACION.md                       contrato de datos y fronteras de cada medición
    │   ├── PARTE_1.md                           análisis del programa original, datos y referencia CPU
    │   └── PARTE_2.md                           entrenamiento con CUDA, diseño paralelo y kernels propios
    ├── notebooks/
    │   └── Proyecto3_Colab_CUDA.ipynb           experimento completo en Google Colab
    ├── reference/
    │   ├── Laboratorio3.original.ipynb          notebook original del laboratorio, sin cambios
    │   └── README.md                            procedencia del notebook original
    ├── results/
    │   ├── capturas/                            corrida de evidencia de la parte 2 con capturas de pantalla
    │   ├── parte3/                              experimento repetido, análisis y comparación entre entornos
    │   ├── cpu_rtx_01/, gpu_baseline_01/, ...   corridas individuales de las partes 1 y 2
    │   └── README.md                            índice de las corridas
    ├── scripts/
    │   ├── evidencia_parte2.sh                  corrida de evidencia de la parte 2
    │   └── parte3_benchmark.sh                  experimento final de la parte 3 en el equipo NVIDIA
    ├── tests/
    │   ├── test_benchmark.py                    plan de corridas, estadística y análisis
    │   ├── test_cuda_kernels.py                 kernels propios frente a NumPy
    │   ├── test_data.py                         particiones y equivalencia entre modos de preparación
    │   ├── test_gpu_device.py                   selección del dispositivo de entrenamiento
    │   └── test_model.py                        arquitectura de la red base
    ├── .gitignore
    ├── Informe.pdf                              informe de rendimiento y análisis
    ├── README.md
    ├── requirements.txt                         dependencias del entorno CPU
    ├── requirements-gpu.txt                     dependencias del entorno CUDA para Linux o WSL2
    ├── requirements-lock.txt                    versiones exactas del entorno CPU validado
    └── requirements-gpu-lock.txt                versiones exactas del entorno CUDA validado

## Requisitos

- Python; se utilizó la versión 3.13 en Windows para la preparación y la referencia CPU, y la versión 3.12 en Linux para el entrenamiento con CUDA.
- Paquetes del entorno CPU: tensorflow 2.21.0, numpy, pandas, opencv-python-headless, pyarrow, psutil y matplotlib.
- Paquetes adicionales del entorno CUDA: tensorflow con el extra and-cuda, numba 0.68.1, cupy-cuda12x, cuda-pathfinder y pynvjitlink-cu12.
- GPU NVIDIA con su driver instalado. No es necesario instalar el CUDA Toolkit, porque CUDA, cuDNN y cuBLAS se instalan como paquetes de pip. No se debe instalar numba-cuda, que exige un driver de la serie 12.5 o superior.
- TensorFlow 2.21 no usa la GPU en Windows nativo, por lo que el entrenamiento con CUDA se ejecuta en Linux o en WSL2. La preparación y la referencia CPU funcionan en ambos sistemas.
- El dataset ASL Alphabet de Kaggle, de aproximadamente 1 GB, descargado según `docs/DATASET.md`.

## Instalación

Entorno CPU en Windows con PowerShell, desde la raíz del repositorio:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
```

Entorno CUDA en Linux o WSL2:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-gpu.txt
.venv/bin/python -m asl_cpu.gpu_check --output results/gpu_environment.json
```

Para reproducir exactamente los entornos validados se usan `requirements-lock.txt` y `requirements-gpu-lock.txt` en lugar de los archivos anteriores.

## Ejecución

Preparación de imágenes en el CPU, con un proceso y con cuatro procesos:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m asl_cpu.prepare --workers 1
.\.venv\Scripts\python.exe -X utf8 -m asl_cpu.prepare --workers 4 --output-dir data/processed/parallel4
.\.venv\Scripts\python.exe -X utf8 -m asl_cpu.compare_preparations data/processed/baseline data/processed/parallel4
```

Entrenamiento de referencia en el CPU, en Windows o Linux:

```powershell
.\.venv\Scripts\python.exe -m asl_cpu.train --device cpu --threads 8 --output-dir results/mi_cpu_01
```

Entrenamiento en la GPU con el Adam de Keras y con el kernel Adam propio, y verificación de que ambas corridas comparten la misma carga:

```bash
.venv/bin/python -m asl_cpu.train --device gpu --output-dir results/mi_gpu_01
.venv/bin/python -m asl_cpu.train --device gpu --optimizer numba --output-dir results/mi_gpu_numba_01
.venv/bin/python -m asl_cpu.compare_runs results/mi_cpu_01 results/mi_gpu_01
```

Kernels propios y perfil de los kernels de biblioteca:

```bash
.venv/bin/python -m asl_cpu.benchmark_kernels --weights results/mi_gpu_01/final.weights.h5 --output-dir results/mi_kernels_numba
.venv/bin/python -m asl_cpu.profile_kernels --output-dir results/mi_kernels
```

Experimento repetido de la parte 3 en el equipo NVIDIA, con cinco repeticiones por configuración, pausas para tomar capturas y análisis al final:

```bash
bash scripts/parte3_benchmark.sh --dry-run
bash scripts/parte3_benchmark.sh
```

El mismo experimento en Google Colab se ejecuta abriendo `notebooks/Proyecto3_Colab_CUDA.ipynb` con un entorno de ejecución con GPU y corriendo sus celdas en orden. La comparación entre ambos entornos se genera con:

```bash
.venv/bin/python -m asl_cpu.analyze_benchmark results/parte3/rtx3060 results/parte3/colab_t4 --output-dir results/parte3/comparacion
```

Las pruebas automatizadas se ejecutan con `python -m unittest discover -s tests`. En un equipo sin GPU las pruebas de los kernels CUDA se omiten. Cada comando de entrenamiento exige una carpeta de salida vacía, de modo que nunca sobrescribe evidencia, y la opción `--smoke` produce una prueba pequeña de integración que no sirve para medir rendimiento.

## Contenido del análisis

- Análisis de la solución secuencial. Descripción del notebook original, tabla de trece operaciones desde la recepción de las imágenes hasta el resultado y tabla de operaciones repetitivas con los datos que procesan y su número de repeticiones.
- Separación del flujo de trabajo. Preparación en el CPU con un proceso o con varios mediante `ProcessPoolExecutor`, partición por bloques idéntica a la del laboratorio, contrato de datos en NPZ con huellas SHA-256 y entrenamiento con el pipeline de entrada en el CPU y el modelo en el dispositivo seleccionado.
- Medición del tiempo secuencial. Fronteras de cada fase, sincronización de la GPU antes de leer el reloj, tiempos por época separados del entrenamiento completo y registro del uso de CPU por núcleo y de la GPU durante cada corrida.
- Oportunidades de paralelización. Estimación de las operaciones de punto flotante por capa y justificación de las operaciones delegadas a la GPU y de las que permanecen en el CPU.
- Diseño del algoritmo paralelo. Distribución de hilos, bloques y grids por tipo de operación, configuración real de lanzamiento de los kernels propios y perfil de los kernels de cuDNN, cuBLAS y TensorFlow con CUPTI.
- Implementación con CUDA Python. Adaptación del entrenamiento a la GPU, kernel Adam propio integrado mediante DLPack, kernels propios de convolución directa y con memoria compartida, MaxPooling, producto de matrices y softmax, transferencias entre CPU y GPU, y funciones y bibliotecas utilizadas.
- Análisis y conclusiones. Cinco repeticiones por configuración, media, desviación estándar e intervalo bootstrap del speedup por fase y total, ley de Amdahl, preparación con varios núcleos del CPU, uso de recursos con capturas de pantalla, calidad del modelo, segundo entorno en Google Colab y discusión técnica sobre la división del trabajo entre CPU y GPU, el efecto de CUDA y sus límites.

## Nota sobre los datos

El conjunto ASL Alphabet de Akash Nagaraj se obtiene de Kaggle y se distribuye con licencia GPL 2. Contiene 87,000 imágenes de entrenamiento de 200 por 200 píxeles, 3,000 por cada una de las 29 clases. Los datos, los pesos entrenados y las predicciones se excluyen del control de versiones; cada integrante descarga su propia copia y la equivalencia entre equipos se comprueba con las huellas de los arreglos preparados.
