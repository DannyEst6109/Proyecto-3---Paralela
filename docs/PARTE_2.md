# Parte 2 Entrenamiento con CUDA

## Alcance

Esta parte entrena la misma CNN base de la parte 1 en una GPU NVIDIA mediante CUDA y demuestra que el hardware se usa realmente. Conserva datos, particiones, arquitectura, optimizador, épocas, lote, semilla y precisión de la referencia CPU. Los archivos, la preparación de imágenes, la normalización y el pipeline de entrada permanecen en CPU; a la GPU se delega solo la matemática de la red (forward, pérdida, gradientes y actualización de Adam).

Entrega el código (`asl_cpu/train.py --device gpu`, `gpu_check.py`, `profile_kernels.py`, `compare_runs.py`, `plot_gpu.py`), la verificación del entorno, el diseño paralelo con la configuración real de hilos/bloques/grids, una ejecución GPU completa y su comparación con una ejecución CPU en el mismo equipo. Los promedios de cinco repeticiones y el speedup final corresponden a la parte 3.

## Hardware y entorno

| Elemento | Valor |
|---|---|
| GPU | NVIDIA GeForce RTX 3060 Laptop GPU, 6,144 MiB, capacidad de cómputo 8.6 (Ampere, 30 SM) |
| Driver NVIDIA | 550.163.01, admite hasta CUDA 12.4 |
| CPU | Intel Core i7-11800H, 8 núcleos / 16 hilos |
| RAM | 16 GB |
| Sistema | Linux 6.8 (Ubuntu), Python 3.12.3 |
| TensorFlow / Keras | 2.21.0 / 3.15.1, compilado con CUDA 12.5.1 y cuDNN 9 |
| Librerías CUDA (pip `tensorflow[and-cuda]`) | cudart 12.9.79, cuBLAS 12.9.3.13, cuDNN 9.28.1.4, CUPTI 12.9.79, cuSOLVER 11.7.5.82 |

Se instala con `pip install -r requirements-gpu.txt`; las versiones exactas están en `requirements-gpu-lock.txt`. No se instaló el CUDA Toolkit del sistema: las librerías llegan como paquetes `nvidia-*` y solo se requiere el driver. CUDA 12.x permite ejecutar un runtime 12.9 con un driver de la serie 12.4 (compatibilidad de versión menor). TensorFlow trae código para sm_80; la RTX 3060 (sm_86) lo ejecuta por compatibilidad binaria dentro de Ampere.

`python -m asl_cpu.gpu_check` guarda `results/gpu_environment.json`. Ejecuta 10 multiplicaciones 2048 × 2048 y una convolución en `/GPU:0`, verifica que el resultado esté en `device:GPU:0` y obtuvo 5,965 GFLOP/s en el matmul.

Durante la configuración se encontraron dos problemas que conviene documentar:

1. **TensorFlow no veía la GPU.** El log decía `Could not load dynamic library 'libcusolver.so.11'` y `Skipping registering GPU devices`. El archivo sí estaba instalado, pero el RPATH del wheel de TensorFlow 2.21 no incluye `nvidia/cusolver/lib`. `common.preload_pip_cuda_libraries()` carga esa librería con `ctypes` (RTLD_GLOBAL) antes de importar TensorFlow; después TF la encuentra por su soname. El modo CPU no la carga.
2. **TensorFloat-32.** En Ampere, TensorFlow usa TF32 por defecto en matmul y convoluciones (mantisa de 10 bits). El matmul de `gpu_check` difería del CPU hasta 7.9 × 10⁻² con TF32 y 4.7 × 10⁻⁴ con float32 estricto. Como la configuración común exige float32 y la parte 1 pide no cambiar la precisión para favorecer una versión, `train.py` desactiva TF32. `--allow-tf32` queda como experimento separado y etiquetado (`run_kind = baseline_gpu_tf32`).

## Operaciones que se ejecutan en GPU

La CNN tiene 1,145,693 parámetros. Para una imagen de 64 × 64 × 3, el forward requiere aproximadamente:

| Capa | Salida | Operaciones de punto flotante por imagen (forward) |
|---|---|---:|
| Conv2D 32, 3 × 3 | 64 × 64 × 32 | 7.08 M |
| Conv2D 64, 3 × 3 | 32 × 32 × 64 | 37.75 M |
| Conv2D 128, 3 × 3 | 16 × 16 × 128 | 37.75 M |
| Dense 8192 → 128 | 128 | 2.10 M |
| Dense 128 → 29 | 29 | 0.01 M |
| Total | | 84.7 M |

Contando forward más backward como ≈ 3 veces el forward, una época (12,180 imágenes de entrenamiento y 2,610 de validación) suma ≈ 3.3 TFLOP, y las 20 épocas ≈ 66 TFLOP. El 99 % está en convoluciones y productos de matrices, que son las operaciones que se delegan a la GPU.

| Operación | Por qué se beneficia del paralelismo GPU |
|---|---|
| Convoluciones (forward, gradiente de entrada y gradiente de filtros) | Cada valor de salida es un producto punto independiente de 27 a 576 términos. Hay millones por lote (64 × 64 × 64 × 32 = 8.4 M salidas solo en la primera capa) y los filtros se reutilizan, de modo que cabe en memoria compartida. cuDNN las transforma en GEMM, Winograd o FFT. |
| Dense (productos de matrices) | GEMM de 64 × 8192 por 8192 × 128 por lote: cada elemento de la salida es independiente y se calcula por bloques. cuBLAS/CUTLASS. |
| ReLU, sesgo, Dropout | Elemento a elemento, sin dependencias: un hilo por valor. |
| MaxPooling y su gradiente | Cada ventana 2 × 2 es independiente. cuDNN. |
| Softmax y pérdida | Reducciones pequeñas por fila (29 clases); poco trabajo, pero evita volver a CPU a mitad del paso. |
| Actualización de Adam | Elemento a elemento sobre 1.15 M parámetros y sus dos momentos; las variables viven en GPU y no cruzan PCIe en cada paso. |

Lo que permanece en CPU y por qué: recorrer carpetas, abrir y decodificar JPEG, redimensionar y exportar NPZ (parte 1) son tareas de E/S y control con poca aritmética por byte. Llevarlas a GPU obligaría a copiar imágenes crudas por PCIe para hacer poco cálculo. La normalización a float32 se hace una vez en CPU antes de entrenar y `tf.data` arma los lotes en CPU (`input_pipeline_device = /CPU:0`).

## Diseño paralelo: hilos, bloques y grids

Esta implementación usa CUDA a través de TensorFlow: cada operación de la red se ejecuta como uno o varios kernels de cuDNN, cuBLAS/CUTLASS o de TensorFlow/Eigen. No escribimos kernels propios; la librería elige la configuración de lanzamiento según la forma de los tensores y la GPU. **Pendiente del equipo:** confirmar con el docente que acepta CUDA mediante el framework. Si exige kernels propios, la tabla siguiente sirve de base para escribirlos con Numba o CuPy.

El diseño conceptual es paralelismo de datos sobre los elementos de salida:

| Operación | Unidad de trabajo de un hilo | Bloque | Grid |
|---|---|---|---|
| Elemento a elemento (ReLU, sesgo, Dropout, Adam) | Un valor del tensor | 1D, 256–1024 hilos | 1D, ⌈n / hilos por bloque⌉ bloques |
| MaxPooling y gradiente | Un valor de salida (o de entrada en el gradiente) | 256 hilos | 3D: posiciones × canales × imágenes del lote |
| Convolución como GEMM (im2col implícito) | Una sub-tesela de la matriz de salida (varios valores por hilo) | 128–256 hilos que cargan en memoria compartida una tesela de entrada y de filtros | 2D/3D: teselas de salida × lote |
| Dense (GEMM) | Una sub-tesela de C = A·B | 128 hilos con tesela 128 × 32 en memoria compartida | Teselas de C |
| Reducciones (gradiente del sesgo, pérdida, métricas) | Una suma parcial | 256 hilos que reducen en memoria compartida | Un bloque por fragmento; una segunda pasada combina los parciales |

Un lote de entrenamiento de 64 imágenes se ejecuta completo en la GPU: el grid de cada kernel cubre todo el lote y la RTX 3060 reparte los bloques entre sus 30 SM. Las épocas y los lotes siguen siendo secuenciales, porque cada actualización de Adam depende de la anterior.

`python -m asl_cpu.profile_kernels` perfila 10 pasos de entrenamiento con CUPTI, después de 3 de calentamiento, en float32 estricto con determinismo. Registró 3,270 lanzamientos de 119 kernels distintos (≈ 327 por paso) y 71.3 ms de GPU, es decir ≈ 7.1 ms por paso. Configuraciones reales (`results/gpu_kernels/kernels.csv`), mostrando el kernel más pesado de cada categoría:

| Categoría (op de TensorFlow) | Kernel | Grid | Block | Hilos totales | Registros por hilo | Memoria compartida (B) | Ocupación % | µs por llamada |
|---|---|---|---|---:|---:|---:|---:|---:|
| Conv2D gradiente de filtros | `ampere_sgemm_128x128_nt` | 1×1×36 | 256×1×1 | 9,216 | 118 | 16,384 | 33 | 1112.6 |
| Gradiente ReLU | `Eigen::EigenMetaKernel` | 45×1×1 | 1024×1×1 | 46,080 | 18 | 0 | 66 | 196.2 |
| MaxPooling gradiente | `cudnn::pooling_bw_kernel_max_deterministic` | 16×32×64 | 256×1×1 | 8,388,608 | 30 | 0 | 100 | 387.3 |
| Conv2D forward fusionada con sesgo+ReLU | `cudnn_ampere_scudnn_winograd_128x128` | 128×4×2 | 256×1×1 | 262,144 | 126 | 49,152 | 33 | 323.8 |
| Conv2D gradiente de entrada | `cudnn_ampere_scudnn_winograd_128x128` | 64×4×2 | 256×1×1 | 131,072 | 126 | 49,152 | 33 | 280.3 |
| MaxPooling forward | `cudnn::pooling_fw_4d_kernel` | 8×8×64 | 256×1×1 | 1,048,576 | 40 | 4,096 | 100 | 134.7 |
| Gradiente del sesgo | `cub::DeviceReduce` | 2048×1×1 | 256×1×1 | 524,288 | 30 | 48 | 100 | 105.0 |
| Actualización Adam | `Eigen::EigenMetaKernel` | 45×1×1 | 1024×1×1 | 46,080 | 20 | 0 | 66 | 23.2 |
| Dense (MatMul) | `cutlass_80_simt_sgemm_128x32` | 1×1×32 | 128×1×1 | 4,096 | 126 | 30,720 | 25 | 43.8 |
| Cambio de formato NCHW/NHWC | `SwapDimension1And2InTensor3` | 512×1×1 | 1024×1×1 | 524,288 | 32 | 12,288 | 66 | 42.9 |

Reparto del tiempo de GPU por paso:

| Categoría | % del tiempo de GPU |
|---|---:|
| Conv2D gradiente de filtros | 40.9 |
| Actualización Adam y escalares | 11.7 |
| Conv2D forward | 11.0 |
| MaxPooling gradiente | 9.8 |
| Gradiente ReLU | 8.3 |
| Conv2D gradiente de entrada | 7.5 |
| MaxPooling forward | 3.3 |
| Gradiente del sesgo | 3.0 |
| Dense | 2.0 |
| Resto (formato, pérdida, Dropout, métricas) | 2.5 |

Observaciones del perfil:

- **Kernels elemento a elemento.** Siguen exactamente el diseño de un hilo por elemento: el gradiente de MaxPooling lanza 8,388,608 hilos para 64 × 64 × 64 × 32 valores, y ReLU y Adam usan grids 1D de bloques de 1024.
- **Convoluciones y GEMM.** Asignan varias salidas por hilo y usan teselas en memoria compartida (hasta 48 KiB por bloque). Esa es la causa de su ocupación de 25–33 %: muchos registros por hilo (118–126) limitan los bloques simultáneos por SM, a cambio de reutilizar datos.
- **Gradiente de filtros.** Domina porque con determinismo activado cuDNN debe elegir algoritmos deterministas (`wgrad_alg1`, sgemm, FFT) que no son los más rápidos.
- **Adam y elemento a elemento pesan.** Ocupan casi 20 % del tiempo pese a su poca aritmética: el modelo es pequeño y cada uno de los ≈ 327 lanzamientos por paso tiene un costo fijo.

## Adaptación del código

Cambios en `asl_cpu/train.py`:

| Parte | CPU (`--device cpu`, por defecto) | GPU (`--device gpu`) |
|---|---|---|
| Visibilidad de la GPU | `CUDA_VISIBLE_DEVICES=-1` y `set_visible_devices([], "GPU")` antes de crear el runtime | Se exige al menos una GPU CUDA visible; si no hay, error explícito |
| Memoria GPU | — | `set_memory_growth(True)` para reservar según uso y medir la memoria real |
| Lectura de NPZ, hashes, normalización | CPU | CPU (sin cambios) |
| `tf.data` (mezcla, lotes, prefetch) | `/CPU:0` | `/CPU:0` |
| Pesos, activaciones, gradientes, Adam | `/CPU:0` | `/GPU:0` (`with tf.device(device)`) |
| Verificación | Pesos en `CPU:0` | Pesos **y** variables de Adam en `GPU:0`; si no, la corrida falla |
| Precisión | float32 | float32 estricto (TF32 desactivado) |
| Determinismo, semilla, `jit_compile=False` | Igual | Igual |
| Monitores | psutil cada 1 s y nvidia-smi cada 500 ms | Igual |
| Sincronización | — | `tf.test.experimental.sync_devices()` al final de cada época, de fit y de la evaluación |

El modelo sigue creándose con `asl_cpu.model.create_cnn_base`, sin cambios. Con la misma semilla, los pesos iniciales de la GPU tienen el mismo SHA-256 que la referencia CPU de la parte 1 (`b1feb980…`), así que ambas versiones parten del mismo punto. `--initial-weights` permite forzarlo si cambia el entorno.

## Transferencias CPU–GPU y medición

El pipeline de entrada está en CPU, así que Keras copia cada lote (float32 e int16) de host a GPU. `metrics.json → transfers` registra los bytes:

| Copia host → GPU | Bytes |
|---|---:|
| Entrenamiento, por época | 598.7 MB |
| Validación, por época | 128.3 MB |
| fit completo (20 épocas) | 14.54 GB |
| Evaluación y predicción posteriores | 384.9 MB |

Hacia el host solo vuelven la pérdida y las métricas de cada lote y, al final, las probabilidades de prueba (2,610 × 29 float32). Los pesos se copian una vez al crearse y regresan solo al guardarlos.

14.54 GB en 42.9 s de fit equivalen a ≈ 0.34 GB/s, muy por debajo del ancho de banda de PCIe 4.0. El perfil no muestra las copias como cuello de botella. No se midió el tiempo de copia por separado: queda **dentro** de `train_with_validation_s` y `evaluate_and_predictions_s` y no debe sumarse otra vez al total.

Los relojes usan `time.perf_counter` con las mismas fronteras que la versión CPU (tabla de `INTEGRACION.md`). Como la GPU ejecuta de forma asíncrona, se llama a `sync_devices()` antes de leer el reloj; además, Keras ya espera el resultado de cada época al leer sus métricas. La primera época incluye el trazado de la función de entrenamiento y la selección de algoritmos de cuDNN: 5.48 s, frente a 1.96 s de media en las épocas 2–20. No se excluyó calentamiento, igual que en la referencia CPU.

## Resultados de la ejecución GPU

Comando: `python -m asl_cpu.train --device gpu --output-dir results/gpu_baseline_01`.

| Medición | GPU |
|---|---:|
| Configuración de TensorFlow | 2.077 s |
| Carga y verificación de datos | 1.486 s |
| Normalización CPU | 0.544 s |
| Construcción del modelo y la entrada | 1.520 s |
| Entrenamiento con validación (20 épocas) | 42.868 s |
| Evaluación y predicciones | 0.469 s |
| Ejecución completa del comando | 49.253 s |
| Primera época / media de las épocas 2–20 | 5.48 s / 1.96 s |
| Exactitud final: entrenamiento / validación / prueba | 96.69 % / 74.67 % / 80.88 % |
| Pico de memoria de TensorFlow en fit | 241.5 MiB |
| Memoria usada en la GPU (nvidia-smi, todo el proceso) | 624 MiB de 6,144 |

Uso de la GPU (`gpu.csv`, nvidia-smi cada 500 ms) durante el régimen estable de fit, desde 4 s después de iniciarlo:

| Métrica | Media | Máximo |
|---|---:|---:|
| Utilización de SM | 82 % | 87 % |
| Utilización del controlador de memoria | 44.5 % | 50 % |
| Potencia | 72.1 W | 80.1 W (límite de la laptop: 80 W) |
| Reloj SM | 1,421 MHz | 1,627 MHz |
| Temperatura | 84.8 °C | 87 °C |

Mientras la GPU entrena, el proceso Python usa en promedio 58 % de un núcleo CPU para armar lotes y lanzar kernels. Los demás núcleos quedan casi libres (8.9 % de media por procesador lógico en todo el sistema). La utilización no llega a 100 % porque el lote de 64 imágenes de 64 × 64 genera kernels cortos, con huecos entre lanzamientos y entre lotes.

## Comparación con la referencia CPU en el mismo equipo

La referencia de la parte 1 (`cpu_baseline_01`) se ejecutó en otra laptop (Windows, Core Ultra 5 125U), así que no sirve como denominador. Se repitió la referencia CPU en este equipo con el código actual, que también corre el monitor de nvidia-smi: `python -m asl_cpu.train --device cpu --threads 8 --output-dir results/cpu_rtx_01`. `python -m asl_cpu.compare_runs results/cpu_rtx_01 results/gpu_baseline_01` verificó que ambas corridas tienen la misma configuración, el mismo manifiesto, los mismos arreglos, los mismos pesos iniciales, el mismo número de parámetros y las mismas 20 épocas.

Tiempos de una sola pareja de corridas (los promedios de cinco repeticiones son de la parte 3):

| Fase | CPU, 8 hilos (s) | GPU (s) | CPU / GPU |
|---|---:|---:|---:|
| Configuración de TensorFlow | 2.049 | 2.077 | 0.99 |
| Carga y verificación de datos | 1.498 | 1.486 | 1.01 |
| Normalización CPU | 0.226 | 0.544 | 0.42 |
| Construcción del modelo y la entrada | 0.545 | 1.520 | 0.36 |
| **Entrenamiento con validación** | **254.059** | **42.868** | **5.93** |
| Evaluación y predicciones | 2.622 | 0.469 | 5.60 |
| **Ejecución completa del comando** | **261.274** | **49.253** | **5.30** |
| Primera época | 13.89 | 5.48 | 2.53 |
| Media de las épocas 2–20 | 12.64 | 1.96 | 6.44 |

- **Fases de entrenamiento y evaluación.** Son las únicas que cambian de dispositivo, y en ellas la GPU es 5.6–5.9× más rápida.
- **Fases que siguen en CPU.** Carga, verificación y normalización quedan iguales o varían por ruido. La normalización (0.23 s frente a 0.54 s) es la misma operación NumPy en ambos casos.
- **Construcción del modelo.** Tarda 1 s más en GPU porque ahí se crea el contexto CUDA y se reserva la memoria.
- **Primera época.** Su speedup (2.5×) es menor que el de las demás (6.4×) porque incluye el trazado y la selección de algoritmos de cuDNN, un costo fijo que pesa más cuando el resto es rápido.
- **Total del comando.** El speedup baja a 5.3× porque las partes en CPU no se aceleran (ley de Amdahl).

Uso de recursos durante fit en la corrida CPU: el proceso usó en promedio 580 % de CPU, cerca de 6 de los 8 hilos asignados, con 35–51 % de media en cada uno de los 16 procesadores lógicos del sistema. En `cpu_rtx_01/gpu.csv` la memoria de la GPU se mantiene en 240 MiB (escritorio y terminal) y `nvidia_smi_epoch1.txt` no lista el proceso de entrenamiento. Los picos aislados de utilización (37 de 521 muestras por encima de 5 %) corresponden a la terminal, que también usa la GPU.

Resultados del modelo:

| Métrica | CPU | GPU | GPU − CPU |
|---|---:|---:|---:|
| Exactitud de entrenamiento (época 20) | 96.59 % | 96.69 % | +0.10 pp |
| Exactitud de validación | 74.48 % | 74.67 % | +0.19 pp |
| Pérdida de validación | 1.307 | 1.366 | +0.058 |
| Exactitud de prueba | 82.61 % | 80.88 % | −1.72 pp |
| Pérdida de prueba | 0.832 | 1.011 | +0.178 |
| Predicciones de prueba con la misma clase | 83.7 % | | |

Ambas corridas parten de los mismos pesos y ven los mismos lotes en el mismo orden, pero no son idénticas bit a bit. CPU (oneDNN) y GPU (cuDNN/cuBLAS) usan algoritmos distintos para convoluciones y GEMM (Winograd, FFT, teselas de distinto tamaño), que suman en distinto orden. En float32 eso produce diferencias de redondeo desde el primer paso, que se acumulan durante 3,820 actualizaciones. `enable_op_determinism()` garantiza repetir el mismo resultado en el mismo dispositivo, no igualdad entre dispositivos.

Después de la primera época la diferencia en la exactitud de entrenamiento es de 1 pp y en la época 20 es de 0.1 pp. Las curvas de entrenamiento son prácticamente iguales; la validación y la prueba difieren entre 0.2 y 1.7 pp. El modelo sobreajusta (96.6 % en entrenamiento frente a 75 % en validación), por lo que dos trayectorias cercanas pueden clasificar distinto los ejemplos dudosos de prueba. Eso explica el 83.7 % de coincidencia. No se interpreta la diferencia de 1.7 pp como un efecto de CUDA sobre la calidad del modelo: con una sola pareja de corridas, es del orden de la variación entre trayectorias. La parte 3 puede confirmarlo con las repeticiones.

## Por qué CUDA reduce el tiempo y qué lo limita

- **Más aritmética en paralelo.** La CPU ejecuta las convoluciones con 8 hilos y sus unidades vectoriales. La GPU reparte cada capa en cientos de miles o millones de hilos sobre 30 SM, con memoria compartida para reutilizar teselas de entrada y filtros. El entrenamiento logró ≈ 66 TFLOP / 42.9 s ≈ 1.5 TFLOP/s efectivos en GPU, frente a ≈ 0.26 TFLOP/s en CPU.
- **Los datos no se mueven de más.** Pesos, gradientes y momentos de Adam viven en la GPU todo el entrenamiento; solo cruzan PCIe los lotes de entrada (≈ 0.34 GB/s, lejos del límite del bus).
- **Límites observados:**
  - El modelo y el lote son pequeños: cada paso lanza ≈ 327 kernels de microsegundos, y la GPU llega a 82 % de utilización, no a 100 %.
  - Adam y las operaciones elemento a elemento consumen ≈ 20 % del tiempo de GPU con poca aritmética.
  - El determinismo obliga a cuDNN a usar algoritmos de gradiente de filtros más lentos (41 % del tiempo de GPU).
  - Desactivar TF32 renuncia a las unidades tensoriales de Ampere.
  - La laptop alcanza su límite de 80 W y 87 °C, por lo que el reloj baja durante la corrida.
  - Las fases en CPU y el arranque de CUDA (≈ 6 s) limitan el speedup total.
- **Qué podría acelerar más (experimentos separados, no usados en la comparación):** TF32 o precisión mixta, un lote mayor, `jit_compile=True` (XLA, que fusiona kernels), `prefetch_to_device` y desactivar el determinismo. Cualquiera de ellos cambia la carga o la numérica, así que debe reportarse aparte.

## Evidencias

| Archivo | Qué demuestra |
|---|---|
| `results/gpu_environment.json` | GPU, driver, versiones CUDA/cuDNN, paquetes `nvidia-*` y prueba de matmul/conv en `GPU:0` |
| `results/smoke_gpu/` | Prueba pequeña de integración (1 época, 16/4/4 imágenes por clase); no es un benchmark |
| `results/gpu_baseline_01/train.log` | Log completo: GPU visible, versiones de CUDA/cuDNN, dispositivos de pesos y de Adam, memoria, épocas |
| `results/gpu_baseline_01/metrics.json` | Tiempos por fase, dispositivos, transferencias, memoria, hashes y marcas de tiempo de cada fase |
| `results/gpu_baseline_01/gpu.csv`, `gpu.png` | Utilización, memoria, potencia y reloj de la GPU durante toda la corrida, con las fases marcadas |
| `results/gpu_baseline_01/resources.csv`, `resources.png` | Uso de CPU por núcleo y del proceso durante la corrida GPU |
| `results/gpu_baseline_01/evidence/nvidia_smi_epoch1.txt` | Salida de `nvidia-smi` durante la primera época: el PID del entrenamiento aparece como proceso de cómputo (C) con memoria asignada |
| `results/gpu_kernels/` | Kernels CUDA reales con grid, block, registros, memoria compartida y ocupación |
| `results/cpu_rtx_01/` | Referencia CPU en el mismo equipo, con su `gpu.csv` mostrando la GPU inactiva |
| `results/comparison_cpu_rtx_01_vs_gpu_baseline_01.json` | Salida de `compare_runs` |

**Pendiente del equipo:** capturas de pantalla reales (nvidia-smi o `watch -n 0.5 nvidia-smi`, y el monitor del sistema) durante una corrida GPU y una CPU. Deben tomarlas una persona en el equipo. Los CSV, gráficos y el texto de nvidia-smi son mediciones reales, pero no sustituyen una captura.

## Entrega a la persona 3

- Corrida GPU lista para medir: `python -m asl_cpu.train --device gpu --output-dir results/gpu_NN`, con una carpeta nueva por repetición.
- Referencia CPU en el mismo equipo: `python -m asl_cpu.train --device cpu --threads 8 --output-dir results/cpu_NN`.
- Validación de cada pareja: `python -m asl_cpu.compare_runs results/cpu_NN results/gpu_NN`. Termina con error si cambian la configuración, los datos, los pesos iniciales, las épocas o las muestras.
- Los campos de `times` tienen las mismas fronteras en ambas versiones. Las transferencias están dentro de fit y evaluación. La primera época contiene el trazado y la selección de algoritmos de cuDNN en ambos casos.
- No comparar contra `cpu_baseline_01`, que es otra laptop (Windows, Core Ultra 5). No mezclar corridas `_tf32` con float32 estricto. No usar `--smoke` para rendimiento.
- La GPU de la laptop llega a su límite de potencia y a 87 °C. Conviene dejarla enfriar entre repeticiones y mantenerla conectada a la corriente.
