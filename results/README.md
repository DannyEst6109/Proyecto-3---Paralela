# Evidencia local de las partes 1 y 2

Esta carpeta contiene ejecuciones reales de preparación, entrenamiento CPU y entrenamiento CUDA. No contiene promedios de cinco entrenamientos (parte 3). Los pesos y predicciones permanecen locales y están excluidos de Git; los logs, metadatos, tablas y capturas sí pueden versionarse.

| Carpeta | Contenido |
|---|---|
| preparation_sequential_01 | Metadatos y monitor de la preparación de 17,400 imágenes con un trabajador |
| preparation_parallel4_01 | Metadatos y monitor de la preparación equivalente con cuatro trabajadores |
| smoke_cpu | Prueba de integración con una época y 16/4/4 imágenes por clase; no es un benchmark |
| cpu_baseline_01 | Entrenamiento CPU de referencia con todas las particiones y 20 épocas |
| gpu_environment.json | Parte 2: GPU, driver, versiones CUDA/cuDNN y prueba de matmul/conv en `/GPU:0` (RTX 3060 Laptop, Linux) |
| smoke_gpu | Parte 2: prueba de integración GPU, una época; no es un benchmark |
| gpu_baseline_01 | Parte 2: entrenamiento CUDA completo, 20 épocas, float32 estricto, con `gpu.csv`/`gpu.png` y salida de nvidia-smi |
| cpu_rtx_01 | Parte 2: referencia CPU repetida en el mismo equipo NVIDIA con el código actual, para comparar con `gpu_baseline_01` |
| gpu_kernels | Parte 2: perfil CUPTI de 10 pasos con grid/block/ocupación de cada kernel CUDA |
| comparison_cpu_rtx_01_vs_gpu_baseline_01.json | Parte 2: misma carga verificada, diferencias de resultados y speedup de una sola pareja |
| gpu_numba_adam_01 | Parte 2: entrenamiento completo con la actualización de Adam hecha por el kernel CUDA propio (Numba) |
| smoke_gpu_numba | Parte 2: prueba de integración del Adam Numba, una época |
| comparison_cpu_rtx_01_vs_gpu_numba_adam_01.json | Parte 2: misma carga verificada para la corrida con kernel propio |
| capturas | Parte 2: corrida de evidencia del 13/11/2026 (CPU, GPU Keras, GPU kernel Numba) con capturas de pantalla de nvidia-smi, logs, gráficos, comparaciones y benchmark de kernels |
| kernels_numba | Parte 2: forward de la CNN con kernels propios (exactitud vs TensorFlow) y tiempos por capa CPU / Numba / cuDNN |

`hardware.json` registra procesador, GPU visible por Windows y RAM del equipo de la parte 1; el equipo de la parte 2 está descrito en `gpu_environment.json` y en `docs/PARTE_2.md`. El entorno Python y las librerías de cada ejecución están en sus propios metadatos. `validation.log` registra las comprobaciones finales del código.

Las carpetas `preparation_*` son copias de los metadatos y monitores originales guardados junto a los datos en `data/processed/`. No se copiaron los arreglos pesados a resultados. La equivalencia se comprueba con `python -m asl_cpu.compare_preparations data/processed/baseline data/processed/parallel4`.

Cada CSV de recursos conserva la fase activa y una muestra aproximadamente cada segundo. Los gráficos derivados no son capturas de pantalla. La captura del Administrador de tareas está en `cpu_baseline_01/evidence/`, junto con su instante de captura, y conserva sus píxeles originales.

Las mediciones locales se obtuvieron en una computadora de uso general, con otras aplicaciones abiertas y tareas de documentación/captura durante el entrenamiento. Son una referencia funcional inicial. No se mantuvo una condición de carga externa controlada y no se deben presentar como benchmark definitivo. El monitor permite revisar parte de esa variación. La fase de compresión incluye su costo real y las ejecuciones de preparación no controlan la caché de archivos.

La referencia completa usa los pesos de la última época, sin parada temprana ni restauración de la mejor época de validación. Sus exactitudes no deben compararse directamente con la selección automática del notebook histórico como si fueran el mismo protocolo.

`cpu_baseline_01` (Windows, Core Ultra 5) no debe usarse como denominador frente a `gpu_baseline_01` (Linux, i7-11800H + RTX 3060); para eso existe `cpu_rtx_01`. Para el speedup final, repetir CPU/GPU en el equipo NVIDIA y conservar la misma configuración, precisión, datos, modelo, política de calentamiento y límites de medición. La persona 3 consolidará los promedios y la dispersión del experimento final.
