# Parte 3: experimento repetido y comparación

| Carpeta | Cómo se genera | Qué contiene |
|---|---|---|
| `rtx3060/` | `bash scripts/parte3_benchmark.sh` en la laptop NVIDIA | 5 repeticiones de CPU, GPU con Adam de Keras y GPU con kernel Adam Numba; preparación con 1 y 4 procesos |
| `colab_t4/` | `notebooks/Proyecto3_Colab_CUDA.ipynb` en Google Colab | El mismo experimento en una GPU T4 (la CPU de Colab usa 3 repeticiones por su duración) |
| `comparacion/` | `python -m asl_cpu.analyze_benchmark results/parte3/rtx3060 results/parte3/colab_t4 --output-dir results/parte3/comparacion` | Tabla y gráfico entre entornos |

Dentro de cada entorno:

- `manifest.json`: plan, comandos exactos, orden, tiempo de cada proceso, estado de la GPU al inicio y al final, y la verificación de que todas las corridas comparten la misma carga.
- `prepare/workers<N>/rep<NN>/`: metadatos y monitor de cada repetición de preparación (los arreglos se verifican contra `data/processed/baseline` y luego se borran).
- `train/<config>/rep<NN>/`: salida completa de `asl_cpu.train` (logs, `metrics.json`, `epochs.csv`, `progress.json`, `resources.csv`, `gpu.csv`, `evidence/`). `warmup01/` es el calentamiento descartado.
- `logs/`: salida de consola de cada proceso.
- `capturas/`: capturas de pantalla tomadas durante las corridas. En `rtx3060/capturas/` están las 43 capturas de la terminal en orden cronológico, con el número de corrida, la configuración y la repetición en el nombre. En `colab_t4/capturas/` están el monitor durante el CPU y la GPU y, en `notebook/`, la captura completa del notebook ejecutado dividida en secciones.
- `analisis/`: `RESULTADOS.md`, `corridas_*.csv`, `resumen_por_configuracion.csv`, `speedup.csv`, `tiempos_por_epoca.csv`, `resumen.json` y gráficos PNG.

Los pesos (`*.weights.h5`) y las predicciones (`predictions.npz`) quedan fuera de Git. Si se borran, el análisis sigue funcionando, solo omite la columna de coincidencia de predicciones con la CPU.
