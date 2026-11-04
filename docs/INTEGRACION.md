# Integración de las partes del proyecto

## Entrega de la persona 1 a la persona 2

Compartir el código y `config/baseline.json`. Cada integrante obtiene el dataset por el enlace documentado en `DATASET.md`. Se puede copiar la carpeta `data/processed/baseline/` de forma externa a Git, o regenerarla con `python -m asl_cpu.prepare --workers 1`.

El contrato de `asl_64x64.npz` es:

| Clave | Forma | Tipo y significado |
|---|---|---|
| X_train | 12180 × 64 × 64 × 3 | uint8 RGB, rango 0..255 |
| y_train | 12180 | int16, índice de clase 0..28 |
| X_val | 2610 × 64 × 64 × 3 | uint8 RGB, rango 0..255 |
| y_val | 2610 | int16, índice de clase 0..28 |
| X_test | 2610 × 64 × 64 × 3 | uint8 RGB, rango 0..255 |
| y_test | 2610 | int16, índice de clase 0..28 |
| clases | 29 | cadenas Unicode, orden alfabético A-Z, del, nothing, space |

Se carga con `np.load(ruta, allow_pickle=False)`. Normalizar en CPU mediante `X.astype(np.float32) / np.float32(255.0)`. No cambiar canales, interpolación, etiquetas ni orden del conjunto de prueba.

Antes de comparar, verificar `config_sha256`, `partition_sha256` y `array_sha256` contra `metadata.json`. El hash de un ZIP puede cambiar aunque sus arreglos sean equivalentes; para comparar preparaciones usar las huellas de los arreglos y el manifiesto.

La arquitectura está en `asl_cpu.model.create_cnn_base(tf, config, 29)`. Esta función no selecciona el dispositivo y puede reutilizarse en la versión CUDA. La versión GPU debe conservar los 1,145,693 parámetros, Adam 0.001, pérdida sparse categorical crossentropy, lote 64, float32 y 20 épocas. Mantener `jit_compile=False`, misma política de determinismo y mismo orden de lotes en la primera comparación. Registrar cualquier cambio posterior como un experimento separado.

Para igualar el punto de partida puede copiarse `initial.weights.h5` de la ejecución CPU y cargarlo antes de entrenar. Reinicializar Adam en cada repetición. La semilla no garantiza por sí sola resultados bit a bit idénticos entre distintas versiones de librerías o dispositivos; comparar las huellas iniciales y reportar exactitud/pérdida finales.

La normalización, los archivos y el pipeline de entrada permanecen en CPU. Si se añade aumentación posteriormente, debe hacerse en CPU y con el mismo procedimiento en ambos casos. La persona 2 debe confirmar con el docente si acepta CUDA a través de TensorFlow o exige kernels propios y cómo espera la explicación de hilos, bloques y grids.

## Entrega de la persona 1 a la persona 3

La ejecución CPU se invoca como proceso nuevo mediante `python -m asl_cpu.train`. Usar una carpeta distinta para cada repetición y conservar `metrics.json`, `epochs.csv`, `train.log`, `resources.csv` y las capturas.

| Campo de tiempo | Qué incluye |
|---|---|
| inventory_s | Recorrido y validación de los nombres de las 87,000 imágenes |
| partition_s | Asignación de bloques, banda de guarda y selección |
| images_s | Lectura, decodificación JPEG, resize, RGB, etiquetas y gestión de trabajadores |
| export_s | Compresión NPZ y exportación de manifiestos |
| prepare_wall_s | Preparación completa, hashes y registro de entorno; no es la suma exacta de los cuatro anteriores |
| tensorflow_setup_s | Import y configuración de TensorFlow y dispositivos |
| load_and_verify_s | Lectura/descompresión de datos preparados y verificación de hashes |
| normalize_cpu_s | Conversión uint8 a float32 y división entre 255 |
| build_model_and_input_s | Construcción de datasets, modelo y optimizador; no incluye el guardado inicial de pesos |
| train_with_validation_s | fit completo: 20 épocas, validación por época y escritura de progreso/logs |
| evaluate_and_predictions_s | Evaluación posterior en validación/prueba, predicción y exportación de predicciones |
| save_final_weights_s | Escritura de los pesos finales |
| run_wall_s | Ejecución del entrenamiento desde el setup hasta el guardado; incluye tareas adicionales entre fases |

Las fases no se deben sumar de nuevo a `run_wall_s`. Para un total que incluya preparar desde JPEG, medir el proceso completo o reportar explícitamente `prepare_wall_s + run_wall_s`; este último es un total compuesto de dos comandos, no una medición continua. La normalización ya está dentro de `run_wall_s`. No comparar una preparación desde JPEG con una carga desde NPZ sin indicarlo.

La ejecución local no tiene calentamiento excluido. El primer trazado está dentro de fit. Si la persona 3 incorpora calentamiento, debe usar la misma política en CPU y GPU y restaurar pesos y estado del optimizador antes de medir. No usar `--smoke` para conclusiones de rendimiento.

En GPU se debe esperar la terminación real del trabajo antes de detener el reloj y explicar cómo se incluyen las transferencias CPU-GPU. Si se desglosan transferencias, no sumarlas dos veces al total. El monitor y el guardado por época introducen sobrecosto: conservarlos también en la comparación.

La persona 3 lidera las cinco repeticiones propuestas, los promedios, dispersión, tablas, gráficos y speedup final. Medir CPU y GPU en la misma computadora NVIDIA y el mismo entorno cuando sea posible. Las cifras de esta laptop no son el denominador apropiado para atribuir una mejora exclusivamente a CUDA en otra computadora.

## Uso de la evidencia

`resources.csv` contiene actividad por procesador lógico de todo el sistema, RSS del árbol Python y CPU del árbol Python. En la columna del proceso 100% significa un procesador lógico ocupado y puede superarse; no equivale al porcentaje global del Administrador de tareas. RSS de procesos e hijos puede contar páginas compartidas más de una vez.

La captura del Administrador de tareas muestra el sistema durante la ejecución y no atribuye toda su carga a Python. El uso de la GPU Intel por otras aplicaciones no demuestra uso de CUDA ni invalida que los pesos del modelo estén en CPU. La evidencia del dispositivo se encuentra en el log y en `weight_devices`/`visible_gpus` de `metrics.json`.

Para las mediciones finales cerrar otras cargas cuando sea posible, mantener la alimentación y configuración de energía y registrar el estado del equipo. No modificar el dataset, la cantidad de épocas ni la precisión para favorecer una versión.
