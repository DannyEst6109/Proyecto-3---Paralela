# Parte 1 Datos y referencia CPU

## Alcance y punto de partida

Esta parte prepara las imágenes ASL en CPU y deja una referencia de entrenamiento CPU reproducible para la comparación posterior con CUDA. Conserva las particiones y la CNN base del laboratorio original. Entrega el análisis del flujo, las dos tablas solicitadas, el código de preparación, la configuración común y evidencia local de ejecución.

El laboratorio de origen es un clasificador de fotografías estáticas. Aunque su motivación menciona reconocimiento frente a una cámara, el código proporcionado no implementa captura continua de video ni un traductor ASL en tiempo real. El Proyecto 3 mide aquí la preparación de un conjunto de imágenes y el entrenamiento de una red; no debe presentar estos tiempos como latencia de una cámara.

El notebook original compara dos CNN, una red densa, PCA/SVM y una CNN con aumentación, además de análisis exploratorio y pruebas con fotografías propias. Se conserva sin cambios en `reference/Laboratorio3.original.ipynb`. Sus salidas históricas no son resultados de esta laptop. La procedencia y el commit están en `reference/README.md`.

## Operaciones del programa original

| No. | Operación identificada | Descripción de su funcionamiento |
|---|---|---|
| 1 | Configuración de rutas y entorno | Localiza carpetas crudas/procesadas, fija semilla y prepara utilidades de exportación. |
| 2 | Inventario de imágenes | Recorre las carpetas de clases y obtiene clase, nombre e índice de cada archivo JPEG. |
| 3 | Inspección y análisis exploratorio | Muestra ejemplos y revisa formato, resolución, distribución, brillo, contraste y similitud entre fotogramas/clases. |
| 4 | División de datos | Asigna bloques contiguos a entrenamiento, validación y prueba, aplica una banda de guarda y selecciona una submuestra uniforme. |
| 5 | Lectura y decodificación | OpenCV abre cada JPEG seleccionado y lo decodifica en memoria CPU. |
| 6 | Reducción y conversión de color | Redimensiona a 64 × 64 con INTER_AREA y convierte BGR a RGB. |
| 7 | Codificación de etiquetas | Asocia cada clase con un entero estable del 0 al 28 y construye X/y de cada partición. |
| 8 | Representaciones numéricas | Convierte RGB uint8 a float32 entre 0 y 1 para las redes; crea gris aplanado para el modelo clásico. |
| 9 | Persistencia y recarga | Guarda arreglos NPZ y tablas Parquet, y los recarga para las etapas de modelado. |
| 10 | Construcción y entrenamiento de modelos | Crea redes Keras y ajusta sus pesos por lotes; también ajusta el modelo PCA/SVM sobre otra representación. |
| 11 | Validación y selección | Evalúa pérdida/exactitud, usa parada temprana en el original y compara modelos por exactitud de validación. |
| 12 | Aumentación y nuevo entrenamiento | Anteponer rotaciones, zoom, traslación y contraste a la mejor CNN genera un experimento adicional en el original. |
| 13 | Evaluación y predicción | Clasifica prueba y fotografías propias y produce métricas, matrices de confusión y figuras. |

En la implementación de esta parte se extraen las operaciones 1, 2 y 4 a 11 necesarias para una comparación de la CNN base. No se ejecutan todos los modelos ni el análisis exploratorio en cada benchmark. La representación gris/PCA/SVM y las fotografías propias no forman parte de la carga medida de la CNN base.

## Operaciones repetitivas

Los conteos siguientes corresponden a una preparación completa y, donde se indica, a un entrenamiento completo de la configuración común. No son conteos de kernels CUDA ni instrucciones de máquina.

| Operación repetitiva | Datos que procesa | Cantidad de repeticiones |
|---|---|---|
| Registro de archivos de entrenamiento | Rutas, clase e índice | 87,000 imágenes, 3,000 por cada una de las 29 clases |
| Selección por clase y partición | Inventario con bloque, posición y banda de guarda | 29 clases × 3 particiones = 87 selecciones |
| Lectura/decodificación JPEG seleccionada | Archivos originales de 200 × 200 RGB | 17,400 archivos por preparación |
| Resize y BGR a RGB | Píxeles de cada imagen seleccionada | 17,400 redimensionamientos y 17,400 conversiones |
| Asignación de etiquetas | Nombre de clase de cada imagen | 17,400 etiquetas por preparación |
| Conversión a float32 y normalización | 64 × 64 × 3 valores por imagen | 17,400 imágenes, 213,811,200 valores, por inicio de entrenamiento completo |
| Lotes de entrenamiento | 12,180 imágenes y etiquetas | ceil(12180 / 64) = 191 lotes por época; 3,820 lotes en 20 épocas |
| Forward, pérdida y actualización Adam | Tres convoluciones y capas finales del modelo por lote | Una actualización del optimizador por lote de entrenamiento: 3,820 actualizaciones |
| Validación durante fit | 2,610 imágenes reservadas | ceil(2610 / 64) = 41 lotes por época; 820 lotes en 20 épocas |
| Evaluación posterior | Validación y prueba, 2,610 imágenes cada una | 41 lotes por conjunto, 82 lotes adicionales en total |
| Predicción posterior | 2,610 imágenes de prueba | 41 lotes; sus probabilidades se guardan para revisión |

El lote final de entrenamiento contiene 20 imágenes y se conserva; no se descartan muestras para hacer divisible el conjunto por 64. Los conteos de entrenamiento no incluyen la ejecución pequeña de integración. En el notebook original la parada temprana cambia el número de épocas efectivo, por lo que no corresponde atribuirle estos conteos de 20 épocas fijas.

## Datos y decisiones de preparación

El dataset ASL Alphabet contiene 29 clases: A-Z, del, nothing y space. La copia local verificada tiene 87,000 imágenes de entrenamiento y 28 imágenes oficiales de prueba. Se construye una prueba interna por bloques; el pequeño conjunto oficial no sustituye la partición de evaluación.

Se conserva la partición original: 30 bloques de 100 índices, permutados con `np.random.default_rng(42)`. Se asignan 21 bloques a entrenamiento, 4 a validación y 5 a prueba. Se excluyen 25 posiciones de cada extremo de todos los bloques. Dentro de cada clase/partición se eligen posiciones uniformemente espaciadas, sin repetir archivos.

| Partición | Por clase | Total |
|---|---:|---:|
| Entrenamiento | 420 | 12,180 |
| Validación | 90 | 2,610 |
| Prueba | 90 | 2,610 |
| Total | 600 | 17,400 |

Las pruebas comprueban que un bloque no aparezca en más de una partición, que no haya archivos repetidos y que las posiciones seleccionadas respeten el margen. También ejecutan las celdas originales 34 y 35 y comparan sus particiones con las nuevas: deben ser exactamente iguales. Esta estrategia reduce el riesgo de repartir fotogramas vecinos entre conjuntos; no demuestra por sí sola independencia de personas, entornos o todas las posibles fuentes de fuga.

La transformación de imágenes conserva `cv2.INTER_AREA`, resolución 64 × 64 y color RGB. Se exporta uint8 para evitar almacenar flotantes innecesariamente. La normalización se ejecuta en CPU una vez por corrida, antes de construir el pipeline de entrenamiento, y tiene un tiempo separado. No se regeneran representaciones grises que la CNN no necesita.

El modo de un trabajador ejecuta el bucle de preparación en un proceso. El modo de varios trabajadores distribuye imágenes independientes entre procesos CPU mediante `ProcessPoolExecutor`, conserva el orden con `map` y restringe OpenCV a un hilo por proceso para evitar sobreasignación. La creación y comunicación de procesos se incluyen en `images_s`. Este paralelismo CPU no utiliza CUDA.

Los hashes de imágenes, etiquetas, clases y manifiesto permiten verificar equivalencia entre modos y equipos. Una coincidencia del manifiesto demuestra la misma selección y orden; las huellas de los arreglos comprueban además el resultado de las transformaciones. Los hashes no sustituyen las verificaciones de calidad de datos.

## Referencia del modelo CPU

La arquitectura conserva tres bloques Conv2D/MaxPooling2D con 32, 64 y 128 filtros, convoluciones 3 × 3 con padding same y ReLU. Después usa Flatten, Dense de 128 con ReLU, Dropout 0.3 y Dense de 29 con softmax. Tiene 1,145,693 parámetros entrenables.

Se mantiene Adam con tasa inicial 0.001, pérdida sparse categorical crossentropy, lotes de 64 y precisión float32. Se fija la semilla Python/NumPy/TensorFlow con `tf.keras.utils.set_random_seed(42)` y se habilitan operaciones deterministas. El entrenamiento mezcla las muestras por época mediante un pipeline `tf.data` determinista con un hilo privado y prefetch de un lote.

Se elimina la parada temprana para que ambas versiones realicen 20 épocas. No se incluye aumentación en esta comparación inicial. Son cambios del protocolo experimental respecto del notebook, no afirmaciones de que el original tuviera esos ajustes. No se exige exactitud idéntica a sus salidas históricas porque cambiaron el entorno, la inicialización controlada, el orden de lotes y el número de épocas.

El script CPU establece `CUDA_VISIBLE_DEVICES=-1` antes de importar TensorFlow, oculta los dispositivos GPU, crea el modelo y ejecuta fit/evaluación en `/CPU:0`. Verifica y registra los dispositivos reales de los pesos. El entrenamiento usa ocho hilos intraoperación de TensorFlow y uno interoperación por defecto; por eso debe describirse como referencia CPU, no como aritmética estrictamente monohilo. La preparación secuencial sí utiliza un solo trabajador.

## Medición y evidencia

Los tiempos se miden con `time.perf_counter`. La preparación registra inventario, partición, lectura/transformación y exportación por separado. El entrenamiento registra setup, carga/verificación, normalización, construcción del modelo/entrada, fit con validación y evaluación posterior. La definición exacta y la prevención de doble conteo están en `INTEGRACION.md`.

El tiempo de fit incluye la primera compilación/trazado y la validación al final de cada época. No se excluye un calentamiento. La escritura de progreso y de logs por época también forma parte de esa medición y debe mantenerse en el protocolo GPU. Las fases de carga/verificación incluyen trabajo de hashes; el costo de ese control debe identificarse al interpretar los resultados.

Una ejecución inicial de preparación produjo:

| Fase | Un trabajador CPU (s) | Cuatro trabajadores CPU (s) |
|---|---:|---:|
| Inventario | 1.788 | 1.603 |
| Partición | 0.689 | 0.525 |
| Lectura, resize, RGB y etiquetas | 15.857 | 7.399 |
| Exportación NPZ y manifiestos | 12.026 | 12.576 |
| Preparación completa, con verificación | 30.966 | 22.723 |

Son observaciones de una ejecución por modo, con el modo secuencial ejecutado primero. No constituyen promedios, un experimento de caché controlada ni un speedup CUDA. La caché del sistema de archivos y la carga externa pueden favorecer la segunda corrida. Las huellas confirman que ambos modos produjeron imágenes y etiquetas exactamente iguales.

La compresión/exportación permanece cerca de doce segundos en ambas observaciones y limita la mejora total aun cuando baja el tiempo de transformación de imágenes. Debe evaluarse con repeticiones antes de concluir qué número de trabajadores conviene; cuatro es una opción implementada y validada, no un óptimo demostrado.

El entrenamiento completo local guarda sus resultados en `results/cpu_baseline_01/metrics.json`. Su log, tiempos por época y monitor constituyen evidencia de la ejecución; la prueba de integración se identifica por `run_kind=smoke` y no se mezcla con ella. Las mediciones finales repetidas CPU/GPU corresponden a la persona 3 y deben hacerse preferentemente en la misma computadora NVIDIA.

La ejecución local completa terminó sus 20 épocas con los siguientes resultados:

| Medición | Resultado |
|---|---:|
| Entrenamiento con validación por época | 631.331 s |
| Carga y verificación de datos preparados | 1.827 s |
| Normalización CPU | 0.874 s |
| Evaluación y exportación de predicciones | 5.529 s |
| Ejecución completa del comando de entrenamiento | 643.229 s |
| Exactitud final de entrenamiento | 96.626 % |
| Exactitud final de validación | 71.034 % |
| Exactitud final de prueba | 80.000 % |

El total de 643.229 s ya contiene las fases del comando; no se deben sumar otra vez. Tampoco incluye la preparación anterior desde JPEG. Estos valores son una sola corrida, no promedios. El equipo fue un Intel Core Ultra 5 125U con 12 núcleos, 14 procesadores lógicos y aproximadamente 32 GB de RAM. Se guardaron 598 muestras del monitor. Hubo otras aplicaciones abiertas y actividades de documentación/captura, por lo que no se debe interpretar esta corrida como una medición de laboratorio con carga externa controlada.

La diferencia entre entrenamiento y validación indica sobreajuste en esta configuración. Se mantienen los pesos de la época 20 para respetar la carga fija; no se seleccionó un checkpoint por el resultado de prueba. El objetivo de esta referencia es comparar rendimiento computacional bajo el mismo trabajo, sin afirmar que 20 épocas sean el mejor ajuste predictivo. Las probabilidades guardadas reproducen la exactitud de prueba de 80% al recalcularla.

Las seis pruebas automatizadas finalizaron correctamente, incluidas equivalencia con las particiones originales, conservación de píxeles/etiquetas entre trabajadores y equivalencia de arquitectura/capas/parámetros con la CNN base original. El entorno no reportó dependencias incompatibles. La prueba pequeña y la corrida completa partieron de la misma huella de pesos iniciales.

`resources.csv` registra cada segundo CPU por procesador lógico, CPU del proceso Python y sus hijos y memoria RSS. El gráfico generado desde ese CSV se identifica como gráfico de mediciones, no como captura. La captura del Administrador de tareas durante el entrenamiento se guarda en `results/cpu_baseline_01/evidence/taskmanager_cpu_training.png` y conserva sus valores originales.

Los gráficos del Administrador de tareas incluyen otras aplicaciones. La GPU Intel puede tener actividad de escritorio mientras el modelo está en CPU; la evidencia de ubicación del entrenamiento es la lista de GPU visibles vacía y los pesos en `/CPU:0`, registrada en el log y los metadatos. Esta laptop no permite probar CUDA.

## Justificación de la separación CPU y GPU

La preparación involucra rutas, metadatos, lectura de archivos y decodificación JPEG. Los núcleos CPU pueden procesar imágenes independientes y gestionar el sistema de archivos sin transferir imágenes crudas a GPU para cada operación. El beneficio depende del almacenamiento, la caché y el costo de iniciar y comunicar trabajadores; no es correcto afirmar que CPU siempre gana para cualquier transformación de imágenes.

El entrenamiento concentra operaciones numéricas regulares y repetidas sobre lotes: convoluciones, productos matriciales, activaciones y cálculo de gradientes. Esa carga es una candidata a la GPU CUDA, con suficientes datos y cómputo para amortizar las transferencias. La persona 2 debe registrar qué operaciones ejecuta realmente TensorFlow en GPU y la persona 3 debe medir su costo completo.

Se entrega a CUDA una carga comparable, no una red más pequeña ni menos épocas. Si la GPU no reduce el tiempo total, el informe debe explicar el resultado a partir de mediciones, considerando transferencias, lote, tamaño del modelo, sincronización y preparación CPU.

## Entrega al equipo

La persona 2 recibe `asl_cpu/model.py`, `config/baseline.json`, el contrato NPZ y los manifiestos/hashes. La persona 3 recibe los tiempos con sus fronteras, logs y monitores, y la referencia CPU ejecutable. Los pesos iniciales pueden compartirse fuera de Git para igualar la inicialización.

Las mediciones CUDA y el informe PDF final se completarán al integrar las tres partes. Cada integrante debe registrar sus cambios y contribuciones en el repositorio compartido.
