# Evidencia local de la parte 1

Esta carpeta contiene ejecuciones reales de preparación y entrenamiento CPU. No contiene resultados CUDA ni promedios de cinco entrenamientos. Los pesos y predicciones permanecen locales y están excluidos de Git; los logs, metadatos, tablas y capturas sí pueden versionarse.

| Carpeta | Contenido |
|---|---|
| preparation_sequential_01 | Metadatos y monitor de la preparación de 17,400 imágenes con un trabajador |
| preparation_parallel4_01 | Metadatos y monitor de la preparación equivalente con cuatro trabajadores |
| smoke_cpu | Prueba de integración con una época y 16/4/4 imágenes por clase; no es un benchmark |
| cpu_baseline_01 | Entrenamiento CPU de referencia con todas las particiones y 20 épocas |

`hardware.json` registra procesador, GPU visible por Windows y RAM del equipo. El entorno Python y las librerías de cada ejecución están en sus propios metadatos. `validation.log` registra las comprobaciones finales del código.

Las carpetas `preparation_*` son copias de los metadatos y monitores originales guardados junto a los datos en `data/processed/`. No se copiaron los arreglos pesados a resultados. La equivalencia se comprueba con `python -m asl_cpu.compare_preparations data/processed/baseline data/processed/parallel4`.

Cada CSV de recursos conserva la fase activa y una muestra aproximadamente cada segundo. Los gráficos derivados no son capturas de pantalla. La captura del Administrador de tareas está en `cpu_baseline_01/evidence/`, junto con su instante de captura, y conserva sus píxeles originales.

Las mediciones locales se obtuvieron en una computadora de uso general, con otras aplicaciones abiertas y tareas de documentación/captura durante el entrenamiento. Son una referencia funcional inicial. No se mantuvo una condición de carga externa controlada y no se deben presentar como benchmark definitivo. El monitor permite revisar parte de esa variación. La fase de compresión incluye su costo real y las ejecuciones de preparación no controlan la caché de archivos.

La referencia completa usa los pesos de la última época, sin parada temprana ni restauración de la mejor época de validación. Sus exactitudes no deben compararse directamente con la selección automática del notebook histórico como si fueran el mismo protocolo.

Para calcular un speedup atribuible a CUDA, repetir CPU/GPU en el equipo NVIDIA y conservar la misma configuración, precisión, datos, modelo, política de calentamiento y límites de medición. La persona 3 consolidará los promedios y la dispersión del experimento final.
