#!/usr/bin/env bash
# Parte 3: experimento final en el equipo NVIDIA (laptop RTX 3060, Linux).
#
# Mide en la MISMA máquina y con el MISMO entorno:
#   - preparación de imágenes con 1 y 4 procesos CPU (5 repeticiones c/u, 1 calentamiento descartado);
#   - entrenamiento CPU, GPU con Adam de Keras y GPU con el kernel Adam en Numba
#     (5 repeticiones c/u, orden intercalado y rotado, 1 corrida smoke de calentamiento por configuración).
# Después genera tablas, speedup por fase y gráficos en $OUT/analisis/.
#
# Uso, desde la raíz del repositorio:   bash scripts/parte3_benchmark.sh
# Variables opcionales: PY (intérprete), OUT (carpeta), REPS (repeticiones). Argumentos extra se pasan
# a asl_cpu.benchmark (por ejemplo --dry-run para ver el plan, o --pause-before '' para no pausar).
#
# Antes de empezar: laptop conectada a la corriente, modo de energía de alto rendimiento, otras
# aplicaciones cerradas. Abrir en otra terminal `watch -n 0.5 nvidia-smi` y el Monitor del sistema
# (pestaña Recursos) para las capturas. Duración aproximada: 60 minutos. Es reanudable: si se
# interrumpe, volver a ejecutar el mismo comando continúa donde quedó.
set -eu
#set -o pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
OUT=${OUT:-results/parte3/rtx3060}
REPS=${REPS:-5}

command -v nvidia-smi >/dev/null || { echo "No se encontró nvidia-smi: este script es para el equipo NVIDIA."; exit 1; }
[ -f data/processed/baseline/metadata.json ] || { echo "Falta data/processed/baseline. Ejecute: $PY -m asl_cpu.prepare --workers 4"; exit 1; }
mkdir -p "$OUT"
[ -f "$OUT/gpu_environment.json" ] || $PY -m asl_cpu.gpu_check --output "$OUT/gpu_environment.json"
[ -f "$OUT/nvidia_smi_inicio.txt" ] || nvidia-smi > "$OUT/nvidia_smi_inicio.txt"

$PY -m asl_cpu.benchmark --output-dir "$OUT" --label rtx3060 --reps "$REPS" --threads 8 \
    --prepare-workers 1,4 --prepare-reps "$REPS" --prepare-warmup 1 --warmup smoke \
    --cooldown 30 --cooldown-gpu-temp 60 --cooldown-max-wait 180 \
    --pause-before workers4:1,cpu:1,gpu_keras:1,gpu_numba:1 "$@"

for arg in "$@"; do [ "$arg" = "--dry-run" ] && exit 0; done
$PY -m asl_cpu.analyze_benchmark "$OUT"
echo
echo "Listo. Reporte: $OUT/analisis/RESULTADOS.md"
echo "Guarde las capturas en $OUT/capturas/ y suba la carpeta al repositorio (los pesos y predicciones quedan fuera por .gitignore)."
