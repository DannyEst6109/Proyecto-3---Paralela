#!/usr/bin/env bash
# Corre las tres ejecuciones de la parte 2 y pausa para tomar capturas de pantalla.
# Uso (desde la raíz del repositorio):  bash scripts/evidencia_parte2.sh
# Antes: abrir otra terminal con `watch -n 0.5 nvidia-smi` y el Monitor del sistema (pestaña Recursos).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
STAMP=$(date +%Y%m%d_%H%M)
OUT=results/capturas_$STAMP
mkdir -p "$OUT"

pause() {
  echo
  echo ">>> $1"
  read -r -p ">>> Presiona Enter para continuar... " _
}

echo "Evidencia de la parte 2 en $OUT"
$PY -m asl_cpu.gpu_check --output "$OUT/gpu_environment.json"
nvidia-smi > "$OUT/nvidia_smi_inicio.txt"

pause "1/3 CPU (~4.5 min). Durante el entrenamiento captura el Monitor del sistema (núcleos ocupados) y nvidia-smi (sin proceso python). Guarda las imágenes en $OUT/"
$PY -m asl_cpu.train --device cpu --threads 8 --output-dir "$OUT/cpu"

pause "2/3 GPU con Adam de Keras (~50 s). Captura nvidia-smi (python usando memoria y ~80 % de utilización) y el Monitor del sistema."
$PY -m asl_cpu.train --device gpu --output-dir "$OUT/gpu_keras"

pause "3/3 GPU con el kernel Adam en Numba (~75 s). Captura nvidia-smi y el Monitor del sistema."
$PY -m asl_cpu.train --device gpu --optimizer numba --output-dir "$OUT/gpu_numba"

echo
echo "Gráficos y comparaciones..."
for run in cpu gpu_keras gpu_numba; do
  $PY -m asl_cpu.plot_gpu "$OUT/$run" > /dev/null
  $PY -m asl_cpu.plot_resources "$OUT/$run/resources.csv" "$OUT/$run/resources.png" > /dev/null
done
$PY -m asl_cpu.compare_runs "$OUT/cpu" "$OUT/gpu_keras" --output "$OUT/comparacion_cpu_vs_gpu_keras.json" > /dev/null
$PY -m asl_cpu.compare_runs "$OUT/cpu" "$OUT/gpu_numba" --output "$OUT/comparacion_cpu_vs_gpu_numba.json" > /dev/null
$PY - "$OUT" <<'EOF'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
for name in ("cpu", "gpu_keras", "gpu_numba"):
    m = json.loads((out / name / "metrics.json").read_text(encoding="utf-8"))
    t = m["times"]
    print(f"{name:10} fit {t['train_with_validation_s']:7.1f} s | total {t['run_wall_s']:7.1f} s | "
          f"prueba {100 * m['test']['accuracy']:.2f} % | {m['run_kind']}")
EOF
echo
echo "Listo. Revisa que tus capturas estén en $OUT/ y agrégalas al repositorio."
