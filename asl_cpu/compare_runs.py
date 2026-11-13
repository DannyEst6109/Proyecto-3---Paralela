"""Compara una ejecución CPU con una GPU: carga de trabajo, resultados del modelo y tiempos.

Uso: python -m asl_cpu.compare_runs results/cpu_rtx_01 results/gpu_01 [--output comparacion.json]

Verifica que ambas corridas usen la misma configuración, datos y pesos iniciales, y reporta
diferencias de pérdida/exactitud por época, métricas finales y acuerdo de predicciones.
El speedup que imprime es de una sola pareja de corridas; los promedios son de la parte 3.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SAME_WORKLOAD = ("config_sha256", "partition_sha256", "used_array_sha256", "initial_weights_sha256",
                 "model_parameters", "epochs_completed", "sample_counts")
PHASES = ("tensorflow_setup_s", "load_and_verify_s", "normalize_cpu_s", "build_model_and_input_s",
          "train_with_validation_s", "evaluate_and_predictions_s", "save_final_weights_s", "run_wall_s")


def load(run: Path) -> dict:
    metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    if metrics.get("status") != "complete":
        raise SystemExit(f"{run} no terminó (status={metrics.get('status')})")
    return metrics


def compare(cpu_dir: Path, gpu_dir: Path) -> dict:
    cpu, gpu = load(cpu_dir), load(gpu_dir)
    report = {"cpu_run": str(cpu_dir), "gpu_run": str(gpu_dir),
              "devices": {"cpu": cpu.get("weight_devices"), "gpu": gpu.get("weight_devices")}}
    report["same_workload"] = {key: cpu.get(key) == gpu.get(key) for key in SAME_WORKLOAD}

    final = {}
    for split in ("validation", "test"):
        for metric in ("accuracy", "loss"):
            a, b = cpu[split][metric], gpu[split][metric]
            final[f"{split}_{metric}"] = {"cpu": a, "gpu": b, "gpu_minus_cpu": b - a}
    report["final_metrics"] = final

    epochs_cpu = pd.read_csv(cpu_dir / "epochs.csv")
    epochs_gpu = pd.read_csv(gpu_dir / "epochs.csv")
    diff = (epochs_gpu.drop(columns="epoch") - epochs_cpu.drop(columns="epoch")).abs()
    report["epoch_abs_diff"] = {
        "max": {c: float(diff[c].max()) for c in diff.columns},
        "first_epoch": {c: float(diff[c].iloc[0]) for c in diff.columns},
        "last_epoch": {c: float(diff[c].iloc[-1]) for c in diff.columns},
    }

    pred_cpu, pred_gpu = cpu_dir / "predictions.npz", gpu_dir / "predictions.npz"
    if pred_cpu.is_file() and pred_gpu.is_file():
        with np.load(pred_cpu) as a, np.load(pred_gpu) as b:
            if not np.array_equal(a["labels"], b["labels"]):
                raise SystemExit("Las etiquetas de prueba no coinciden: no es la misma carga.")
            pa, pb = a["probabilities"], b["probabilities"]
            report["test_predictions"] = {
                "argmax_agreement": float((pa.argmax(1) == pb.argmax(1)).mean()),
                "max_abs_prob_diff": float(np.abs(pa - pb).max()),
                "mean_abs_prob_diff": float(np.abs(pa - pb).mean()),
            }

    report["single_pair_speedup"] = {
        phase: {"cpu_s": cpu["times"][phase], "gpu_s": gpu["times"][phase],
                "speedup": cpu["times"][phase] / gpu["times"][phase]}
        for phase in PHASES if phase in cpu["times"] and phase in gpu["times"]
    }
    report["note"] = ("Speedup de una sola pareja de corridas (CPU/GPU). "
                      "Las cinco repeticiones y promedios corresponden a la parte 3.")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cpu_run", type=Path)
    parser.add_argument("gpu_run", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    report = compare(args.cpu_run, args.gpu_run)
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    if not all(report["same_workload"].values()):
        raise SystemExit("Las corridas no comparten la misma carga de trabajo; ver same_workload.")


if __name__ == "__main__":
    main()
