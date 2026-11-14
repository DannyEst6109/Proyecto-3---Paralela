"""Grafica gpu.csv (nvidia-smi) junto con las fases registradas en metrics.json.

Uso: python -m asl_cpu.plot_gpu results/gpu_01 [salida.png]
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main():
    run = Path(sys.argv[1])
    output = Path(sys.argv[2]) if len(sys.argv) > 2 else run / "gpu.png"
    data = pd.read_csv(run / "gpu.csv", skipinitialspace=True)
    data = data[data["timestamp"].astype(str).str.match(r"\d{4}/")]
    data["t"] = [datetime.strptime(v.strip(), "%Y/%m/%d %H:%M:%S.%f").timestamp() for v in data["timestamp"]]
    for column in data.columns.drop(["timestamp", "t"]):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    marks = json.loads((run / "metrics.json").read_text(encoding="utf-8")).get("phase_marks", [])
    origin = marks[0]["unix_time"] if marks else data["t"].iloc[0]
    data["elapsed_s"] = data["t"] - origin

    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    axes[0].plot(data["elapsed_s"], data["gpu_util_percent"], label="Utilización SM (%)")
    axes[0].plot(data["elapsed_s"], data["gpu_mem_util_percent"], label="Utilización controlador memoria (%)", alpha=0.7)
    axes[0].set_ylim(0, 105)
    axes[1].plot(data["elapsed_s"], data["gpu_mem_used_mib"], color="tab:green", label="Memoria usada (MiB)")
    axes[2].plot(data["elapsed_s"], data["power_w"], color="tab:red", label="Potencia (W)")
    for ax in axes:
        for i, mark in enumerate(marks[:-1]):
            start = mark["unix_time"] - origin
            end = marks[i + 1]["unix_time"] - origin
            ax.axvspan(start, end, color=f"C{i % 10}", alpha=0.06)
            if ax is axes[0] and end - start > 5:
                ax.text(start, 102, mark["phase"], fontsize=7, va="top")
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("segundos desde el inicio de la ejecución")
    fig.suptitle(f"GPU durante {run.name} (nvidia-smi cada 500 ms)")
    fig.tight_layout()
    fig.savefig(output, dpi=120)
    print(output)


if __name__ == "__main__":
    main()
