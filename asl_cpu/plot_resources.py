"""Grafica mediciones reales del monitor; la figura no es una captura de pantalla."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    data = pd.read_csv(args.input)
    if data.empty:
        parser.error("El monitor no contiene muestras suficientes.")
    cores = [name for name in data if name.startswith("cpu_")]
    seconds = data["elapsed_s"]
    fig, axes = plt.subplots(3, 1, figsize=(11, 7), constrained_layout=True, sharex=True,
                             gridspec_kw={"height_ratios": [2, 1, 1]})
    heat = axes[0].pcolormesh(seconds.to_numpy(), range(len(cores)), data[cores].to_numpy().T,
                             shading="nearest", vmin=0, vmax=100, cmap="viridis")
    axes[0].set_ylabel("Procesador lógico")
    axes[0].set_yticks(range(len(cores)))
    fig.colorbar(heat, ax=axes[0], label="CPU del sistema (%)")
    axes[1].plot(seconds, data["process_tree_cpu_percent"], color="#2166ac")
    axes[1].set_ylabel("CPU proceso\ne hijos (%)")
    axes[1].grid(alpha=0.2)
    axes[2].plot(seconds, data["process_tree_rss_mb"], color="#1b7837")
    axes[2].set_ylabel("RSS proceso\ne hijos (MiB)")
    axes[2].set_xlabel("Tiempo desde el inicio del monitor (s)")
    axes[2].grid(alpha=0.2)
    fig.suptitle("Monitor CPU y memoria · mediciones psutil cada segundo\nCPU del proceso: 100 % equivale a un procesador lógico")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
