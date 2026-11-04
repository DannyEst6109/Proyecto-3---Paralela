"""CLI: python -m asl_cpu.prepare --workers 1 (o varios procesos CPU)."""
from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np

from .common import ROOT, SPLITS, environment, file_sha256, fingerprint, load_config, write_json
from .data import array_hashes, inventory_images, prepare_arrays, select_partitions
from .resources import ResourceMonitor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/processed/baseline")
    parser.add_argument("--config", type=Path, default=ROOT / "config/baseline.json")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers debe ser al menos 1")
    config = load_config(args.config)
    # Evita mezclar artefactos de distintas preparaciones.
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("La carpeta de salida debe estar vacía. Elija otra para conservar las mediciones.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    times = {}
    overall = perf_counter()
    with ResourceMonitor(args.output_dir / "resources.csv") as monitor:
        monitor.phase = "inventory"
        start = perf_counter()
        inventory, train = inventory_images(args.raw_dir)
        times["inventory_s"] = perf_counter() - start
        monitor.phase = "partition"
        start = perf_counter()
        inventory, partition = select_partitions(inventory, config)
        times["partition_s"] = perf_counter() - start
        monitor.phase = "decode_resize_rgb"
        start = perf_counter()
        arrays = prepare_arrays(partition, train, config["image_size"], args.workers)
        times["images_s"] = perf_counter() - start
        monitor.phase = "export"
        start = perf_counter()
        np.savez_compressed(args.output_dir / "asl_64x64.npz", **arrays)
        inventory.to_parquet(args.output_dir / "inventario.parquet", index=False)
        partition.to_parquet(args.output_dir / "particion.parquet", index=False)
        partition.to_csv(args.output_dir / "particion.csv", index=False, lineterminator="\n")
        times["export_s"] = perf_counter() - start
        monitor.phase = "verification"
        hashes = array_hashes(arrays)
        metadata = {
            "status": "complete", "config": config, "config_sha256": fingerprint(config),
            "workers": args.workers, "train_dir": str(train.resolve()),
            "inventory_count": len(inventory), "selected_count": len(partition),
            "split_counts": {split: int((partition["conjunto"] == split).sum()) for split in SPLITS},
            "array_sha256": hashes,
            "partition_sha256": file_sha256(args.output_dir / "particion.csv"),
            "archive_sha256": file_sha256(args.output_dir / "asl_64x64.npz"),
            "times": times, "environment": environment(),
            "timing_notes": "images_s incluye lectura, decodificación, resize, RGB, etiquetas y gestión de procesos; export_s incluye compresión y manifiestos. No incluye normalización float32 ni entrenamiento.",
        }
        times["prepare_wall_s"] = perf_counter() - overall
        write_json(args.output_dir / "metadata.json", metadata)
    print(f"Preparación completa: {args.output_dir}", flush=True)
    print(times, flush=True)


if __name__ == "__main__":
    main()
