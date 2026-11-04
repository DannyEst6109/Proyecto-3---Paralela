"""Inventario, partición original y preparación de imágenes exclusivamente en CPU."""
from __future__ import annotations

import hashlib
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .common import EXPECTED_CLASSES, SPLITS


def resolve_train_dir(raw: Path) -> Path:
    for candidate in (raw / "asl_alphabet_train" / "asl_alphabet_train", raw / "asl_alphabet_train", raw):
        if all((candidate / name).is_dir() for name in EXPECTED_CLASSES):
            return candidate
    raise FileNotFoundError(f"No se encontraron las 29 clases ASL en {raw}.")


def inventory_images(raw: Path) -> tuple[pd.DataFrame, Path]:
    train = resolve_train_dir(raw)
    rows = []
    for label in EXPECTED_CLASSES:
        for path in sorted((train / label).glob("*.jpg")):
            match = re.fullmatch(re.escape(label) + r"(\d+)", path.stem)
            if match is None:
                raise ValueError(f"Nombre de imagen inesperado: {path}")
            rows.append({"clase": label, "indice": int(match[1]), "archivo": path.name,
                         "relative_path": path.relative_to(train).as_posix()})
    inventory = pd.DataFrame(rows).sort_values(["clase", "indice"]).reset_index(drop=True)
    for label, group in inventory.groupby("clase", sort=True):
        if group["indice"].tolist() != list(range(1, 3001)):
            raise ValueError(f"La clase {label} debe contener exactamente los índices 1..3000.")
    return inventory, train


def select_partitions(inventory: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reproduce las celdas 34 y 35 del notebook base, incluido su orden."""
    inventory = inventory.copy()
    size = config["block_size"]
    blocks = 3000 // size
    if 3000 % size or sum(config[f"{split}_blocks"] for split in SPLITS) != blocks:
        raise ValueError("Los bloques deben cubrir exactamente los 3000 fotogramas.")
    permutation = np.random.default_rng(config["seed"]).permutation(blocks)
    assignment = {}
    offset = 0
    for split in SPLITS:
        count = config[f"{split}_blocks"]
        assignment.update({int(b): split for b in permutation[offset:offset + count]})
        offset += count
    inventory["bloque"] = (inventory["indice"] - 1) // size
    inventory["conjunto"] = inventory["bloque"].map(assignment)
    inventory["posicion"] = (inventory["indice"] - 1) % size
    margin = config["block_margin"]
    inventory["util"] = inventory["posicion"].between(margin, size - 1 - margin)
    selected = []
    for label in EXPECTED_CLASSES:
        for split in SPLITS:
            candidates = inventory[(inventory["clase"] == label) & (inventory["conjunto"] == split)
                                   & inventory["util"]].sort_values("indice")
            count = config["samples_per_class"][split]
            if count > len(candidates):
                raise ValueError(f"No hay suficientes imágenes de {label}/{split}: {count} > {len(candidates)}")
            positions = np.linspace(0, len(candidates) - 1, count).astype(int)
            selected.append(candidates.iloc[positions])
    partition = pd.concat(selected).reset_index(drop=True)
    if partition["relative_path"].duplicated().any():
        raise ValueError("La partición contiene imágenes repetidas.")
    if partition.groupby(["clase", "bloque"])["conjunto"].nunique().max() != 1:
        raise ValueError("Un bloque aparece en más de una partición.")
    return inventory, partition


def _worker_init():
    # Evita que cada proceso abra además su propio grupo de hilos OpenCV.
    cv2.setNumThreads(1)


def read_image(task: tuple[str, int]) -> np.ndarray:
    filename, size = task
    image = cv2.imread(filename, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"No se pudo leer la imagen: {filename}")
    image = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def prepare_arrays(partition: pd.DataFrame, train: Path, size: int, workers: int) -> dict:
    """map conserva el orden: el modo multiproceso produce los mismos arreglos."""
    if workers < 1:
        raise ValueError("workers debe ser al menos 1.")
    _worker_init()
    pool = ProcessPoolExecutor(max_workers=workers, initializer=_worker_init) if workers > 1 else None
    arrays = {"clases": np.array(EXPECTED_CLASSES)}
    try:
        for split in SPLITS:
            subset = partition[partition["conjunto"] == split].reset_index(drop=True)
            tasks = [(str(train / p), size) for p in subset["relative_path"]]
            images = map(read_image, tasks) if pool is None else pool.map(read_image, tasks, chunksize=32)
            x = np.empty((len(subset), size, size, 3), dtype=np.uint8)
            for i, image in enumerate(images):
                x[i] = image
            arrays[f"X_{split}"] = x
            mapping = {label: i for i, label in enumerate(EXPECTED_CLASSES)}
            arrays[f"y_{split}"] = subset["clase"].map(mapping).to_numpy(dtype=np.int16)
            print(f"{split}: {x.shape}, {x.nbytes / 1024**2:.1f} MiB", flush=True)
    finally:
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
    return arrays


def array_hashes(arrays: dict) -> dict:
    return {key: hashlib.sha256(value.tobytes(order="C")).hexdigest() for key, value in arrays.items()}
