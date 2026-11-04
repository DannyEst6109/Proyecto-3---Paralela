"""Configuración y utilidades compartidas sin importar TensorFlow."""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "val", "test")
EXPECTED_CLASSES = sorted([chr(n) for n in range(ord("A"), ord("Z") + 1)] + ["del", "nothing", "space"])


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config["model"] != "cnn_base" or config["precision"] != "float32":
        raise ValueError("Esta referencia implementa cnn_base con precisión float32.")
    if config["augmentation"] or config["early_stopping"]:
        raise ValueError("La referencia medida no usa aumentación ni parada temprana.")
    if config["image_size"] <= 0 or config["batch_size"] <= 0 or config["epochs"] <= 0:
        raise ValueError("Resolución, lote y épocas deben ser positivos.")
    if config["block_size"] <= 0 or not 0 <= config["block_margin"] < config["block_size"] / 2:
        raise ValueError("Tamaño o margen de bloque inválido.")
    for split in SPLITS:
        if config["samples_per_class"][split] <= 0 or config[f"{split}_blocks"] <= 0:
            raise ValueError("Cada partición requiere bloques y muestras positivos.")
    return config


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def fingerprint(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def environment() -> dict:
    import psutil
    packages = {}
    for package in ("tensorflow", "keras", "numpy", "pandas", "opencv-python-headless", "pyarrow", "psutil"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    return {
        "recorded_at": datetime.now().astimezone().isoformat(),
        "platform": platform.platform(), "python": sys.version,
        "python_executable": sys.executable,
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cpus": psutil.cpu_count(), "ram_bytes": psutil.virtual_memory().total,
        "packages": packages,
    }
