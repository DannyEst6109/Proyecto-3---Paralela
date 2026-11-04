"""Comprueba equivalencia exacta entre dos preparaciones, sin comparar tiempos de entrenamiento."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    args = parser.parse_args()
    left = json.loads((args.left / "metadata.json").read_text(encoding="utf-8"))
    right = json.loads((args.right / "metadata.json").read_text(encoding="utf-8"))
    for key in ("status", "config_sha256", "partition_sha256", "array_sha256", "split_counts"):
        if left[key] != right[key] or left["status"] != "complete":
            raise SystemExit(f"No equivalentes: {key}")
    print("Equivalencia exacta: configuración, particiones, imágenes, clases y etiquetas.")
    print(f"images_s: {left['times']['images_s']:.3f} vs {right['times']['images_s']:.3f}")
    print("Una ejecución por modo no permite concluir una aceleración promedio.")


if __name__ == "__main__":
    main()
