"""Perfila pasos de entrenamiento en GPU y extrae grid/block reales de cada kernel CUDA.

Uso: python -m asl_cpu.profile_kernels --output-dir results/gpu_kernels

TensorFlow no lanza kernels escritos por nosotros: llama a cuDNN (convoluciones, pooling),
cuBLAS (Dense) y a kernels propios de TF (ReLU, softmax, Adam, reducciones). Cada librería
elige su configuración de lanzamiento. Este script la registra con el profiler de TensorFlow
(CUPTI) para documentar cómo se distribuye realmente el trabajo en hilos, bloques y grids.

Se usa un lote sintético con la misma forma que el entrenamiento (64 × 64 × 64 × 3, 29 clases):
la configuración de los kernels depende de las formas, no del valor de los píxeles.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
from collections import defaultdict
from pathlib import Path

from .common import ROOT, load_config, preload_pip_cuda_libraries, write_json
from .model import create_cnn_base

DETAILS = re.compile(r"(\w+):([\d,]+)")


def find_xplanes(log_dir: Path) -> list[Path]:
    return sorted(log_dir.rglob("*.xplane.pb"))


def stat_text(plane, stat) -> str | None:
    if stat.str_value:
        return stat.str_value
    if stat.ref_value:
        return plane.stat_metadata[stat.ref_value].name
    return None


def parse_kernels(xplane_path: Path) -> list[dict]:
    try:
        from tsl.profiler.protobuf import xplane_pb2
    except ImportError:
        from tensorflow.tsl.profiler.protobuf import xplane_pb2
    space = xplane_pb2.XSpace()
    space.ParseFromString(xplane_path.read_bytes())
    stats = defaultdict(lambda: {"calls": 0, "total_ns": 0, "ops": set()})
    for plane in space.planes:
        if not plane.name.startswith("/device:GPU:"):
            continue
        stat_names = {k: v.name for k, v in plane.stat_metadata.items()}
        for line in plane.lines:
            for event in line.events:
                metadata = plane.event_metadata[event.metadata_id]
                found = {}
                for stat in list(event.stats) + list(metadata.stats):
                    key = stat_names.get(stat.metadata_id)
                    if key in ("kernel_details", "tf_op"):
                        found[key] = stat_text(plane, stat)
                details = found.get("kernel_details")
                if details is None:
                    continue  # memcpy/memset u otros eventos que no son kernels
                entry = stats[(metadata.name, details)]
                entry["calls"] += 1
                entry["total_ns"] += event.duration_ps // 1000
                if found.get("tf_op"):
                    entry["ops"].add(found["tf_op"])
    rows = []
    for (name, details), entry in stats.items():
        fields = dict(DETAILS.findall(details))
        grid = [int(v) for v in fields.get("grid", "0,0,0").split(",")]
        block = [int(v) for v in fields.get("block", "0,0,0").split(",")]
        blocks = grid[0] * grid[1] * grid[2]
        threads_per_block = block[0] * block[1] * block[2]
        op_types = sorted({op.rsplit(":", 1)[-1] for op in entry["ops"]})
        rows.append({
            "kernel": name, "category": classify(name, op_types), "tf_op_types": ";".join(op_types),
            "tf_ops": ";".join(sorted(entry["ops"]))[:500],
            "calls": entry["calls"], "total_ms": entry["total_ns"] / 1e6,
            "mean_us": entry["total_ns"] / entry["calls"] / 1e3,
            "grid": "x".join(map(str, grid)), "block": "x".join(map(str, block)),
            "blocks": blocks, "threads_per_block": threads_per_block, "total_threads": blocks * threads_per_block,
            "registers_per_thread": int(fields.get("regs", 0)),
            "static_shared_bytes": int(fields.get("static_shared", 0)),
            "dynamic_shared_bytes": int(fields.get("dynamic_shared", 0)),
            "occupancy_percent": float(fields.get("occ_pct", 0)),
        })
    return sorted(rows, key=lambda r: r["total_ms"], reverse=True)


CATEGORIES = (
    (("Conv2DBackpropFilter",), "Conv2D gradiente de filtros (cuDNN)"),
    (("Conv2DBackpropInput",), "Conv2D gradiente de entrada (cuDNN)"),
    (("Conv2D", "_FusedConv2D"), "Conv2D forward (cuDNN, fusionada con sesgo+ReLU)"),
    (("MaxPoolGrad",), "MaxPooling gradiente (cuDNN)"),
    (("MaxPool",), "MaxPooling forward (cuDNN)"),
    (("MatMul", "BatchMatMulV2"), "Dense: producto de matrices (cuBLAS)"),
    (("BiasAddGrad",), "Gradiente del sesgo (reducción TF)"),
    (("BiasAdd",), "Suma de sesgo (TF/Eigen)"),
    (("ReluGrad",), "Gradiente ReLU (TF/Eigen)"),
    (("Relu",), "ReLU (TF/Eigen)"),
    (("Softmax", "SparseSoftmaxCrossEntropyWithLogits", "Log", "LogSoftmax"), "Softmax y pérdida (TF)"),
    (("AssignAddVariableOp", "AssignSubVariableOp", "AssignVariableOp", "Sqrt", "RealDiv", "Pow",
      "Mul", "AddV2", "Sub", "Square", "Maximum", "Cast"), "Actualización Adam y escalares (TF/Eigen)"),
    (("StatelessRandomUniformV2", "GreaterEqual", "SelectV2", "Select"), "Dropout (TF/Eigen)"),
    (("UnsortedSegmentSum", "GatherV2"), "Gradiente de la pérdida por etiqueta (TF)"),
    (("Transpose",), "Cambio de formato NCHW/NHWC (cuDNN/TF)"),
    (("Sum", "Mean", "ArgMax", "Equal", "DivNoNan"), "Reducciones y métricas (TF)"),
)


def classify(name: str, op_types: list[str]) -> str:
    for types, label in CATEGORIES:
        if any(op in types for op in op_types):
            return label
    lower = name.lower()
    if "nchwtonhwc" in lower or "nhwctonchw" in lower or "transpose" in lower:
        return "Cambio de formato NCHW/NHWC (cuDNN/TF)"
    if "pooling" in lower:
        return "MaxPooling (cuDNN)"
    if "gemm" in lower or "winograd" in lower or "conv" in lower:
        return "Convolución/GEMM sin op TF asociada (cuDNN)"
    return "otro: " + (";".join(op_types) or "sin tf_op")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=ROOT / "config/baseline.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    preload_pip_cuda_libraries()
    import tensorflow as tf

    if not tf.config.list_physical_devices("GPU"):
        raise SystemExit("No hay GPU visible para TensorFlow.")
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.keras.utils.set_random_seed(config["seed"])
    tf.config.experimental.enable_op_determinism()
    # Igual que train.py: float32 estricto, sin TensorFloat-32.
    tf.config.experimental.enable_tensor_float_32_execution(False)
    size, batch = config["image_size"], config["batch_size"]
    with tf.device("/CPU:0"):
        x = tf.random.stateless_uniform((batch, size, size, 3), seed=(1, 2))
        y = tf.random.stateless_uniform((batch,), seed=(3, 4), maxval=29, dtype=tf.int32)
    with tf.device("/GPU:0"):
        model = create_cnn_base(tf, config, 29)
        model.compile(optimizer=tf.keras.optimizers.Adam(config["learning_rate"]),
                      loss="sparse_categorical_crossentropy", metrics=["accuracy"], jit_compile=False)
        for _ in range(args.warmup_steps):
            model.train_on_batch(x, y)
        log_dir = args.output_dir / "trace"
        tf.profiler.experimental.start(str(log_dir))
        for _ in range(args.steps):
            model.train_on_batch(x, y)
        tf.test.experimental.sync_devices()
        tf.profiler.experimental.stop()

    xplanes = find_xplanes(log_dir)
    if not xplanes:
        raise SystemExit(f"El profiler no generó xplane.pb en {log_dir}")
    rows = parse_kernels(xplanes[-1])
    if not rows:
        raise SystemExit("No se encontraron kernels con kernel_details (¿CUPTI disponible?).")
    with (args.output_dir / "kernels.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    by_library = defaultdict(float)
    for row in rows:
        by_library[row["category"]] += row["total_ms"]
    total = sum(by_library.values())
    summary = {
        "steps_profiled": args.steps, "warmup_steps": args.warmup_steps, "batch_shape": [batch, size, size, 3],
        "gpu": tf.config.experimental.get_device_details(tf.config.list_physical_devices("GPU")[0]),
        "distinct_kernels": len(rows), "kernel_launches": sum(r["calls"] for r in rows),
        "gpu_kernel_time_ms": total,
        "time_share_by_category": {k: {"ms": v, "percent": 100 * v / total} for k, v in
                                  sorted(by_library.items(), key=lambda kv: -kv[1])},
        "xplane": str(xplanes[-1]),
    }
    write_json(args.output_dir / "summary.json", summary)
    print(f"{summary['kernel_launches']} lanzamientos, {len(rows)} kernels distintos, "
          f"{total:.1f} ms de GPU en {args.steps} pasos")
    for row in rows[:15]:
        print(f"{row['total_ms']:8.2f} ms  x{row['calls']:<4} grid {row['grid']:>12} block {row['block']:>9} "
              f"occ {row['occupancy_percent']:5.1f}%  {row['category'][:38]:38}  {row['kernel'][:60]}")


if __name__ == "__main__":
    main()
