"""Verifica la GPU y el entorno CUDA y registra hardware y versiones.

Uso: python -m asl_cpu.gpu_check --output results/gpu_environment.json
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from importlib.metadata import distributions
from pathlib import Path
from time import perf_counter

from .common import ROOT, environment, preload_pip_cuda_libraries, write_json


def nvidia_query() -> dict:
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return {"available": False}
    fields = "name,driver_version,memory.total,compute_cap,pcie.link.gen.max,pcie.link.width.max,power.limit,clocks.max.sm"
    out = subprocess.run([exe, f"--query-gpu={fields}", "--format=csv,noheader"],
                         capture_output=True, text=True, timeout=20).stdout.strip()
    header = subprocess.run([exe], capture_output=True, text=True, timeout=20).stdout.splitlines()
    cuda_driver = next((line.split("CUDA Version:")[1].strip(" |") for line in header if "CUDA Version:" in line), None)
    return {"available": True, "query": dict(zip(fields.split(","), (v.strip() for v in out.split(",")))),
            "driver_max_cuda": cuda_driver}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/gpu_environment.json")
    args = parser.parse_args()
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    report = {"environment": environment(), "nvidia_smi": nvidia_query()}
    report["nvidia_pip_packages"] = sorted(
        f"{d.metadata['Name']}=={d.version}" for d in distributions()
        if d.metadata["Name"] and d.metadata["Name"].lower().startswith("nvidia-"))

    preload_pip_cuda_libraries()
    import tensorflow as tf
    build = tf.sysconfig.get_build_info()
    gpus = tf.config.list_physical_devices("GPU")
    report["tensorflow"] = {
        "version": tf.__version__, "is_cuda_build": build.get("is_cuda_build"),
        "cuda_version": build.get("cuda_version"), "cudnn_version": build.get("cudnn_version"),
        "cuda_compute_capabilities": build.get("cuda_compute_capabilities"),
        "physical_gpus": [str(g) for g in gpus],
        "gpu_details": [tf.config.experimental.get_device_details(g) for g in gpus],
    }
    if not gpus:
        report["status"] = "sin GPU visible para TensorFlow"
        write_json(args.output, report)
        raise SystemExit("TensorFlow no detecta GPU CUDA; ver " + str(args.output))

    # Prueba de humo: una multiplicación de matrices y una convolución en /GPU:0.
    with tf.device("/GPU:0"):
        a = tf.random.stateless_normal((2048, 2048), seed=(1, 2))
        x = tf.random.stateless_normal((64, 64, 64, 3), seed=(3, 4))
        k = tf.random.stateless_normal((3, 3, 3, 32), seed=(5, 6))
        tf.linalg.matmul(a, a)
        tf.nn.conv2d(x, k, 1, "SAME")
        start = perf_counter()
        for _ in range(10):
            c = tf.linalg.matmul(a, a)
        _ = c.numpy()  # copia device->host: fuerza a esperar a la GPU
        elapsed = perf_counter() - start
        y = tf.nn.conv2d(x, k, 1, "SAME")
    with tf.device("/CPU:0"):
        cpu_c = tf.linalg.matmul(a, a)
    # Ampere usa TF32 por defecto; se compara también con float32 estricto en GPU.
    tf32_default = tf.config.experimental.tensor_float_32_execution_enabled()
    tf.config.experimental.enable_tensor_float_32_execution(False)
    with tf.device("/GPU:0"):
        strict = tf.linalg.matmul(a, a)
    tf.config.experimental.enable_tensor_float_32_execution(tf32_default)
    report["smoke_test"] = {
        "matmul_device": c.device, "conv2d_device": y.device,
        "matmul_2048_x10_s": elapsed,
        "matmul_gflops": 10 * 2 * 2048**3 / elapsed / 1e9,
        "tensor_float_32_default": tf32_default,
        "max_abs_diff_vs_cpu": float(tf.reduce_max(tf.abs(cpu_c - c)).numpy()),
        "max_abs_diff_vs_cpu_strict_fp32": float(tf.reduce_max(tf.abs(cpu_c - strict)).numpy()),
        "gpu_memory_tf_bytes": tf.config.experimental.get_memory_info("GPU:0"),
    }
    report["status"] = "ok" if "GPU:0" in c.device and "GPU:0" in y.device else "operaciones fuera de GPU"
    write_json(args.output, report)
    print(f"GPU: {report['nvidia_smi'].get('query', {}).get('name')} | driver "
          f"{report['nvidia_smi'].get('query', {}).get('driver_version')} | TF {tf.__version__} "
          f"CUDA {build.get('cuda_version')} cuDNN {build.get('cudnn_version')}")
    print(f"matmul en {c.device}: {report['smoke_test']['matmul_gflops']:.0f} GFLOP/s | "
          f"dif. máx. vs CPU {report['smoke_test']['max_abs_diff_vs_cpu']:.2e} (TF32={tf32_default}), "
          f"{report['smoke_test']['max_abs_diff_vs_cpu_strict_fp32']:.2e} (fp32 estricto) | estado: {report['status']}")


if __name__ == "__main__":
    main()
