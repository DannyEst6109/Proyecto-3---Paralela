"""Valida y mide los kernels CUDA propios (Numba) contra NumPy en CPU y TensorFlow/cuDNN en GPU.

Uso: python -m asl_cpu.benchmark_kernels --weights results/gpu_baseline_01/final.weights.h5 \
         --output-dir results/kernels_numba

1. Exactitud: pasa todo el conjunto de prueba por la CNN entrenada usando solo los kernels
   propios y compara probabilidades y clases con model.predict de TensorFlow.
2. Tiempo por capa con un lote de entrenamiento (64 imágenes): NumPy (CPU, BLAS multihilo),
   kernels Numba (GPU) y TensorFlow (GPU, cuDNN/cuBLAS).
3. Paso de Adam sobre los 1,145,693 parámetros: NumPy, kernel Numba y Adam de Keras.

Los tiempos GPU esperan a que termine el kernel (sincronización) y no incluyen copias,
salvo la fila de transferencia host→GPU, que se reporta aparte.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
from pathlib import Path
from time import perf_counter

import numpy as np

from .common import ROOT, environment, load_config, preload_pip_cuda_libraries, write_json
from .model import create_cnn_base


def timed(function, repeats: int, sync=None, warmup: int = 3) -> dict:
    for _ in range(warmup):
        function()
        if sync:
            sync()
    samples = []
    for _ in range(repeats):
        start = perf_counter()
        function()
        if sync:
            sync()
        samples.append(perf_counter() - start)
    return {"median_ms": 1000 * statistics.median(samples), "mean_ms": 1000 * statistics.fmean(samples),
            "stdev_ms": 1000 * statistics.pstdev(samples), "repeats": repeats}


# ------------------------------------------------------------ referencia CPU (NumPy)
def np_conv_bias_relu(x, w, b):
    padded = np.pad(x, ((0, 0), (1, 1), (1, 1), (0, 0)))
    windows = np.lib.stride_tricks.sliding_window_view(padded, (3, 3), axis=(1, 2))  # N,H,W,C,3,3
    return np.maximum(np.tensordot(windows, w, axes=([4, 5, 3], [0, 1, 2])) + b, 0).astype(np.float32)


def np_maxpool(x):
    n, h, w, c = x.shape
    return x.reshape(n, h // 2, 2, w // 2, 2, c).max(axis=(2, 4))


def np_softmax(z):
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def np_adam(w, g, m, v, lr_t, b1=0.9, b2=0.999, eps=1e-7):
    m += (g - m) * (1 - b1)
    v += (g * g - v) * (1 - b2)
    return w - lr_t * m / (np.sqrt(v) + eps)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", type=Path, default=ROOT / "results/gpu_baseline_01/final.weights.h5")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/processed/baseline")
    parser.add_argument("--config", type=Path, default=ROOT / "config/baseline.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("La carpeta de salida debe estar vacía para no sobrescribir evidencia.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    preload_pip_cuda_libraries()
    import tensorflow as tf

    tf.config.experimental.enable_tensor_float_32_execution(False)   # float32 estricto
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
    from .cuda_kernels import (NUMBA_SETUP, NumbaCNN, adam_launch_config, adam_update, conv2d_3x3_bias_relu,
                               conv2d_3x3_tiled, conv_tiled_config, cuda, grid_2d, matmul_bias_act, matmul_config, maxpool_2x2, softmax_config,
                               softmax_rows)

    with np.load(args.data_dir / "asl_64x64.npz", allow_pickle=False) as data:
        x_test = data["X_test"].astype(np.float32) / np.float32(255.0)     # normalización en CPU
        y_test = data["y_test"].astype(np.int64)
    batch = config["batch_size"]
    with tf.device("/GPU:0"):
        model = create_cnn_base(tf, config, 29)
        model.load_weights(args.weights)
    weights = [w.astype(np.float32) for w in model.get_weights()]
    cnn = NumbaCNN(weights, conv="tiled")
    cnn_naive = NumbaCNN(weights, conv="naive")
    report = {"weights": str(args.weights), "numba_setup": NUMBA_SETUP, "environment": environment(),
              "gpu": tf.config.experimental.get_device_details(tf.config.list_physical_devices("GPU")[0]),
              "tensor_float_32": False}

    # ----------------------------------------------------------- 1. exactitud
    tf_probs = model.predict(x_test, batch_size=batch, verbose=0)
    numba_probs = np.empty_like(tf_probs)
    naive_probs = np.empty_like(tf_probs)
    for start in range(0, len(x_test), batch):
        chunk = cuda.to_device(np.ascontiguousarray(x_test[start:start + batch]))
        numba_probs[start:start + batch] = cnn.forward(chunk, record=start == 0).copy_to_host()
        naive_probs[start:start + batch] = cnn_naive.forward(chunk, record=start == 0).copy_to_host()
    report["accuracy_check"] = {
        "test_images": int(len(x_test)),
        "tensorflow_accuracy": float((tf_probs.argmax(1) == y_test).mean()),
        "numba_kernels_accuracy": float((numba_probs.argmax(1) == y_test).mean()),
        "argmax_agreement": float((tf_probs.argmax(1) == numba_probs.argmax(1)).mean()),
        "max_abs_prob_diff": float(np.abs(tf_probs - numba_probs).max()),
        "naive_conv_accuracy": float((naive_probs.argmax(1) == y_test).mean()),
        "naive_conv_max_abs_prob_diff": float(np.abs(tf_probs - naive_probs).max()),
    }
    report["launch_configuration_batch64"] = cnn.launches
    report["launch_configuration_batch64_naive_conv"] = cnn_naive.launches

    # ------------------------------------------------ 2. tiempo por capa (lote 64)
    x_host = np.ascontiguousarray(x_test[:batch])
    k = [weights[i] for i in range(10)]
    sync_numba = cuda.synchronize
    sync_tf = tf.test.experimental.sync_devices
    rows = []

    def add(layer, implementation, stats, grid="", block=""):
        rows.append({"layer": layer, "implementation": implementation, "grid": grid, "block": block, **stats})

    # Entradas de cada capa calculadas una vez en CPU, GPU-Numba y GPU-TF.
    h_cpu = [x_host]
    for i in range(3):
        h_cpu.append(np_maxpool(np_conv_bias_relu(h_cpu[-1], k[2 * i], k[2 * i + 1])))
    flat_cpu = h_cpu[3].reshape(batch, -1)
    dense1_cpu = np.maximum(flat_cpu @ k[6] + k[7], 0)
    logits_cpu = dense1_cpu @ k[8] + k[9]

    dev = {name: cuda.to_device(np.ascontiguousarray(arr)) for name, arr in
           {"h0": h_cpu[0], "h1": h_cpu[1], "h2": h_cpu[2], "flat": flat_cpu, "d1": dense1_cpu,
            "logits": logits_cpu.astype(np.float32)}.items()}
    dw = [cuda.to_device(w) for w in k]
    with tf.device("/GPU:0"):
        tfh = {name: tf.constant(arr) for name, arr in
               {"h0": h_cpu[0], "h1": h_cpu[1], "h2": h_cpu[2], "flat": flat_cpu, "d1": dense1_cpu,
                "logits": logits_cpu.astype(np.float32)}.items()}
        tfw = [tf.constant(w) for w in k]

    for i in range(3):
        name = f"Conv2D {i + 1} + sesgo + ReLU"
        inp_cpu, inp_dev, inp_tf = h_cpu[i], dev[f"h{i}"], tfh[f"h{i}"]
        n, height, width, _ = inp_cpu.shape
        filters = k[2 * i].shape[3]
        out = cuda.device_array((n, height, width, filters), dtype=np.float32)
        grid, block = grid_2d(width, height, n * filters)
        add(name, "CPU NumPy", timed(lambda: np_conv_bias_relu(inp_cpu, k[2 * i], k[2 * i + 1]), args.repeats))
        add(name, "GPU Numba v1 (directo)", timed(lambda: conv2d_3x3_bias_relu[grid, block](
            inp_dev, dw[2 * i], dw[2 * i + 1], out, 1), args.repeats, sync_numba),
            "x".join(map(str, grid)), "x".join(map(str, block)))
        grid2, block2 = conv_tiled_config(width, height, n, filters)
        add(name, "GPU Numba v2 (memoria compartida)", timed(lambda: conv2d_3x3_tiled[grid2, block2](
            inp_dev, dw[2 * i], dw[2 * i + 1], out, 1), args.repeats, sync_numba),
            "x".join(map(str, grid2)), "x".join(map(str, block2)))
        with tf.device("/GPU:0"):
            add(name, "GPU TensorFlow (cuDNN)", timed(lambda: tf.nn.relu(tf.nn.bias_add(
                tf.nn.conv2d(inp_tf, tfw[2 * i], 1, "SAME"), tfw[2 * i + 1])), args.repeats, sync_tf))

        name = f"MaxPooling {i + 1}"
        conv_cpu = np_conv_bias_relu(inp_cpu, k[2 * i], k[2 * i + 1])
        conv_dev = cuda.to_device(conv_cpu)
        pooled = cuda.device_array((n, height // 2, width // 2, filters), dtype=np.float32)
        grid, block = grid_2d(width // 2, height // 2, n * filters)
        add(name, "CPU NumPy", timed(lambda: np_maxpool(conv_cpu), args.repeats))
        add(name, "GPU kernel Numba", timed(lambda: maxpool_2x2[grid, block](conv_dev, pooled), args.repeats,
                                            sync_numba), "x".join(map(str, grid)), "x".join(map(str, block)))
        with tf.device("/GPU:0"):
            conv_tf = tf.constant(conv_cpu)
            add(name, "GPU TensorFlow (cuDNN)", timed(lambda: tf.nn.max_pool2d(conv_tf, 2, 2, "VALID"),
                                                      args.repeats, sync_tf))

    for index, (a_name, w_i, relu, cols) in enumerate((("flat", 6, 1, 128), ("d1", 8, 0, 29)), start=1):
        name = f"Dense {index} ({'ReLU' if relu else 'logits'})"
        a_cpu = flat_cpu if a_name == "flat" else dense1_cpu
        out = cuda.device_array((batch, cols), dtype=np.float32)
        grid, block = matmul_config(batch, cols)
        add(name, "CPU NumPy", timed(lambda: np.maximum(a_cpu @ k[w_i] + k[w_i + 1], 0) if relu
                                     else a_cpu @ k[w_i] + k[w_i + 1], args.repeats))
        add(name, "GPU kernel Numba", timed(lambda: matmul_bias_act[grid, block](
            dev[a_name], dw[w_i], dw[w_i + 1], out, relu), args.repeats, sync_numba),
            "x".join(map(str, grid)), "x".join(map(str, block)))
        with tf.device("/GPU:0"):
            add(name, "GPU TensorFlow (cuBLAS)", timed(
                lambda: (tf.nn.relu if relu else tf.identity)(tf.matmul(tfh[a_name], tfw[w_i]) + tfw[w_i + 1]),
                args.repeats, sync_tf))

    name = "Softmax"
    probs = cuda.device_array((batch, 29), dtype=np.float32)
    grid, block = softmax_config(batch)
    add(name, "CPU NumPy", timed(lambda: np_softmax(logits_cpu), args.repeats))
    add(name, "GPU kernel Numba", timed(lambda: softmax_rows[grid, block](dev["logits"], probs), args.repeats,
                                        sync_numba), str(grid), str(block))
    with tf.device("/GPU:0"):
        add(name, "GPU TensorFlow", timed(lambda: tf.nn.softmax(tfh["logits"]), args.repeats, sync_tf))

    def cpu_forward():
        h = x_host
        for i in range(3):
            h = np_maxpool(np_conv_bias_relu(h, k[2 * i], k[2 * i + 1]))
        h = np.maximum(h.reshape(batch, -1) @ k[6] + k[7], 0)
        return np_softmax(h @ k[8] + k[9])

    name = "Forward completo (lote 64)"
    add(name, "CPU NumPy", timed(cpu_forward, args.repeats))
    add(name, "GPU kernels Numba (conv v1)", timed(lambda: cnn_naive.forward(dev["h0"]), args.repeats, sync_numba))
    add(name, "GPU kernels Numba (conv v2)", timed(lambda: cnn.forward(dev["h0"]), args.repeats, sync_numba))
    with tf.device("/GPU:0"):
        add(name, "GPU TensorFlow (cuDNN/cuBLAS)", timed(lambda: model(tfh["h0"], training=False),
                                                         args.repeats, sync_tf))
    add("Copia host→GPU del lote (3.1 MB)", "cuda.to_device", timed(lambda: cuda.to_device(x_host), args.repeats,
                                                                     sync_numba))

    # ----------------------------------------------------- 3. paso de Adam
    total = sum(w.size for w in weights)
    rng = np.random.default_rng(0)
    w_host = np.concatenate([w.ravel() for w in weights])
    g_host = (rng.standard_normal(total) * 1e-3).astype(np.float32)
    m_host = np.zeros(total, np.float32)
    v_host = np.zeros(total, np.float32)
    lr_t = np.float32(0.001 * np.sqrt(1 - 0.999) / (1 - 0.9))
    blocks, threads = adam_launch_config(total)
    d = {n: cuda.to_device(a) for n, a in (("w", w_host), ("g", g_host), ("m", m_host), ("v", v_host))}
    d_out = cuda.device_array(total, np.float32)
    name = f"Adam, {total:,} parámetros"
    add(name, "CPU NumPy", timed(lambda: np_adam(w_host, g_host, m_host.copy(), v_host.copy(), lr_t), args.repeats))
    add(name, "GPU kernel Numba", timed(lambda: adam_update[blocks, threads](
        d["w"], d["g"], d["m"], d["v"], d_out, lr_t, np.float32(0.9), np.float32(0.999), np.float32(1e-7)),
        args.repeats, sync_numba), str(blocks), str(threads))
    with tf.device("/GPU:0"):
        variables = [tf.Variable(w) for w in weights]
        grads = [tf.constant(rng.standard_normal(w.shape).astype(np.float32) * 1e-3) for w in weights]
        keras_adam = tf.keras.optimizers.Adam(0.001)
        step = tf.function(lambda: keras_adam.apply_gradients(zip(grads, variables)))
        add(name, "GPU TensorFlow (Keras Adam)", timed(step, args.repeats, sync_tf))

    # Comprobación numérica de un paso de Adam frente a NumPy.
    d_check = {n: cuda.to_device(a) for n, a in (("w", w_host), ("g", g_host), ("m", m_host), ("v", v_host))}
    adam_update[blocks, threads](d_check["w"], d_check["g"], d_check["m"], d_check["v"], d_out, lr_t,
                                 np.float32(0.9), np.float32(0.999), np.float32(1e-7))
    report["adam_check_max_abs_diff_vs_numpy"] = float(np.abs(
        d_out.copy_to_host() - np_adam(w_host, g_host, m_host.copy(), v_host.copy(), lr_t)).max())

    report["timings"] = rows
    report["numpy_threads_note"] = "NumPy usa BLAS multihilo (tensordot/matmul) en la referencia CPU."
    write_json(args.output_dir / "benchmark.json", report)
    with (args.output_dir / "timings.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    check = report["accuracy_check"]
    print(f"Prueba ({check['test_images']} imágenes): TF {check['tensorflow_accuracy']:.4f} | kernels Numba v2 "
          f"{check['numba_kernels_accuracy']:.4f} | coincidencia {check['argmax_agreement']:.4f} | "
          f"dif. máx. prob {check['max_abs_prob_diff']:.2e} | Adam vs NumPy "
          f"{report['adam_check_max_abs_diff_vs_numpy']:.2e} | conv v1 {check['naive_conv_accuracy']:.4f}")
    for row in rows:
        print(f"{row['layer'][:34]:34} {row['implementation'][:30]:30} {row['median_ms']:9.3f} ms  "
              f"grid {row['grid']:>12} block {row['block']:>6}")


if __name__ == "__main__":
    main()
