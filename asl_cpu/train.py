"""Entrenamiento reproducible en CPU (referencia) o en GPU con CUDA.

--device cpu  oculta las GPU antes de importar TensorFlow (referencia de la parte 1).
--device gpu  coloca pesos, gradientes y optimizador en /GPU:0 (CUDA/cuDNN/cuBLAS).
En ambos casos la lectura del NPZ, la normalización y tf.data se ejecutan en CPU;
solo cambia el dispositivo del modelo, de modo que la carga de trabajo es la misma.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from time import perf_counter

import numpy as np

from .common import (ROOT, SPLITS, environment, file_sha256, fingerprint, load_config,
                     preload_pip_cuda_libraries, write_json)
from .data import array_hashes
from .model import create_cnn_base
from .resources import GpuMonitor, ResourceMonitor, nvidia_smi_snapshot


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, message):
        for stream in self.streams:
            if getattr(stream, "closed", False):
                continue
            stream.write(message)
            stream.flush()
        return len(message)

    def flush(self):
        for stream in self.streams:
            if not getattr(stream, "closed", False):
                stream.flush()

    def close(self):
        # absl conserva este destino de logs hasta terminar el intérprete.
        # Nunca cerramos stdout/stderr desde el duplicador.
        self.flush()

    def isatty(self):
        return False


def weight_hash(model) -> str:
    digest = hashlib.sha256()
    for value in model.get_weights():
        digest.update(value.tobytes())
    return digest.hexdigest()


def resolve_training_device(tf, mode: str) -> str:
    """Traduce --device a un dispositivo TensorFlow, fallando si se pide GPU sin CUDA."""
    if mode == "cpu":
        return "/CPU:0"
    gpus = tf.config.list_physical_devices("GPU")
    if mode == "auto":
        return "/GPU:0" if gpus else "/CPU:0"
    if mode != "gpu":
        raise ValueError(f"Dispositivo desconocido: {mode}")
    if not gpus:
        raise RuntimeError("No CUDA-capable GPU visible to TensorFlow. Revise el driver NVIDIA "
                           "y que TensorFlow se instaló con tensorflow[and-cuda].")
    if not tf.config.get_visible_devices("GPU"):
        raise RuntimeError("Hay una GPU física, pero está oculta para TensorFlow.")
    return "/GPU:0"


def sync_device(tf) -> None:
    """Espera a que termine todo el trabajo encolado en GPU antes de leer el reloj."""
    sync = getattr(getattr(tf.test, "experimental", None), "sync_devices", None)
    if sync is not None:
        sync()


def input_transfer_bytes(arrays: dict, epochs: int) -> dict:
    """Bytes de entrada que Keras copia host->device (lotes float32 e int16)."""
    per_pass = {split: int(arrays[f"X_{split}"].nbytes + arrays[f"y_{split}"].nbytes) for split in SPLITS}
    return {
        "per_pass_bytes": per_pass,
        "fit_h2d_bytes": epochs * (per_pass["train"] + per_pass["val"]),
        "evaluate_predict_h2d_bytes": per_pass["val"] + 2 * per_pass["test"],
        "note": "fit copia train y val en cada época; después evaluate(val), evaluate(test) y "
                "predict(test). Las copias device->host (pérdida/métricas por lote y "
                "probabilidades finales) son de pocos bytes por lote. Los pesos se copian a la "
                "GPU una sola vez al crearlos y vuelven al host solo al guardarlos.",
    }


def run(args):
    config = load_config(args.config)
    metadata = json.loads((args.data_dir / "metadata.json").read_text(encoding="utf-8"))
    if metadata["status"] != "complete" or fingerprint(config) != metadata["config_sha256"]:
        raise ValueError("Los datos y el entrenamiento deben compartir la misma configuración.")
    result = {
        "status": "running", "run_kind": "smoke" if args.smoke else f"baseline_{args.device}",
        "device_requested": args.device, "device": None, "process_pid": os.getpid(),
        "config": config, "config_sha256": fingerprint(config),
        "partition_sha256": metadata["partition_sha256"],
        "source_array_sha256": metadata["array_sha256"],
        "epochs_requested": 1 if args.smoke else config["epochs"],
        "cpu_threads": args.threads, "data_threads": 1,
        "smoke_samples_per_class": {"train": 16, "val": 4, "test": 4} if args.smoke else None,
        "warmup": "none; la primera época incluye trazado y compilación de las funciones de entrenamiento",
        "environment": environment(), "times": {}, "phase_marks": [],
    }
    write_json(args.output_dir / "metrics.json", result)
    times = result["times"]
    overall = perf_counter()
    evidence = args.output_dir / "evidence"
    evidence.mkdir(exist_ok=True)
    try:
        with ResourceMonitor(args.output_dir / "resources.csv") as monitor, \
                GpuMonitor(args.output_dir / "gpu.csv") as gpu_monitor:
            def mark(phase):
                monitor.phase = phase
                result["phase_marks"].append({"phase": phase, "unix_time": time.time()})

            mark("tensorflow_setup")
            start = perf_counter()
            if "tensorflow" in sys.modules:
                raise RuntimeError("Ejecute este módulo en un proceso nuevo para controlar el dispositivo.")
            if args.device == "cpu":
                os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
            os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
            os.environ["KERAS_BACKEND"] = "tensorflow"
            # Estas variables se establecen antes del import y del runtime de TF.
            os.environ["OMP_NUM_THREADS"] = str(args.threads)
            if args.device != "cpu":
                result["preloaded_cuda_libraries"] = preload_pip_cuda_libraries()
            import tensorflow as tf
            if args.device == "cpu":
                tf.config.set_visible_devices([], "GPU")
            device = resolve_training_device(tf, args.device)
            if device == "/CPU:0":
                tf.config.set_visible_devices([], "GPU")
            else:
                # Reserva memoria según se necesite para poder medir el uso real.
                for gpu in tf.config.list_physical_devices("GPU"):
                    tf.config.experimental.set_memory_growth(gpu, True)
            tf.config.threading.set_intra_op_parallelism_threads(args.threads)
            tf.config.threading.set_inter_op_parallelism_threads(1)
            tf.keras.utils.set_random_seed(config["seed"])
            tf.config.experimental.enable_op_determinism()
            tf.keras.mixed_precision.set_global_policy("float32")
            # En Ampere (RTX 30xx) TF usa TensorFloat-32 por defecto en matmul/conv: mantisa de 10 bits.
            # La comparación exige float32 estricto; TF32 solo como experimento etiquetado.
            tf.config.experimental.enable_tensor_float_32_execution(args.allow_tf32)
            result["tensor_float_32"] = tf.config.experimental.tensor_float_32_execution_enabled()
            visible_gpus = tf.config.get_visible_devices("GPU")
            if device == "/CPU:0" and visible_gpus:
                raise RuntimeError(f"La referencia CPU tiene GPU visibles: {visible_gpus}")
            result["device"] = "GPU" if device == "/GPU:0" else "CPU"
            if args.smoke:
                result["run_kind"] = "smoke"
            else:
                result["run_kind"] = f"baseline_{result['device'].lower()}" + ("_tf32" if args.allow_tf32 else "")
            result["tf_device"] = device
            result["visible_gpus"] = [str(d) for d in visible_gpus]
            result["gpu_details"] = [tf.config.experimental.get_device_details(d) for d in visible_gpus]
            result["logical_devices"] = [str(d) for d in tf.config.list_logical_devices()]
            result["tensorflow_build"] = tf.sysconfig.get_build_info()
            times["tensorflow_setup_s"] = perf_counter() - start
            if device == "/GPU:0":
                print(f"GPU visibles: {result['visible_gpus']} | detalles: {result['gpu_details']}", flush=True)
                print(f"CUDA {result['tensorflow_build'].get('cuda_version')} | "
                      f"cuDNN {result['tensorflow_build'].get('cudnn_version')} | modelo en {device}", flush=True)
            else:
                print("GPU visibles: [] | entrenamiento forzado a /CPU:0", flush=True)

            mark("load_prepared_data")
            start = perf_counter()
            if file_sha256(args.data_dir / "asl_64x64.npz") != metadata["archive_sha256"]:
                raise ValueError("El archivo de datos cambió desde la preparación.")
            if file_sha256(args.data_dir / "particion.csv") != metadata["partition_sha256"]:
                raise ValueError("El manifiesto de particiones cambió desde la preparación.")
            with np.load(args.data_dir / "asl_64x64.npz", allow_pickle=False) as source:
                arrays = {key: source[key] for key in source.files}
            if array_hashes(arrays) != metadata["array_sha256"]:
                raise ValueError("Las huellas de los arreglos no coinciden con la preparación.")
            if args.smoke:
                for split, count in (("train", 16), ("val", 4), ("test", 4)):
                    labels = arrays[f"y_{split}"]
                    indices = np.concatenate([np.flatnonzero(labels == i)[:count]
                                              for i in range(len(arrays["clases"]))])
                    arrays[f"X_{split}"] = arrays[f"X_{split}"][indices]
                    arrays[f"y_{split}"] = labels[indices]
            times["load_and_verify_s"] = perf_counter() - start
            result["used_array_sha256"] = array_hashes(arrays)
            result["sample_counts"] = {s: len(arrays[f"y_{s}"]) for s in SPLITS}

            mark("normalize_cpu")
            start = perf_counter()
            for split in SPLITS:
                arrays[f"X_{split}"] = arrays[f"X_{split}"].astype(np.float32) / np.float32(255.0)
            times["normalize_cpu_s"] = perf_counter() - start
            print(f"Muestras: {result['sample_counts']}", flush=True)
            if device == "/GPU:0":
                result["transfers"] = input_transfer_bytes(arrays, result["epochs_requested"])

            mark("build_model_and_input")
            start = perf_counter()
            datasets = {}
            # El pipeline de entrada se construye siempre en CPU.
            with tf.device("/CPU:0"):
                for split in SPLITS:
                    dataset = tf.data.Dataset.from_tensor_slices((arrays[f"X_{split}"], arrays[f"y_{split}"]))
                    if split == "train":
                        dataset = dataset.shuffle(len(arrays["y_train"]), seed=config["seed"], reshuffle_each_iteration=True)
                    options = tf.data.Options()
                    options.experimental_deterministic = True
                    options.threading.private_threadpool_size = 1
                    options.threading.max_intra_op_parallelism = 1
                    datasets[split] = dataset.batch(config["batch_size"]).with_options(options).prefetch(1)
            result["input_pipeline_device"] = "/CPU:0"
            # El modelo (pesos, activaciones, gradientes y estado de Adam) vive en `device`.
            with tf.device(device):
                model = create_cnn_base(tf, config, len(arrays["clases"]))
                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config["learning_rate"]),
                              loss="sparse_categorical_crossentropy", metrics=["accuracy"], jit_compile=False)
                if args.initial_weights:
                    model.load_weights(args.initial_weights)
                    result["initial_weights_source"] = str(args.initial_weights)
            expected = device.lstrip("/")
            result["weight_devices"] = sorted({v.handle.device for v in model.weights})
            if any(expected not in d for d in result["weight_devices"]):
                raise RuntimeError(f"Se detectaron variables del modelo fuera de {device}: {result['weight_devices']}")
            result["model_parameters"] = model.count_params()
            result["initial_weights_sha256"] = weight_hash(model)
            times["build_model_and_input_s"] = perf_counter() - start
            model.save_weights(args.output_dir / "initial.weights.h5")
            (args.output_dir / "model.json").write_text(model.to_json(), encoding="utf-8")
            model.summary()
            print(f"Dispositivos de los pesos: {result['weight_devices']}", flush=True)

            class Progress(tf.keras.callbacks.Callback):
                def on_epoch_begin(self, epoch, logs=None):
                    self.start = perf_counter()

                def on_epoch_end(self, epoch, logs=None):
                    sync_device(tf)
                    row = {"epoch": epoch + 1, "epoch_wall_s": perf_counter() - self.start}
                    row.update({k: float(v) for k, v in (logs or {}).items()})
                    progress.append(row)
                    write_json(args.output_dir / "progress.json", {"epochs": progress})
                    if epoch == 0 and gpu_monitor.available:
                        # Evidencia: qué procesos usan la GPU mientras entrena este proceso.
                        (evidence / "nvidia_smi_epoch1.txt").write_text(
                            f"PID del entrenamiento: {os.getpid()}\n\n" + nvidia_smi_snapshot(), encoding="utf-8")

            progress = []
            if device == "/GPU:0":
                tf.config.experimental.reset_memory_stats("GPU:0")
            mark("train_with_validation")
            start = perf_counter()
            with tf.device(device):
                history = model.fit(datasets["train"], validation_data=datasets["val"],
                                    epochs=result["epochs_requested"], verbose=2,
                                    shuffle=False,
                                    callbacks=[Progress(), tf.keras.callbacks.CSVLogger(str(args.output_dir / "epochs.csv"))])
            sync_device(tf)
            times["train_with_validation_s"] = perf_counter() - start
            result["epochs_completed"] = len(history.history["loss"])
            # Adam crea sus momentos dentro de fit; también deben estar en el dispositivo elegido.
            result["optimizer_variable_devices"] = sorted(
                {v.handle.device for v in model.optimizer.variables if hasattr(v, "handle")})
            if any(expected not in d for d in result["optimizer_variable_devices"]):
                raise RuntimeError(f"Variables de Adam fuera de {device}: {result['optimizer_variable_devices']}")
            print(f"Dispositivos de las variables de Adam: {result['optimizer_variable_devices']}", flush=True)
            if device == "/GPU:0":
                info = tf.config.experimental.get_memory_info("GPU:0")
                result["gpu_memory_tf_bytes"] = {"current": int(info["current"]), "peak_during_fit": int(info["peak"])}
                print(f"Memoria GPU de TensorFlow: pico {info['peak'] / 1024**2:.1f} MiB, "
                      f"actual {info['current'] / 1024**2:.1f} MiB", flush=True)

            mark("evaluate")
            start = perf_counter()
            with tf.device(device):
                result["validation"] = {k: float(v) for k, v in model.evaluate(datasets["val"], return_dict=True, verbose=0).items()}
                result["test"] = {k: float(v) for k, v in model.evaluate(datasets["test"], return_dict=True, verbose=0).items()}
                probabilities = model.predict(datasets["test"], verbose=0)
            sync_device(tf)
            np.savez_compressed(args.output_dir / "predictions.npz", probabilities=probabilities,
                                labels=arrays["y_test"], classes=arrays["clases"])
            times["evaluate_and_predictions_s"] = perf_counter() - start
            mark("save_model")
            start = perf_counter()
            model.save_weights(args.output_dir / "final.weights.h5")
            times["save_final_weights_s"] = perf_counter() - start
            result["final_weights_sha256"] = weight_hash(model)
            times["run_wall_s"] = perf_counter() - overall
            mark("done")
            result["status"] = "complete"
            result["timing_notes"] = (
                "train_with_validation_s es fit completo e incluye validación de cada época. "
                "evaluate_and_predictions_s es evaluación posterior, predicción y exportación. "
                "load_and_verify_s incluye descompresión y hashes. run_wall_s incluye setup, E/S y monitor; "
                "no incluye la preparación previa ni el arranque del intérprete. "
                "Los monitores CPU (psutil) y GPU (nvidia-smi) se ejecutan en ambas versiones. "
                "En GPU se llama tf.test.experimental.sync_devices() al final de cada época, de fit y "
                "de la evaluación antes de detener el reloj; las copias host->device de cada lote "
                "están incluidas en los tiempos de fit y evaluación."
            )
            write_json(args.output_dir / "metrics.json", result)
            print(json.dumps({"times": times, "validation": result["validation"], "test": result["test"]}, indent=2), flush=True)
    except Exception as error:
        result["status"] = "failed"
        result["error"] = str(error)
        write_json(args.output_dir / "metrics.json", result)
        raise


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/processed/baseline")
    parser.add_argument("--config", type=Path, default=ROOT / "config/baseline.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "gpu", "auto"), default="cpu",
                        help="cpu: referencia (por defecto); gpu: exige CUDA; auto: GPU si existe")
    parser.add_argument("--initial-weights", type=Path, default=None,
                        help="initial.weights.h5 de otra ejecución para partir de los mismos pesos")
    parser.add_argument("--allow-tf32", action="store_true",
                        help="permite TensorFloat-32 en GPU Ampere; experimento separado, no float32 estricto")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--smoke", action="store_true", help="1 época, 16/4/4 imágenes por clase; no es un benchmark")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads debe ser positivo")
    if args.initial_weights and not args.initial_weights.is_file():
        parser.error(f"No existe {args.initial_weights}")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("La carpeta de salida debe estar vacía para no sobrescribir evidencia.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "train.log").open("w", encoding="utf-8") as log:
        with contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
            run(args)


if __name__ == "__main__":
    main()
