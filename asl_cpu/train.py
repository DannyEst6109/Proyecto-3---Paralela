"""Entrenamiento CPU reproducible. No importa TensorFlow hasta ocultar las GPU."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

from .common import ROOT, SPLITS, environment, file_sha256, fingerprint, load_config, write_json
from .data import array_hashes
from .model import create_cnn_base
from .resources import ResourceMonitor


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


def run(args):
    config = load_config(args.config)
    metadata = json.loads((args.data_dir / "metadata.json").read_text(encoding="utf-8"))
    if metadata["status"] != "complete" or fingerprint(config) != metadata["config_sha256"]:
        raise ValueError("Los datos y el entrenamiento deben compartir la misma configuración.")
    result = {
        "status": "running", "run_kind": "smoke" if args.smoke else "baseline_cpu",
        "device": "CPU", "config": config, "config_sha256": fingerprint(config),
        "partition_sha256": metadata["partition_sha256"],
        "source_array_sha256": metadata["array_sha256"],
        "epochs_requested": 1 if args.smoke else config["epochs"],
        "cpu_threads": args.threads, "data_threads": 1,
        "smoke_samples_per_class": {"train": 16, "val": 4, "test": 4} if args.smoke else None,
        "warmup": "none; la primera época incluye trazado y compilación de las funciones de entrenamiento",
        "environment": environment(), "times": {},
    }
    write_json(args.output_dir / "metrics.json", result)
    times = result["times"]
    overall = perf_counter()
    try:
        with ResourceMonitor(args.output_dir / "resources.csv") as monitor:
            monitor.phase = "tensorflow_setup"
            start = perf_counter()
            if "tensorflow" in sys.modules:
                raise RuntimeError("Ejecute este módulo en un proceso nuevo para controlar el dispositivo.")
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
            os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
            os.environ["KERAS_BACKEND"] = "tensorflow"
            # Estas variables se establecen antes del import y del runtime de TF.
            os.environ["OMP_NUM_THREADS"] = str(args.threads)
            import tensorflow as tf
            tf.config.set_visible_devices([], "GPU")
            tf.config.threading.set_intra_op_parallelism_threads(args.threads)
            tf.config.threading.set_inter_op_parallelism_threads(1)
            tf.keras.utils.set_random_seed(config["seed"])
            tf.config.experimental.enable_op_determinism()
            tf.keras.mixed_precision.set_global_policy("float32")
            visible_gpus = tf.config.get_visible_devices("GPU")
            if visible_gpus:
                raise RuntimeError(f"La referencia CPU tiene GPU visibles: {visible_gpus}")
            result["visible_gpus"] = [str(d) for d in visible_gpus]
            result["logical_devices"] = [str(d) for d in tf.config.list_logical_devices()]
            result["tensorflow_build"] = tf.sysconfig.get_build_info()
            times["tensorflow_setup_s"] = perf_counter() - start
            print("GPU visibles: [] | entrenamiento forzado a /CPU:0", flush=True)

            monitor.phase = "load_prepared_data"
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

            monitor.phase = "normalize_cpu"
            start = perf_counter()
            for split in SPLITS:
                arrays[f"X_{split}"] = arrays[f"X_{split}"].astype(np.float32) / np.float32(255.0)
            times["normalize_cpu_s"] = perf_counter() - start
            print(f"Muestras: {result['sample_counts']}", flush=True)

            monitor.phase = "build_model_and_input"
            start = perf_counter()
            datasets = {}
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
                model = create_cnn_base(tf, config, len(arrays["clases"]))
                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config["learning_rate"]),
                              loss="sparse_categorical_crossentropy", metrics=["accuracy"], jit_compile=False)
            result["weight_devices"] = sorted({v.handle.device for v in model.weights})
            if any("CPU:0" not in device for device in result["weight_devices"]):
                raise RuntimeError("Se detectaron variables del modelo fuera de CPU.")
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
                    row = {"epoch": epoch + 1, "epoch_wall_s": perf_counter() - self.start}
                    row.update({k: float(v) for k, v in (logs or {}).items()})
                    progress.append(row)
                    write_json(args.output_dir / "progress.json", {"epochs": progress})

            progress = []
            monitor.phase = "train_with_validation"
            start = perf_counter()
            with tf.device("/CPU:0"):
                history = model.fit(datasets["train"], validation_data=datasets["val"],
                                    epochs=result["epochs_requested"], verbose=2,
                                    shuffle=False,
                                    callbacks=[Progress(), tf.keras.callbacks.CSVLogger(str(args.output_dir / "epochs.csv"))])
            times["train_with_validation_s"] = perf_counter() - start
            result["epochs_completed"] = len(history.history["loss"])

            monitor.phase = "evaluate"
            start = perf_counter()
            with tf.device("/CPU:0"):
                result["validation"] = {k: float(v) for k, v in model.evaluate(datasets["val"], return_dict=True, verbose=0).items()}
                result["test"] = {k: float(v) for k, v in model.evaluate(datasets["test"], return_dict=True, verbose=0).items()}
                probabilities = model.predict(datasets["test"], verbose=0)
            np.savez_compressed(args.output_dir / "predictions.npz", probabilities=probabilities,
                                labels=arrays["y_test"], classes=arrays["clases"])
            times["evaluate_and_predictions_s"] = perf_counter() - start
            monitor.phase = "save_model"
            start = perf_counter()
            model.save_weights(args.output_dir / "final.weights.h5")
            times["save_final_weights_s"] = perf_counter() - start
            result["final_weights_sha256"] = weight_hash(model)
            times["run_wall_s"] = perf_counter() - overall
            result["status"] = "complete"
            result["timing_notes"] = (
                "train_with_validation_s es fit completo e incluye validación de cada época. "
                "evaluate_and_predictions_s es evaluación posterior, predicción y exportación. "
                "load_and_verify_s incluye descompresión y hashes. run_wall_s incluye setup, E/S y monitor; "
                "no incluye la preparación previa ni el arranque del intérprete. "
                "El monitor introduce sobrecosto y debe usarse también en la comparación GPU."
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
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--smoke", action="store_true", help="1 época, 16/4/4 imágenes por clase; no es un benchmark")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads debe ser positivo")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("La carpeta de salida debe estar vacía para no sobrescribir evidencia.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "train.log").open("w", encoding="utf-8") as log:
        with contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
            run(args)


if __name__ == "__main__":
    main()
