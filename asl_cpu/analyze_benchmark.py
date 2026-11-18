"""Parte 3: convierte las corridas de `asl_cpu.benchmark` en tablas, speedup y gráficos.

Uso:
    python -m asl_cpu.analyze_benchmark results/parte3/rtx3060
    python -m asl_cpu.analyze_benchmark results/parte3/rtx3060 results/parte3/colab_t4 \
        --output-dir results/parte3/comparacion

Con un experimento escribe en `<experimento>/analisis/`; con varios, un análisis por entorno y una
comparación entre entornos. Solo usa las corridas medidas (las de calentamiento se descartan) y se
detiene si alguna corrida medida no comparte la misma carga de trabajo.

Definiciones:
- Speedup de una fase = tiempo promedio de la referencia (CPU) / tiempo promedio de la versión
  acelerada, cada uno con sus propias repeticiones. El intervalo es un bootstrap percentil 95 %
  de esa razón (remuestreo independiente de cada grupo, semilla fija).
- Las transferencias host→GPU están dentro de fit y evaluación: se reportan en bytes y GB/s, pero
  nunca se suman otra vez al total.
- Los tiempos por época (primera época y media de las épocas 2..N) van en una tabla aparte y no
  se mezclan con el entrenamiento completo.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

SAME_WORKLOAD = ("config_sha256", "partition_sha256", "used_array_sha256", "initial_weights_sha256",
                 "model_parameters", "epochs_completed", "sample_counts")

# (clave, nombre en español, dónde se ejecuta)
TRAIN_PHASES = [
    ("tensorflow_setup_s", "Import y configuración de TensorFlow/CUDA", "CPU"),
    ("load_and_verify_s", "Carga y verificación del NPZ", "CPU"),
    ("normalize_cpu_s", "Normalización uint8→float32", "CPU"),
    ("build_model_and_input_s", "Modelo, optimizador y tf.data", "CPU + contexto"),
    ("train_with_validation_s", "Entrenamiento con validación (fit)", "dispositivo"),
    ("evaluate_and_predictions_s", "Evaluación y predicción", "dispositivo"),
    ("save_final_weights_s", "Guardado de pesos finales", "CPU"),
    ("run_wall_s", "Total del comando (reloj interno)", "total"),
    ("process_wall_s", "Total del proceso (reloj del orquestador)", "total"),
]
EPOCH_METRICS = [
    ("first_epoch_s", "Primera época (incluye trazado y selección de algoritmos)"),
    ("later_epochs_mean_s", "Media de las épocas 2..N (régimen estable)"),
]
PREP_PHASES = [
    ("inventory_s", "Inventario de las 87,000 imágenes"),
    ("partition_s", "Partición por bloques"),
    ("images_s", "Lectura JPEG, resize y RGB"),
    ("export_s", "Compresión NPZ y manifiestos"),
    ("prepare_wall_s", "Preparación completa (reloj interno)"),
    ("process_wall_s", "Preparación, proceso completo"),
]
CONFIG_LABELS = {
    "cpu": "CPU", "gpu_keras": "GPU CUDA (Keras)", "gpu_numba": "GPU CUDA + Adam Numba",
    "cpu_1thread": "CPU 1 hilo",
}
PARALLEL_PREP = ("images_s", "prepare_wall_s", "process_wall_s")


def procs(w: int) -> str:
    return f"{w} proceso" + ("s" if w != 1 else "")


# Paleta categórica validada (skill dataviz, slots 1-3 válidos en todos los pares) y tinta de texto.
SERIES = {"cpu": "#2a78d6", "gpu_keras": "#eb6834", "gpu_numba": "#1baf7a", "cpu_1thread": "#6250d6"}
EXTRA = ["#6250d6", "#eda100", "#e87ba4", "#008300", "#e34948"]
SURFACE, INK, INK_2, GRID, NEUTRAL = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3de", "#a3a29c"
# Valores críticos t de Student a dos colas, 95 %, para gl = 1..30.
T95 = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160,
       2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056,
       2.052, 2.048, 2.045, 2.042]


# ----------------------------------------------------------------------------- estadística
def describe(values) -> dict:
    x = np.asarray([v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))], dtype=float)
    n = len(x)
    if n == 0:
        return {"n": 0, "mean": None, "std": None, "min": None, "max": None, "cv_percent": None, "ci95": None}
    std = float(x.std(ddof=1)) if n > 1 else 0.0
    mean = float(x.mean())
    t = T95[n - 2] if 2 <= n <= 31 else 1.96
    return {"n": n, "mean": mean, "std": std, "min": float(x.min()), "max": float(x.max()),
            "cv_percent": 100 * std / mean if mean else None,
            "ci95": t * std / math.sqrt(n) if n > 1 else None}


def speedup(reference, accelerated, n_boot: int = 10000, seed: int = 0) -> dict:
    """Razón de medias y su intervalo bootstrap percentil 95 %."""
    a = np.asarray(reference, dtype=float)
    b = np.asarray(accelerated, dtype=float)
    if len(a) == 0 or len(b) == 0 or b.mean() == 0:
        return {"speedup": None, "ci_low": None, "ci_high": None}
    result = {"speedup": float(a.mean() / b.mean()), "ci_low": None, "ci_high": None}
    if len(a) > 1 and len(b) > 1:
        rng = np.random.default_rng(seed)
        ra = a[rng.integers(0, len(a), (n_boot, len(a)))].mean(axis=1)
        rb = b[rng.integers(0, len(b), (n_boot, len(b)))].mean(axis=1)
        ratios = ra / rb
        result["ci_low"], result["ci_high"] = (float(v) for v in np.percentile(ratios, [2.5, 97.5]))
    return result


# ----------------------------------------------------------------------------- lectura
def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def gpu_samples(run: Path) -> pd.DataFrame | None:
    path = run / "gpu.csv"
    if not path.is_file():
        return None
    data = pd.read_csv(path, skipinitialspace=True)
    if data.empty or "timestamp" not in data:
        return None
    data = data[data["timestamp"].astype(str).str.match(r"\d{4}/")].copy()
    if data.empty:
        return None
    data["t"] = [datetime.strptime(v.strip(), "%Y/%m/%d %H:%M:%S.%f").timestamp() for v in data["timestamp"]]
    for column in data.columns.drop(["timestamp", "t"]):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    return data


def phase_window(metrics: dict, phase: str) -> tuple[float, float] | None:
    marks = metrics.get("phase_marks") or []
    for i, mark in enumerate(marks[:-1]):
        if mark["phase"] == phase:
            return mark["unix_time"], marks[i + 1]["unix_time"]
    return None


def train_row(base: Path, step: dict, label: str) -> dict:
    run = base / step["run_dir"]
    metrics = read_json(run / "metrics.json")
    times = metrics["times"]
    row = {"entorno": label, "config": step["config"], "rep": step["rep"], "order": step["order"],
           "run_dir": step["run_dir"], "run_kind": metrics.get("run_kind"), "device": metrics.get("device"),
           "tensor_float_32": metrics.get("tensor_float_32"), "threads": metrics.get("cpu_threads"),
           "process_wall_s": step.get("process_wall_s"),
           "gpu_temp_start_c": (step.get("gpu_state_start") or {}).get("temperature_c")}
    for key, _, _ in TRAIN_PHASES:
        if key != "process_wall_s":
            row[key] = times.get(key)
    progress_path = run / "progress.json"
    epochs = read_json(progress_path)["epochs"] if progress_path.is_file() else []
    epoch_times = [e["epoch_wall_s"] for e in epochs]
    row["first_epoch_s"] = epoch_times[0] if epoch_times else None
    row["later_epochs_mean_s"] = float(np.mean(epoch_times[1:])) if len(epoch_times) > 1 else None
    row["epoch_times"] = epoch_times
    for split in ("validation", "test"):
        for metric in ("accuracy", "loss"):
            row[f"{split}_{metric}"] = (metrics.get(split) or {}).get(metric)
    for key in SAME_WORKLOAD:
        value = metrics.get(key)
        row[key] = json.dumps(value, sort_keys=True) if isinstance(value, dict) else value
    transfers = metrics.get("transfers") or {}
    fit_bytes = transfers.get("fit_h2d_bytes")
    row["fit_h2d_gb"] = fit_bytes / 1e9 if fit_bytes else None
    row["fit_h2d_gb_per_s"] = fit_bytes / 1e9 / times["train_with_validation_s"] if fit_bytes else None
    row["eval_h2d_gb"] = (transfers.get("evaluate_predict_h2d_bytes") or 0) / 1e9 or None
    numba = metrics.get("numba_adam") or {}
    row["numba_kernel_and_sync_s"] = numba.get("kernel_and_sync_s")
    row["numba_interop_s"] = numba.get("dlpack_interop_and_assign_s")
    peak = (metrics.get("gpu_memory_tf_bytes") or {}).get("peak_during_fit")
    row["tf_gpu_peak_mib"] = peak / 2**20 if peak else None

    resources = run / "resources.csv"
    if resources.is_file():
        data = pd.read_csv(resources)
        fit = data[data["phase"] == "train_with_validation"]
        cores = [c for c in data.columns if c.startswith("cpu_") and c.endswith("_percent")]
        if not fit.empty:
            row["process_cpu_mean_percent"] = float(fit["process_tree_cpu_percent"].mean())
            row["system_core_mean_percent"] = float(fit[cores].to_numpy().mean())
            row["busiest_core_mean_percent"] = float(fit[cores].mean().max())
            row["process_rss_max_mib"] = float(fit["process_tree_rss_mb"].max())
            row["logical_cpus"] = len(cores)
    gpu = gpu_samples(run)
    window = phase_window(metrics, "train_with_validation")
    if gpu is not None and window:
        start = window[0] + (row["first_epoch_s"] or 0)   # régimen estable: sin la primera época
        steady = gpu[(gpu["t"] >= start) & (gpu["t"] <= window[1])]
        whole = gpu[(gpu["t"] >= window[0]) & (gpu["t"] <= window[1])]
        if not steady.empty:
            row["gpu_util_mean_percent"] = float(steady["gpu_util_percent"].mean())
            row["gpu_util_max_percent"] = float(steady["gpu_util_percent"].max())
            row["gpu_mem_util_mean_percent"] = float(steady["gpu_mem_util_percent"].mean())
            row["gpu_power_mean_w"] = float(steady["power_w"].mean())
            row["gpu_temp_mean_c"] = float(steady["temperature_c"].mean())
            row["gpu_temp_max_c"] = float(steady["temperature_c"].max())
            row["gpu_sm_clock_mean_mhz"] = float(steady["sm_clock_mhz"].mean())
        if not whole.empty:
            row["gpu_mem_used_max_mib"] = float(whole["gpu_mem_used_mib"].max())
    return row


def prepare_row(base: Path, step: dict, label: str) -> dict:
    run = base / step["run_dir"]
    meta = read_json(run / "metadata.json")
    row = {"entorno": label, "workers": step["workers"], "rep": step["rep"], "order": step["order"],
           "run_dir": step["run_dir"], "process_wall_s": step.get("process_wall_s"),
           "same_arrays": (step.get("prepare_check") or {}).get("same_arrays_as_training_data")}
    row.update({k: meta["times"].get(k) for k, _ in PREP_PHASES if k != "process_wall_s"})
    resources = run / "resources.csv"
    if resources.is_file():
        data = pd.read_csv(resources)
        images = data[data["phase"] == "decode_resize_rgb"]
        cores = [c for c in data.columns if c.startswith("cpu_") and c.endswith("_percent")]
        if not images.empty:
            row["process_cpu_mean_percent"] = float(images["process_tree_cpu_percent"].mean())
            row["system_core_mean_percent"] = float(images[cores].to_numpy().mean())
            row["cores_above_50_mean"] = float((images[cores] > 50).sum(axis=1).mean())
            row["logical_cpus"] = len(cores)
    return row


def load_experiment(path: Path) -> dict:
    manifest = read_json(path / "manifest.json")
    label = manifest.get("label") or path.name
    measured = [s for s in manifest["runs"] if s.get("role") == "measured" and s.get("status") == "complete"]
    train = pd.DataFrame([train_row(path, s, label) for s in measured if s["kind"] == "train"])
    prep = pd.DataFrame([prepare_row(path, s, label) for s in measured if s["kind"] == "prepare"])
    warmups = [s["name"] for s in manifest["runs"] if s.get("role") == "warmup"]
    failed = [s["name"] for s in manifest["runs"] if s.get("status") != "complete"]
    return {"path": path, "label": label, "manifest": manifest, "train": train, "prep": prep,
            "warmups": warmups, "failed": failed}


# ----------------------------------------------------------------------------- análisis
def workload_check(train: pd.DataFrame) -> dict:
    if train.empty:
        return {"ok": False, "detail": "sin corridas de entrenamiento medidas"}
    detail = {key: int(train[key].astype(str).nunique()) for key in SAME_WORKLOAD}
    expected = {"cpu": "CPU", "cpu_1thread": "CPU", "gpu_keras": "GPU", "gpu_numba": "GPU"}
    devices_ok = bool((train["config"].map(expected) == train["device"]).all())
    tf32 = sorted(train["tensor_float_32"].dropna().astype(str).unique().tolist())
    return {"ok": all(v == 1 for v in detail.values()) and devices_ok and tf32 in ([], ["False"]),
            "distinct_values": detail, "devices_ok": devices_ok, "tensor_float_32": tf32}


def config_order(train: pd.DataFrame, manifest: dict) -> list[str]:
    wanted = manifest["settings"]["configs"]
    return [c for c in wanted if c in set(train["config"])]


def analyze(exp: dict, baseline: str) -> dict:
    train, prep = exp["train"], exp["prep"]
    configs = config_order(train, exp["manifest"]) if not train.empty else []
    out = {"label": exp["label"], "configs": configs, "baseline": baseline}
    out["workload"] = workload_check(train)
    smoke = bool(exp["manifest"]["settings"].get("smoke"))
    out["smoke"] = smoke

    keys = [k for k, _, _ in TRAIN_PHASES] + [k for k, _ in EPOCH_METRICS] + [
        "validation_accuracy", "validation_loss", "test_accuracy", "test_loss", "fit_h2d_gb",
        "fit_h2d_gb_per_s", "eval_h2d_gb", "process_cpu_mean_percent", "system_core_mean_percent",
        "busiest_core_mean_percent", "process_rss_max_mib", "gpu_util_mean_percent", "gpu_util_max_percent",
        "gpu_mem_util_mean_percent", "gpu_mem_used_max_mib", "gpu_power_mean_w", "gpu_temp_mean_c",
        "gpu_temp_max_c", "gpu_sm_clock_mean_mhz", "tf_gpu_peak_mib", "numba_kernel_and_sync_s",
        "numba_interop_s", "gpu_temp_start_c"]
    stats = {}
    for config in configs:
        group = train[train["config"] == config]
        stats[config] = {k: describe(group[k]) for k in keys if k in group}
    out["stats"] = stats

    out["speedup"] = {}
    if baseline in configs:
        base = train[train["config"] == baseline]
        for config in configs:
            if config == baseline:
                continue
            other = train[train["config"] == config]
            out["speedup"][config] = {
                k: speedup(base[k].dropna(), other[k].dropna())
                for k in [k for k, _, _ in TRAIN_PHASES] + [k for k, _ in EPOCH_METRICS] if k in base}
        # Ley de Amdahl: fracción del tiempo de la referencia que corre en el dispositivo acelerado.
        mean = lambda frame, key: float(frame[key].mean())
        accelerated = ["train_with_validation_s", "evaluate_and_predictions_s"]
        p = sum(mean(base, k) for k in accelerated) / mean(base, "run_wall_s")
        out["amdahl"] = {}
        for config in out["speedup"]:
            other = train[train["config"] == config]
            s_acc = sum(mean(base, k) for k in accelerated) / sum(mean(other, k) for k in accelerated)
            out["amdahl"][config] = {
                "fraction_accelerated_p": p, "speedup_accelerated_phases": s_acc,
                "amdahl_predicted_total": 1 / ((1 - p) + p / s_acc),
                "amdahl_limit_infinite_speedup": 1 / (1 - p),
                "measured_total_run_wall": out["speedup"][config]["run_wall_s"]["speedup"],
                "cpu_only_phases_reference_s": mean(base, "run_wall_s") - sum(mean(base, k) for k in accelerated),
                "cpu_only_phases_accelerated_s": mean(other, "run_wall_s") - sum(mean(other, k) for k in accelerated),
            }

    # Calidad: reproducibilidad y coincidencia de predicciones con la referencia.
    out["quality"] = {}
    ref_pred = None
    if baseline in configs:
        first = train[train["config"] == baseline].sort_values("rep").iloc[0]
        path = exp["path"] / first["run_dir"] / "predictions.npz"
        if path.is_file():
            with np.load(path) as data:
                ref_pred = data["probabilities"].argmax(1)
    for config in configs:
        group = train[train["config"] == config]
        agreement = []
        if ref_pred is not None:
            for run_dir in group["run_dir"]:
                path = exp["path"] / run_dir / "predictions.npz"
                if path.is_file():
                    with np.load(path) as data:
                        agreement.append(float((data["probabilities"].argmax(1) == ref_pred).mean()))
        out["quality"][config] = {
            "test_accuracy_distinct_values": int(group["test_accuracy"].round(10).nunique()),
            "agreement_with_reference_rep01": describe(agreement) if agreement else None}

    # Preparación multinúcleo.
    out["prepare"] = {}
    if not prep.empty:
        workers = sorted(prep["workers"].unique())
        out["prepare"]["workers"] = [int(w) for w in workers]
        out["prepare"]["stats"] = {int(w): {k: describe(prep[prep["workers"] == w][k]) for k, _ in PREP_PHASES}
                                   | {k: describe(prep[prep["workers"] == w][k]) for k in
                                      ("process_cpu_mean_percent", "system_core_mean_percent", "cores_above_50_mean")
                                      if k in prep}
                                   for w in workers}
        out["prepare"]["same_arrays_all"] = bool(prep["same_arrays"].fillna(False).all())
        low = prep[prep["workers"] == workers[0]]
        out["prepare"]["speedup"] = {}
        for w in workers[1:]:
            high = prep[prep["workers"] == w]
            out["prepare"]["speedup"][int(w)] = {
                k: {**speedup(low[k].dropna(), high[k].dropna()), "efficiency": None} for k, _ in PREP_PHASES}
            for k in out["prepare"]["speedup"][int(w)]:
                s = out["prepare"]["speedup"][int(w)][k]["speedup"]
                ratio = w / workers[0]
                out["prepare"]["speedup"][int(w)][k]["efficiency"] = s / ratio if s else None

        # Flujo completo compuesto: preparación + entrenamiento (suma de medias de dos comandos).
        if baseline in configs:
            seq = float(low["process_wall_s"].mean())
            par = float(prep[prep["workers"] == workers[-1]]["process_wall_s"].mean())
            base_run = float(train[train["config"] == baseline]["process_wall_s"].mean())
            pipeline = {"original": {"description": f"Preparación con {procs(workers[0])} + entrenamiento {CONFIG_LABELS.get(baseline, baseline)}",
                                     "total_s": seq + base_run}}
            for config in configs:
                if config == baseline:
                    continue
                run = float(train[train["config"] == config]["process_wall_s"].mean())
                pipeline[config] = {"description": f"Preparación con {procs(workers[-1])} + entrenamiento {CONFIG_LABELS.get(config, config)}",
                                    "prepare_s": par, "train_process_s": run, "total_s": par + run,
                                    "speedup_vs_original": (seq + base_run) / (par + run)}
            out["pipeline"] = pipeline
    return out


# ----------------------------------------------------------------------------- formato
def fmt(value, digits=3, unit=""):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    return f"{value:,.{digits}f}{unit}"


def fmt_stat(stat: dict, digits=3) -> str:
    if not stat or stat["n"] == 0:
        return "—"
    text = fmt(stat["mean"], digits)
    if stat["n"] > 1:
        text += f" ± {fmt(stat['std'], digits)}"
    return text


def fmt_speedup(entry: dict) -> str:
    if not entry or entry["speedup"] is None:
        return "—"
    text = f"{entry['speedup']:.2f}×"
    if entry.get("ci_low") is not None:
        text += f" [{entry['ci_low']:.2f}–{entry['ci_high']:.2f}]"
    return text


def md_table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---:" if i else "---" for i in range(len(header))) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def report_markdown(exp: dict, res: dict, figures: dict) -> str:
    m = exp["manifest"]
    s = m["settings"]
    host = m.get("host", {})
    gpu = host.get("gpu") or {}
    configs = res["configs"]
    lines = [f"# Resultados de la parte 3 · {res['label']}", ""]
    if res["smoke"]:
        lines += ["> **PRUEBA DE INTEGRACIÓN (`--smoke`).** Estas cifras no sirven para conclusiones de rendimiento.", ""]
    if not res["workload"]["ok"]:
        lines += ["> **ATENCIÓN:** las corridas medidas no comparten exactamente la misma carga de trabajo. "
                  "Revise `resumen.json → workload`.", ""]
    lines += ["## Entorno y método", "",
              md_table(["Elemento", "Valor"], [
                  ["CPU", f"{host.get('cpu_model') or '—'} · {host.get('physical_cores')} núcleos / {host.get('logical_cpus')} hilos"],
                  ["RAM", fmt((host.get('ram_bytes') or 0) / 2**30, 1, " GiB")],
                  ["GPU", f"{gpu.get('name', '—')} · driver {gpu.get('driver_version', '—')} · {fmt(gpu.get('memory_total_mib'), 0, ' MiB')}"],
                  ["Sistema / Python", f"{host.get('platform') or '—'} · {(str(host.get('python') or '—').split() or ['—'])[0]}"],
                  ["TensorFlow", str((host.get('packages') or {}).get('tensorflow'))],
                  ["Commit", str(host.get("git_commit"))],
                  ["Repeticiones medidas", ", ".join(f"{CONFIG_LABELS.get(c, c)}: {res['stats'][c]['train_with_validation_s']['n']}" for c in configs)],
                  ["Calentamiento", f"{s['warmup']} (descartado, una corrida por configuración)"],
                  ["Orden", s["order_policy"]],
                  ["Enfriamiento entre corridas", f"{s['cooldown_s']} s" + (f", y GPU ≤ {s['cooldown_gpu_temp_c']} °C" if s.get('cooldown_gpu_temp_c') else "")],
                  ["Hilos intra-op de TensorFlow", str(s["threads"])],
                  ["Carga de trabajo", "17,400 imágenes 64×64 RGB · CNN base de 1,145,693 parámetros · Adam 0.001 · lote 64 · 20 épocas · float32 estricto (sin TF32)"
                   if not res["smoke"] else "smoke: 1 época, 16/4/4 imágenes por clase"],
                  ["Misma carga en todas las corridas", "sí" if res["workload"]["ok"] else "NO"],
              ]), "",
              "Cada valor es media ± desviación estándar muestral de las repeticiones medidas. "
              "El speedup es la media de la referencia CPU entre la media de la versión con GPU; entre corchetes, "
              "intervalo bootstrap del 95 %.", ""]

    lines += ["## Tiempos por fase (s)", ""]
    header = ["Fase", "Se ejecuta en"] + [CONFIG_LABELS.get(c, c) for c in configs]
    rows = [[name, where] + [fmt_stat(res["stats"][c].get(key)) for c in configs] for key, name, where in TRAIN_PHASES]
    lines += [md_table(header, rows), "",
              "Las fases no se suman otra vez al total: `run_wall_s` ya incluye todas y además el monitor y tareas entre fases. "
              "`process_wall_s` es el proceso completo visto desde fuera (incluye arrancar Python e importar módulos).", ""]

    lines += ["## Tiempo por época (s)", "", "Separado del entrenamiento completo, como pide el acuerdo del equipo.", ""]
    rows = [[name] + [fmt_stat(res["stats"][c].get(key)) for c in configs] for key, name in EPOCH_METRICS]
    lines += [md_table(["Medición"] + [CONFIG_LABELS.get(c, c) for c in configs], rows), ""]

    if res["speedup"]:
        lines += ["## Speedup por fase (CPU / GPU)", ""]
        acc = list(res["speedup"])
        rows = [[name] + [fmt_speedup(res["speedup"][c].get(key)) for c in acc]
                for key, name, _ in TRAIN_PHASES + [(k, n, "") for k, n in EPOCH_METRICS]]
        lines += [md_table(["Fase"] + [CONFIG_LABELS.get(c, c) for c in acc], rows), "",
                  "Un valor menor que 1 indica que esa fase tarda más con la versión GPU (por ejemplo, crear el "
                  "contexto CUDA dentro de la construcción del modelo).", ""]
        lines += ["### Ley de Amdahl", ""]
        rows = []
        for c, a in res["amdahl"].items():
            rows.append([CONFIG_LABELS.get(c, c), fmt(100 * a["fraction_accelerated_p"], 1, " %"),
                         fmt(a["speedup_accelerated_phases"], 2, "×"), fmt(a["amdahl_predicted_total"], 2, "×"),
                         fmt(a["measured_total_run_wall"], 2, "×"), fmt(a["amdahl_limit_infinite_speedup"], 1, "×"),
                         f"{fmt(a['cpu_only_phases_reference_s'], 2)} → {fmt(a['cpu_only_phases_accelerated_s'], 2)}"])
        lines += [md_table(["Versión", "p (fracción acelerable en CPU)", "Speedup de fit + evaluación",
                            "Total previsto por Amdahl", "Total medido", "Límite si fit fuese instantáneo",
                            "Fases no aceleradas (s), CPU → GPU"], rows), "",
                  "p es la fracción de `run_wall_s` de la referencia CPU que pasa al dispositivo (fit y evaluación). "
                  "El total medido queda por debajo del previsto cuando las fases que siguen en CPU también crecen con la GPU "
                  "(inicialización de CUDA y del contexto).", ""]

    gpu_configs = [c for c in configs if res["stats"][c].get("fit_h2d_gb", {}).get("n")]
    if gpu_configs:
        lines += ["## Transferencias host → GPU", ""]
        rows = [[CONFIG_LABELS.get(c, c), fmt(res["stats"][c]["fit_h2d_gb"]["mean"], 2, " GB"),
                 fmt(res["stats"][c]["fit_h2d_gb_per_s"]["mean"], 3, " GB/s"),
                 fmt(res["stats"][c]["eval_h2d_gb"]["mean"], 3, " GB")] for c in gpu_configs]
        lines += [md_table(["Versión", "Copiado durante fit", "Tasa efectiva en fit", "Copiado en evaluación"], rows), "",
                  "Los bytes salen del tamaño real de los lotes que Keras copia en cada época. El tiempo de copia está "
                  "dentro de fit y de la evaluación y no se suma de nuevo al total.", ""]

    lines += ["## Uso de recursos durante fit", ""]
    rows = []
    for c in configs:
        st = res["stats"][c]
        cores = st.get("process_cpu_mean_percent", {}).get("mean")
        rows.append([CONFIG_LABELS.get(c, c), fmt(cores, 0, " %") + (f" (≈{cores / 100:.1f} hilos)" if cores else ""),
                     fmt(st.get("system_core_mean_percent", {}).get("mean"), 1, " %"),
                     fmt(st.get("gpu_util_mean_percent", {}).get("mean"), 0, " %"),
                     fmt(st.get("gpu_mem_used_max_mib", {}).get("mean"), 0, " MiB"),
                     fmt(st.get("gpu_power_mean_w", {}).get("mean"), 1, " W"),
                     fmt(st.get("gpu_temp_max_c", {}).get("mean"), 0, " °C")])
    lines += [md_table(["Versión", "CPU del proceso", "Media por procesador lógico", "Utilización SM (épocas 2..N)",
                        "Memoria GPU (máx.)", "Potencia GPU", "Temperatura máx."], rows), "",
              "CPU del proceso: 100 % equivale a un procesador lógico. La memoria GPU de nvidia-smi incluye el "
              "escritorio y otros procesos; en la versión CPU muestra solo ese consumo de fondo.", ""]

    lines += ["## Calidad del modelo", ""]
    rows = []
    def pct(stat):
        scaled = {k: (v * 100 if k in ("mean", "std") and v is not None else v) for k, v in stat.items()}
        return fmt_stat(scaled, 2) + " %"

    for c in configs:
        st, q = res["stats"][c], res["quality"][c]
        agree = q["agreement_with_reference_rep01"]
        rows.append([CONFIG_LABELS.get(c, c), pct(st["test_accuracy"]), pct(st["validation_accuracy"]),
                     str(q["test_accuracy_distinct_values"]),
                     fmt(100 * agree["mean"], 1, " %") if agree else "—"])
    lines += [md_table(["Versión", "Exactitud de prueba", "Exactitud de validación",
                        "Valores distintos entre repeticiones", "Predicciones iguales a la CPU rep. 1"], rows), "",
              "Un solo valor distinto confirma que cada versión es reproducible en el mismo dispositivo "
              "(`enable_op_determinism`). CPU y GPU suman en distinto orden, así que entre dispositivos se esperan "
              "diferencias pequeñas de redondeo y no idénticas.", ""]

    if res.get("prepare"):
        p = res["prepare"]
        lines += ["## Preparación de imágenes en CPU (multiproceso)", ""]
        header = ["Fase"] + [procs(w) for w in p["workers"]] + [f"Speedup {w} vs {p['workers'][0]}" for w in p["speedup"]]
        rows = []
        for key, name in PREP_PHASES:
            rows.append([name] + [fmt_stat(p["stats"][w][key]) for w in p["workers"]] +
                        [fmt_speedup(p["speedup"][w][key]) + (f" · E = {100 * p['speedup'][w][key]['efficiency']:.0f} %"
                                                              if p["speedup"][w][key]["efficiency"] and key in PARALLEL_PREP
                                                              else "")
                         for w in p["speedup"]])
        rows.append(["CPU del árbol de procesos durante la lectura"] +
                     [fmt(p["stats"][w].get("process_cpu_mean_percent", {}).get("mean"), 0, " %") for w in p["workers"]] +
                     [""] * len(p["speedup"]))
        rows.append(["Procesadores lógicos > 50 % (media)"] +
                     [fmt(p["stats"][w].get("cores_above_50_mean", {}).get("mean"), 1) for w in p["workers"]] +
                     [""] * len(p["speedup"]))
        lines += [md_table(header, rows), "",
                  "Todas las repeticiones reproducen exactamente los arreglos usados para entrenar: " +
                  ("sí." if p["same_arrays_all"] else "**NO**."),
                  "E es la eficiencia paralela (speedup entre el aumento de procesos); solo se muestra donde hay trabajo "
                  "repartido entre procesos. Inventario, partición y exportación (compresión del NPZ) se ejecutan en el "
                  "proceso principal en ambos modos, por eso no se aceleran.", ""]
        if res.get("pipeline"):
            rows = [[v["description"], fmt(v["total_s"], 1, " s"),
                     fmt(v.get("speedup_vs_original"), 2, "×") if k != "original" else "1.00×"]
                    for k, v in res["pipeline"].items()]
            lines += ["### Flujo completo: preparación + entrenamiento", "",
                      md_table(["Flujo", "Tiempo (s)", "Speedup"], rows), "",
                      "Total compuesto: suma de las medias de dos comandos medidos por separado (proceso completo), no "
                      "una medición continua.", ""]

    if figures:
        lines += ["## Gráficos", ""] + [f"![{title}]({name})" for name, title in figures.items()] + [""]
    if exp["warmups"]:
        lines += [f"Corridas de calentamiento descartadas: {', '.join(exp['warmups'])}.", ""]
    if exp["failed"]:
        lines += [f"Corridas no completadas (excluidas): {', '.join(exp['failed'])}.", ""]
    return "\n".join(lines)


# ----------------------------------------------------------------------------- gráficos
def setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
        "text.color": INK, "axes.titlecolor": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
        "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold", "legend.frameon": False,
    })
    return plt


def color_of(config: str, index: int = 0) -> str:
    return SERIES.get(config, EXTRA[index % len(EXTRA)])


def make_figures(exp: dict, res: dict, out: Path) -> dict:
    plt = setup_matplotlib()
    train, configs = exp["train"], res["configs"]
    figures = {}
    tag = " · PRUEBA SMOKE" if res["smoke"] else ""
    if not configs:
        return figures

    # 1. Entrenamiento completo por configuración, con cada repetición.
    fig, ax = plt.subplots(figsize=(9, 1.3 + 0.62 * len(configs)))
    for i, c in enumerate(configs):
        values = train[train["config"] == c]["train_with_validation_s"]
        st = res["stats"][c]["train_with_validation_s"]
        y = len(configs) - 1 - i
        ax.barh(y, st["mean"], height=0.36, color=color_of(c, i))
        if st["n"] > 1:
            ax.errorbar(st["mean"], y, xerr=st["std"], color=INK, capsize=4, lw=1.2)
        ax.scatter(values, [y] * len(values), s=26, facecolor=SURFACE, edgecolor=INK, zorder=3, lw=1.1)
        right = max([st["mean"] + (st["std"] or 0)] + list(values))
        ax.text(right, y, f"   {st['mean']:.1f} s" + (f" ± {st['std']:.1f}" if st["n"] > 1 else ""),
                color=INK, fontsize=9, va="center")
    ax.set_yticks(range(len(configs)), [CONFIG_LABELS.get(c, c) for c in reversed(configs)])
    ax.set_xlabel("Tiempo de entrenamiento con validación, fit completo (s)")
    ax.set_title(f"Entrenamiento: media ± desviación y cada repetición{tag}", loc="left")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, ax.get_xlim()[1] * 1.25)
    fig.tight_layout()
    name = "entrenamiento_por_configuracion.png"
    fig.savefig(out / name, dpi=160)
    plt.close(fig)
    figures[name] = "Tiempo de entrenamiento por configuración"

    # 2. Desglose del total del comando.
    parts = [("Fases que siguen en CPU", NEUTRAL),
             ("Entrenamiento con validación (fit)", "#6250d6"), ("Evaluación y predicción", "#eda100")]
    fig, ax = plt.subplots(figsize=(9.5, 1.9 + 0.6 * len(configs)))
    for i, c in enumerate(configs):
        st = res["stats"][c]
        fit, ev, total = (st[k]["mean"] for k in ("train_with_validation_s", "evaluate_and_predictions_s", "run_wall_s"))
        rest = total - fit - ev
        y = len(configs) - 1 - i
        left = 0
        for (label, color), width in zip(parts, (rest, fit, ev)):
            ax.barh(y, width, left=left, height=0.36, color=color, edgecolor=SURFACE, linewidth=2,
                    label=label if i == 0 else None)
            left += width
        ax.text(total, y, f"  {total:.1f} s", va="center", fontsize=9, color=INK)
    ax.set_yticks(range(len(configs)), [CONFIG_LABELS.get(c, c) for c in reversed(configs)])
    ax.set_xlabel("Tiempo medio del comando de entrenamiento, run_wall_s (s)")
    ax.set_title(f"En qué se va el tiempo del comando{tag}", loc="left")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, ax.get_xlim()[1] * 1.1)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, fontsize=8.5)
    ax.set_title(ax.get_title(loc="left"), loc="left", pad=26)
    fig.tight_layout()
    name = "desglose_total.png"
    fig.savefig(out / name, dpi=160)
    plt.close(fig)
    figures[name] = "Desglose del tiempo total por fase"

    # 3. Speedup por fase.
    if res["speedup"]:
        acc = list(res["speedup"])
        phases = [(k, n) for k, n, _ in TRAIN_PHASES] + EPOCH_METRICS
        phases = [(k, n) for k, n in phases if any(res["speedup"][c].get(k, {}).get("speedup") for c in acc)]
        fig, ax = plt.subplots(figsize=(10, 1.5 + 0.3 * len(phases) * len(acc)))
        height = 0.8 / len(acc)
        for j, c in enumerate(acc):
            for i, (k, _) in enumerate(phases):
                e = res["speedup"][c][k]
                y = len(phases) - 1 - i + (len(acc) - 1 - j - (len(acc) - 1) / 2) * height
                ax.barh(y, e["speedup"], height=height * 0.7, color=color_of(c, j),
                        label=CONFIG_LABELS.get(c, c) if i == 0 else None)
                if e["ci_low"] is not None:
                    ax.plot([e["ci_low"], e["ci_high"]], [y, y], color=INK, lw=1.1)
                ax.text(max(e["speedup"], e["ci_high"] or 0), y, f"  {e['speedup']:.2f}×", va="center", fontsize=8, color=INK)
        ax.axvline(1, color=INK_2, lw=1, ls="--")
        ax.text(1, len(phases) - 0.35, " 1× = igual que CPU", color=INK_2, fontsize=8, va="bottom")
        ax.set_yticks(range(len(phases)), [n for _, n in reversed(phases)], fontsize=8.5)
        ax.set_xlabel("Speedup = tiempo medio CPU / tiempo medio GPU (línea: IC bootstrap 95 %)")
        ax.set_title(f"Speedup por fase frente a la referencia CPU{tag}", loc="left")
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, ax.get_xlim()[1] * 1.12)
        ax.legend(loc="lower right", fontsize=9)
        fig.tight_layout()
        name = "speedup_por_fase.png"
        fig.savefig(out / name, dpi=160)
        plt.close(fig)
        figures[name] = "Speedup por fase"

    # 4. Tiempo por época.
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for i, c in enumerate(configs):
        epochs = train[train["config"] == c]["epoch_times"].tolist()
        length = min(len(e) for e in epochs) if epochs else 0
        if length == 0:
            continue
        matrix = np.array([e[:length] for e in epochs])
        x = np.arange(1, length + 1)
        mean = matrix.mean(0)
        ax.fill_between(x, matrix.min(0), matrix.max(0), color=color_of(c, i), alpha=0.15, lw=0)
        ax.plot(x, mean, color=color_of(c, i), lw=2, marker="o", ms=4, label=CONFIG_LABELS.get(c, c))
        ax.text(x[-1] + 0.3, mean[-1], f"{CONFIG_LABELS.get(c, c)}  {mean[-1]:.2f} s", va="center", fontsize=8.5, color=INK)
    ax.set_yscale("log")
    from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter
    lo, hi = ax.get_ylim()
    ticks = [t for t in (0.1, 0.2, 0.5, 1, 1.5, 2, 3, 5, 8, 10, 15, 20, 30, 50, 100, 200) if lo <= t <= hi]
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("Época")
    ax.set_ylabel("Tiempo de la época (s, escala log)")
    ax.set_title(f"Tiempo por época (media y rango de las repeticiones){tag}", loc="left")
    ax.set_xlim(0.5, ax.get_xlim()[1] + 5)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=len(configs), fontsize=8.5)
    ax.set_title(ax.get_title(loc="left"), loc="left", pad=24)
    fig.tight_layout()
    name = "tiempo_por_epoca.png"
    fig.savefig(out / name, dpi=160)
    plt.close(fig)
    figures[name] = "Tiempo por época"

    # 5. Repeticiones en orden de ejecución (deriva térmica / variación).
    fig, axes = plt.subplots(1, len(configs), figsize=(3.4 * len(configs), 3.2), squeeze=False)
    for i, c in enumerate(configs):
        ax = axes[0][i]
        group = train[train["config"] == c].sort_values("rep")
        ax.plot(group["rep"], group["train_with_validation_s"], color=color_of(c, i), lw=2, marker="o", ms=6)
        ax.set_title(CONFIG_LABELS.get(c, c), fontsize=10, loc="left")
        ax.set_xlabel("Repetición")
        ax.set_xticks(group["rep"])
        if i == 0:
            ax.set_ylabel("fit (s)")
        st = res["stats"][c]["train_with_validation_s"]
        if st["cv_percent"] is not None:
            ax.text(0.02, 0.04, f"CV = {st['cv_percent']:.1f} %", transform=ax.transAxes, fontsize=8.5, color=INK_2)
    fig.suptitle(f"Variación entre repeticiones (cada panel con su propia escala){tag}", x=0.01, ha="left",
                 fontweight="bold", fontsize=11)
    fig.tight_layout()
    name = "repeticiones.png"
    fig.savefig(out / name, dpi=160)
    plt.close(fig)
    figures[name] = "Variación entre repeticiones"

    # 6. Recursos: CPU y GPU en paneles separados (unidades distintas, sin doble eje).
    fig, axes = plt.subplots(1, 2, figsize=(10, 1.5 + 0.6 * len(configs)), sharey=True)
    panels = [("process_cpu_mean_percent", "CPU del proceso en fit (% de un procesador lógico)"),
              ("gpu_util_mean_percent", "Utilización de la GPU en fit, épocas 2..N (%)")]
    for ax, (key, title) in zip(axes, panels):
        for i, c in enumerate(configs):
            st = res["stats"][c].get(key, {})
            y = len(configs) - 1 - i
            value = st.get("mean") if st else None
            if value is None:
                ax.text(0, y, "  sin datos", va="center", fontsize=8.5, color=INK_2)
                continue
            ax.barh(y, value, height=0.36, color=color_of(c, i))
            ax.text(value, y, f"  {value:.0f} %", va="center", fontsize=9, color=INK)
        ax.set_title(title, fontsize=9.5, loc="left")
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, max(ax.get_xlim()[1] * 1.18, 10))
    axes[0].set_yticks(range(len(configs)), [CONFIG_LABELS.get(c, c) for c in reversed(configs)])
    fig.suptitle(f"Quién trabaja durante el entrenamiento{tag}", x=0.01, ha="left", fontweight="bold", fontsize=11)
    fig.tight_layout()
    name = "recursos_cpu_gpu.png"
    fig.savefig(out / name, dpi=160)
    plt.close(fig)
    figures[name] = "Uso de CPU y GPU durante fit"

    # 7. Preparación multinúcleo.
    p = res.get("prepare")
    if p and len(p.get("workers", [])) >= 1:
        keys = [(k, n) for k, n in PREP_PHASES if k != "process_wall_s"]
        workers = p["workers"]
        fig, ax = plt.subplots(figsize=(9.5, 1.4 + 0.75 * len(keys)))
        height = 0.8 / len(workers)
        colors = [NEUTRAL, "#6250d6", "#eda100", "#e87ba4"]
        for j, w in enumerate(workers):
            for i, (k, _) in enumerate(keys):
                st = p["stats"][w][k]
                y = len(keys) - 1 - i + (len(workers) - 1 - j - (len(workers) - 1) / 2) * height
                ax.barh(y, st["mean"], height=height * 0.7, color=colors[j % len(colors)],
                        label=procs(w) if i == 0 else None)
                ax.text(st["mean"], y, f"  {st['mean']:.2f} s", va="center", fontsize=8, color=INK)
        ax.set_yticks(range(len(keys)), [n for _, n in reversed(keys)], fontsize=9)
        ax.set_xlabel("Tiempo medio (s)")
        last = workers[-1]
        if len(workers) > 1:
            s = p["speedup"][last]["images_s"]["speedup"]
            subtitle = f"\nLectura y resize {s:.2f}× más rápida con {last} procesos"
        else:
            subtitle = ""
        ax.set_title(f"Preparación de imágenes en CPU{tag}{subtitle}", loc="left", fontsize=11, pad=24)
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, ax.get_xlim()[1] * 1.15)
        ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=len(workers), fontsize=9)
        fig.tight_layout()
        name = "preparacion_multinucleo.png"
        fig.savefig(out / name, dpi=160)
        plt.close(fig)
        figures[name] = "Preparación con uno y varios procesos CPU"
    return figures


# ----------------------------------------------------------------------------- comparación entre entornos
def compare_environments(results: list[tuple[dict, dict]], out: Path) -> None:
    plt = setup_matplotlib()
    configs = [c for c in results[0][1]["configs"] if all(c in r["configs"] for _, r in results)]
    rows, table = [], []
    for exp, res in results:
        host = exp["manifest"].get("host", {})
        gpu = (host.get("gpu") or {}).get("name", "—")
        for c in configs:
            st = res["stats"][c]
            sp = res["speedup"].get(c, {}) if c != res["baseline"] else {}
            rows.append({"entorno": res["label"], "cpu": host.get("cpu_model"), "logical_cpus": host.get("logical_cpus"),
                         "gpu": gpu, "config": c, "n": st["train_with_validation_s"]["n"],
                         "fit_mean_s": st["train_with_validation_s"]["mean"],
                         "fit_std_s": st["train_with_validation_s"]["std"],
                         "later_epoch_mean_s": st["later_epochs_mean_s"]["mean"],
                         "run_wall_mean_s": st["run_wall_s"]["mean"],
                         "speedup_fit_vs_cpu": sp.get("train_with_validation_s", {}).get("speedup") if sp else 1.0,
                         "speedup_total_vs_cpu": sp.get("run_wall_s", {}).get("speedup") if sp else 1.0,
                         "test_accuracy": st["test_accuracy"]["mean"]})
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "comparacion_entornos.csv", index=False)
    labels = [r["label"] for _, r in results]
    for _, row in frame.iterrows():
        table.append([row["entorno"], CONFIG_LABELS.get(row["config"], row["config"]), str(row["n"]),
                      fmt(row["fit_mean_s"], 2) + (f" ± {row['fit_std_s']:.2f}" if row["n"] > 1 else ""),
                      fmt(row["later_epoch_mean_s"], 3), fmt(row["run_wall_mean_s"], 2),
                      fmt(row["speedup_fit_vs_cpu"], 2, "×"), fmt(row["speedup_total_vs_cpu"], 2, "×"),
                      fmt(100 * row["test_accuracy"], 2, " %")])
    hw = []
    for exp, res in results:
        host = exp["manifest"].get("host", {})
        gpu = host.get("gpu") or {}
        hw.append([res["label"], f"{host.get('cpu_model') or '—'} ({host.get('logical_cpus')} hilos)",
                   f"{gpu.get('name', '—')} ({fmt(gpu.get('memory_total_mib'), 0, ' MiB')})",
                   str((host.get("packages") or {}).get("tensorflow"))])
    cross = []
    for c in configs:
        if c.startswith("gpu") and len(results) > 1:
            a, b = results[0][1]["stats"][c]["train_with_validation_s"], results[1][1]["stats"][c]["train_with_validation_s"]
            cross.append([CONFIG_LABELS.get(c, c), fmt(a["mean"], 2, " s"), fmt(b["mean"], 2, " s"),
                          fmt(b["mean"] / a["mean"], 2, "×")])
    text = ["# Comparación entre entornos", "",
            "Cada entorno se compara primero consigo mismo (CPU y GPU en la misma máquina). Comparar una GPU con "
            "otra mide hardware distinto con el mismo código y los mismos datos; no se calcula un speedup CUDA "
            "mezclando la CPU de un equipo con la GPU de otro.", "",
            md_table(["Entorno", "CPU", "GPU", "TensorFlow"], hw), "",
            md_table(["Entorno", "Versión", "n", "fit (s)", "Época 2..N (s)", "Total del comando (s)",
                      "Speedup fit vs CPU propia", "Speedup total vs CPU propia", "Exactitud de prueba"], table), ""]
    if cross:
        text += ["Tiempo de fit entre las dos GPU (mismo código, mismos datos; compara hardware, no es un speedup CUDA):", "",
                 md_table(["Versión", f"fit {labels[0]}", f"fit {labels[1]}", f"{labels[1]} / {labels[0]}"], cross), ""]

    fig, axes = plt.subplots(1, 2, figsize=(11, 1.6 + 0.6 * len(configs) * len(results)))
    height = 0.8 / len(configs)
    for j, c in enumerate(configs):
        for i, label in enumerate(labels):
            row = frame[(frame["entorno"] == label) & (frame["config"] == c)].iloc[0]
            y = len(labels) - 1 - i + (len(configs) - 1 - j - (len(configs) - 1) / 2) * height
            axes[0].barh(y, row["fit_mean_s"], height=height * 0.85, color=color_of(c, j),
                         label=CONFIG_LABELS.get(c, c) if i == 0 else None)
            axes[0].text(row["fit_mean_s"], y, f"  {row['fit_mean_s']:.1f} s", va="center", fontsize=8, color=INK)
            axes[1].barh(y, row["speedup_fit_vs_cpu"], height=height * 0.85, color=color_of(c, j))
            axes[1].text(row["speedup_fit_vs_cpu"], y, f"  {row['speedup_fit_vs_cpu']:.2f}×", va="center", fontsize=8, color=INK)
    for ax in axes:
        ax.set_yticks(range(len(labels)), list(reversed(labels)))
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, ax.get_xlim()[1] * 1.2)
    axes[0].set_title("Entrenamiento (fit), tiempo medio (s)", loc="left", fontsize=10)
    axes[1].set_title("Speedup de fit frente a la CPU del mismo entorno", loc="left", fontsize=10)
    axes[1].axvline(1, color=INK_2, lw=1, ls="--")
    axes[0].legend(loc="lower right", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out / "comparacion_entornos.png", dpi=160)
    plt.close(fig)
    text += ["![Comparación entre entornos](comparacion_entornos.png)", ""]
    (out / "COMPARACION.md").write_text("\n".join(text), encoding="utf-8")


# ----------------------------------------------------------------------------- salida
def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def write_outputs(exp: dict, res: dict, out: Path, plots: bool) -> None:
    out.mkdir(parents=True, exist_ok=True)
    train = exp["train"].drop(columns=["epoch_times"], errors="ignore")
    train.to_csv(out / "corridas_entrenamiento.csv", index=False)
    if not exp["prep"].empty:
        exp["prep"].to_csv(out / "corridas_preparacion.csv", index=False)
    epochs = [{"entorno": r["entorno"], "config": r["config"], "rep": r["rep"], "epoch": i + 1, "epoch_wall_s": t}
              for _, r in exp["train"].iterrows() for i, t in enumerate(r["epoch_times"])]
    pd.DataFrame(epochs).to_csv(out / "tiempos_por_epoca.csv", index=False)
    summary_rows = []
    for c in res["configs"]:
        for key, st in res["stats"][c].items():
            summary_rows.append({"config": c, "metric": key, **st})
    pd.DataFrame(summary_rows).to_csv(out / "resumen_por_configuracion.csv", index=False)
    sp_rows = [{"config": c, "phase": k, **v} for c, phases in res["speedup"].items() for k, v in phases.items()]
    if res.get("prepare", {}).get("speedup"):
        sp_rows += [{"config": f"prepare_workers{w}", "phase": k, **v}
                    for w, phases in res["prepare"]["speedup"].items() for k, v in phases.items()]
    pd.DataFrame(sp_rows).to_csv(out / "speedup.csv", index=False)
    figures = make_figures(exp, res, out) if plots else {}
    (out / "resumen.json").write_text(json.dumps(jsonable(res), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out / "RESULTADOS.md").write_text(report_markdown(exp, res, figures), encoding="utf-8")


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experiments", type=Path, nargs="+", help="carpetas con manifest.json de asl_cpu.benchmark")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--baseline", default="cpu", help="configuración de referencia para el speedup")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--allow-mismatch", action="store_true",
                        help="escribir el análisis aunque las cargas de trabajo no coincidan")
    args = parser.parse_args(argv)
    results = []
    for path in args.experiments:
        exp = load_experiment(path)
        res = analyze(exp, args.baseline)
        if len(args.experiments) == 1:
            out = args.output_dir or path / "analisis"
        else:
            out = (args.output_dir or path.parent / "comparacion") / res["label"]
        write_outputs(exp, res, out, plots=not args.no_plots)
        results.append((exp, res))
        print(f"{res['label']}: {out / 'RESULTADOS.md'}")
        for c in res["configs"]:
            st = res["stats"][c]["train_with_validation_s"]
            sp = res["speedup"].get(c, {}).get("train_with_validation_s", {})
            print(f"  {c:12} n={st['n']}  fit {st['mean']:9.2f} ± {st['std']:.2f} s" +
                  (f"  speedup {fmt_speedup(sp)}" if sp else ""))
        if not res["workload"]["ok"]:
            print(f"  ATENCIÓN: la carga de trabajo no coincide: {res['workload']}")
            if not args.allow_mismatch:
                raise SystemExit(2)
    if len(results) > 1:
        out = args.output_dir or args.experiments[0].parent / "comparacion"
        compare_environments(results, out)
        print(f"Comparación: {out / 'COMPARACION.md'}")


if __name__ == "__main__":
    main()
