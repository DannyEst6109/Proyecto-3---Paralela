"""Parte 3: repite las mediciones de preparación y entrenamiento y registra cada corrida.

Uso típico, en el equipo NVIDIA y desde la raíz del repositorio:

    python -m asl_cpu.benchmark --output-dir results/parte3/rtx3060 --label rtx3060 \
        --reps 5 --threads 8 --prepare-workers 1,4

Cada corrida es un proceso nuevo de `asl_cpu.prepare` o `asl_cpu.train` con su propia carpeta,
así cada medición arranca con el intérprete, TensorFlow y el contexto CUDA limpios. El
orquestador no cambia ningún parámetro del modelo ni de los datos: solo decide el orden, repite,
mide el proceso completo desde fuera y guarda `manifest.json` con todo lo ejecutado.

Política de medición (la misma para CPU y GPU):
- Calentamiento: antes de medir se ejecuta una corrida descartada por configuración
  (`--warmup smoke`, por defecto: 1 época con 16/4/4 imágenes por clase). Calienta la caché de
  archivos del NPZ, los módulos de Python y la caché JIT del driver. No se excluye nada dentro de
  las corridas medidas: la primera época sigue incluyendo el trazado de TensorFlow y la selección
  de algoritmos de cuDNN, y se reporta por separado de las épocas 2..N.
- Orden intercalado y rotado: la repetición r ejecuta las configuraciones desplazadas r lugares,
  para que ninguna quede siempre primera (GPU fría) o última (GPU caliente).
- Enfriamiento fijo entre corridas (`--cooldown`) y, opcionalmente, espera a una temperatura de GPU.
- Preparación: `--prepare-warmup` corridas descartadas por modo (calientan la caché de los JPEG) y
  `--prepare-reps` medidas. Cada repetición debe reproducir exactamente los arreglos usados para
  entrenar; después se borran los archivos pesados y se conservan metadatos y monitor.

Es reanudable: al repetir el mismo comando se omiten las corridas que ya terminaron.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from time import perf_counter

from .common import ROOT, environment, fingerprint, load_config, write_json

SCHEMA = "parte3-benchmark-v1"
# Mismas claves que compare_runs.SAME_WORKLOAD (se repiten aquí para no importar pandas/numpy).
SAME_WORKLOAD = ("config_sha256", "partition_sha256", "used_array_sha256", "initial_weights_sha256",
                 "model_parameters", "epochs_completed", "sample_counts")

CONFIGS = {
    "cpu": {"train_args": ["--device", "cpu"], "device": "CPU",
            "label": "CPU (TensorFlow, oneDNN)"},
    "gpu_keras": {"train_args": ["--device", "gpu"], "device": "GPU",
                  "label": "GPU CUDA (cuDNN/cuBLAS, Adam de Keras)"},
    "gpu_numba": {"train_args": ["--device", "gpu", "--optimizer", "numba"], "device": "GPU",
                  "label": "GPU CUDA + kernel Adam propio (Numba)"},
    # Opcional y lento: aritmética de TensorFlow limitada a un hilo.
    "cpu_1thread": {"train_args": ["--device", "cpu"], "device": "CPU", "threads": 1,
                    "label": "CPU con 1 hilo (TensorFlow)"},
}
PREPARE_HEAVY_FILES = ("asl_64x64.npz", "inventario.parquet", "particion.parquet", "particion.csv")


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def gpu_state() -> dict | None:
    """Temperatura, reloj y potencia de la GPU 0 en este instante (None si no hay nvidia-smi)."""
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return None
    try:
        out = subprocess.run([exe, "--query-gpu=name,driver_version,temperature.gpu,clocks.sm,power.draw,"
                                   "utilization.gpu,memory.used,memory.total",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return None
    line = out.stdout.strip().splitlines()[0] if out.stdout.strip() else ""
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 8:
        return None
    keys = ("name", "driver_version", "temperature_c", "sm_clock_mhz", "power_w",
            "utilization_percent", "memory_used_mib", "memory_total_mib")
    state = dict(zip(keys, parts))
    for key in keys[2:]:
        try:
            state[key] = float(state[key])
        except ValueError:
            state[key] = None
    return state


def cpu_model() -> str | None:
    import platform
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or None


def git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def parse_overrides(text: str) -> dict:
    overrides = {}
    for item in filter(None, (s.strip() for s in text.split(","))):
        name, _, value = item.partition("=")
        if name not in CONFIGS or not value.isdigit() or int(value) < 1:
            raise argparse.ArgumentTypeError(f"Override inválido: {item!r} (use config=N, p. ej. cpu=3)")
        overrides[name] = int(value)
    return overrides


def build_plan(settings: dict) -> list[dict]:
    """Lista ordenada de corridas. Es determinista: el mismo comando produce el mismo plan."""
    plan = []
    workers = settings["prepare_workers"]
    for w in workers:
        for i in range(settings["prepare_warmup"]):
            plan.append({"name": f"prepare/workers{w}/warmup{i + 1:02d}", "kind": "prepare",
                         "role": "warmup", "workers": w, "rep": i + 1})
    for r in range(settings["prepare_reps"]):
        shift = r % len(workers) if workers else 0
        for w in workers[shift:] + workers[:shift]:
            plan.append({"name": f"prepare/workers{w}/rep{r + 1:02d}", "kind": "prepare",
                         "role": "measured", "workers": w, "rep": r + 1})

    configs = settings["configs"]
    if settings["warmup"] != "none":
        for c in configs:
            plan.append({"name": f"train/{c}/warmup01", "kind": "train", "role": "warmup", "config": c,
                         "rep": 1, "smoke": settings["warmup"] == "smoke" or settings["smoke"]})
    reps = {c: settings["reps_override"].get(c, settings["reps"]) for c in configs}
    for r in range(max(reps.values(), default=0)):
        shift = r % len(configs)
        for c in configs[shift:] + configs[:shift]:
            if r < reps[c]:
                plan.append({"name": f"train/{c}/rep{r + 1:02d}", "kind": "train", "role": "measured",
                             "config": c, "rep": r + 1, "smoke": settings["smoke"]})
    for order, step in enumerate(plan, start=1):
        step["order"] = order
    return plan


def step_command(step: dict, settings: dict, run_dir: Path) -> list[str]:
    py = [sys.executable, "-X", "utf8", "-m"]
    if step["kind"] == "prepare":
        return py + ["asl_cpu.prepare", "--workers", str(step["workers"]), "--raw-dir", settings["raw_dir"],
                     "--config", settings["config_path"], "--output-dir", str(run_dir)]
    config = CONFIGS[step["config"]]
    threads = config.get("threads", settings["threads"])
    command = py + ["asl_cpu.train", *config["train_args"], "--threads", str(threads),
                    "--data-dir", settings["data_dir"], "--config", settings["config_path"],
                    "--output-dir", str(run_dir)]
    if step["smoke"]:
        command.append("--smoke")
    return command


def step_finished(run_dir: Path, kind: str) -> bool:
    path = run_dir / ("metadata.json" if kind == "prepare" else "metrics.json")
    try:
        return read_json(path).get("status") == "complete"
    except (OSError, ValueError):
        return False


def run_streaming(command: list[str], log_path: Path, echo: bool) -> int:
    """Ejecuta el proceso, duplica su salida en consola y en un log, y devuelve el código de salida."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(command) + "\n")
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding="utf-8", errors="replace", bufsize=1)
        for line in process.stdout:
            log.write(line)
            if echo:
                sys.stdout.write(line)
                sys.stdout.flush()
        return process.wait()


def summarize(run_dir: Path, step: dict) -> dict:
    """Valores clave de una corrida terminada, solo para seguimiento rápido en el manifiesto."""
    if step["kind"] == "prepare":
        meta = read_json(run_dir / "metadata.json")
        return {"times": meta["times"], "array_sha256": meta["array_sha256"],
                "partition_sha256": meta["partition_sha256"], "workers": meta["workers"]}
    metrics = read_json(run_dir / "metrics.json")
    return {"run_kind": metrics.get("run_kind"), "device": metrics.get("device"),
            "train_with_validation_s": metrics["times"].get("train_with_validation_s"),
            "run_wall_s": metrics["times"].get("run_wall_s"),
            "test_accuracy": (metrics.get("test") or {}).get("accuracy"),
            **{key: metrics.get(key) for key in SAME_WORKLOAD}}


def check_prepared(run_dir: Path, reference: dict, keep_arrays: bool) -> dict:
    meta = read_json(run_dir / "metadata.json")
    result = {"same_arrays_as_training_data": meta["array_sha256"] == reference["array_sha256"],
              "same_partition_as_training_data": meta["partition_sha256"] == reference["partition_sha256"],
              "same_config": meta["config_sha256"] == reference["config_sha256"]}
    if not keep_arrays:
        removed = []
        for name in PREPARE_HEAVY_FILES:
            path = run_dir / name
            if path.is_file():
                path.unlink()
                removed.append(name)
        result["removed_files"] = removed
    return result


def check_workload(manifest: dict, output_dir: Path) -> dict:
    """Todas las corridas medidas deben compartir la misma carga de trabajo y su dispositivo."""
    rows, reference, reference_name = [], None, None
    for step in manifest["runs"]:
        if step["kind"] != "train" or step["role"] != "measured" or step.get("status") != "complete":
            continue
        metrics = read_json(output_dir / step["run_dir"] / "metrics.json")
        if reference is None:
            reference, reference_name = metrics, step["name"]
        same = {key: metrics.get(key) == reference.get(key) for key in SAME_WORKLOAD}
        expected = CONFIGS[step["config"]]["device"]
        device_ok = metrics.get("device") == expected and all(
            expected in d for d in metrics.get("weight_devices") or [""])
        rows.append({"run": step["name"], "device_ok": device_ok, "same_as_reference": same,
                     "ok": device_ok and all(same.values())})
    prep = [{"run": s["name"], **s.get("prepare_check", {})} for s in manifest["runs"]
            if s["kind"] == "prepare" and s["role"] == "measured" and s.get("status") == "complete"]
    prep_ok = all(p.get("same_arrays_as_training_data") and p.get("same_partition_as_training_data") for p in prep)
    return {"reference_run": reference_name, "train": rows, "prepare": prep,
            "all_ok": all(r["ok"] for r in rows) and prep_ok}


def wait_cooldown(seconds: float, max_temp: float | None, max_wait: float) -> dict:
    start = perf_counter()
    if seconds > 0:
        time.sleep(seconds)
    if max_temp is not None:
        while perf_counter() - start < seconds + max_wait:
            state = gpu_state()
            if state is None or state["temperature_c"] is None or state["temperature_c"] <= max_temp:
                break
            time.sleep(5)
    return {"cooldown_s": round(perf_counter() - start, 1)}


def maybe_pause(step: dict, pauses: set[str]) -> None:
    key = f"{step.get('config') or 'workers' + str(step['workers'])}:{step['rep']}"
    if step["role"] != "measured" or key not in pauses:
        return
    message = (f"\n>>> Antes de {step['name']}: prepare las capturas (nvidia-smi / Monitor del sistema / "
               f"Administrador de tareas) y tómelas MIENTRAS corre.")
    if sys.stdin.isatty():
        input(message + "\n>>> Presione Enter para iniciar... ")
    else:
        print(message + " (sin terminal interactiva: no se pausa)", flush=True)


def mirror_copy(source: Path, target: Path) -> None:
    if source.is_dir():
        shutil.copytree(source, target, dirs_exist_ok=True)
    elif source.is_file():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, required=True, help="carpeta del experimento (una por equipo)")
    parser.add_argument("--label", default=None, help="nombre corto del entorno, p. ej. rtx3060 o colab_t4")
    parser.add_argument("--configs", default="cpu,gpu_keras,gpu_numba",
                        help=f"configuraciones separadas por coma: {', '.join(CONFIGS)}")
    parser.add_argument("--reps", type=int, default=5, help="repeticiones medidas por configuración")
    parser.add_argument("--reps-override", type=parse_overrides, default={},
                        help="repeticiones distintas por configuración, p. ej. cpu=3 (equipos con CPU lenta)")
    parser.add_argument("--warmup", choices=("smoke", "full", "none"), default="smoke",
                        help="corrida descartada por configuración antes de medir (misma política en CPU y GPU)")
    parser.add_argument("--threads", type=int, default=8, help="hilos intra-op de TensorFlow, iguales en todas")
    parser.add_argument("--prepare-workers", default="1,4",
                        help="modos de preparación a medir (procesos CPU), p. ej. 1,4. Vacío para omitir")
    parser.add_argument("--prepare-reps", type=int, default=5)
    parser.add_argument("--prepare-warmup", type=int, default=1)
    parser.add_argument("--keep-prepared-arrays", action="store_true",
                        help="conservar NPZ y Parquet de cada repetición de preparación (pesan ~100 MB c/u)")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/processed/baseline")
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--config", type=Path, default=ROOT / "config/baseline.json")
    parser.add_argument("--cooldown", type=float, default=20.0, help="segundos de pausa entre corridas")
    parser.add_argument("--cooldown-gpu-temp", type=float, default=None,
                        help="además, esperar a que la GPU baje de esta temperatura (°C)")
    parser.add_argument("--cooldown-max-wait", type=float, default=180.0)
    parser.add_argument("--pause-before", default="",
                        help="pausas para capturas, p. ej. cpu:1,gpu_keras:1,gpu_numba:1,workers4:1")
    parser.add_argument("--mirror-dir", type=Path, default=None,
                        help="copia de seguridad tras cada corrida (p. ej. Google Drive en Colab)")
    parser.add_argument("--smoke", action="store_true",
                        help="SOLO prueba de integración: todas las corridas con --smoke; no sirve para rendimiento")
    parser.add_argument("--keep-going", action="store_true", help="continuar si una corrida falla")
    parser.add_argument("--quiet", action="store_true", help="no repetir en consola la salida de cada corrida")
    parser.add_argument("--dry-run", action="store_true", help="mostrar el plan sin ejecutar")
    args = parser.parse_args(argv)
    args.configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    unknown = [c for c in args.configs if c not in CONFIGS]
    if unknown or not args.configs:
        parser.error(f"Configuraciones desconocidas: {unknown}")
    if len(set(args.configs)) != len(args.configs):
        parser.error("Configuración repetida en --configs")
    try:
        args.prepare_workers = [int(w) for w in args.prepare_workers.split(",") if w.strip()]
    except ValueError:
        parser.error("--prepare-workers debe ser una lista de enteros")
    if any(w < 1 for w in args.prepare_workers) or len(set(args.prepare_workers)) != len(args.prepare_workers):
        parser.error("--prepare-workers: enteros positivos sin repetir")
    if args.reps < 1 or args.threads < 1 or args.prepare_reps < 0 or args.prepare_warmup < 0:
        parser.error("--reps y --threads deben ser positivos; las repeticiones de preparación, >= 0")
    if not args.prepare_workers:
        args.prepare_reps = args.prepare_warmup = 0
    args.pause_before = {p.strip() for p in args.pause_before.split(",") if p.strip()}
    return args


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    args = parse_args(argv)
    output_dir = args.output_dir.resolve()
    config = load_config(args.config)
    meta_path = args.data_dir / "metadata.json"
    if not meta_path.is_file():
        raise SystemExit(f"No existe {meta_path}. Prepare primero los datos: python -m asl_cpu.prepare --workers 4")
    training_data = read_json(meta_path)
    if training_data.get("status") != "complete" or training_data["config_sha256"] != fingerprint(config):
        raise SystemExit("Los datos preparados no corresponden a la configuración común.")
    if args.prepare_workers and not args.raw_dir.is_dir():
        raise SystemExit(f"No existe {args.raw_dir}; use --prepare-workers '' para omitir la preparación.")

    if args.mirror_dir and (args.mirror_dir / "manifest.json").is_file() and not (output_dir / "manifest.json").is_file():
        print(f"Restaurando corridas previas desde {args.mirror_dir}", flush=True)
        shutil.copytree(args.mirror_dir, output_dir, dirs_exist_ok=True)

    settings = {
        "configs": args.configs, "reps": args.reps, "reps_override": args.reps_override, "warmup": args.warmup,
        "threads": args.threads, "prepare_workers": args.prepare_workers, "prepare_reps": args.prepare_reps,
        "prepare_warmup": args.prepare_warmup, "cooldown_s": args.cooldown,
        "cooldown_gpu_temp_c": args.cooldown_gpu_temp, "smoke": args.smoke,
        "data_dir": str(args.data_dir.resolve()), "raw_dir": str(args.raw_dir.resolve()),
        "config_path": str(args.config.resolve()), "config_sha256": fingerprint(config),
        "training_array_sha256": training_data["array_sha256"],
        "order_policy": "preparación primero; después calentamiento por configuración y repeticiones "
                        "intercaladas con orden rotado (la repetición r desplaza la lista r lugares)",
    }
    plan = build_plan(settings)
    if args.dry_run:
        for step in plan:
            print(f"{step['order']:3d}  {step['role']:8}  {step['name']}")
        print(f"{len(plan)} corridas")
        return

    manifest_path = output_dir / "manifest.json"
    if manifest_path.is_file():
        manifest = read_json(manifest_path)
        if manifest.get("schema") != SCHEMA:
            raise SystemExit(f"{manifest_path} no es un manifiesto de la parte 3.")
        old = manifest["settings"]
        for key in ("threads", "smoke", "config_sha256", "training_array_sha256", "warmup"):
            if old.get(key) != settings[key]:
                raise SystemExit(f"El experimento existente usa {key}={old.get(key)!r}; use otra --output-dir.")
        manifest["settings"] = settings
    else:
        manifest = {"schema": SCHEMA, "created_at": now_iso(), "label": args.label or output_dir.name,
                    "settings": settings, "runs": []}
    manifest["label"] = args.label or manifest.get("label") or output_dir.name
    manifest["host"] = {**environment(), "cpu_model": cpu_model(), "gpu": gpu_state(), "git_commit": git_commit()}
    manifest["configs"] = {c: {**CONFIGS[c], "threads": CONFIGS[c].get("threads", args.threads)} for c in args.configs}
    done = {s["name"]: s for s in manifest["runs"]}
    output_dir.mkdir(parents=True, exist_ok=True)

    def save():
        manifest["updated_at"] = now_iso()
        manifest["runs"] = sorted(done.values(), key=lambda s: s["order"])
        write_json(manifest_path, manifest)
        if args.mirror_dir:
            mirror_copy(manifest_path, args.mirror_dir / "manifest.json")

    print(f"Experimento {manifest['label']}: {len(plan)} corridas en {output_dir}", flush=True)
    executed = 0
    for step in plan:
        run_dir = output_dir / step["name"]
        previous = done.get(step["name"])
        if previous and previous.get("status") == "complete" and step_finished(run_dir, step["kind"]):
            previous["order"] = step["order"]
            continue
        if run_dir.exists():
            # Corrida incompleta de una ejecución anterior: se conserva aparte, no se mezcla.
            run_dir.rename(run_dir.with_name(f"{run_dir.name}.incompleta-{datetime.now():%Y%m%d_%H%M%S}"))
        record = {**step, "run_dir": step["name"], "status": "running"}
        if executed:
            record.update(wait_cooldown(args.cooldown, args.cooldown_gpu_temp, args.cooldown_max_wait))
        maybe_pause(step, args.pause_before)
        command = step_command(step, settings, run_dir)
        record.update({"command": command, "started_at": now_iso(), "gpu_state_start": gpu_state()})
        done[step["name"]] = record
        save()
        print(f"\n=== [{step['order']}/{len(plan)}] {step['name']} ({step['role']}) ===", flush=True)
        run_dir.parent.mkdir(parents=True, exist_ok=True)
        log_path = output_dir / "logs" / (step["name"].replace("/", "__") + ".console.log")
        start = perf_counter()
        code = run_streaming(command, log_path, echo=not args.quiet)
        record["process_wall_s"] = perf_counter() - start
        record.update({"finished_at": now_iso(), "exit_code": code, "console_log": str(log_path.relative_to(output_dir)),
                       "gpu_state_end": gpu_state()})
        executed += 1
        if code == 0 and step_finished(run_dir, step["kind"]):
            record["status"] = "complete"
            if step["kind"] == "prepare":
                record["prepare_check"] = check_prepared(run_dir, training_data, args.keep_prepared_arrays)
            record["summary"] = summarize(run_dir, step)
            print(f"--- {step['name']}: proceso {record['process_wall_s']:.1f} s", flush=True)
        else:
            record["status"] = "failed"
            print(f"!!! {step['name']} falló (código {code}). Revise {log_path}", flush=True)
        save()
        if args.mirror_dir:
            mirror_copy(run_dir, args.mirror_dir / step["name"])
            mirror_copy(log_path, args.mirror_dir / record["console_log"])
        if record["status"] == "failed" and not args.keep_going:
            raise SystemExit(1)

    manifest["workload_check"] = check_workload(manifest, output_dir)
    save()
    check = manifest["workload_check"]
    print(f"\nCorridas ejecutadas ahora: {executed}; total en el manifiesto: {len(done)}", flush=True)
    print("Carga de trabajo idéntica en todas las corridas medidas: " + ("sí" if check["all_ok"] else "NO"), flush=True)
    print(f"Siguiente paso: python -m asl_cpu.analyze_benchmark {args.output_dir}", flush=True)
    if not check["all_ok"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
