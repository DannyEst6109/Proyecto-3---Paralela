"""Pruebas de la parte 3: plan de corridas, estadística, speedup y validación de la carga."""
import json
import tempfile
import unittest
from pathlib import Path

from asl_cpu import analyze_benchmark as ab
from asl_cpu import benchmark as bm

SETTINGS = {"configs": ["cpu", "gpu_keras", "gpu_numba"], "reps": 5, "reps_override": {}, "warmup": "smoke",
            "smoke": False, "prepare_workers": [1, 4], "prepare_reps": 5, "prepare_warmup": 1}


class PlanTests(unittest.TestCase):
    def test_counts_roles_and_unique_names(self):
        plan = bm.build_plan(SETTINGS)
        names = [s["name"] for s in plan]
        self.assertEqual(len(names), len(set(names)))
        measured = [s for s in plan if s["role"] == "measured" and s["kind"] == "train"]
        self.assertEqual(len(measured), 15)
        self.assertEqual(len([s for s in plan if s["role"] == "warmup" and s["kind"] == "train"]), 3)
        self.assertTrue(all(s["smoke"] for s in plan if s["role"] == "warmup" and s["kind"] == "train"))
        self.assertFalse(any(s["smoke"] for s in measured))
        self.assertEqual(len([s for s in plan if s["kind"] == "prepare" and s["role"] == "measured"]), 10)
        # La preparación va antes que todo entrenamiento.
        kinds = [s["kind"] for s in plan]
        self.assertEqual(kinds, sorted(kinds))

    def test_rotation_changes_first_configuration(self):
        plan = [s for s in bm.build_plan(SETTINGS) if s["kind"] == "train" and s["role"] == "measured"]
        firsts = [plan[i * 3]["config"] for i in range(5)]
        self.assertEqual(firsts, ["cpu", "gpu_keras", "gpu_numba", "cpu", "gpu_keras"])
        for config in SETTINGS["configs"]:
            reps = [s["rep"] for s in plan if s["config"] == config]
            self.assertEqual(reps, [1, 2, 3, 4, 5])

    def test_overrides_and_no_warmup(self):
        settings = dict(SETTINGS, reps_override={"cpu": 3}, warmup="none", prepare_workers=[], prepare_reps=0)
        plan = bm.build_plan(settings)
        self.assertEqual(sum(s["config"] == "cpu" for s in plan), 3)
        self.assertEqual(sum(s["config"] == "gpu_keras" for s in plan), 5)
        self.assertFalse(any(s["role"] == "warmup" for s in plan))

    def test_train_command_keeps_common_parameters(self):
        settings = dict(SETTINGS, threads=8, data_dir="d", config_path="c.json", raw_dir="r")
        step = {"kind": "train", "config": "gpu_numba", "smoke": False}
        command = bm.step_command(step, settings, Path("out"))
        self.assertIn("--optimizer", command)
        self.assertEqual(command[command.index("--threads") + 1], "8")
        self.assertNotIn("--smoke", command)
        step = {"kind": "train", "config": "cpu_1thread", "smoke": True}
        command = bm.step_command(step, settings, Path("out"))
        self.assertEqual(command[command.index("--threads") + 1], "1")
        self.assertIn("--smoke", command)


class StatisticsTests(unittest.TestCase):
    def test_describe(self):
        st = ab.describe([10.0, 12.0, 14.0])
        self.assertAlmostEqual(st["mean"], 12.0)
        self.assertAlmostEqual(st["std"], 2.0)
        self.assertAlmostEqual(st["ci95"], 4.303 * 2.0 / 3 ** 0.5, places=6)
        self.assertEqual(ab.describe([])["n"], 0)

    def test_speedup_is_ratio_of_means_with_interval(self):
        result = ab.speedup([100.0, 110.0, 90.0], [20.0, 22.0, 18.0])
        self.assertAlmostEqual(result["speedup"], 5.0)
        self.assertLessEqual(result["ci_low"], 5.0)
        self.assertGreaterEqual(result["ci_high"], 5.0)
        self.assertIsNone(ab.speedup([10.0], [2.0])["ci_low"])


def write_run(base: Path, name: str, device: str, fit: float, init_hash: str = "abc", epochs: int = 3):
    run = base / name
    run.mkdir(parents=True)
    times = {"tensorflow_setup_s": 2.0, "load_and_verify_s": 1.0, "normalize_cpu_s": 0.2,
             "build_model_and_input_s": 1.0, "train_with_validation_s": fit, "evaluate_and_predictions_s": fit / 50,
             "save_final_weights_s": 0.05}
    times["run_wall_s"] = sum(times.values()) + 0.5
    metrics = {"status": "complete", "device": device, "run_kind": f"baseline_{device.lower()}",
               "tensor_float_32": False, "cpu_threads": 8, "times": times,
               "weight_devices": [f"/job:localhost/replica:0/task:0/device:{device}:0"],
               "validation": {"accuracy": 0.75, "loss": 1.3}, "test": {"accuracy": 0.81, "loss": 0.9},
               "config_sha256": "c", "partition_sha256": "p", "used_array_sha256": {"X": "x"},
               "initial_weights_sha256": init_hash, "model_parameters": 1145693, "epochs_completed": epochs,
               "sample_counts": {"train": 12180, "val": 2610, "test": 2610}}
    if device == "GPU":
        metrics["transfers"] = {"fit_h2d_bytes": 14.54e9, "evaluate_predict_h2d_bytes": 0.38e9}
    (run / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    progress = [{"epoch": i + 1, "epoch_wall_s": fit / epochs * (2 if i == 0 else 1)} for i in range(epochs)]
    (run / "progress.json").write_text(json.dumps({"epochs": progress}), encoding="utf-8")


def write_experiment(base: Path, gpu_init_hash: str = "abc") -> Path:
    runs, order = [], 0
    for rep in (1, 2, 3):
        for config, device, fit in (("cpu", "CPU", 270.0 + rep), ("gpu_keras", "GPU", 45.0 + rep / 10)):
            order += 1
            name = f"train/{config}/rep{rep:02d}"
            write_run(base, name, device, fit, gpu_init_hash if device == "GPU" else "abc")
            runs.append({"name": name, "kind": "train", "role": "measured", "config": config, "rep": rep,
                         "order": order, "run_dir": name, "status": "complete", "process_wall_s": fit + 10})
    runs.append({"name": "train/cpu/warmup01", "kind": "train", "role": "warmup", "config": "cpu", "rep": 1,
                 "order": 0, "run_dir": "train/cpu/warmup01", "status": "complete"})
    manifest = {"schema": bm.SCHEMA, "label": "prueba", "runs": runs, "host": {},
                "settings": {"configs": ["cpu", "gpu_keras"], "smoke": False, "warmup": "smoke",
                             "order_policy": "rotado", "cooldown_s": 20, "threads": 8}}
    (base / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return base


class AnalysisTests(unittest.TestCase):
    def test_speedup_amdahl_and_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            exp = ab.load_experiment(write_experiment(Path(tmp)))
            self.assertEqual(len(exp["train"]), 6)          # el calentamiento no se analiza
            res = ab.analyze(exp, "cpu")
            self.assertTrue(res["workload"]["ok"])
            fit = res["speedup"]["gpu_keras"]["train_with_validation_s"]["speedup"]
            self.assertAlmostEqual(fit, 272.0 / 45.2, places=6)
            amdahl = res["amdahl"]["gpu_keras"]
            self.assertLess(amdahl["amdahl_predicted_total"], amdahl["amdahl_limit_infinite_speedup"])
            self.assertAlmostEqual(res["stats"]["gpu_keras"]["fit_h2d_gb"]["mean"], 14.54)
            ab.write_outputs(exp, res, Path(tmp) / "analisis", plots=False)
            report = (Path(tmp) / "analisis" / "RESULTADOS.md").read_text(encoding="utf-8")
            self.assertIn("Speedup por fase", report)
            self.assertTrue((Path(tmp) / "analisis" / "speedup.csv").is_file())

    def test_detects_different_initial_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            exp = ab.load_experiment(write_experiment(Path(tmp), gpu_init_hash="otro"))
            self.assertFalse(ab.analyze(exp, "cpu")["workload"]["ok"])


if __name__ == "__main__":
    unittest.main()
