"""Monitor real de CPU por procesador lógico y memoria del proceso y sus hijos."""
from __future__ import annotations

import csv
import threading
import time
from pathlib import Path

import psutil


class ResourceMonitor:
    def __init__(self, path: Path, interval: float = 1.0):
        self.path = path
        self.interval = interval
        self.phase = "startup"
        self._stop = threading.Event()
        self._thread = None
        self._process = psutil.Process()

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *args):
        self._stop.set()
        self._thread.join()

    def _sample(self):
        start = time.perf_counter()
        logical = psutil.cpu_count()
        fields = ["elapsed_s", "phase", "process_tree_cpu_percent", "process_tree_rss_mb", "system_ram_percent"]
        fields += [f"cpu_{i}_percent" for i in range(logical)]
        previous = {}
        previous_time = start
        psutil.cpu_percent(percpu=True)
        with self.path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            # La primera muestra se usa para establecer el contador de CPU.
            while True:
                now = time.perf_counter()
                rss = 0
                delta_cpu = 0.0
                current = {}
                try:
                    processes = [self._process] + self._process.children(recursive=True)
                except psutil.Error:
                    processes = []
                for process in processes:
                    try:
                        cpu = process.cpu_times()
                        total = cpu.user + cpu.system
                        key = (process.pid, process.create_time())
                        if key in previous:
                            delta_cpu += max(0, total - previous[key])
                        current[key] = total
                        rss += process.memory_info().rss
                    except psutil.Error:
                        continue
                cores = psutil.cpu_percent(percpu=True)
                if previous:
                    row = {
                        "elapsed_s": round(now - start, 4), "phase": self.phase,
                        # Puede superar 100%: 100% equivale a un procesador lógico.
                        "process_tree_cpu_percent": round(100 * delta_cpu / (now - previous_time), 3),
                        "process_tree_rss_mb": round(rss / 1024**2, 3),
                        "system_ram_percent": psutil.virtual_memory().percent,
                    }
                    row.update({f"cpu_{i}_percent": value for i, value in enumerate(cores)})
                    writer.writerow(row)
                    stream.flush()
                previous = current
                previous_time = now
                if self._stop.wait(self.interval):
                    break


GPU_QUERY = ("timestamp,utilization.gpu,utilization.memory,memory.used,memory.total,"
             "power.draw,temperature.gpu,clocks.sm")
GPU_FIELDS = ["timestamp", "gpu_util_percent", "gpu_mem_util_percent", "gpu_mem_used_mib",
              "gpu_mem_total_mib", "power_w", "temperature_c", "sm_clock_mhz"]


def nvidia_smi_snapshot() -> str:
    """Salida textual de nvidia-smi y de los procesos con contexto CUDA."""
    import shutil
    import subprocess
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return "nvidia-smi no disponible\n"
    parts = []
    for command in ([exe], [exe, "--query-compute-apps=pid,process_name,used_memory", "--format=csv"]):
        try:
            parts.append(subprocess.run(command, capture_output=True, text=True, timeout=20).stdout)
        except (OSError, subprocess.SubprocessError) as error:
            parts.append(f"{command}: {error}\n")
    return "\n".join(parts)


class GpuMonitor:
    """Muestrea la GPU NVIDIA con `nvidia-smi --loop-ms` en un proceso aparte.

    Se usa igual en CPU y GPU: en la referencia CPU sirve como evidencia de que la GPU
    permanece inactiva. Si no hay nvidia-smi, no hace nada (available=False).
    """

    def __init__(self, path: Path, interval_ms: int = 500):
        self.path = path
        self.interval_ms = interval_ms
        self.available = False
        self._process = None
        self._stream = None

    def __enter__(self):
        import shutil
        import subprocess
        exe = shutil.which("nvidia-smi")
        if exe is None:
            return self
        self._stream = self.path.open("w", encoding="utf-8", newline="")
        self._stream.write(",".join(GPU_FIELDS) + "\n")
        self._stream.flush()
        self._process = subprocess.Popen(
            [exe, f"--query-gpu={GPU_QUERY}", "--format=csv,noheader,nounits", f"--loop-ms={self.interval_ms}"],
            stdout=self._stream, stderr=subprocess.DEVNULL)
        self.available = True
        return self

    def __exit__(self, *args):
        if self._process is not None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except Exception:
                self._process.kill()
        if self._stream is not None:
            self._stream.close()
