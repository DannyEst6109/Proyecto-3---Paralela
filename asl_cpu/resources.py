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
