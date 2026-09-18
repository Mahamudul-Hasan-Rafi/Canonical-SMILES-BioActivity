"""GPU thermal guard and monitor (nvidia-smi based; no admin rights, no settings changed)."""
import subprocess
import time


def gpu_query(fields="temperature.gpu,power.draw,utilization.gpu,memory.used,fan.speed"):
    try:
        return subprocess.run(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True, timeout=10).stdout.strip().splitlines()[0]
    except Exception:
        return None


def gpu_temp():
    q = gpu_query("temperature.gpu")
    try:
        return int(q)
    except (TypeError, ValueError):
        return None


class ThermalGuard:
    """Call .check() every training step. Every `every_s` seconds it reads the GPU
    temperature; at >= hot_c it sleeps until the GPU has cooled to cool_c."""

    def __init__(self, hot_c=78, cool_c=70, every_s=20, log=print):
        self.hot_c, self.cool_c, self.every_s, self.log = hot_c, cool_c, every_s, log
        self._last = 0.0
        self.paused_s = 0.0

    def check(self):
        now = time.time()
        if now - self._last < self.every_s:
            return
        self._last = now
        t = gpu_temp()
        if t is None or t < self.hot_c:
            return
        import torch
        self.log(f"[thermal] GPU {t}C >= {self.hot_c}C - pausing until <= {self.cool_c}C")
        t0 = time.time()
        while True:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            time.sleep(15)
            t = gpu_temp()
            if t is None or t <= self.cool_c:
                break
        self.paused_s += time.time() - t0
        self.log(f"[thermal] resumed at {t}C after {time.time() - t0:.0f}s")


def monitor(stop_event, path, every=60):
    """Appends 'time, temp, power, util, mem, fan' to path every `every` seconds."""
    while not stop_event.is_set():
        q = gpu_query()
        if q:
            with open(path, "a") as fh:
                fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}, {q}\n")
        stop_event.wait(every)
