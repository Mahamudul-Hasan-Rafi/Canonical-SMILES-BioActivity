#!/usr/bin/env bash
# Waits for the legacy ablation run (run_experiments.py) to exit, migrates its results,
# then runs the new experiment queue (scaffold -> unbalanced -> untuned -> backbones).
cd "$(dirname "$0")"
PY=E:/ML/BioActivity/.venv/Scripts/python.exe
running() {
  powershell -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'run_experiments.py' }).Count"
}
echo "$(date '+%F %T') waiting for run_experiments.py to finish"
while [ "$(running | tr -d '\r')" != "0" ]; do sleep 120; done
echo "$(date '+%F %T') ablation run finished; migrating"
PYTHONIOENCODING=utf-8 $PY migrate_legacy.py
PYTHONIOENCODING=utf-8 $PY runner.py --exp repro ablation --status | head -3
echo "$(date '+%F %T') starting queue"
PYTHONIOENCODING=utf-8 $PY runner.py --queue --workers 3
echo "$(date '+%F %T') queue finished"
