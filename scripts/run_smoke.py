#!/usr/bin/env python3
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
output = ROOT / "examples" / "generated" / "synthetic_panel.csv"
subprocess.run([sys.executable, str(ROOT / "examples" / "make_synthetic_panel.py"), "--output", str(output)], check=True)
subprocess.run([sys.executable, str(ROOT / "scripts" / "run_model.py"), "check", "--config", str(ROOT / "examples" / "smoke.yaml")], check=True)
subprocess.run([sys.executable, str(ROOT / "scripts" / "run_model.py"), "train", "--config", str(ROOT / "examples" / "smoke.yaml")], check=True)
