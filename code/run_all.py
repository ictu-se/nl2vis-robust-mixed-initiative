from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"

SCRIPTS = [
    "run_escalation.py",
    "run_clarification_robustness.py",
    "run_repair_robustness.py",
    "run_budget_sweep.py",
    "run_extended_analysis.py",
]


def main() -> None:
    for script in SCRIPTS:
        print(f"\n[run_all] Running {script}")
        subprocess.run([sys.executable, str(CODE / script)], check=True, cwd=ROOT)
    print("\n[run_all] Finished. Artifacts are in artifacts_runtime/.")


if __name__ == "__main__":
    main()
