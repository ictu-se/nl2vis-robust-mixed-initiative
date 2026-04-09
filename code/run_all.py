from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"

SCRIPTS = [
    "run_paper12_escalation.py",
    "run_paper14_noisy_clarification.py",
    "run_paper15_noisy_repair.py",
    "run_paper16_budgeted.py",
    "run_paper16_q1_extensions.py",
]


def main() -> None:
    for script in SCRIPTS:
        print(f"\n[run_all] Running {script}")
        subprocess.run([sys.executable, str(CODE / script)], check=True, cwd=ROOT)
    print("\n[run_all] Finished. Artifacts are in artifacts_runtime/.")


if __name__ == "__main__":
    main()
