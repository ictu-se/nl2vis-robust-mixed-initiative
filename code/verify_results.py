from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "artifacts_runtime"
REFERENCE = ROOT / "reference_results"

CHECKS = {
    "paper12_escalation/paper12_escalation_summary.json": "paper12_escalation_summary.json",
    "paper14_noisy_clarify/paper14_noisy_clarify_summary.json": "paper14_noisy_clarify_summary.json",
    "paper15_noisy_repair/paper15_noisy_repair_summary.json": "paper15_noisy_repair_summary.json",
    "paper16_budgeted/paper16_budgeted_summary.json": "paper16_budgeted_summary.json",
    "paper16_q1_extensions/paper16_q1_regime_summary.json": "paper16_q1_regime_summary.json",
    "paper16_q1_extensions/paper16_q1_significance_summary.json": "paper16_q1_significance_summary.json",
}


def main() -> None:
    mismatches: list[str] = []
    for runtime_rel, ref_name in CHECKS.items():
        runtime_path = RUNTIME / runtime_rel
        ref_path = REFERENCE / ref_name
        if not runtime_path.exists():
            mismatches.append(f"Missing runtime file: {runtime_rel}")
            continue
        runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
        reference = json.loads(ref_path.read_text(encoding="utf-8"))
        if runtime != reference:
            mismatches.append(f"Content mismatch: {runtime_rel}")
        else:
            print(f"[ok] {runtime_rel}")

    if mismatches:
        print("\nVerification failed:")
        for row in mismatches:
            print(f"- {row}")
        raise SystemExit(1)

    print("\nAll generated summaries match the reference results.")


if __name__ == "__main__":
    main()
