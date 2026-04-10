from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "code"))

from interaction_utils import apply_edit, load_ranked_rows, slot_differences, slotify, unique_golds


ARTIFACT_DIR = ROOT / "artifacts_runtime" / "repair_robustness"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


def canonical(obj: dict) -> str:
    return json.dumps(obj, sort_keys=True)


def simulate(start: dict, target: dict, noise: float, recover: bool, rng: random.Random) -> float:
    cur = json.loads(json.dumps(start))
    diffs = slot_differences(start, target)[:2]
    target_slots = slotify(target)
    for slot in diffs:
        if rng.random() < noise:
            if recover:
                cur = apply_edit(cur, slot, target_slots.get(slot))
            continue
        cur = apply_edit(cur, slot, target_slots.get(slot))
    return float(canonical(cur) == canonical(target))


def build_summary() -> dict:
    summary = {}
    for system in ["heuristic", "reranked"]:
        rows = load_ranked_rows(system, "test")
        summary[system] = {}
        for noise in [0.0, 0.1, 0.2, 0.3, 0.4]:
            base = []
            recover = []
            for seed in range(10):
                rng = random.Random(seed)
                rr = random.Random(seed + 999)
                for row in rows:
                    start = row["predictions"][0]
                    for target in unique_golds(row):
                        base.append(simulate(start, target, noise, False, rng))
                        recover.append(simulate(start, target, noise, True, rr))
            summary[system][str(noise)] = {"base": sum(base) / len(base), "recover": sum(recover) / len(recover)}
    return summary


def plot(summary: dict) -> None:
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    xs = [0.0, 0.1, 0.2, 0.3, 0.4]
    for ax, system in zip(axes, ["heuristic", "reranked"]):
        ax.plot(xs, [summary[system][str(x)]["base"] for x in xs], marker="o", label="No recovery", color="#d62728")
        ax.plot(xs, [summary[system][str(x)]["recover"] for x in xs], marker="s", label="Verification recovery", color="#2ca02c")
        ax.set_title(system)
        ax.set_xlabel("Edit noise")
        ax.set_ylabel("Intent@1")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(ARTIFACT_DIR / "figure1_noise_curves.png", dpi=220)
    plt.close(fig)


def main() -> None:
    summary = build_summary()
    (ARTIFACT_DIR / "repair_robustness_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plot(summary)
    print("Generated repair-robustness artifacts in", ARTIFACT_DIR)


if __name__ == "__main__":
    main()
