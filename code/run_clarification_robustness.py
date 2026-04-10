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

from interaction_utils import ask_greedy, filter_candidates, load_ranked_rows, slotify, unique_golds


ARTIFACT_DIR = ROOT / "artifacts_runtime" / "clarification_robustness"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


def canonical(obj: dict) -> str:
    return json.dumps(obj, sort_keys=True)


def simulate(predictions: list[dict], target: dict, noise: float, robust: bool, rng: random.Random) -> float:
    preds = list(predictions)
    target_slots = slotify(target)
    for _ in range(2):
        q = ask_greedy(preds, target)
        if q is None:
            break
        if rng.random() < noise:
            values = []
            for pred in preds:
                val = slotify(pred).get(q["slot"])
                if val is not None and val != target_slots.get(q["slot"]) and val not in values:
                    values.append(val)
            choices = [{"slot": q["slot"], "value": val} for val in values]
            if choices:
                q = rng.choice(choices)
                preds = filter_candidates(preds, q)
                if robust:
                    # confirmation-style recovery on the surviving pool
                    q2 = ask_greedy(preds, target)
                    if q2 is not None:
                        preds = filter_candidates(preds, q2)
                continue
        preds = filter_candidates(preds, q)
    return float(canonical(preds[0]) == canonical(target))


def build_summary() -> dict:
    summary = {}
    for system in ["heuristic", "reranked"]:
        rows = load_ranked_rows(system, "test")
        summary[system] = {}
        for noise in [0.0, 0.1, 0.2, 0.3, 0.4]:
            base = []
            robust = []
            for seed in range(10):
                rng = random.Random(seed)
                for row in rows:
                    for target in unique_golds(row):
                        base.append(simulate(row["predictions"], target, noise, False, rng))
                        robust.append(simulate(row["predictions"], target, noise, True, random.Random(seed + 1000)))
            summary[system][str(noise)] = {
                "base": sum(base) / len(base),
                "robust": sum(robust) / len(robust),
            }
    return summary


def plot(summary: dict) -> None:
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    xs = [0.0, 0.1, 0.2, 0.3, 0.4]
    for ax, system in zip(axes, ["heuristic", "reranked"]):
        ax.plot(xs, [summary[system][str(x)]["base"] for x in xs], marker="o", label="No recovery", color="#d62728")
        ax.plot(xs, [summary[system][str(x)]["robust"] for x in xs], marker="s", label="Recovering", color="#2ca02c")
        ax.set_title(system)
        ax.set_xlabel("Answer noise")
        ax.set_ylabel("Intent@1")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(ARTIFACT_DIR / "figure1_noise_curves.png", dpi=220)
    plt.close(fig)


def main() -> None:
    summary = build_summary()
    (ARTIFACT_DIR / "clarification_robustness_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plot(summary)
    print("Generated clarification-robustness artifacts in", ARTIFACT_DIR)


if __name__ == "__main__":
    main()
