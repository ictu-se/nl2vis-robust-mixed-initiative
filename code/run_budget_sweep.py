from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "code"))

ARTIFACT_DIR = ROOT / "artifacts_runtime" / "budget_sweep"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

LAMBDA_VALUES = [0.02, 0.04, 0.08, 0.12, 0.16, 0.20]
CONTROLLERS = [
    ("rank_only", "#1f77b4"),
    ("clarify_only", "#ff7f0e"),
    ("repair_only", "#2ca02c"),
    ("memory_clarify", "#9467bd"),
    ("full_controller", "#d62728"),
]


def load_controller_summary() -> dict:
    path = ROOT / "data" / "reference_inputs" / "controller_summary.json"
    return json.loads(path.read_text(encoding="utf-8"))


def build_summary(source: dict) -> dict:
    summary: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for system in ["heuristic", "reranked"]:
        summary[system] = {}
        for lam in LAMBDA_VALUES:
            summary[system][str(lam)] = {}
            for name, row in source[system]["controllers"].items():
                summary[system][str(lam)][name] = {
                    "intent": row["intent"],
                    "actions": row["actions"],
                    "utility": row["intent"] - lam * row["actions"],
                }
    return summary


def plot_summary(summary: dict) -> None:
    sns.set_theme(style="whitegrid")
    figures = [
        ("utility", "figure1_utility_lambda.png"),
        ("intent", "figure2_intent.png"),
        ("actions", "figure3_actions.png"),
    ]
    for metric, filename in figures:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=(metric != "actions"))
        for ax, system in zip(axes, ["heuristic", "reranked"]):
            for name, color in CONTROLLERS:
                ys = [summary[system][str(lam)][name][metric] for lam in LAMBDA_VALUES]
                ax.plot(LAMBDA_VALUES, ys, marker="o", label=name, color=color)
            ax.set_title(system)
            ax.set_xlabel("lambda")
            ax.set_ylabel(metric)
        axes[1].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(ARTIFACT_DIR / filename, dpi=220)
        plt.close(fig)


def main() -> None:
    source = load_controller_summary()
    summary = build_summary(source)
    plot_summary(summary)
    (ARTIFACT_DIR / "budget_sweep_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Generated budget-sweep artifacts in", ARTIFACT_DIR)


if __name__ == "__main__":
    main()
