from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "code"))

from interaction_utils import apply_edit, ask_greedy, filter_candidates, load_ranked_rows, pool_uncertainty_features, slot_differences, slotify, unique_golds


ARTIFACT_DIR = ROOT / "artifacts_runtime" / "escalation"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


def canonical(obj: dict) -> str:
    return json.dumps(obj, sort_keys=True)


def clarify_outcome(predictions: list[dict], target: dict, budget: int = 2) -> float:
    preds = list(predictions)
    for _ in range(budget):
        q = ask_greedy(preds, target)
        if q is None:
            break
        preds = filter_candidates(preds, q)
    return float(canonical(preds[0]) == canonical(target))


def repair_outcome(start: dict, target: dict, budget: int = 2) -> float:
    cur = json.loads(json.dumps(start))
    target_slots = slotify(target)
    for slot in slot_differences(start, target)[:budget]:
        cur = apply_edit(cur, slot, target_slots.get(slot))
    return float(canonical(cur) == canonical(target))


def build_dataset(rows: list[dict]) -> list[dict]:
    data = []
    for row in rows:
        base = pool_uncertainty_features(row["predictions"])
        start = row["predictions"][0]
        for target in unique_golds(row):
            c = clarify_outcome(row["predictions"], target)
            r = repair_outcome(start, target)
            label = 1 if r > c else 0
            feats = dict(base)
            feats["start_slot_distance"] = float(len(slot_differences(start, target)))
            feats["target_in_pool"] = float(any(canonical(p) == canonical(target) for p in row["predictions"]))
            data.append({"features": feats, "label": label, "clarify": c, "repair": r})
    return data


def fit_policy(train: list[dict], test: list[dict]) -> tuple[list[int], list[str], list[float]]:
    names = sorted(train[0]["features"].keys())
    X_train = [[row["features"][n] for n in names] for row in train]
    y_train = [row["label"] for row in train]
    X_test = [[row["features"][n] for n in names] for row in test]
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train, y_train)
    preds = model.predict(X_test).tolist()
    return preds, names, model.coef_[0].tolist()


def build_summary() -> dict:
    summary = {}
    for system in ["heuristic", "reranked"]:
        train = build_dataset(load_ranked_rows(system, "dev"))
        test = build_dataset(load_ranked_rows(system, "test"))
        preds, names, weights = fit_policy(train, test)
        methods = {"always_clarify": [], "always_repair": [], "oracle": [], "escalate": []}
        for pred, row in zip(preds, test):
            methods["always_clarify"].append((row["clarify"], 2.0))
            methods["always_repair"].append((row["repair"], 2.0))
            oracle_best = max(row["clarify"], row["repair"])
            methods["oracle"].append((oracle_best, 2.0))
            if pred == 1:
                methods["escalate"].append((row["repair"], 2.0))
            else:
                methods["escalate"].append((row["clarify"], 2.0))
        summary[system] = {
            "results": {
                name: {
                    "intent": sum(v for v, _ in vals) / len(vals),
                    "actions": sum(a for _, a in vals) / len(vals),
                    "utility": sum(v - 0.08 * a for v, a in vals) / len(vals),
                }
                for name, vals in methods.items()
            },
            "weights": dict(sorted(zip(names, weights), key=lambda item: abs(item[1]), reverse=True)),
        }
    return summary


def plot_accuracy(summary: dict) -> None:
    sns.set_theme(style="whitegrid")
    order = ["always_clarify", "always_repair", "escalate", "oracle"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, system in zip(axes, ["heuristic", "reranked"]):
        vals = [summary[system]["results"][k]["intent"] for k in order]
        ax.bar(order, vals, color="#1f77b4")
        ax.set_title(system)
        ax.tick_params(axis="x", rotation=25)
        ax.set_ylabel("Intent@1")
    fig.tight_layout()
    fig.savefig(ARTIFACT_DIR / "figure1_accuracy.png", dpi=220)
    plt.close(fig)


def plot_utility(summary: dict) -> None:
    sns.set_theme(style="whitegrid")
    order = ["always_clarify", "always_repair", "escalate", "oracle"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, system in zip(axes, ["heuristic", "reranked"]):
        vals = [summary[system]["results"][k]["utility"] for k in order]
        ax.bar(order, vals, color="#2ca02c")
        ax.set_title(system)
        ax.tick_params(axis="x", rotation=25)
        ax.set_ylabel("Utility")
    fig.tight_layout()
    fig.savefig(ARTIFACT_DIR / "figure2_utility.png", dpi=220)
    plt.close(fig)


def plot_weights(summary: dict) -> None:
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, system in zip(axes, ["heuristic", "reranked"]):
        rows = list(summary[system]["weights"].items())[:8]
        ax.barh([k for k, _ in rows][::-1], [v for _, v in rows][::-1], color="#9467bd")
        ax.set_title(system)
    fig.tight_layout()
    fig.savefig(ARTIFACT_DIR / "figure3_weights.png", dpi=220)
    plt.close(fig)


def main() -> None:
    summary = build_summary()
    (ARTIFACT_DIR / "escalation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plot_accuracy(summary)
    plot_utility(summary)
    plot_weights(summary)
    print("Generated escalation artifacts in", ARTIFACT_DIR)


if __name__ == "__main__":
    main()
