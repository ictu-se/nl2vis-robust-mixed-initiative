from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "code"))

from interaction_utils import (
    apply_edit,
    ask_greedy,
    filter_candidates,
    generic_profile_features,
    load_ranked_rows,
    pool_uncertainty_features,
    slot_differences,
    slotify,
    unique_golds,
)


ARTIFACT_DIR = ROOT / "artifacts_runtime" / "extended_analysis"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

ARCHETYPES = {
    "bar_aggregate": {"mark_bar": 1.5, "has_aggregate": 1.2, "has_color": -0.4},
    "color_breakdown": {"has_color": 1.5, "has_aggregate": -0.5, "mark_bar": 0.4},
    "distribution": {"mark_boxplot": 1.8, "has_filter": 0.8},
    "minimal_line": {"mark_line": 1.6, "has_color": -0.8, "has_filter": -0.4},
    "binned_summary": {"has_bin": 1.5, "has_aggregate": 0.9},
    "part_to_whole": {"has_theta": 1.8, "has_color": 0.6},
}


def canonical(obj: dict) -> str:
    return json.dumps(obj, sort_keys=True)


def mean(xs: list[float]) -> float:
    return sum(xs) / max(1, len(xs))


def bootstrap_ci(values: list[float], reps: int = 2000, seed: int = 13) -> dict[str, float]:
    rng = random.Random(seed)
    if not values:
        return {"mean": 0.0, "lo": 0.0, "hi": 0.0}
    boots = []
    n = len(values)
    for _ in range(reps):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        boots.append(mean(sample))
    boots.sort()
    lo = boots[int(0.025 * len(boots))]
    hi = boots[int(0.975 * len(boots))]
    return {"mean": mean(values), "lo": lo, "hi": hi}


def preference_score(spec: dict, weights: dict[str, float]) -> float:
    feats = generic_profile_features(spec)
    return sum(weights.get(name, 0.0) * value for name, value in feats.items())


def slot_tokens(spec: dict) -> list[str]:
    return [f"{slot}={value}" for slot, value in slotify(spec).items() if value is not None]


def build_hidden_profile(seed_golds: list[dict]) -> dict[str, float]:
    counts: dict[str, float] = defaultdict(float)
    for gold in seed_golds:
        for token in slot_tokens(gold):
            counts[token] += 1.0
    return counts


def overlap_score(spec: dict, token_weights: dict[str, float]) -> float:
    return sum(token_weights.get(token, 0.0) for token in slot_tokens(spec))


def choose_target(golds: list[dict], weights: dict[str, float], hidden_profile: dict[str, float]) -> dict:
    return max(golds, key=lambda gold: (preference_score(gold, weights) + 0.7 * overlap_score(gold, hidden_profile), canonical(gold)))


def average_memory(golds: list[dict]) -> tuple[dict[str, float], dict[str, float]]:
    generic = defaultdict(float)
    tokens = defaultdict(float)
    if not golds:
        return {}, {}
    for gold in golds:
        for name, value in generic_profile_features(gold).items():
            generic[name] += value
        for token in slot_tokens(gold):
            tokens[token] += 1.0
    denom = float(len(golds))
    return ({name: value / denom for name, value in generic.items()}, {token: value / denom for token, value in tokens.items()})


def rank_with_memory(predictions: list[dict], generic_mem: dict[str, float], token_mem: dict[str, float], beta: float, gamma: float) -> list[dict]:
    indexed = list(enumerate(predictions))
    ranked = sorted(
        indexed,
        key=lambda pair: (
            beta * sum(generic_mem.get(name, 0.0) * value for name, value in generic_profile_features(pair[1]).items())
            + gamma * overlap_score(pair[1], token_mem)
            - 0.1 * pair[0],
            -pair[0],
        ),
        reverse=True,
    )
    return [pred for _, pred in ranked]


def simulate_clarification(predictions: list[dict], target: dict, budget: int) -> tuple[dict, int]:
    remaining = list(predictions)
    asked = 0
    for _ in range(budget):
        question = ask_greedy(remaining, target)
        if question is None:
            break
        asked += 1
        remaining = filter_candidates(remaining, question)
    return remaining[0], asked


def repair_to_target(spec: dict, target: dict, budget: int) -> tuple[dict, int]:
    current = json.loads(json.dumps(spec))
    target_slots = slotify(target)
    diffs = slot_differences(spec, target)
    used = 0
    for slot in diffs[:budget]:
        current = apply_edit(current, slot, target_slots.get(slot))
        used += 1
    return current, used


def build_adaptive_model(rows: list[dict]) -> tuple[LogisticRegression, list[str], dict[str, float]]:
    dataset = []
    for row in rows:
        base_features = pool_uncertainty_features(row["predictions"])
        for target in unique_golds(row):
            dataset.append({"features": base_features, "label": float(canonical(row["predictions"][0]) == canonical(target))})
    feature_names = sorted(dataset[0]["features"].keys())
    X = [[item["features"][name] for name in feature_names] for item in dataset]
    y = [int(item["label"]) for item in dataset]
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X, y)
    return model, feature_names, {"t1": 0.65, "t2": 0.7}


def chosen_budget(prob: float, thresholds: dict[str, float]) -> int:
    if prob >= thresholds["t2"]:
        return 0
    if prob >= thresholds["t1"]:
        return 1
    return 2


def build_controller_logs() -> dict:
    logs = {}
    for system in ["heuristic", "reranked"]:
        dev_rows = load_ranked_rows(system, "dev")
        test_rows = load_ranked_rows(system, "test")
        model, feature_names, thresholds = build_adaptive_model(dev_rows)
        beta = 0.0
        gamma = 2.0
        system_logs: dict[str, list[dict[str, float]]] = defaultdict(list)
        for archetype, weights in ARCHETYPES.items():
            seed_targets = [choose_target(unique_golds(row), weights, {}) for row in dev_rows[:10]]
            hidden = build_hidden_profile(seed_targets)
            history_targets = [choose_target(unique_golds(row), weights, hidden) for row in dev_rows[:20]]
            generic_mem, token_mem = average_memory(history_targets)
            for row in test_rows:
                golds = unique_golds(row)
                target = choose_target(golds, weights, hidden)
                personalized = rank_with_memory(row["predictions"], generic_mem, token_mem, beta, gamma)
                features = pool_uncertainty_features(personalized)
                prob = model.predict_proba([[features[name] for name in feature_names]])[0, 1]
                q_budget = chosen_budget(prob, thresholds)

                rank_only = personalized[0]
                clarify_only, q_used = simulate_clarification(row["predictions"], target, q_budget)
                repair_only, r_used = repair_to_target(row["predictions"][0], target, 2)
                mem_clarify, mq_used = simulate_clarification(personalized, target, q_budget)
                full_after_clarify, fq_used = simulate_clarification(personalized, target, q_budget)
                if canonical(full_after_clarify) != canonical(target):
                    full_final, fr_used = repair_to_target(full_after_clarify, target, 2)
                else:
                    full_final, fr_used = full_after_clarify, 0

                system_logs["rank_only"].append({"intent": float(canonical(rank_only) == canonical(target)), "actions": 0.0})
                system_logs["clarify_only"].append({"intent": float(canonical(clarify_only) == canonical(target)), "actions": float(q_used)})
                system_logs["repair_only"].append({"intent": float(canonical(repair_only) == canonical(target)), "actions": float(r_used)})
                system_logs["memory_clarify"].append({"intent": float(canonical(mem_clarify) == canonical(target)), "actions": float(mq_used)})
                system_logs["full_controller"].append({"intent": float(canonical(full_final) == canonical(target)), "actions": float(fq_used + fr_used)})
        logs[system] = system_logs
    return logs


def build_significance_summary(controller_logs: dict) -> dict:
    summary: dict[str, dict] = {"controllers": {}, "escalation": {}, "clarify_noise": {}, "repair_noise": {}}

    # Controller-level paired utility gains vs strongest non-full baseline
    for system in ["heuristic", "reranked"]:
        full_utils = [row["intent"] - 0.08 * row["actions"] for row in controller_logs[system]["full_controller"]]
        baseline_candidates = {
            name: [row["intent"] - 0.08 * row["actions"] for row in rows]
            for name, rows in controller_logs[system].items()
            if name != "full_controller"
        }
        best_name, best_utils = max(baseline_candidates.items(), key=lambda item: mean(item[1]))
        gains = [f - b for f, b in zip(full_utils, best_utils)]
        summary["controllers"][system] = {
            "best_baseline": best_name,
            "full_utility": bootstrap_ci(full_utils, seed=101),
            "baseline_utility": bootstrap_ci(best_utils, seed=102),
            "gain_over_best_baseline": bootstrap_ci(gains, seed=103),
        }

    # Escalation significance
    from run_escalation import build_dataset, fit_policy

    for system in ["heuristic", "reranked"]:
        train = build_dataset(load_ranked_rows(system, "dev"))
        test = build_dataset(load_ranked_rows(system, "test"))
        preds, _, _ = fit_policy(train, test)
        always_clarify = [row["clarify"] for row in test]
        always_repair = [row["repair"] for row in test]
        escalate = [row["repair"] if pred == 1 else row["clarify"] for pred, row in zip(preds, test)]
        best_fixed_name, best_fixed = max(
            {"clarify": always_clarify, "repair": always_repair}.items(),
            key=lambda item: mean(item[1]),
        )
        gains = [e - b for e, b in zip(escalate, best_fixed)]
        summary["escalation"][system] = {
            "best_global_fixed": best_fixed_name,
            "clarify": bootstrap_ci(always_clarify, seed=201),
            "repair": bootstrap_ci(always_repair, seed=202),
            "escalate": bootstrap_ci(escalate, seed=203),
            "gain_over_best_global_fixed": bootstrap_ci(gains, seed=204),
        }

    # Noise robustness at eta=0.4 as paired gains
    from run_clarification_robustness import simulate as clarify_sim
    from run_repair_robustness import simulate as repair_sim

    for system in ["heuristic", "reranked"]:
        rows = load_ranked_rows(system, "test")
        clarify_base = []
        clarify_robust = []
        repair_base = []
        repair_robust = []
        for seed in range(10):
            clarify_rng = random.Random(seed)
            repair_rng = random.Random(seed)
            repair_rr = random.Random(seed + 999)
            for row in rows:
                start = row["predictions"][0]
                for target in unique_golds(row):
                    clarify_base.append(clarify_sim(row["predictions"], target, 0.4, False, clarify_rng))
                    clarify_robust.append(clarify_sim(row["predictions"], target, 0.4, True, random.Random(seed + 1000)))
                    repair_base.append(repair_sim(start, target, 0.4, False, repair_rng))
                    repair_robust.append(repair_sim(start, target, 0.4, True, repair_rr))
        summary["clarify_noise"][system] = {
            "base": bootstrap_ci(clarify_base, seed=301),
            "robust": bootstrap_ci(clarify_robust, seed=302),
            "gain": bootstrap_ci([r - b for r, b in zip(clarify_robust, clarify_base)], seed=303),
        }
        summary["repair_noise"][system] = {
            "base": bootstrap_ci(repair_base, seed=304),
            "robust": bootstrap_ci(repair_robust, seed=305),
            "gain": bootstrap_ci([r - b for r, b in zip(repair_robust, repair_base)], seed=306),
        }
    return summary


def build_regime_summary() -> dict:
    from run_escalation import clarify_outcome, repair_outcome

    summary = {}
    for system in ["heuristic", "reranked"]:
        rows = load_ranked_rows(system, "test")
        buckets: dict[tuple[int, str], dict[str, list[float]]] = defaultdict(lambda: {"clarify": [], "repair": [], "count": []})
        for row in rows:
            start = row["predictions"][0]
            for target in unique_golds(row):
                in_pool = int(any(canonical(p) == canonical(target) for p in row["predictions"]))
                dist = len(slot_differences(start, target))
                if dist <= 1:
                    dist_bucket = "1"
                elif dist == 2:
                    dist_bucket = "2"
                else:
                    dist_bucket = "3+"
                key = (in_pool, dist_bucket)
                buckets[key]["clarify"].append(clarify_outcome(row["predictions"], target))
                buckets[key]["repair"].append(repair_outcome(start, target))
                buckets[key]["count"].append(1.0)
        summary[system] = {
            f"inpool_{in_pool}_dist_{dist_bucket}": {
                "clarify": mean(vals["clarify"]),
                "repair": mean(vals["repair"]),
                "gap_repair_minus_clarify": mean(vals["repair"]) - mean(vals["clarify"]),
                "count": int(sum(vals["count"])),
            }
            for (in_pool, dist_bucket), vals in buckets.items()
        }
    return summary


def plot_regime_heatmaps(summary: dict) -> None:
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for row_idx, system in enumerate(["heuristic", "reranked"]):
        grid_gap = []
        grid_count = []
        for in_pool in [0, 1]:
            gap_row = []
            count_row = []
            for dist_bucket in ["1", "2", "3+"]:
                item = summary[system].get(f"inpool_{in_pool}_dist_{dist_bucket}", None)
                gap_row.append(item["gap_repair_minus_clarify"] if item else 0.0)
                count_row.append(item["count"] if item else 0)
            grid_gap.append(gap_row)
            grid_count.append(count_row)
        sns.heatmap(grid_gap, annot=True, fmt=".3f", cmap="coolwarm", center=0.0, ax=axes[row_idx, 0], cbar=row_idx == 0)
        axes[row_idx, 0].set_title(f"{system}: repair - clarify")
        axes[row_idx, 0].set_xlabel("start slot distance")
        axes[row_idx, 0].set_ylabel("target in pool")
        axes[row_idx, 0].set_xticklabels(["1", "2", "3+"])
        axes[row_idx, 0].set_yticklabels(["0", "1"], rotation=0)

        sns.heatmap(grid_count, annot=True, fmt="d", cmap="Blues", ax=axes[row_idx, 1], cbar=row_idx == 0)
        axes[row_idx, 1].set_title(f"{system}: case count")
        axes[row_idx, 1].set_xlabel("start slot distance")
        axes[row_idx, 1].set_ylabel("target in pool")
        axes[row_idx, 1].set_xticklabels(["1", "2", "3+"])
        axes[row_idx, 1].set_yticklabels(["0", "1"], rotation=0)
    fig.savefig(ARTIFACT_DIR / "figure1_regime_heatmaps.png", dpi=220)
    plt.close(fig)


def plot_gain_intervals(summary: dict) -> None:
    sns.set_theme(style="whitegrid")
    labels = []
    means = []
    los = []
    his = []
    for system in ["heuristic", "reranked"]:
        for block, key in [
            ("controllers", "gain_over_best_baseline"),
            ("escalation", "gain_over_best_global_fixed"),
            ("clarify_noise", "gain"),
            ("repair_noise", "gain"),
        ]:
            item = summary[block][system][key]
            labels.append(f"{system}\n{block}")
            means.append(item["mean"])
            los.append(item["mean"] - item["lo"])
            his.append(item["hi"] - item["mean"])
    fig, ax = plt.subplots(figsize=(10, 5))
    xs = list(range(len(labels)))
    ax.errorbar(xs, means, yerr=[los, his], fmt="o", capsize=4, color="#d62728")
    ax.axhline(0.0, color="black", linewidth=1, linestyle="--")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Gain with 95% bootstrap CI")
    ax.set_title("Statistical reliability of controller-level gains")
    fig.tight_layout()
    fig.savefig(ARTIFACT_DIR / "figure2_gain_intervals.png", dpi=220)
    plt.close(fig)


def main() -> None:
    controller_logs = build_controller_logs()
    significance = build_significance_summary(controller_logs)
    regime = build_regime_summary()
    (ARTIFACT_DIR / "significance_summary.json").write_text(json.dumps(significance, indent=2), encoding="utf-8")
    (ARTIFACT_DIR / "regime_summary.json").write_text(json.dumps(regime, indent=2), encoding="utf-8")
    plot_regime_heatmaps(regime)
    plot_gain_intervals(significance)
    print("Generated extended-analysis artifacts in", ARTIFACT_DIR)


if __name__ == "__main__":
    main()
