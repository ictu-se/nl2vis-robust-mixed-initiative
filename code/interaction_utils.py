from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RANKED_POOL_DIR = ROOT / "data" / "ranked_pools"

QUESTION_ORDER = ["filter", "mark", "x", "y", "aggregate", "color", "theta", "size", "sort", "bin"]
EDIT_ORDER = ["x", "y", "color", "aggregate", "filter", "mark", "theta", "size", "sort", "bin"]


def load_ranked_rows(system: str, split: str) -> list[dict[str, Any]]:
    suffix = "reranked_results" if system == "reranked" else "results"
    path = RANKED_POOL_DIR / f"{split}_{suffix}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["rows"]


def freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple((key, freeze(value[key])) for key in sorted(value))
    if isinstance(value, list):
        return tuple(freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(freeze(item) for item in value)
    return value


def thaw(value: Any) -> Any:
    if isinstance(value, tuple):
        if value and all(isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str) for item in value):
            return {key: thaw(val) for key, val in value}
        return [thaw(item) for item in value]
    return value


def canonicalize(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True)


def unique_golds(row: dict[str, Any]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    output: list[dict[str, Any]] = []
    for gold in row["gold_answer"]:
        key = canonicalize(gold)
        if key not in seen:
            seen.add(key)
            output.append(gold)
    return output


def slotify(spec: dict[str, Any]) -> dict[str, Any]:
    encoding = spec.get("encoding", {})

    def field(name: str) -> Any:
        if name in encoding and isinstance(encoding[name], dict):
            return freeze(encoding[name].get("field"))
        return None

    def aggregate(name: str) -> Any:
        if name in encoding and isinstance(encoding[name], dict):
            return freeze(encoding[name].get("aggregate"))
        return None

    filters = [freeze(step["filter"]) for step in spec.get("transform", []) if "filter" in step]
    sorts = []
    bins = []
    for channel in ["x", "y", "color", "theta", "size"]:
        if channel in encoding and isinstance(encoding[channel], dict) and "sort" in encoding[channel]:
            sorts.append((channel, freeze(encoding[channel]["sort"])))
    for channel in ["x", "y"]:
        if channel in encoding and isinstance(encoding[channel], dict) and "bin" in encoding[channel]:
            bins.append((channel, freeze(encoding[channel]["bin"])))

    return {
        "mark": freeze(spec.get("mark")),
        "x": field("x"),
        "y": field("y"),
        "color": field("color"),
        "theta": field("theta"),
        "size": field("size"),
        "aggregate": tuple((channel, aggregate(channel)) for channel in ["x", "y", "theta", "size"] if aggregate(channel)) or None,
        "filter": tuple(sorted(filters)) if filters else None,
        "sort": tuple(sorted(sorts)) if sorts else None,
        "bin": tuple(sorted(bins)) if bins else None,
    }


def candidate_pool_contains_target(predictions: list[dict[str, Any]], target: dict[str, Any]) -> bool:
    target_key = canonicalize(target)
    return target_key in {canonicalize(pred) for pred in predictions}


def slot_differences(source: dict[str, Any], target: dict[str, Any]) -> list[str]:
    source_slots = slotify(source)
    target_slots = slotify(target)
    return [slot for slot in EDIT_ORDER if source_slots.get(slot) != target_slots.get(slot)]


def apply_edit(spec: dict[str, Any], slot: str, target_value: Any) -> dict[str, Any]:
    updated = json.loads(json.dumps(spec))
    encoding = updated.setdefault("encoding", {})

    def ensure_channel(channel: str) -> dict[str, Any]:
        return encoding.setdefault(channel, {})

    if slot == "mark":
        if target_value is None:
            updated.pop("mark", None)
        else:
            updated["mark"] = thaw(target_value)
        return updated

    if slot in {"x", "y", "color", "theta", "size"}:
        if target_value is None:
            encoding.pop(slot, None)
        else:
            channel = ensure_channel(slot)
            channel["field"] = thaw(target_value)
        return updated

    if slot == "aggregate":
        target_map = thaw(target_value) if target_value is not None else {}
        if not isinstance(target_map, dict):
            target_map = {}
        for channel in ["x", "y", "theta", "size"]:
            if channel in encoding and isinstance(encoding[channel], dict):
                if channel in target_map and target_map[channel] is not None:
                    encoding[channel]["aggregate"] = target_map[channel]
                else:
                    encoding[channel].pop("aggregate", None)
        return updated

    if slot == "filter":
        transforms = updated.get("transform", [])
        transforms = [step for step in transforms if "filter" not in step]
        if target_value is not None:
            thawed = thaw(target_value)
            if isinstance(thawed, list):
                transforms.extend({"filter": item} for item in thawed)
            else:
                transforms.append({"filter": thawed})
        if transforms:
            updated["transform"] = transforms
        else:
            updated.pop("transform", None)
        return updated

    if slot == "sort":
        target_map = thaw(target_value) if target_value is not None else {}
        if not isinstance(target_map, dict):
            target_map = {}
        for channel in ["x", "y", "color", "theta", "size"]:
            if channel in encoding and isinstance(encoding[channel], dict):
                if channel in target_map and target_map[channel] is not None:
                    encoding[channel]["sort"] = target_map[channel]
                else:
                    encoding[channel].pop("sort", None)
        return updated

    if slot == "bin":
        target_map = thaw(target_value) if target_value is not None else {}
        if not isinstance(target_map, dict):
            target_map = {}
        for channel in ["x", "y"]:
            if channel in encoding and isinstance(encoding[channel], dict):
                if channel in target_map and target_map[channel] is not None:
                    encoding[channel]["bin"] = target_map[channel]
                else:
                    encoding[channel].pop("bin", None)
        return updated

    return updated


def repair_order_greedy(source: dict[str, Any], target: dict[str, Any]) -> list[str]:
    return slot_differences(source, target)


def repair_order_random(source: dict[str, Any], target: dict[str, Any], rng: random.Random) -> list[str]:
    diffs = slot_differences(source, target)
    rng.shuffle(diffs)
    return diffs


def current_status(spec: dict[str, Any], target: dict[str, Any], golds: list[dict[str, Any]]) -> dict[str, Any]:
    spec_key = canonicalize(spec)
    target_key = canonicalize(target)
    gold_keys = {canonicalize(gold) for gold in golds}
    return {
        "intent_top1": float(spec_key == target_key),
        "gold_acceptance_top1": float(spec_key in gold_keys),
        "slot_distance": float(len(slot_differences(spec, target))),
    }


def valid_questions(remaining_specs: list[dict[str, Any]], target_spec: dict[str, Any]) -> list[dict[str, Any]]:
    remaining_slots = [slotify(spec) for spec in remaining_specs]
    target_slots = slotify(target_spec)
    questions: list[dict[str, Any]] = []
    for slot in QUESTION_ORDER:
        values = [slots.get(slot) for slots in remaining_slots]
        target_value = target_slots.get(slot)
        if len(set(values)) <= 1:
            continue
        if target_value is None or target_value not in set(values):
            continue
        questions.append({"slot": slot, "value": target_value})
    return questions


def ask_greedy(remaining_specs: list[dict[str, Any]], target_spec: dict[str, Any]) -> dict[str, Any] | None:
    best: tuple[tuple[int, int, int], dict[str, Any]] | None = None
    for question in valid_questions(remaining_specs, target_spec):
        values = [slotify(spec).get(question["slot"]) for spec in remaining_specs]
        keep = sum(value == question["value"] for value in values)
        removed = len(values) - keep
        score = (removed, -keep, -QUESTION_ORDER.index(question["slot"]))
        if best is None or score > best[0]:
            best = (score, question)
    return None if best is None else best[1]


def filter_candidates(remaining_specs: list[dict[str, Any]], question: dict[str, Any]) -> list[dict[str, Any]]:
    filtered = [spec for spec in remaining_specs if slotify(spec).get(question["slot"]) == question["value"]]
    return filtered or remaining_specs


def generic_profile_features(spec: dict[str, Any]) -> dict[str, float]:
    slots = slotify(spec)
    return {
        "mark_bar": float(slots["mark"] == "bar"),
        "mark_line": float(slots["mark"] == "line"),
        "mark_boxplot": float(slots["mark"] == "boxplot"),
        "has_color": float(slots["color"] is not None),
        "has_filter": float(slots["filter"] is not None),
        "has_aggregate": float(slots["aggregate"] is not None),
        "has_bin": float(slots["bin"] is not None),
        "has_theta": float(slots["theta"] is not None),
    }


def pool_uncertainty_features(predictions: list[dict[str, Any]]) -> dict[str, float]:
    slots = [slotify(pred) for pred in predictions]
    features: dict[str, float] = {"candidate_count": float(len(predictions))}
    for slot in ["mark", "x", "y", "color", "aggregate", "filter", "theta", "bin"]:
        features[f"unique_{slot}"] = float(len({candidate.get(slot) for candidate in slots}))
    features["mean_non_null_slots"] = float(
        sum(sum(value is not None for value in candidate.values()) for candidate in slots) / max(1, len(slots))
    )
    return features


def mean(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def counter_to_sorted_rows(counter: Counter[str], success_counter: Counter[str] | None = None) -> list[dict[str, Any]]:
    rows = []
    for key, count in counter.most_common():
        row: dict[str, Any] = {"slot": key, "count": int(count)}
        if success_counter is not None:
            row["success_rate"] = float(success_counter[key] / count) if count else 0.0
        rows.append(row)
    return rows


def summarize_budget_records(records: list[dict[str, Any]], budget_key: str) -> dict[str, dict[str, float]]:
    metrics: dict[str, defaultdict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        budget = int(record[budget_key])
        for key, value in record["status"].items():
            metrics[key][budget].append(float(value))
    summary: dict[str, dict[str, float]] = {}
    for key, budget_values in metrics.items():
        summary[key] = {str(budget): mean(values) for budget, values in sorted(budget_values.items())}
    return summary
