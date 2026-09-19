"""Bootstrap a labeled retrieval eval set (and disjoint contrastive training
pairs) from exercise metadata.

Labels are heuristic, not hand-verified by a fitness domain expert: a query
like "barbell exercise for quadriceps" is marked relevant for every exercise
whose `equipment`/`primaryMuscles` fields match exactly. That's a reasonable
proxy (metadata-driven "relevant" sets are how BEIR-style benchmarks bootstrap
too) but it's worth spot-checking a sample of `data/eval/test_queries.jsonl`
before trusting it as ground truth.

Leakage control: queries are built from metadata "combos" (e.g. one
(muscle, equipment) pair). Combos are split into disjoint eval-only and
train-only pools *per axis* before any query text is generated, so no query
template used for eval scoring ever also appears as a contrastive training
anchor.

Usage:
    python -m src.eval.build_eval_set \
        --dataset data/processed/exercises_dataset.jsonl \
        --eval_out data/eval/test_queries.jsonl \
        --train_out data/training/contrastive_pairs.jsonl
"""

import argparse
import hashlib
import json
import random
from pathlib import Path

MIN_COMBO_SUPPORT = 3
MAX_COMBO_SUPPORT = 40
EVAL_TARGET = 80
MAX_ITEMS_PER_TRAIN_COMBO = 8
MAX_TRAIN_PAIRS = 500
EVAL_SPLIT_FRACTION = 0.4  # fraction of each axis's combos held out for eval

EQUIPMENT_DISPLAY = {
    "body only": "bodyweight",
    "e-z curl bar": "EZ curl bar",
    "foam roll": "foam roller",
    "other": "other equipment",
}


def _display(value: str) -> str:
    return EQUIPMENT_DISPLAY.get(value, value)


def _combo_bucket(axis: str, combo: tuple, num_buckets: int = 100) -> int:
    """Deterministic hash bucket for a combo, used to split eval/train pools."""
    key = f"{axis}:{combo}".encode()
    digest = hashlib.sha256(key).hexdigest()
    return int(digest, 16) % num_buckets


def load_dataset(path: Path) -> list[dict]:
    items = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def primary_muscle(attributes: dict) -> str | None:
    muscles = attributes.get("primaryMuscles") or []
    return muscles[0] if muscles else None


# Each axis: (name, template_variants, key_fn)
# key_fn extracts the combo tuple from an item's attributes, or None if the
# item doesn't have the fields this axis needs.
AXES = [
    (
        "muscle_equipment",
        [
            "{equipment} exercise for {muscle}",
            "{muscle} exercise using {equipment}",
        ],
        lambda a: (
            (primary_muscle(a), a.get("equipment"))
            if primary_muscle(a) and a.get("equipment")
            else None
        ),
    ),
    (
        "muscle_mechanic",
        [
            "{mechanic} exercise for {muscle}",
        ],
        lambda a: (
            (primary_muscle(a), a.get("mechanic"))
            if primary_muscle(a) and a.get("mechanic")
            else None
        ),
    ),
    (
        "muscle_category",
        [
            "{category} exercise for {muscle}",
            "{muscle} {category} exercise",
        ],
        lambda a: (
            (primary_muscle(a), a.get("category"))
            if primary_muscle(a) and a.get("category")
            else None
        ),
    ),
    (
        "equipment_category",
        [
            "{category} exercise using {equipment}",
        ],
        lambda a: (
            (a.get("equipment"), a.get("category"))
            if a.get("equipment") and a.get("category")
            else None
        ),
    ),
    (
        "level_muscle",
        [
            "{level} {muscle} exercise",
        ],
        lambda a: (
            (a.get("level"), primary_muscle(a)) if a.get("level") and primary_muscle(a) else None
        ),
    ),
]


def _render(template: str, axis: str, combo: tuple) -> str:
    if axis == "muscle_equipment":
        muscle, equipment = combo
        return template.format(muscle=muscle, equipment=_display(equipment))
    if axis == "muscle_mechanic":
        muscle, mechanic = combo
        return template.format(muscle=muscle, mechanic=mechanic)
    if axis == "muscle_category":
        muscle, category = combo
        return template.format(muscle=muscle, category=category)
    if axis == "equipment_category":
        equipment, category = combo
        return template.format(equipment=_display(equipment), category=category)
    if axis == "level_muscle":
        level, muscle = combo
        return template.format(level=level, muscle=muscle)
    raise ValueError(f"Unknown axis: {axis}")


def build_combo_index(items: list[dict]) -> dict[str, dict[tuple, list[str]]]:
    """axis -> combo -> list of item ids matching that combo exactly."""
    index: dict[str, dict[tuple, list[str]]] = {axis: {} for axis, _, _ in AXES}
    for item in items:
        attrs = item.get("attributes", {})
        for axis, _templates, key_fn in AXES:
            combo = key_fn(attrs)
            if combo is None:
                continue
            index[axis].setdefault(combo, []).append(item["id"])
    return index


def split_combos(index: dict[str, dict[tuple, list[str]]], rng: random.Random) -> tuple[dict, dict]:
    """Split each axis's combos into disjoint eval/train pools by hash bucket."""
    eval_pools: dict[str, dict[tuple, list[str]]] = {}
    train_pools: dict[str, dict[tuple, list[str]]] = {}
    threshold = int(EVAL_SPLIT_FRACTION * 100)

    for axis, combos in index.items():
        eval_pools[axis] = {}
        train_pools[axis] = {}
        for combo, ids in combos.items():
            support = len(ids)
            if support < MIN_COMBO_SUPPORT or support > MAX_COMBO_SUPPORT:
                continue
            bucket = _combo_bucket(axis, combo)
            if bucket < threshold:
                eval_pools[axis][combo] = ids
            else:
                train_pools[axis][combo] = ids
    return eval_pools, train_pools


def build_eval_queries(eval_pools: dict, rng: random.Random, target: int) -> list[dict]:
    axis_names = [axis for axis, _, _ in AXES]
    templates_by_axis = {axis: templates for axis, templates, _ in AXES}

    all_candidates = []
    for axis in axis_names:
        for combo, ids in eval_pools[axis].items():
            template = rng.choice(templates_by_axis[axis])
            query = _render(template, axis, combo)
            all_candidates.append(
                {
                    "query": query,
                    "relevant": sorted(ids),
                    "axis": axis,
                    "combo": list(combo),
                }
            )

    rng.shuffle(all_candidates)

    seen_queries = set()
    selected = []
    for cand in all_candidates:
        if cand["query"] in seen_queries:
            continue
        seen_queries.add(cand["query"])
        selected.append(cand)
        if len(selected) >= target:
            break
    return selected


def build_train_pairs(
    train_pools: dict, items_by_id: dict[str, dict], rng: random.Random
) -> list[dict]:
    templates_by_axis = {axis: templates for axis, templates, _ in AXES}

    pairs = []
    combos_flat = [
        (axis, combo, ids) for axis, combos in train_pools.items() for combo, ids in combos.items()
    ]
    rng.shuffle(combos_flat)

    for axis, combo, ids in combos_flat:
        template = rng.choice(templates_by_axis[axis])
        query = _render(template, axis, combo)
        sample_ids = rng.sample(ids, k=min(len(ids), MAX_ITEMS_PER_TRAIN_COMBO))
        for item_id in sample_ids:
            item = items_by_id[item_id]
            pairs.append(
                {
                    "query": query,
                    "positive_id": item_id,
                    "positive_text": item.get("combined_text", "")[:500],
                }
            )
        if len(pairs) >= MAX_TRAIN_PAIRS:
            break

    return pairs[:MAX_TRAIN_PAIRS]


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} rows to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="data/processed/exercises_dataset.jsonl")
    parser.add_argument("--eval_out", default="data/eval/test_queries.jsonl")
    parser.add_argument("--train_out", default="data/training/contrastive_pairs.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    items = load_dataset(Path(args.dataset))
    items_by_id = {item["id"]: item for item in items}
    print(f"Loaded {len(items)} items from {args.dataset}")

    combo_index = build_combo_index(items)
    for axis, combos in combo_index.items():
        supported = sum(
            1 for ids in combos.values() if MIN_COMBO_SUPPORT <= len(ids) <= MAX_COMBO_SUPPORT
        )
        print(f"  axis={axis}: {len(combos)} combos, {supported} in support range")

    eval_pools, train_pools = split_combos(combo_index, rng)

    eval_queries = build_eval_queries(eval_pools, rng, target=EVAL_TARGET)
    # Drop the bookkeeping fields before writing — eval_harness.py only needs
    # {"query", "relevant"}; axis/combo were only useful for debugging above.
    eval_rows = [{"query": q["query"], "relevant": q["relevant"]} for q in eval_queries]
    write_jsonl(eval_rows, Path(args.eval_out))

    train_pairs = build_train_pairs(train_pools, items_by_id, rng)
    write_jsonl(train_pairs, Path(args.train_out))

    eval_query_texts = {q["query"] for q in eval_queries}
    train_query_texts = {p["query"] for p in train_pairs}
    overlap = eval_query_texts & train_query_texts
    print(f"Query text overlap between eval and train sets: {len(overlap)} (should be 0)")


if __name__ == "__main__":
    main()
