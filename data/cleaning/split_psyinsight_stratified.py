import json
import argparse
import random
from pathlib import Path
from collections import defaultdict, Counter

def normalize_label(label):
    if not label or not str(label).strip():
        return "Unknown"
    return str(label).strip()

def get_id(ex):
    did = ex.get("dialog_id")
    return None if did is None else str(did).strip()

def count_by_label(items):
    c = Counter()
    for ex in items:
        c[normalize_label(ex.get("psychotherapy"))] += 1
    return c

def print_dist(title, counter):
    total = sum(counter.values())
    print(f"\n== {title} (n={total}) ==")
    for label, n in sorted(counter.items(), key=lambda x: (-x[1], x[0])):
        pct = (n / total * 100) if total else 0.0
        print(f"{label}: {n} ({pct:.1f}%)")

def main():
    p = argparse.ArgumentParser(description="Stratified 80/20 split with rare-label exclusion + forced train IDs.")
    p.add_argument("--input", default="en_data_cleaned.json", help="Input cleaned dataset (list of full objects)")
    p.add_argument("--train_out", default="en_data_cleaned_train.json", help="Train output JSON")
    p.add_argument("--test_out", default="en_data_cleaned_test.json", help="Test output JSON")
    p.add_argument("--test_ratio", type=float, default=0.20, help="Target test ratio overall (default 0.20)")
    p.add_argument("--exclude_test_leq", type=int, default=5,
                   help="Exclude labels with count <= this from test (default 5)")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--force_train_ids", nargs="*", default=["000563", "000462", "000865"],
                   help="Dialog IDs that must be in train (never in test)")
    args = p.parse_args()

    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Expected a top-level list of dialogue objects.")

    rng = random.Random(args.seed)
    force_train_ids = set(str(x).strip() for x in args.force_train_ids)

    # Build groups and basic counts
    groups = defaultdict(list)
    id_to_item = {}
    missing_id = 0

    for ex in data:
        did = get_id(ex)
        if not did:
            missing_id += 1
            continue
        # If duplicates exist, keep first occurrence 
        if did in id_to_item:
            continue
        id_to_item[did] = ex
        label = normalize_label(ex.get("psychotherapy"))
        groups[label].append(ex)

    filtered_data = list(id_to_item.values())
    total_n = len(filtered_data)
    if total_n == 0:
        raise ValueError("No usable items found (check dialog_id presence).")

    full_counts = count_by_label(filtered_data)
    print_dist("FULL DATASET label distribution", full_counts)
    if missing_id:
        print(f"\nNote: Dropped {missing_id} items with missing/empty dialog_id.")

    # Determine which labels are allowed to contribute to test
    excluded_labels = {lab for lab, n in full_counts.items() if n <= args.exclude_test_leq}

    print(f"\nLabels excluded from test (count <= {args.exclude_test_leq}):")
    for lab in sorted(excluded_labels, key=lambda x: (full_counts[x], x)):
        print(f"  - {lab}: {full_counts[lab]}")

    # Split per label
    train, test = [], []
    forced_train_hits = []

    # First, ensure forced IDs go to train (and remove them from their label pools)
    for lab in list(groups.keys()):
        kept = []
        for ex in groups[lab]:
            did = get_id(ex)
            if did in force_train_ids:
                train.append(ex)
                forced_train_hits.append(did)
            else:
                kept.append(ex)
        groups[lab] = kept

    # Now stratified assignment
    for lab, items in groups.items():
        items = items[:]
        rng.shuffle(items)
        n = len(items)

        # If excluded label, everything goes to train
        if lab in excluded_labels:
            train.extend(items)
            continue

        # Otherwise allocate ~20% to test for this label
        n_test = int(round(n * args.test_ratio))
        # Ensure at least 1 test item if label has enough examples (after forced-train removal)
        if n >= 2 and n_test == 0:
            n_test = 1
        # But don't create empty train for small labels
        if n_test >= n:
            n_test = max(0, n - 1)

        test.extend(items[:n_test])
        train.extend(items[n_test:])

    # Final shuffle
    rng.shuffle(train)
    rng.shuffle(test)

    # Sanity: forced IDs must not be in test
    test_ids = {get_id(x) for x in test}
    leaked = sorted(list(force_train_ids.intersection(test_ids)))
    if leaked:
        raise RuntimeError(f"Forced-train IDs leaked into test: {leaked}")

    # Report resulting sizes
    print("\n==== SPLIT SUMMARY ====")
    print(f"Total used (after dropping missing/duplicate ids): {total_n}")
    print(f"Train: {len(train)} ({len(train)/total_n:.2%})")
    print(f"Test:  {len(test)} ({len(test)/total_n:.2%})")
    print(f"Target test ratio: {args.test_ratio:.2%}")

    # Report forced-train coverage
    missing_forced = sorted(list(force_train_ids.difference(set(forced_train_hits))))
    print(f"\nForced train IDs requested: {sorted(force_train_ids)}")
    print(f"Found & forced into train:   {sorted(set(forced_train_hits))}")
    if missing_forced:
        print(f"WARNING: These forced IDs were not found in dataset: {missing_forced}")

    # Print distributions
    train_counts = count_by_label(train)
    test_counts = count_by_label(test)
    print_dist("TRAIN label distribution", train_counts)
    print_dist("TEST label distribution", test_counts)

    # Save
    Path(args.train_out).write_text(json.dumps(train, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(args.test_out).write_text(json.dumps(test, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved train to: {args.train_out}")
    print(f"Saved test  to: {args.test_out}")

if __name__ == "__main__":
    main()
