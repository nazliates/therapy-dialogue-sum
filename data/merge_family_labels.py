import json
from collections import Counter
from pathlib import Path

# ---- EDIT THESE FILENAMES ----
TRAIN_IN = "en_data_cleaned_train.json"
TEST_IN  = "en_data_cleaned_test.json"

TRAIN_OUT = "en_data_cleaned_train_final.json"
TEST_OUT  = "en_data_cleaned_test_final.json"

OLD = "Marriage and Family Systems Therapy"
NEW = "Family Therapy"

def merge_family_label(item: dict) -> bool:
    """
    Returns True if a change was made.
    Updates either item['psychotherapy'] or item['label'] if present.
    """
    changed = False

    if "psychotherapy" in item and item["psychotherapy"] == OLD:
        item["psychotherapy"] = NEW
        changed = True

    if "label" in item and item["label"] == OLD:
        item["label"] = NEW
        changed = True

    return changed

def get_label(item: dict) -> str:
    """Best-effort label getter for reporting."""
    if "psychotherapy" in item:
        return item.get("psychotherapy")
    if "label" in item:
        return item.get("label")
    return None

def process_file(in_path: str, out_path: str):
    in_path = Path(in_path)
    out_path = Path(out_path)

    with in_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    before = Counter(get_label(x) for x in data if get_label(x) is not None)

    changes = 0
    for item in data:
        if merge_family_label(item):
            changes += 1

    after = Counter(get_label(x) for x in data if get_label(x) is not None)

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"\n=== {in_path.name} -> {out_path.name} ===")
    print(f"Total items: {len(data)}")
    print(f"Changed labels: {changes}")
    print(f"Before: {OLD}={before.get(OLD, 0)}, {NEW}={before.get(NEW, 0)}")
    print(f"After : {OLD}={after.get(OLD, 0)}, {NEW}={after.get(NEW, 0)}")

if __name__ == "__main__":
    process_file(TRAIN_IN, TRAIN_OUT)
    process_file(TEST_IN, TEST_OUT)
