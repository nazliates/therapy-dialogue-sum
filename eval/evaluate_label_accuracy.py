"""
Evaluate psychotherapy label recognition (explicit label mentions) in generated summaries.

- Aligns by dialog_id against PsyInsight reference JSON.
- Detects label mentions using conservative regex patterns (case-insensitive via lowercasing).
- Writes ONE simple .txt report per prediction JSON file.
- No CSV output.
"""

import argparse
import json
import os
import re
from glob import glob
from collections import defaultdict
import unicodedata

LABEL_PATTERNS = {
    "Solution-Focused Brief Therapy": [
    r"solution[-\s]+focused[-\s]+brief[-\s]+therapy",
    r"solution[-\s]+focused[-\s]+therapy",
    r"\bsfbt\b",
    ],
    "Cognitive Behavioral Therapy": [
        r"cognitive behavioral therapy",
        r"cognitive behaviour therapy",
        r"cognitive behavior therapy",
        r"\bcbt\b",
    ],
    "Rational Emotive Behavior Therapy": [
        r"rational emotive behavior therapy",
        r"rational emotive behavioural therapy",
        r"\brebt\b",
    ],
    "Adlerian Therapy": [
        r"adlerian therapy",
        r"individual psychology",
    ],
    "Client-Centered Therapy": [
        r"client[-\s]?centered therapy",
        r"person[-\s]?centered therapy",
    ],
    "Family Therapy": [
        r"family therapy",
    ],
    "Psychoanalytic Therapy": [
        r"psychoanalytic therapy",
        r"psychoanalysis",
    ],
    "Acceptance Commitment Therapy": [
        r"acceptance commitment therapy",
        r"acceptance and commitment therapy",
        r"\bact\b",
    ],
}

# If the reference file uses slightly different strings, they are mapped here.
GOLD_NORMALIZATION = {
    "Cognitive Behavior Therapy": "Cognitive Behavioral Therapy",
    "Cognitive Behaviour Therapy": "Cognitive Behavioral Therapy",
    "Cognitive Behavioural Therapy": "Cognitive Behavioral Therapy",
    "Cognitive Behavioral Therapy": "Cognitive Behavioral Therapy",
}

def normalize(text: str) -> str:
    text = text or ""
    text = unicodedata.normalize("NFKC", text)

    dash_chars = ["\u2010", "\u2011", "\u2012", "\u2013", "\u2014", "\u2015", "\u2212"]
    for d in dash_chars:
        text = text.replace(d, "-")

    text = re.sub(r"\s+", " ", text)
    return text.lower().strip()

def detect_labels(summary: str):
    """Return set of labels mentioned in the summary."""
    found = set()
    text = normalize(summary)
    for label, patterns in LABEL_PATTERNS.items():
        for p in patterns:
            if re.search(p, text):
                found.add(label)
                break
    return found

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_ref_map(ref_path: str):
    refs = load_json(ref_path)
    ref_map = {}
    for d in refs:
        did = d.get("dialog_id")
        gold = d.get("psychotherapy")
        if did is None or gold is None:
            continue

        gold = gold.strip()
        gold = GOLD_NORMALIZATION.get(gold, gold)
        ref_map[did] = gold
    return ref_map

def safe_div(num: int, den: int) -> float:
    return (num / den) if den else 0.0

def sorted_items(d: dict):
    return sorted(d.items(), key=lambda x: (-x[1], x[0]))

def write_txt_report(
    out_path: str,
    pred_file_name: str,
    aligned: int,
    overall_correct: int,
    overall_total: int,
    gold_count: dict,
    predicted_count: dict,
    correct_count: dict,
):
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("=" * 68 + "\n")
        f.write(f"FILE: {pred_file_name}\n")
        f.write("=" * 68 + "\n\n")

        f.write(f"Aligned dialogues: {aligned}\n\n")

        f.write("Overall label recognition accuracy:\n")
        f.write(f"  {overall_correct} / {overall_total} = {safe_div(overall_correct, overall_total):.3f}\n\n")

        f.write("Gold label distribution (from reference, aligned set):\n")
        if gold_count:
            for label, count in sorted_items(gold_count):
                f.write(f"  {label}: {count}\n")
        else:
            f.write("  (none)\n")
        f.write("\n")

        f.write("Model label mentions (any mention in summary; can exceed total if multi-label):\n")
        if predicted_count:
            for label, count in sorted_items(predicted_count):
                f.write(f"  {label}: {count}\n")
        else:
            f.write("  (none)\n")
        f.write("\n")

        f.write("Correct label mentions (gold label mentioned in summary):\n")
        if correct_count:
            for label, count in sorted_items(correct_count):
                f.write(f"  {label}: {count}\n")
        else:
            f.write("  (none)\n")
        f.write("\n")

        f.write("Per-label accuracy (correct / gold):\n")
        for label in LABEL_PATTERNS.keys():
            g = gold_count.get(label, 0)
            c = correct_count.get(label, 0)
            f.write(f"  {label}: {c} / {g} = {safe_div(c, g):.3f}\n")

def eval_one_file(pred_path: str, ref_map: dict):
    preds = load_json(pred_path)

    gold_count = defaultdict(int)       # distribution of gold labels in aligned set
    predicted_count = defaultdict(int)  # how often model mentions each label
    correct_count = defaultdict(int)    # how often gold label is mentioned

    aligned = 0
    overall_total = 0
    overall_correct = 0

    for item in preds:
        dialog_id = item.get("dialog_id")
        if dialog_id is None:
            continue
        if dialog_id not in ref_map:
            continue

        aligned += 1
        gold = ref_map[dialog_id]
        summary = item.get("summary", "") or ""

        predicted_labels = detect_labels(summary)
        if "solution" in summary.lower():
            print("\n--- DEBUG ---")
            print("dialog_id:", dialog_id)
            print("RAW:", repr(summary))
            print("NORMALIZED:", repr(normalize(summary)))
            print("LABELS:", predicted_labels)

        gold_count[gold] += 1
        overall_total += 1

        for lbl in predicted_labels:
            predicted_count[lbl] += 1

        if gold in predicted_labels:
            overall_correct += 1
            correct_count[gold] += 1

    return aligned, overall_correct, overall_total, gold_count, predicted_count, correct_count

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="Reference JSON, e.g. en_data_cleaned_test.json")
    ap.add_argument("--pred_dir", required=True, help="Directory containing prediction JSONs, e.g. results/mistral7b")
    ap.add_argument("--pattern", default="*.json", help="Glob pattern (default: *.json)")
    ap.add_argument("--out_dir", default="results/label_eval", help="Output directory for .txt reports")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    ref_map = build_ref_map(args.ref)

    pred_files = sorted(glob(os.path.join(args.pred_dir, "**", args.pattern), recursive=True))
    if not pred_files:
        raise SystemExit(f"No files found under {args.pred_dir} with pattern {args.pattern}")

    for pred_path in pred_files:
        if not pred_path.lower().endswith(".json"):
            continue

        base = os.path.basename(pred_path)
        txt_name = base.replace(".json", "_label_report.txt")
        out_path = os.path.join(args.out_dir, txt_name)

        try:
            aligned, overall_correct, overall_total, gold_count, predicted_count, correct_count = \
                eval_one_file(pred_path, ref_map)
        except json.JSONDecodeError:
            print(f"[SKIP] Not valid JSON: {pred_path}")
            continue
        except Exception as e:
            print(f"[SKIP] Error reading {pred_path}: {e}")
            continue

        write_txt_report(
            out_path=out_path,
            pred_file_name=base,
            aligned=aligned,
            overall_correct=overall_correct,
            overall_total=overall_total,
            gold_count=gold_count,
            predicted_count=predicted_count,
            correct_count=correct_count,
        )

        print(f"[OK] Wrote {out_path}")

if __name__ == "__main__":
    main()
