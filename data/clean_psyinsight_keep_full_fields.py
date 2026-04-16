import json
import argparse
from pathlib import Path

def normalize_space(s: str) -> str:
    return " ".join((s or "").strip().split())

def main():
    p = argparse.ArgumentParser(
        description="Filter PsyInsight to dialogues with sufficiently long summaries; keep full fields; remove duplicate dialog_id."
    )
    p.add_argument("--input", default="en_data.json",
                   help="Input PsyInsight JSON (top-level list of dialogue objects).")
    p.add_argument("--output", default="en_data_clean.json",
                   help="Output cleaned JSON (full objects kept).")
    p.add_argument("--min_summary_chars", type=int, default=80,
                   help="Keep only dialogues whose summary length >= this threshold after whitespace normalization.")
    p.add_argument("--require_dialog_id", action="store_true",
                   help="If set, drop items with missing/empty dialog_id. (Recommended)")
    args = p.parse_args()

    data = json.loads(Path(args.input).read_text(encoding="utf-8"))

    seen_ids = set()
    out = []

    dropped_short_summary = 0
    dropped_missing_id = 0
    dropped_duplicates = 0

    for ex in data:
        dialog_id = ex.get("dialog_id")

        if dialog_id is None or str(dialog_id).strip() == "":
            if args.require_dialog_id:
                dropped_missing_id += 1
                continue
            dialog_id_str = None
        else:
            dialog_id_str = str(dialog_id).strip()

        # De-dup by dialog_id (keep first)
        if dialog_id_str is not None:
            if dialog_id_str in seen_ids:
                dropped_duplicates += 1
                continue
            seen_ids.add(dialog_id_str)

        summary = normalize_space(ex.get("summary", ""))
        if len(summary) < args.min_summary_chars:
            dropped_short_summary += 1
            continue

        # Keep the full object unchanged
        out.append(ex)

    Path(args.output).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print("==== Clean dataset created ====")
    print(f"Input dialogues: {len(data)}")
    print(f"Kept dialogues:  {len(out)}")
    print(f"Dropped (summary < {args.min_summary_chars} chars): {dropped_short_summary}")
    if args.require_dialog_id:
        print(f"Dropped (missing/empty dialog_id): {dropped_missing_id}")
    print(f"Dropped (duplicate dialog_id, kept first): {dropped_duplicates}")
    print(f"Saved to: {args.output}")

if __name__ == "__main__":
    main()
