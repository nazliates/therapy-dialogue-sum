import argparse
import json
import os
from glob import glob

from rouge_score import rouge_scorer


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def build_ref_map(ref_path: str):
    refs = load_json(ref_path)
    return {d["dialog_id"]: (d.get("summary") or "") for d in refs if "dialog_id" in d}


def align_pairs(pred_path: str, ref_map: dict, min_chars: int):
    preds = load_json(pred_path)

    refs_out, hyps_out, ids_out = [], [], []
    for item in preds:
        did = item.get("dialog_id")
        if did is None or did not in ref_map:
            continue
        hyp = (item.get("summary") or "").strip()
        ref = (ref_map[did] or "").strip()

        if min_chars > 0 and len(hyp) < min_chars:
            continue

        refs_out.append(ref)
        hyps_out.append(hyp)
        ids_out.append(did)

    return ids_out, refs_out, hyps_out


def mean(xs):
    return safe_div(sum(xs), len(xs))


def write_txt_report(out_path: str, pred_file: str, aligned_total: int, used: int, rouge_stats: dict):
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("=" * 68 + "\n")
        f.write(f"FILE: {pred_file}\n")
        f.write("=" * 68 + "\n\n")

        f.write(f"Aligned dialogues (dialog_id overlap): {aligned_total}\n")
        f.write(f"Evaluated pairs (after filters):        {used}\n\n")

        if used == 0:
            f.write("No pairs to evaluate.\n")
            return

        f.write("ROUGE (mean over evaluated pairs):\n")
        for k in ["rouge1", "rouge2", "rougeL", "rougeLsum"]:
            P = rouge_stats[k]["precision"]
            R = rouge_stats[k]["recall"]
            F = rouge_stats[k]["fmeasure"]
            f.write(f"  {k}: P={P:.4f}  R={R:.4f}  F1={F:.4f}\n")


def eval_one_file(pred_path: str, ref_map: dict, use_stemmer: bool, min_chars: int):
    preds = load_json(pred_path)
    aligned_total = sum(1 for it in preds if it.get("dialog_id") in ref_map)

    ids, refs, hyps = align_pairs(pred_path, ref_map, min_chars=min_chars)

    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL", "rougeLsum"],
        use_stemmer=use_stemmer
    )

    agg = {
        "rouge1": {"precision": [], "recall": [], "fmeasure": []},
        "rouge2": {"precision": [], "recall": [], "fmeasure": []},
        "rougeL": {"precision": [], "recall": [], "fmeasure": []},
        "rougeLsum": {"precision": [], "recall": [], "fmeasure": []},
    }

    for ref, hyp in zip(refs, hyps):
        scores = scorer.score(ref, hyp) 
        for k in agg.keys():
            agg[k]["precision"].append(scores[k].precision)
            agg[k]["recall"].append(scores[k].recall)
            agg[k]["fmeasure"].append(scores[k].fmeasure)

    rouge_stats = {
        k: {
            "precision": mean(v["precision"]),
            "recall": mean(v["recall"]),
            "fmeasure": mean(v["fmeasure"]),
        }
        for k, v in agg.items()
    }

    return aligned_total, len(refs), rouge_stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="Reference JSON, e.g. en_data_cleaned_test_final.json")
    ap.add_argument("--pred_dir", required=True, help="Directory with prediction JSONs, e.g. results/llama")
    ap.add_argument("--pattern", default="*.json", help="Glob pattern (default: *.json)")
    ap.add_argument("--out_dir", default="eval/rouge_eval", help="Output directory for .txt reports")
    ap.add_argument("--use_stemmer", action="store_true", help="Use Porter stemmer in ROUGE (recommended)")
    ap.add_argument("--min_chars", type=int, default=0, help="Skip predictions shorter than this many characters (default: 0)")
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
        out_path = os.path.join(args.out_dir, base.replace(".json", "_rouge_report.txt"))

        try:
            aligned_total, used, rouge_stats = eval_one_file(
                pred_path, ref_map, use_stemmer=args.use_stemmer, min_chars=args.min_chars
            )
        except json.JSONDecodeError:
            print(f"[SKIP] Not valid JSON: {pred_path}")
            continue
        except Exception as e:
            print(f"[SKIP] Error reading {pred_path}: {e}")
            continue

        write_txt_report(out_path, base, aligned_total, used, rouge_stats)
        print(f"[OK] Wrote {out_path}")


if __name__ == "__main__":
    main()
