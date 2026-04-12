import json
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import torch
import nltk
from nltk.tokenize import sent_tokenize

from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForSequenceClassification

nltk.download("punkt")

REFERENCE_FILE = "en_data_cleaned_test_final.json"
GENERATED_DIR = "results_clean_final/gptossnew"
OUTPUT_DIR = "sts_entailment_outputs_deberta"

STS_MODEL_NAME = "sentence-transformers/stsb-roberta-base"
NLI_MODEL_NAME = "potsawee/deberta-v3-large-mnli"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ENTAILMENT_THRESHOLD = 0.5


# === CHANGED === supports both .json and .jsonl
def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        if path.endswith(".jsonl"):
            return [json.loads(line) for line in f if line.strip()]
        else:
            return json.load(f)


def save_json(path: str, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def split_sentences(text: str) -> List[str]:
    if not text or not text.strip():
        return []
    return [s.strip() for s in sent_tokenize(text) if s.strip()]


def build_reference_map(reference_data: List[Dict[str, Any]]) -> Dict[str, str]:
    ref_map = {}
    for item in reference_data:
        dialog_id = item.get("dialog_id")
        summary = item.get("summary", "")
        if dialog_id is not None:
            ref_map[dialog_id] = summary
    return ref_map


def supported_ratio(scores: List[float], threshold: float = 0.5) -> float:
    if not scores:
        return None
    return float(sum(score >= threshold for score in scores) / len(scores))


# === NEW === helper for DeBERTa entailment
def compute_deberta_probs(
    pairs: List[tuple[str, str]],
    tokenizer: AutoTokenizer,
    model: AutoModelForSequenceClassification,
    device: str,
) -> np.ndarray:
    """
    pairs: list of (hypothesis, premise)

    For potsawee/deberta-v3-large-mnli:
    index 0 = entailment
    index 1 = contradiction
    There is no neutral output head.
    """
    if not pairs:
        return np.empty((0, 3))

    inputs = tokenizer.batch_encode_plus(
        pairs,
        return_tensors="pt",
        truncation=True,
        padding=True,
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits

    probs = torch.softmax(logits, dim=-1).cpu().numpy()
    return probs


def evaluate_summary_pair(
    reference_summary: str,
    generated_summary: str,
    sts_model: SentenceTransformer,
    nli_tokenizer: AutoTokenizer,                   
    nli_model: AutoModelForSequenceClassification,  
) -> Dict[str, Any]:
    ref_sents = split_sentences(reference_summary)
    gen_sents = split_sentences(generated_summary)

    if len(ref_sents) == 0 or len(gen_sents) == 0:
        return {
            "reference_sentences": ref_sents,
            "generated_sentences": gen_sents,
            "alignments_gen_to_ref": [],
            "alignments_ref_to_gen": [],
            "avg_max_similarity_gen_to_ref": None,
            "avg_entailment_gen_to_ref": None,
            "supported_ratio_gen_to_ref": None,
            "avg_max_similarity_ref_to_gen": None,
            "avg_entailment_ref_to_gen": None,
            "supported_ratio_ref_to_gen": None,
        }

    ref_emb = sts_model.encode(ref_sents, convert_to_tensor=True, device=DEVICE)
    gen_emb = sts_model.encode(gen_sents, convert_to_tensor=True, device=DEVICE)

    sim_matrix = util.cos_sim(gen_emb, ref_emb).cpu().numpy()

    # -----------------------------------------------------
    # Direction 1: generated -> reference
    # -----------------------------------------------------
    alignments_gen_to_ref = []
    gen_to_ref_sims = []
    gen_to_ref_entailments = []

    nli_pairs_gen_to_ref = []
    best_ref_indices = []

    for i, gen_sent in enumerate(gen_sents):
        best_ref_idx = int(np.argmax(sim_matrix[i]))
        best_ref_sent = ref_sents[best_ref_idx]
        best_sim = float(sim_matrix[i][best_ref_idx])

        best_ref_indices.append(best_ref_idx)
        gen_to_ref_sims.append(best_sim)

        # premise = reference, hypothesis = generated
        nli_pairs_gen_to_ref.append((gen_sent, best_ref_sent))

    # === CHANGED === DeBERTa inference instead of CrossEncoder.predict
    nli_probs = compute_deberta_probs(
        nli_pairs_gen_to_ref,
        nli_tokenizer,
        nli_model,
        DEVICE,
    )

    # === CHANGED === label order for this DeBERTa model
    ENTAILMENT_INDEX = 0
    CONTRADICTION_INDEX = 1

    for i, gen_sent in enumerate(gen_sents):
        probs = nli_probs[i]
        ref_idx = best_ref_indices[i]

        entailment_prob = float(probs[ENTAILMENT_INDEX])
        contradiction_prob = float(probs[CONTRADICTION_INDEX])
        neutral_prob = None
        gen_to_ref_entailments.append(entailment_prob)

        alignments_gen_to_ref.append({
            "generated_sentence": gen_sent,
            "matched_reference_sentence": ref_sents[ref_idx],
            "sts_similarity": gen_to_ref_sims[i],
            "entailment_probability": entailment_prob,
            "neutral_probability": neutral_prob,
            "contradiction_probability": contradiction_prob,
        })

    # -----------------------------------------------------
    # Direction 2: reference -> generated
    # -----------------------------------------------------
    alignments_ref_to_gen = []
    ref_to_gen_sims = []
    ref_to_gen_entailments = []

    sim_matrix_rev = sim_matrix.T
    nli_pairs_ref_to_gen = []
    best_gen_indices = []

    for i, ref_sent in enumerate(ref_sents):
        best_gen_idx = int(np.argmax(sim_matrix_rev[i]))
        best_gen_sent = gen_sents[best_gen_idx]
        best_sim = float(sim_matrix_rev[i][best_gen_idx])

        best_gen_indices.append(best_gen_idx)
        ref_to_gen_sims.append(best_sim)

        # premise = generated, hypothesis = reference
        nli_pairs_ref_to_gen.append((ref_sent, best_gen_sent))

    # === CHANGED === DeBERTa inference instead of CrossEncoder.predict
    nli_probs_rev = compute_deberta_probs(
        nli_pairs_ref_to_gen,
        nli_tokenizer,
        nli_model,
        DEVICE,
    )

    for i, ref_sent in enumerate(ref_sents):
        probs = nli_probs_rev[i]
        gen_idx = best_gen_indices[i]

        entailment_prob = float(probs[ENTAILMENT_INDEX])
        contradiction_prob = float(probs[CONTRADICTION_INDEX])
        neutral_prob = None

        ref_to_gen_entailments.append(entailment_prob)

        alignments_ref_to_gen.append({
            "reference_sentence": ref_sent,
            "matched_generated_sentence": gen_sents[gen_idx],
            "sts_similarity": ref_to_gen_sims[i],
            "entailment_probability": entailment_prob,
            "neutral_probability": neutral_prob,
            "contradiction_probability": contradiction_prob,
        })

    return {
        "reference_sentences": ref_sents,
        "generated_sentences": gen_sents,
        "alignments_gen_to_ref": alignments_gen_to_ref,
        "alignments_ref_to_gen": alignments_ref_to_gen,
        "avg_max_similarity_gen_to_ref": float(np.mean(gen_to_ref_sims)),
        "avg_entailment_gen_to_ref": float(np.mean(gen_to_ref_entailments)),
        "supported_ratio_gen_to_ref": supported_ratio(gen_to_ref_entailments, ENTAILMENT_THRESHOLD),
        "avg_max_similarity_ref_to_gen": float(np.mean(ref_to_gen_sims)),
        "avg_entailment_ref_to_gen": float(np.mean(ref_to_gen_entailments)),
        "supported_ratio_ref_to_gen": supported_ratio(ref_to_gen_entailments, ENTAILMENT_THRESHOLD),
    }


def evaluate_generated_file(
    generated_file: Path,
    ref_map: Dict[str, str],
    sts_model: SentenceTransformer,
    nli_tokenizer: AutoTokenizer,                  
    nli_model: AutoModelForSequenceClassification, 
) -> Dict[str, Any]:
    generated_data = load_json(str(generated_file))

    results = []
    missing_ids = []

    for item in generated_data:
        dialog_id = item.get("dialog_id")
        # === CHANGED === supports both fine-tuned and old files
        generated_summary = item.get("pred_summary", item.get("summary", ""))

        if dialog_id not in ref_map:
            missing_ids.append(dialog_id)
            continue

        reference_summary = ref_map[dialog_id]

        metrics = evaluate_summary_pair(
            reference_summary=reference_summary,
            generated_summary=generated_summary,
            sts_model=sts_model,
            nli_tokenizer=nli_tokenizer, 
            nli_model=nli_model,         
        )

        results.append({
            "dialog_id": dialog_id,
            "reference_summary": reference_summary,
            "generated_summary": generated_summary,
            **metrics
        })

    valid_g2r_sim = [r["avg_max_similarity_gen_to_ref"] for r in results if r["avg_max_similarity_gen_to_ref"] is not None]
    valid_g2r_ent = [r["avg_entailment_gen_to_ref"] for r in results if r["avg_entailment_gen_to_ref"] is not None]
    valid_r2g_sim = [r["avg_max_similarity_ref_to_gen"] for r in results if r["avg_max_similarity_ref_to_gen"] is not None]
    valid_r2g_ent = [r["avg_entailment_ref_to_gen"] for r in results if r["avg_entailment_ref_to_gen"] is not None]
    valid_g2r_sup = [r["supported_ratio_gen_to_ref"] for r in results if r["supported_ratio_gen_to_ref"] is not None]
    valid_r2g_sup = [r["supported_ratio_ref_to_gen"] for r in results if r["supported_ratio_ref_to_gen"] is not None]

    summary = {
        "file_name": generated_file.name,
        "num_samples_evaluated": len(results),
        "num_missing_dialog_ids": len(missing_ids),
        "missing_dialog_ids": missing_ids,
        "dataset_metrics": {
            "avg_max_similarity_gen_to_ref": float(np.mean(valid_g2r_sim)) if valid_g2r_sim else None,
            "avg_entailment_gen_to_ref": float(np.mean(valid_g2r_ent)) if valid_g2r_ent else None,
            "avg_supported_ratio_gen_to_ref": float(np.mean(valid_g2r_sup)) if valid_g2r_sup else None,
            "avg_max_similarity_ref_to_gen": float(np.mean(valid_r2g_sim)) if valid_r2g_sim else None,
            "avg_entailment_ref_to_gen": float(np.mean(valid_r2g_ent)) if valid_r2g_ent else None,
            "avg_supported_ratio_ref_to_gen": float(np.mean(valid_r2g_sup)) if valid_r2g_sup else None,
        },
        "sample_results": results,
    }

    return summary


def main():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    print(f"Using device: {DEVICE}")
    print(f"Loading reference file: {REFERENCE_FILE}")
    reference_data = load_json(REFERENCE_FILE)
    ref_map = build_reference_map(reference_data)
    print(f"Loaded {len(ref_map)} reference summaries")

    print(f"Loading STS model: {STS_MODEL_NAME}")
    sts_model = SentenceTransformer(STS_MODEL_NAME, device=DEVICE)

    print(f"Loading NLI model: {NLI_MODEL_NAME}")
    nli_tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_NAME, use_fast=False)  
    nli_model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_NAME) 
    nli_model.to(DEVICE)
    nli_model.eval()

    # === CHANGED === proper debug test for DeBERTa
    test_pairs = [
        ("A man is eating food.", "A person is eating."),
        ("A man is eating food.", "Nobody is eating."),
        ("The therapist asks the client to rate progress.", "The therapist uses a scaling question.")
    ]

    test_probs = compute_deberta_probs(test_pairs, nli_tokenizer, nli_model, DEVICE)

    print("\n=== NLI DEBUG TEST (DeBERTa) ===")
    for pair, p in zip(test_pairs, test_probs):
        print("\nPremise:", pair[0])
        print("Hypothesis:", pair[1])
        print("Probabilities [entailment, neutral, contradiction]:", p)

    # === CHANGED === supports both .json and .jsonl
    generated_files = sorted(Path(GENERATED_DIR).rglob("*"))
    generated_files = [p for p in generated_files if p.suffix in [".json", ".jsonl"]]
    print(f"Found {len(generated_files)} generated summary files")

    all_file_summaries = []

    for generated_file in generated_files:
        print(f"\nEvaluating: {generated_file.name}")
        file_result = evaluate_generated_file(
            generated_file=generated_file,
            ref_map=ref_map,
            sts_model=sts_model,
            nli_tokenizer=nli_tokenizer,  
            nli_model=nli_model,          
        )

        output_path = Path(OUTPUT_DIR) / f"{generated_file.stem}_sts_entailment.json"
        save_json(str(output_path), file_result)
        print(f"Saved detailed results to: {output_path}")

        metrics = file_result["dataset_metrics"]
        print("Dataset metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")

        all_file_summaries.append({
            "file_name": file_result["file_name"],
            "num_samples_evaluated": file_result["num_samples_evaluated"],
            **file_result["dataset_metrics"]
        })

    summary_path = Path(OUTPUT_DIR) / "all_files_summary.json"
    save_json(str(summary_path), all_file_summaries)
    print(f"\nSaved cross-file summary to: {summary_path}")


if __name__ == "__main__":
    main()
