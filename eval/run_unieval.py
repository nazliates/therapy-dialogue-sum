import os
import sys
import argparse
from glob import glob

UNIEVAL_REPO = "/Users/anazliates/Desktop/final_prompting_experiments/UniEval"  

sys.path.insert(0, UNIEVAL_REPO)

from metric.scorer import UniEvaluator
from utils import add_question, print_scores

import json  
from pathlib import Path  
from typing import Dict, Any, List 

import numpy as np
from nltk import sent_tokenize

sys.path.append("..")
from utils import add_question, print_scores


# >>> ADDED: Helper to format PsyInsight dialogue turns into a single source string
def format_turns_as_source(turns: List[Dict[str, Any]]) -> str:
    lines = []
    for t in turns or []:
        speaker = (t.get("speaker") or t.get("participant") or "Speaker").strip()
        text = (t.get("text") or t.get("content") or "").strip()
        if text:
            lines.append(f"{speaker}: {text}")
    return "\n".join(lines).strip()


# >>> ADDED: Load PsyInsight reference file (en_data_cleaned_test_final.json)
def load_psyinsight_references(psy_path: str) -> Dict[str, Dict[str, str]]:
    """
    Returns: dialog_id -> {"source": ..., "reference": ...}
    """
    items = json.loads(Path(psy_path).read_text(encoding="utf-8"))
    by_id: Dict[str, Dict[str, str]] = {}

    for ex in items:
        did = str(ex.get("dialog_id") or ex.get("id") or ex.get("dialogue_id") or "")
        if not did:
            continue

        turns = ex.get("turns")
        if turns is None:
            turns = ex.get("dialog")  # fallback for other formats

        source = format_turns_as_source(turns or [])
        reference = (ex.get("summary") or "").strip()

        by_id[did] = {"source": source, "reference": reference}

    return by_id


# >>> ADDED: Load fine-tuned model outputs (JSONL)
def load_model_outputs(preds_path: str) -> Dict[str, str]:
    """
    Returns: dialog_id -> predicted summary
    Supports:
    - JSONL: one object per line
    - JSON:  list of objects
    """
    preds: Dict[str, str] = {}

    path = Path(preds_path)
    text = path.read_text(encoding="utf-8").strip()

    if path.suffix.lower() == ".jsonl":
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            did = str(obj.get("dialog_id") or obj.get("id") or obj.get("dialogue_id") or "")
            if not did:
                continue
            pred = (obj.get("pred_summary") or obj.get("summary") or obj.get("prediction") or "").strip()
            if pred:
                preds[did] = pred

    elif path.suffix.lower() == ".json":
        items = json.loads(text)
        for obj in items:
            did = str(obj.get("dialog_id") or obj.get("id") or obj.get("dialogue_id") or "")
            if not did:
                continue
            pred = (obj.get("pred_summary") or obj.get("summary") or obj.get("prediction") or "").strip()
            if pred:
                preds[did] = pred

    else:
        raise ValueError(f"Unsupported file type: {preds_path}")

    return preds


# >>> ADDED: Build UniEval input list aligned by dialog_id
def build_unieval_data_from_psyinsight(
    psy_path: str,
    preds_path: str,
    skip_missing: bool = True
) -> List[Dict[str, str]]:
    """
    UniEval expects list of dicts: {source, system_output, reference}
    We also include dialog_id so you can map scores back.
    """
    refs_by_id = load_psyinsight_references(psy_path)
    preds_by_id = load_model_outputs(preds_path)

    data: List[Dict[str, str]] = []
    missing_preds = 0
    missing_refs = 0

    for did, ref_obj in refs_by_id.items():
        pred = preds_by_id.get(did)
        if not pred:
            missing_preds += 1
            if skip_missing:
                continue

        reference = ref_obj.get("reference", "")
        if not reference:
            missing_refs += 1
            if skip_missing:
                continue

        data.append({
            "dialog_id": did,                    # >>> ADDED
            "source": ref_obj["source"],
            "system_output": pred,
            "reference": reference
        })

    print(f"[UniEval] Built {len(data)} aligned samples.")
    if missing_preds:
        print(f"[UniEval] Missing predictions for {missing_preds} dialog_ids.")
    if missing_refs:
        print(f"[UniEval] Missing references for {missing_refs} dialog_ids.")

    return data


class SumEvaluator:
    def __init__(self, max_length=1024, device='cuda:0', cache_dir=None):
        """ Set up evaluator for text summarization """
        self.scorer = UniEvaluator(model_name_or_path='MingZhong/unieval-sum',
                                   max_length=max_length,
                                   device=device, cache_dir=cache_dir)
        self.task = 'summarization'
        self.dimensions = ['coherence', 'consistency', 'fluency', 'relevance']

    def evaluate(self, data, dims=None, overall=True, print_result=False):
        """
            Get the scores of all the given dimensions

            dims: A list of dimensions to be evaluated. If dims is None, SumEvaluator will evaluate
                  four dimensions: coherence, consistency, fluency, relevance.

            overall: indicates whether the overall score is to be calculated.
                     Overall score can be customized to a combination of scores based on different
                     dimensions. The default here is the average score of all the given dimensions.

            print_result: whether to print the average score of each dimension on the screen
        """
        n_data = len(data)
        eval_scores = [{} for _ in range(n_data)]

        # >>> CHANGED: carry dialog_id through into outputs (if present)
        for i in range(n_data):
            if "dialog_id" in data[i]:
                eval_scores[i]["dialog_id"] = data[i]["dialog_id"]

        if dims == None:
            eval_dims = self.dimensions
        else:
            assert isinstance(dims, list)
            eval_dims = dims

        for dim in eval_dims:
            print('Evaluating {} of {} samples !!!'.format(dim, n_data))

            # Calculate average sentence-level scores for 'consistency' and 'fluency'
            if dim == 'consistency' or dim == 'fluency':
                src_list, output_list = [], []
                n_sents = []  # the number of sentences in each generated summary
                for i in range(n_data):
                    if dim == 'consistency':
                        source = data[i]['source']
                    else:
                        source = ''
                    system_outputs = sent_tokenize(data[i]['system_output'])
                    n_sents.append(len(system_outputs))

                    # >>> CHANGED: guard against empty summaries (avoid division by 0)
                    if len(system_outputs) == 0:
                        # Put a dummy sentence so scorer returns something;
                        # we will handle it as 0.0 later.
                        system_outputs = [""]

                    for j in range(len(system_outputs)):
                        src_list.append(source)
                        output_list.append(system_outputs[j])

                input_list = add_question(dimension=dim, output=output_list,
                                          src=src_list, task=self.task)
                sent_score = self.scorer.score(input_list)

                # Get average score for each sample
                start_idx = 0
                score = []
                for cur_n_sent in n_sents:
                    # >>> CHANGED: if original summary had 0 sentences, set score=0.0
                    if cur_n_sent == 0:
                        score.append(0.0)
                        # we added 1 dummy sentence, so advance by 1
                        start_idx += 1
                    else:
                        score.append(sum(sent_score[start_idx: start_idx + cur_n_sent]) / cur_n_sent)
                        start_idx += cur_n_sent

            # Calculate summary-level score for 'coherence' and 'relevance'
            elif dim == 'coherence' or dim == 'relevance':
                src_list, output_list, ref_list = [], [], []
                for i in range(n_data):
                    src_list.append(data[i]['source'])
                    output_list.append(data[i]['system_output'])
                    if dim == 'relevance':
                        ref_list.append(data[i]['reference'])
                input_list = add_question(dimension=dim, output=output_list,
                                          src=src_list, ref=ref_list, task=self.task)
                score = self.scorer.score(input_list)

            # Please customize other dimensions here for summarization
            else:
                raise NotImplementedError('The input format for this dimension is still undefined. \
                                           Please customize it first.')

            for i in range(n_data):
                eval_scores[i][dim] = score[i]

        # Customize your overall score here.
        if overall == True:
            for i in range(n_data):
                # >>> CHANGED: exclude dialog_id from overall computation if present
                vals = [v for k, v in eval_scores[i].items() if k != "dialog_id"]
                eval_scores[i]['overall'] = np.mean(vals)

        if print_result == True:
            print_scores(eval_scores)

        return eval_scores


class DialogEvaluator:
    def __init__(self, max_length=1024, device='cuda:0', cache_dir=None):
        """ Set up evaluator for dialogues """
        self.scorer = UniEvaluator(model_name_or_path='MingZhong/unieval-dialog',
                                   max_length=max_length,
                                   device=device, cache_dir=cache_dir)
        self.task = 'dialogue'
        self.dimensions = ['naturalness', 'coherence', 'engagingness',
                           'groundedness', 'understandability']

    def evaluate(self, data, dims=None, overall=True, print_result=False):
        n_data = len(data)
        eval_scores = [{} for _ in range(n_data)]

        if dims == None:
            eval_dims = self.dimensions
        else:
            assert isinstance(dims, list)
            eval_dims = dims

        for dim in eval_dims:
            print('Evaluating {} of {} samples !!!'.format(dim, n_data))

            if dim == 'engagingness':
                src_list, output_list, context_list = [], [], []
                n_sents = []
                for i in range(n_data):
                    source = data[i]['source']
                    context = data[i]['context']
                    system_outputs = sent_tokenize(data[i]['system_output'])
                    n_sents.append(len(system_outputs))
                    for j in range(len(system_outputs)):
                        src_list.append(source)
                        context_list.append(context)
                        output_list.append(system_outputs[j])
                input_list = add_question(dimension=dim, output=output_list,
                                          src=src_list, context=context_list, task=self.task)
                sent_score = self.scorer.score(input_list)

                start_idx = 0
                score = []
                for cur_n_sent in n_sents:
                    score.append(sum(sent_score[start_idx: start_idx + cur_n_sent]))
                    start_idx += cur_n_sent

            elif dim in ['naturalness', 'coherence', 'groundedness', 'understandability']:
                src_list, output_list, context_list = [], [], []
                for i in range(n_data):
                    if dim == 'coherence':
                        src_list.append(data[i]['source'])
                    else:
                        src_list.append('')
                    output_list.append(data[i]['system_output'])
                    if dim == 'groundedness':
                        context_list.append(data[i]['context'])
                    else:
                        context_list.append('')
                input_list = add_question(dimension=dim, output=output_list,
                                          src=src_list, context=context_list, task=self.task)
                score = self.scorer.score(input_list)

            else:
                raise NotImplementedError('The input format for this dimension is still undefined. \
                                           Please customize it first.')

            for i in range(n_data):
                eval_scores[i][dim] = score[i]

        if overall == True:
            for i in range(n_data):
                eval_scores[i]['overall'] = np.mean(list(eval_scores[i].values()))

        # >>> CHANGED: print only numeric dimensions (exclude dialog_id)
        if print_result == True:
            numeric_dims = [d for d in eval_dims]
            # print simple averages
            for d in numeric_dims + (["overall"] if overall else []):
                vals = [s[d] for s in eval_scores if isinstance(s.get(d), (int, float, np.floating))]
                if vals:
                    print(f"{d}: {float(np.mean(vals)):.4f}")

        return eval_scores


class D2tEvaluator:
    def __init__(self, max_length=1024, device='cuda:0', cache_dir=None):
        """ Set up evaluator for data-to-text """
        self.scorer = UniEvaluator(model_name_or_path='MingZhong/unieval-sum',
                                   max_length=max_length,
                                   device=device, cache_dir=cache_dir)
        self.task = 'data2text'
        self.dimensions = ['naturalness', 'informativeness']

    def evaluate(self, data, dims=None, overall=True, print_result=False):
        n_data = len(data)
        eval_scores = [{} for _ in range(n_data)]

        if dims == None:
            eval_dims = self.dimensions
        else:
            assert isinstance(dims, list)
            eval_dims = dims

        for dim in eval_dims:
            print('Evaluating {} of {} samples !!!'.format(dim, n_data))

            output_list, ref_list = [], []
            for i in range(n_data):
                output_list.append(data[i]['system_output'])
                ref_list.append(data[i]['reference'])

            input_list = add_question(dimension=dim, output=output_list,
                                      ref=ref_list, task=self.task)
            score = self.scorer.score(input_list)

            for i in range(n_data):
                eval_scores[i][dim] = score[i]

        if overall == True:
            for i in range(n_data):
                eval_scores[i]['overall'] = np.mean(list(eval_scores[i].values()))

        if print_result == True:
            print_scores(eval_scores)

        return eval_scores


class FactEvaluator:
    def __init__(self, max_length=1024, device='cuda:0', cache_dir=None):
        """ Set up evaluator for factual consistency detection """
        self.scorer = UniEvaluator(model_name_or_path='MingZhong/unieval-fact',
                                   max_length=max_length,
                                   device=device, cache_dir=cache_dir)
        self.task = 'fact'
        self.dim = 'consistency'

    def evaluate(self, data, print_result=False):
        n_data = len(data)
        eval_scores = [{} for _ in range(n_data)]

        print('Evaluating {} of {} samples !!!'.format(self.dim, n_data))

        src_list, output_list = [], []
        n_sents = []
        for i in range(n_data):
            source = data[i]['source']
            system_outputs = sent_tokenize(data[i]['system_output'])
            n_sents.append(len(system_outputs))
            for j in range(len(system_outputs)):
                src_list.append(source)
                output_list.append(system_outputs[j])
        input_list = add_question(dimension=self.dim, output=output_list,
                                  src=src_list, task=self.task)
        sent_score = self.scorer.score(input_list)

        start_idx = 0
        score = []
        for cur_n_sent in n_sents:
            score.append(sum(sent_score[start_idx: start_idx + cur_n_sent]) / cur_n_sent)
            start_idx += cur_n_sent

        for i in range(n_data):
            eval_scores[i][self.dim] = score[i]

        if print_result == True:
            print_scores(eval_scores)

        return eval_scores


def get_evaluator(task, max_length=1024, device='cuda:0', cache_dir=None):
    assert task in ['summarization', 'dialogue', 'data2text', 'fact']
    if task == 'summarization':
        return SumEvaluator(max_length=max_length,
                            device=device,
                            cache_dir=cache_dir)
    elif task == 'dialogue':
        return DialogEvaluator(max_length=max_length,
                               device=device,
                               cache_dir=cache_dir)
    elif task == 'data2text':
        return D2tEvaluator(max_length=max_length,
                            device=device,
                            cache_dir=cache_dir)
    elif task == 'fact':
        return FactEvaluator(max_length=max_length,
                             device=device,
                             cache_dir=cache_dir)
    else:
        raise NotImplementedError('Other tasks are not implemented, \
                                   please customize specific tasks here.')

def mean(xs):
    vals = []
    for x in xs:
        try:
            vals.append(float(x))
        except (TypeError, ValueError):
            pass
    return sum(vals) / len(vals) if vals else 0.0


def write_unieval_txt_report(out_path: str, pred_file: str, aligned_n: int, scores: list):
    dims = ["coherence", "consistency", "fluency", "relevance", "overall"]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("=" * 68 + "\n")
        f.write(f"FILE: {pred_file}\n")
        f.write("=" * 68 + "\n\n")
        f.write(f"Evaluated pairs: {aligned_n}\n\n")

        if aligned_n == 0:
            f.write("No pairs to evaluate.\n")
            return

        f.write("UniEval (mean over evaluated pairs):\n")
        for d in dims:
            vals = [s.get(d) for s in scores]
            f.write(f"  {d}: {mean(vals):.4f}\n")
            
# >>> ADDED: Example entry-point to run PsyInsight evaluation (optional)
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="Reference JSON, e.g. en_data_cleaned_test_final.json")
    ap.add_argument("--pred_dir", required=True, help="Directory with prediction JSON/JSONL files")
    ap.add_argument("--pattern", default="*.json", help="Glob pattern, e.g. *.json or *.jsonl")
    ap.add_argument("--out_dir", default="eval/unieval_scores", help="Output directory")
    ap.add_argument("--device", default="cpu", help="cpu, cuda:0, or mps")
    ap.add_argument("--max_length", type=int, default=1024)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    pred_files = sorted(glob(os.path.join(args.pred_dir, "**", args.pattern), recursive=True))
    if not pred_files:
        raise SystemExit(f"No files found under {args.pred_dir} with pattern {args.pattern}")

    evaluator = get_evaluator("summarization", device=args.device, max_length=args.max_length)

    for pred_path in pred_files:
        if not (pred_path.lower().endswith(".json") or pred_path.lower().endswith(".jsonl")):
            continue

        base = os.path.basename(pred_path)
        stem = Path(base).stem

        try:
            data = build_unieval_data_from_psyinsight(args.ref, pred_path, skip_missing=True)
            scores = evaluator.evaluate(data, dims=None, overall=True, print_result=False)
        except json.JSONDecodeError:
            print(f"[SKIP] Not valid JSON/JSONL: {pred_path}")
            continue
        except Exception as e:
            print(f"[SKIP] Error reading {pred_path}: {e}")
            continue

        out_jsonl = os.path.join(args.out_dir, f"{stem}_unieval.jsonl")
        with open(out_jsonl, "w", encoding="utf-8") as f:
            for s in scores:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

        out_txt = os.path.join(args.out_dir, f"{stem}_unieval_report.txt")
        write_unieval_txt_report(out_txt, base, len(scores), scores)

        print(f"[OK] Wrote {out_jsonl}")
        print(f"[OK] Wrote {out_txt}")