import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests


def load_text(path: Path) -> str:
    with path.open("r", encoding="utf-8") as f:
        return f.read()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def build_prompt(template: str, item: Dict[str, Any]) -> str:
    summaries = item["shuffled_summaries"]
    return template.format(
        dialogue=item["dialogue"],
        summary_1=summaries["S1"],
        summary_2=summaries["S2"],
        summary_3=summaries["S3"],
        summary_4=summaries["S4"],
        summary_5=summaries["S5"],
        summary_6=summaries["S6"],
    )


def extract_json_block(text: str) -> Optional[str]:
    """
    Robustly extract the first valid top-level JSON object from model output.
    Handles: direct JSON, fenced code blocks, JSON embedded in prose/thinking text.
    Uses depth-aware scanning so nested braces in surrounding prose don't confuse it.
    """
    text = text.strip()

    # 1. Direct parse
    try:
        json.loads(text)
        return text
    except Exception:
        pass

    # 2. Fenced code block ```json ... ``` or ``` ... ```
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fence_match:
        candidate = fence_match.group(1).strip()
        try:
            json.loads(candidate)
            return candidate
        except Exception:
            pass

    # 3. Depth-aware scan: collect ALL top-level {...} blocks, try each
    candidates = []
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                candidates.append(text[start:i + 1])
                start = None

    for candidate in candidates:
        try:
            json.loads(candidate)
            return candidate
        except Exception:
            continue

    return None


def validate_judge_output(obj: Dict[str, Any]) -> None:
    required_top = {"best_id", "second_best_id"}
    missing_top = required_top - set(obj.keys())
    if missing_top:
        raise ValueError(f"Missing top-level fields: {sorted(missing_top)}")

    valid_ids = {"S1", "S2", "S3", "S4", "S5", "S6"}

    if obj["best_id"] not in valid_ids:
        raise ValueError(f"Invalid best_id: {obj['best_id']}")
    if obj["second_best_id"] not in valid_ids:
        raise ValueError(f"Invalid second_best_id: {obj['second_best_id']}")

    if obj["best_id"] == obj["second_best_id"]:
        raise ValueError("best_id and second_best_id must be different")
    expected_ids = {"S1", "S2", "S3", "S4", "S5", "S6"}

def call_ollama(
    prompt: str,
    model: str,
    host: str,
    temperature: float,
    num_predict: int,
    timeout: int,
):
    url = host.rstrip("/") + "/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "think": "low", 
        "options": {
            "temperature": temperature,
            "num_predict": num_predict,
            "stop": ["\n\n```", "\n\nNote:", "\n\nExplanation:"]
        },
    }
    response = requests.post(url, json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()


def main():
    parser = argparse.ArgumentParser(description="Run LLM judge over shuffled summary inputs via Ollama.")
    parser.add_argument("--input", type=Path, required=True, help="Path to judge_input_shuffled.json")
    parser.add_argument("--prompt_file", type=Path, required=True, help="Path to judge prompt .md file")
    parser.add_argument("--output", type=Path, required=True, help="Path to save full judge results JSON")
    parser.add_argument("--model", type=str, default="gpt-oss:20b", help="Ollama model name")
    parser.add_argument("--host", type=str, default="http://localhost:11434", help="Ollama host")
    parser.add_argument("--temperature", type=float, default=0.1, help="Generation temperature")
    parser.add_argument("--num_predict", type=int, default=4000, help="Max generated tokens (use 4000+ for thinking models)")
    parser.add_argument("--timeout", type=int, default=300, help="Request timeout in seconds")
    parser.add_argument("--sleep", type=float, default=0.0, help="Sleep between requests in seconds")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for testing")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output file if present")
    args = parser.parse_args()

    items: List[Dict[str, Any]] = load_json(args.input)
    prompt_template = load_text(args.prompt_file)

    if args.limit is not None:
        items = items[:args.limit]

    results: List[Dict[str, Any]] = []
    completed_ids = set()

    if args.resume and args.output.exists():
        existing = load_json(args.output)
        if isinstance(existing, list):
            results = existing
            completed_ids = {str(x["dialog_id"]) for x in existing if "dialog_id" in x}
            print(f"Resuming: found {len(completed_ids)} completed dialogues in existing output.")

    errors: List[Dict[str, Any]] = []

    total = len(items)
    for idx, item in enumerate(items, start=1):
        dialog_id = str(item["dialog_id"])

        if dialog_id in completed_ids:
            print(f"[{idx}/{total}] Skipping {dialog_id} (already done)")
            continue

        print(f"[{idx}/{total}] Processing {dialog_id}")

        prompt = build_prompt(prompt_template, item)

        raw_response = None
        parsed_response = None
        parse_error = None
        ollama_data = None
        used_thinking_fallback = False

        try:
            ollama_data = call_ollama(
                prompt=prompt,
                model=args.model,
                host=args.host,
                temperature=args.temperature,
                num_predict=args.num_predict,
                timeout=args.timeout,
            )

            raw_response = ollama_data.get("response", "")
            thinking = ollama_data.get("thinking", "")

            print(f"  done: {ollama_data.get('done')}")
            print(f"  done_reason: {ollama_data.get('done_reason')}")
            print(f"  eval_count: {ollama_data.get('eval_count')}")
            print(f"  response length: {len(raw_response)}")
            print(f"  thinking length: {len(thinking)}")

            # Thinking models put reasoning in `thinking` and final answer in `response`.
            # If response is empty (all tokens used by thinking), fall back to thinking.
            if not raw_response.strip() and thinking.strip():
                print("  WARNING: response empty, falling back to thinking field for JSON extraction")
                raw_response = thinking
                used_thinking_fallback = True

            if not raw_response.strip():
                raise ValueError("Both response and thinking fields are empty.")

            json_block = extract_json_block(raw_response)
            if json_block is None:
                raise ValueError("Could not extract valid JSON from model output.")

            parsed_response = json.loads(json_block)
            validate_judge_output(parsed_response)

            result_item = {
                "dialog_id": dialog_id,
                "mapping": item["mapping"],
                "judge_response": parsed_response,
                "raw_response": ollama_data.get("response", ""), 
                "thinking": thinking,
                "used_thinking_fallback": used_thinking_fallback,
            }
            results.append(result_item)
            print(f"  OK (thinking_fallback={used_thinking_fallback})")

        except Exception as e:
            parse_error = str(e)

            prompt_dump_dir = args.output.parent / "failed_prompts"
            prompt_dump_dir.mkdir(parents=True, exist_ok=True)
            (prompt_dump_dir / f"{dialog_id}.txt").write_text(prompt, encoding="utf-8")

            error_item = {
                "dialog_id": dialog_id,
                "mapping": item.get("mapping", {}),
                "error": parse_error,
                "raw_response": ollama_data.get("response", "") if ollama_data else "",
                "thinking": ollama_data.get("thinking", "") if ollama_data else "",
            }
            errors.append(error_item)
            print(f"  ERROR: {parse_error}")

        save_json(results, args.output)

        error_path = args.output.with_name(args.output.stem + "_errors.json")
        save_json(errors, error_path)

        if args.sleep > 0:
            time.sleep(args.sleep)

    print(f"\nDone. Successful: {len(results)} / {total}")
    print(f"Errors: {len(errors)}")
    print(f"Saved results to: {args.output}")
    print(f"Saved errors to: {args.output.with_name(args.output.stem + '_errors.json')}")


if __name__ == "__main__":
    main()