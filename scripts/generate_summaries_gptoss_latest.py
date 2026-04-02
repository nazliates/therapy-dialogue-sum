import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

import torch
import types

if not hasattr(torch, "accelerator"):
    accelerator = types.SimpleNamespace()

    accelerator.is_available = lambda: torch.cuda.is_available()

    accelerator.current_accelerator = lambda check_available=False: (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    accelerator.current_device_index = lambda: (
        torch.cuda.current_device() if torch.cuda.is_available() else 0
    )

    accelerator.device_count = lambda: (
        torch.cuda.device_count() if torch.cuda.is_available() else 0
    )

    def set_device_index(device):
        if not torch.cuda.is_available():
            return
        if isinstance(device, torch.device):
            if device.index is not None:
                torch.cuda.set_device(device.index)
        elif isinstance(device, str):
            if ":" in device:
                torch.cuda.set_device(int(device.split(":")[1]))
            elif device == "cuda":
                torch.cuda.set_device(0)
        else:
            torch.cuda.set_device(device)

    accelerator.set_device_index = set_device_index
    torch.accelerator = accelerator

from transformers import pipeline


def format_dialogue(dialogue):
    lines = []
    for turn in dialogue or []:
        speaker = (turn.get("speaker") or "Client").strip()
        s_low = speaker.lower()
        speaker = "Therapist" if "therapist" in s_low else "Client"
        text = (turn.get("text") or "").strip()
        if text:
            lines.append(f"{speaker}: {text}")
    return "\n".join(lines)


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def build_user_content(dialogue_text: str) -> str:
    return (
        f"Dialogue:\n<<<\n{dialogue_text}\n>>>\n\n"
        "Follow the prompt requirements exactly. "
        "Output only the summary. "
        "The summary must explicitly name exactly one psychotherapy orientation in the paragraph."
    )


def clean_summary(text: str) -> str:
    text = text.strip()

    prefixes = [
        "Summary:",
        "Here is a summary of the therapy session:",
        "Here is the summary:",
        "Therapist-oriented summary:",
        "Final summary:",
    ]
    for p in prefixes:
        if text.lower().startswith(p.lower()):
            text = text[len(p):].strip()

    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="JSON list of dialogue objects")
    parser.add_argument("--prompt", required=True, help="Path to prompt file, e.g. prompts/zero_shot_v1.md")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="openai/gpt-oss-20b")
    parser.add_argument("--limit", type=int, default=91)
    parser.add_argument("--max_new_tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max_input_tokens", type=int, default=4096)
    parser.add_argument("--reasoning_effort", default="low", choices=["low", "medium", "high"])
    args = parser.parse_args()

    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    system_prompt = load_text(Path(args.prompt))

    pipe = pipeline(
        "text-generation",
        model=args.model,
        torch_dtype="auto",
        device_map="auto",
    )

    tok = pipe.tokenizer
    results: List[Dict[str, str]] = []

    for idx, item in enumerate(data[: args.limit], start=1):
        turns = item.get("dialog") or item.get("turns") or item.get("dialogue")
        dialogue_text = format_dialogue(turns)
        user_content = build_user_content(dialogue_text)

        msgs = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        # Apply chat template first (as a string), then inject final channel tag
        # to suppress reasoning leakage — GPT-OSS otherwise exposes chain-of-thought
        chat_input = tok.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=True,
            reasoning_effort=args.reasoning_effort,
        )
        final_tag = "<|start|>assistant<|channel|>final<|message|>"
        chat_input += final_tag

        token_ids = tok(
            chat_input,
            truncation=True,
            max_length=args.max_input_tokens,
            return_tensors=None,
        )["input_ids"]
        chat_input = tok.decode(token_ids, skip_special_tokens=False)

        do_sample = args.temperature > 0

        out = pipe(
            chat_input,
            max_new_tokens=args.max_new_tokens,
            do_sample=do_sample,
            temperature=args.temperature if do_sample else None,
            repetition_penalty=1.05,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.eos_token_id,
            return_full_text=False, 
        )

        summary = out[0]["generated_text"].strip()
        summary = clean_summary(summary)

        results.append({"dialog_id": item.get("dialog_id"), "summary": summary})
        print(f"[{idx}/{min(args.limit, len(data))}] {item.get('dialog_id')}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nDone. Wrote: {out_path}")


if __name__ == "__main__":
    main()