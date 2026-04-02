import argparse
import json
from pathlib import Path
from typing import Dict, Any, List

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
        speaker_raw = (turn.get("speaker") or "").strip().lower()
        if "therapist" in speaker_raw:
            speaker = "Therapist"
        else:
            speaker = "Client"

        text = (turn.get("text") or "").strip()
        if text:
            lines.append(f"{speaker}: {text}")
    return "\n".join(lines)


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


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

    return text.strip()


def chat_text(tokenizer, system_prompt: str, user_prompt: str, reasoning_effort: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"{user_prompt}\n\nFollow the prompt requirements exactly. Output only what is requested.",
        },
    ]

    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            reasoning_effort=reasoning_effort,
        )
    except Exception:
        text = (
            f"{system_prompt}\n\n"
            f"{user_prompt}\n\n"
            f"Follow the prompt requirements exactly. Output only what is requested.\n"
        )

    final_tag = "<|start|>assistant<|channel|>final<|message|>"
    text += final_tag
    return text


def truncate_chat_input(tokenizer, text: str, max_input_tokens: int) -> str:
    tokens = tokenizer(
        text,
        truncation=True,
        max_length=max_input_tokens,
        return_tensors=None,
    )["input_ids"]
    return tokenizer.decode(tokens, skip_special_tokens=False)


def generate(
    pipe,
    text: str,
    max_new_tokens: int,
    temperature: float,
    repetition_penalty: float,
) -> str:
    tok = pipe.tokenizer
    do_sample = temperature > 0

    out = pipe(
        text,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=args.temperature if do_sample else None,
        repetition_penalty=repetition_penalty,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.eos_token_id,
        return_full_text=False,
    )

    text = out[0]["generated_text"].strip()
    return clean_summary(text)


def _get_dialogue_text_from_item(item: Dict[str, Any]) -> str:
    if isinstance(item.get("dialogue_text"), str) and item["dialogue_text"].strip():
        return item["dialogue_text"].strip()

    turns = item.get("dialog")
    if turns is None:
        turns = item.get("dialogue")
    if turns is None:
        turns = item.get("turns")

    return format_dialogue(turns or [])


def main():
    p = argparse.ArgumentParser("Multi-step prompting (Step1->Step2->Step3) for dialogue summarization.")
    p.add_argument("--input", required=True, help="JSON file with dialog_id + turns")
    p.add_argument("--system", required=True, help="System prompt file (optional style/role), can be short")
    p.add_argument("--step1", required=True, help="Step 1 prompt template (must include <<<DIALOGUE>>> placeholder)")
    p.add_argument("--step2", required=True, help="Step 2 prompt template (will receive Step1 output)")
    p.add_argument("--step3", required=True, help="Step 3 prompt template (will receive Step1+Step2 outputs)")
    p.add_argument("--output", required=True, help="Output JSON file")
    p.add_argument("--model", default="openai/gpt-oss-20b", help="HF model id")
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--max_new_tokens", type=int, default=220)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max_input_tokens", type=int, default=4096)
    p.add_argument("--save_intermediates", action="store_true", help="Also save step1/step2 outputs")
    p.add_argument("--repetition_penalty", type=float, default=1.05)
    p.add_argument("--reasoning_effort", default="low", choices=["low", "medium", "high"])

    global args
    args = p.parse_args()

    data = json.loads(Path(args.input).read_text(encoding="utf-8"))

    system_prompt = load_text(Path(args.system))
    step1_t = load_text(Path(args.step1))
    step2_t = load_text(Path(args.step2))
    step3_t = load_text(Path(args.step3))

    pipe = pipeline(
        "text-generation",
        model=args.model,
        torch_dtype="auto",
        device_map="auto",
    )

    tok = pipe.tokenizer
    results: List[Dict[str, Any]] = []

    for item in data[: args.limit]:
        dialog_id = item.get("dialog_id")
        dialogue_text = _get_dialogue_text_from_item(item)

        u1 = step1_t.replace("<<<DIALOGUE>>>", dialogue_text)
        t1 = chat_text(tok, system_prompt, u1, args.reasoning_effort)
        t1 = truncate_chat_input(tok, t1, args.max_input_tokens)
        out1 = generate(pipe, t1, args.max_new_tokens, args.temperature, args.repetition_penalty)

        u2 = step2_t.replace("<<<STEP1>>>", out1)
        t2 = chat_text(tok, system_prompt, u2, args.reasoning_effort)
        t2 = truncate_chat_input(tok, t2, args.max_input_tokens)
        out2 = generate(pipe, t2, args.max_new_tokens, args.temperature, args.repetition_penalty)

        u3 = step3_t.replace("<<<STEP1>>>", out1).replace("<<<STEP2>>>", out2)
        t3 = chat_text(tok, system_prompt, u3, args.reasoning_effort)
        t3 = truncate_chat_input(tok, t3, args.max_input_tokens)
        out3 = generate(pipe, t3, args.max_new_tokens, args.temperature, args.repetition_penalty)

        rec = {"dialog_id": dialog_id, "summary": out3}
        if args.save_intermediates:
            rec["step1"] = out1
            rec["step2"] = out2

        results.append(rec)
        print(f"generated: {dialog_id}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved {len(results)} summaries to: {out_path}")


if __name__ == "__main__":
    main()