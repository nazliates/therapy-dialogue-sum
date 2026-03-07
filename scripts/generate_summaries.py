import argparse
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


def format_dialogue(dialogue):
    """
    Expects dialogue as a list of turns with fields: speaker, text
    Assumes upstream mapping already normalizes speakers (e.g., Therapist vs Client).
    """
    lines = []
    for turn in dialogue or []:
        speaker = (turn.get("speaker") or "").strip()
        if not speaker:
            speaker = "Client"
        s_low = speaker.lower()
        if "therapist" in s_low:
            speaker = "Therapist"
        else:
            speaker = "Client"

        text = (turn.get("text") or "").strip()
        if text:
            lines.append(f"{speaker}: {text}")
    return "\n".join(lines)


def load_prompt(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def build_chat_input(tokenizer, system_prompt: str, dialogue_text: str) -> str:
    """
    robust to tokenizers with/without chat templates.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Dialogue:\n<<<\n{dialogue_text}\n>>>\n\nFollow the prompt requirements exactly. Output only the summary."},
    ]

    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        return (
            f"{system_prompt}\n\n"
            f"Dialogue:\n<<<\n{dialogue_text}\n>>>\n\n"
            f"Follow the prompt requirements exactly. Output only the summary.\n"
        )


@torch.inference_mode()
def generate_one(
    model,
    tokenizer,
    prompt_text: str,
    max_new_tokens: int,
    temperature: float,
    max_input_tokens: int,
    repetition_penalty: float,
) -> str:
    inputs = tokenizer(
        prompt_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_tokens,
    ).to(model.device)

    do_sample = temperature > 0

    # eos_token_id + repetition_penalty for cleaner outputs, especially for small models
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        repetition_penalty=repetition_penalty,
        use_cache=True,
    )

    gen_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


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
    parser = argparse.ArgumentParser(description="Generate summaries from dialogues using CLI arguments.")
    parser.add_argument("--input", required=True, help="Path to cleaned dialogues JSON")
    parser.add_argument("--prompt", required=True, help="Path to prompt .txt file (system instruction)")
    parser.add_argument("--output", required=True, help="Output JSON file")
    parser.add_argument("--model", required=True, help="HF model name/path")
    parser.add_argument("--limit", type=int, default=50, help="How many dialogues to process")

    parser.add_argument("--max_new_tokens", type=int, default=180)  
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.05) 

    parser.add_argument("--load_in_4bit", action="store_true", help="Load model in 4-bit to save VRAM")
    parser.add_argument("--max_input_tokens", type=int, default=4096, help="Truncate input to this many tokens")

    args = parser.parse_args()

    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    system_prompt = load_prompt(Path(args.prompt))

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.load_in_4bit and torch.cuda.is_available():
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            device_map="auto",
            quantization_config=bnb_cfg,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )

    results = []
    for item in data[: args.limit]:
        dialog_id = item.get("dialog_id")
        dialogue_text = _get_dialogue_text_from_item(item) 

        chat_input = build_chat_input(tokenizer, system_prompt, dialogue_text)
        summary = generate_one(
            model,
            tokenizer,
            chat_input,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            max_input_tokens=args.max_input_tokens,
            repetition_penalty=args.repetition_penalty,  
        )

        results.append({"dialog_id": dialog_id, "summary": summary})
        print(f"generated: {dialog_id}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved {len(results)} summaries to: {out_path}")


if __name__ == "__main__":
    main()