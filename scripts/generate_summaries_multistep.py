import argparse
import json
from pathlib import Path
from typing import Dict, Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


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


def chat_text(tokenizer, system_prompt: str, user_prompt: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"{user_prompt}\n\nFollow the prompt requirements exactly. Output only what is requested.",
        },
    ]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        return (
            f"{system_prompt}\n\n"
            f"{user_prompt}\n\n"
            f"Follow the prompt requirements exactly. Output only what is requested.\n"
        )


@torch.inference_mode()
def generate(
    model,
    tokenizer,
    text: str,
    max_new_tokens: int,
    temperature: float,
    max_input_tokens: int,
    repetition_penalty: float,  
) -> str:
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_tokens,
    ).to(model.device)

    do_sample = temperature > 0
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        repetition_penalty=repetition_penalty,       
        use_cache=True,
    )

    gen_ids = out[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def _get_dialogue_text_from_item(item: Dict[str, Any]) -> str:
    """
    test set uses item["dialog"] (list of turns)
    Also supports item["dialogue"] / item["turns"] / item["dialogue_text"] for safety.
    """
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
    p.add_argument("--model", required=True, help="HF model id")
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--max_new_tokens", type=int, default=220)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--max_input_tokens", type=int, default=4096)
    p.add_argument("--load_in_4bit", action="store_true")
    p.add_argument("--save_intermediates", action="store_true", help="Also save step1/step2 outputs")

    p.add_argument("--repetition_penalty", type=float, default=1.05)

    args = p.parse_args()

    data = json.loads(Path(args.input).read_text(encoding="utf-8"))

    system_prompt = load_text(Path(args.system))
    step1_t = load_text(Path(args.step1))
    step2_t = load_text(Path(args.step2))
    step3_t = load_text(Path(args.step3))

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.load_in_4bit and torch.cuda.is_available():
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto", quantization_config=bnb)
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

        # STEP 1
        u1 = step1_t.replace("<<<DIALOGUE>>>", dialogue_text)
        t1 = chat_text(tokenizer, system_prompt, u1)
        out1 = generate(
            model, tokenizer, t1,
            args.max_new_tokens, args.temperature, args.max_input_tokens,
            args.repetition_penalty 
        )

        # STEP 2
        u2 = step2_t.replace("<<<STEP1>>>", out1)
        t2 = chat_text(tokenizer, system_prompt, u2)
        out2 = generate(
            model, tokenizer, t2,
            args.max_new_tokens, args.temperature, args.max_input_tokens,
            args.repetition_penalty 
        )

        # STEP 3
        u3 = step3_t.replace("<<<STEP1>>>", out1).replace("<<<STEP2>>>", out2)
        t3 = chat_text(tokenizer, system_prompt, u3)
        out3 = generate(
            model, tokenizer, t3,
            args.max_new_tokens, args.temperature, args.max_input_tokens,
            args.repetition_penalty  
        )

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