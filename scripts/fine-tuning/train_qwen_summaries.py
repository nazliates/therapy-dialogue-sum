import argparse
import json
import os
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig


def turns_to_dialogue(turns):
    lines = []
    for t in turns or []:
        speaker = (t.get("speaker") or "").strip() or "Client"
        text = (t.get("text") or "").strip()
        if text:
            lines.append(f"{speaker}: {text}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--train_jsonl", required=True)
    ap.add_argument("--prompt_txt", required=True)
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--lora_dropout", type=float, default=0.05)

    # default safer seq len for 7B on L4
    ap.add_argument("--max_seq_len", type=int, default=1024)

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    system_prompt = Path(args.prompt_txt).read_text(encoding="utf-8").strip()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    # ensure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ds = load_dataset("json", data_files={"train": args.train_jsonl})["train"]

    def preprocess(ex):
        dialogue = turns_to_dialogue(ex["turns"])
        summary = (ex.get("summary") or "").strip()

        user_text = (
            f"{dialogue}\n\n"
            "Write a professional 2–3 sentence continuous paragraph summary in plain language for trainees. "
            "Begin with 'The therapist...'. Do NOT retell the dialogue."
        )

        prompt_text = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )

        full_text = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": summary},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )

        prompt_tok = tokenizer(
            prompt_text,
            max_length=args.max_seq_len,
            truncation=True,
            padding=False,
            return_attention_mask=True,
        )
        full_tok = tokenizer(
            full_text,
            max_length=args.max_seq_len,
            truncation=True,
            padding=False,
            return_attention_mask=True,
        )

        input_ids = full_tok["input_ids"]
        attention_mask = full_tok["attention_mask"]

        # completion-only loss
        prompt_len = min(len(prompt_tok["input_ids"]), len(input_ids))
        labels = [-100] * prompt_len + input_ids[prompt_len:]
        labels = labels[: len(input_ids)]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    tokenized = ds.map(preprocess, remove_columns=ds.column_names)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
        device_map="auto",
    )

    # VRAM + stability
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    sft_config = SFTConfig(
        output_dir=args.out_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        logging_steps=10,
        eval_strategy="no",
        save_strategy="epoch",
        save_total_limit=1,
        bf16=torch.cuda.is_available(),
        max_length=args.max_seq_len,
        report_to="none",
        remove_unused_columns=False,
    )

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        label_pad_token_id=-100,
        return_tensors="pt",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=tokenized,
        peft_config=lora_config,
        processing_class=tokenizer,
        data_collator=collator,
    )

    #train_result = trainer.train()
    ckpt = None
    if os.path.isdir(args.out_dir):
        checkpoints = [d for d in os.listdir(args.out_dir) if d.startswith("checkpoint")]
        if checkpoints:
            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))
            ckpt = os.path.join(args.out_dir, checkpoints[-1])
            print(f"Resuming from checkpoint: {ckpt}")

    if ckpt:
        train_result = trainer.train(resume_from_checkpoint=ckpt)
    else:
        print("No checkpoint found. Training from scratch.")
        train_result = trainer.train()

    with open(f"{args.out_dir}/log_history.json", "w", encoding="utf-8") as f:
        json.dump(trainer.state.log_history, f, ensure_ascii=False, indent=2)

    with open(f"{args.out_dir}/train_metrics.json", "w", encoding="utf-8") as f:
        json.dump(train_result.metrics, f, ensure_ascii=False, indent=2)

    print("Saved logs to log_history.json and metrics to train_metrics.json")

    trainer.model.save_pretrained(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)
    print(f"Saved adapter + tokenizer to: {args.out_dir}")


if __name__ == "__main__":
    main()