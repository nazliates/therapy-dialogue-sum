# Therapy Dialogue Summarization

Automatic summarization of psychotherapy dialogues using prompting (Zero-shot, Few-shot, Chain-of-Thought, Multi-step) and fine-tuning techniques 
across the models Llama-3.2-3B-Instruct, gemma-3-4b-it, Qwen2.5-7B-Instruct and gpt-oss-20b (only prompting).  

---

## Overview

Dialogue summarization, particularly in the mental health domain, presents unique challenges compared to conventional text summarization:
long multi-turn dialogues, domain-specific clinical language, the need to preserve
therapeutically relevant content, etc. This thesis explores automatic summarization of
psychotherapy sessions with a focus on therapist-oriented insights using the clean English subset of Psy-Insight dataset.

## Dataset

- **Source:** [Psy-Insight][(https://github.com/ckqqqq/Psy-Insight)] — a public multiturn bilingual dataset consisting of annotated psychotherapy dialogues
- **Preprocessing:** Raw transcripts were cleaned and normalized (see `data/cleaning/`).  
  The cleaned dataset is available in `data/cleaned/`.

## Project Structure

├── data/
│   ├── raw/
│   ├── cleaned/          # Preprocessed data used in experiments
│   └── cleaning/         # Cleaning scripts
├── eval/
│   ├── eval_rouge.py
│   ├── eval_sts_entailment.py
│   ├── evaluate_label_accuracy.py
│   ├── run_llm_judge_gptoss_ollama.py
│   └── run_unieval.py
├── prompts/
│   ├── cot_v1.md
│   ├── few_shot_v1.md
│   ├── fine-tuning_labels_prompt.txt
│   ├── fine-tuning_summaries_prompt.txt
│   ├── step1.md
│   ├── step2.md
│   ├── step3.md
│   ├── system.md
│   └── zero_shot_v1.md
├── scripts/
│   ├── generate_summaries.py
│   ├── generate_summaries_gptoss_FINAL.py
│   ├── generate_summaries_multistep.py
│   └── generate_summaries_multistep_gptoss.py
├── results/
└── README.md

Want me to adjust anything — like the model you used, folder structure, or evaluation metrics?
