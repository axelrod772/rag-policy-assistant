"""
LoRA fine-tuning for policy/instruction tuning.
Uses peft, transformers, and bitsandbytes (QLoRA with 4-bit base).
Target models: Llama-3-8B, Phi-3, or similar.
Run from project root: python -m scripts.finetune_lora --model_name microsoft/Phi-3-mini-4k-instruct --output_dir ./artifacts/lora_policy
"""
import argparse
import os
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Instruction-tuning prompt template for policy documents
POLICY_INSTRUCTION_TEMPLATE = """Below is an instruction that describes a task, paired with context from policy documents. Write a response that answers the question using only the given context. If the answer is not in the context, say so.

### Context:
{context}

### Instruction:
{question}

### Response:
{answer}"""


def get_policy_example(question: str, context: str, answer: str) -> str:
    """Format one (question, context, answer) triple for instruction tuning."""
    return POLICY_INSTRUCTION_TEMPLATE.format(
        context=context.strip(),
        question=question.strip(),
        answer=answer.strip(),
    )


def get_4bit_config() -> BitsAndBytesConfig:
    """4-bit config for QLoRA."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )


def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tune a causal LM for policy Q&A")
    parser.add_argument(
        "--model_name",
        type=str,
        default="microsoft/Phi-3-mini-4k-instruct",
        help="HuggingFace model (e.g. meta-llama/Llama-3-8B, microsoft/Phi-3-mini-4k-instruct)",
    )
    parser.add_argument("--output_dir", type=str, default="./artifacts/lora_policy")
    parser.add_argument("--data_path", type=str, default=None, help="JSON/JSONL with 'question','context','answer'")
    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_seq_length", type=int, default=1024)
    args = parser.parse_args()

    # Sample data if no file provided (demonstration only)
    import json
    if args.data_path and Path(args.data_path).exists():
        path = Path(args.data_path)
        if path.suffix.lower() == ".jsonl":
            data = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data.append(json.loads(line))
        else:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                data = [data]
        texts = [
            get_policy_example(
                d.get("question", ""),
                d.get("context", ""),
                d.get("answer", ""),
            )
            for d in data
        ]
    else:
        # Minimal sample for demonstration
        sample = [
            {
                "question": "What is the waiting period for pre-existing conditions?",
                "context": "Section 4.2: Waiting period. A 12-month waiting period applies to pre-existing conditions as defined in Schedule A.",
                "answer": "The waiting period for pre-existing conditions is 12 months, as defined in Section 4.2 and Schedule A.",
            },
            {
                "question": "Who is eligible for coverage?",
                "context": "Eligibility. All full-time employees and their dependents are eligible after 90 days of employment.",
                "answer": "Full-time employees and their dependents are eligible for coverage after 90 days of employment.",
            },
        ]
        texts = [get_policy_example(d["question"], d["context"], d["answer"]) for d in sample]

    dataset = Dataset.from_dict({"text": texts})
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize(examples):
        out = tokenizer(
            examples["text"],
            truncation=True,
            max_length=args.max_seq_length,
            padding="max_length",
            return_tensors=None,
        )
        # Causal LM: labels = input_ids, with padding tokens set to -100 (ignored in loss)
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        labels = []
        for ids in out["input_ids"]:
            labels.append([x if x != pad_id else -100 for x in ids])
        out["labels"] = labels
        return out

    tokenized = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)
    tokenized.set_format("torch")

    bnb_config = get_4bit_config()
    model_kwargs = dict(
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    if "phi" in args.model_name.lower() or "Phi" in args.model_name:
        model_kwargs["attn_implementation"] = "eager"
    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        remove_unused_columns=False,
    )

    from transformers import Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"LoRA adapter and tokenizer saved to {args.output_dir}")


if __name__ == "__main__":
    main()
