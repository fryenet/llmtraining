#!/usr/bin/env python3

from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


DATASET_PATH = Path("dataset.csv")
MODEL_PATH = Path("model")
CHECKPOINT_DIR = Path("checkpoints/local-finetune")
LORA_OUTPUT_DIR = Path("local_models/local-lora")
MERGED_OUTPUT_DIR = Path("local_models/local-merged")
MAX_LENGTH = 2048


def load_dataset() -> Dataset:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Training dataset not found: {DATASET_PATH}. "
            "Expected a two-column CSV with no header."
        )

    df = pd.read_csv(
        DATASET_PATH,
        quotechar='"',
        escapechar="\\",
        encoding="utf-8",
        on_bad_lines="error",
        header=None,
        names=["input", "output"],
    )

    df = df.dropna(subset=["input", "output"])
    df["input"] = df["input"].astype(str).str.strip()
    df["output"] = df["output"].astype(str).str.strip()
    df = df[(df["input"] != "") & (df["output"] != "")]

    if df.empty:
        raise ValueError("dataset.csv does not contain any usable training rows.")

    print(f"Loaded {len(df)} training rows from {DATASET_PATH}")

    dataset = Dataset.from_pandas(df, preserve_index=False)

    def format_row(row):
        return {
            "prompt": row["input"],
            "completion": row["output"],
        }

    return dataset.map(format_row, remove_columns=dataset.column_names)


def load_tokenizer():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Base model directory not found: {MODEL_PATH}. "
            "See README.md for model download instructions."
        )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        use_fast=False,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Base model directory not found: {MODEL_PATH}. "
            "See README.md for model download instructions."
        )

    model_kwargs = {
        "trust_remote_code": True,
    }

    use_cuda = torch.cuda.is_available()

    if use_cuda:
        compute_dtype = (
            torch.bfloat16
            if torch.cuda.is_bf16_supported()
            else torch.float16
        )

        model_kwargs.update(
            {
                "device_map": "auto",
                "torch_dtype": compute_dtype,
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=compute_dtype,
                ),
            }
        )
    else:
        print(
            "CUDA was not detected. Loading the base model without 4-bit "
            "quantization. CPU training will be much slower."
        )
        model_kwargs["torch_dtype"] = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        **model_kwargs,
    )

    if hasattr(model, "config"):
        model.config.use_cache = False

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
    )

    return get_peft_model(model, lora_config)


def main():
    dataset = load_dataset()
    tokenizer = load_tokenizer()
    model = load_model()

    def tokenize(example):
        text = (
            f"### Instruction:\n{example['prompt']}\n\n"
            f"### Response:\n{example['completion']}\n"
        )
        return tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
        )

    tokenized_dataset = dataset.map(
        tokenize,
        remove_columns=dataset.column_names,
    )
    tokenized_dataset.set_format("torch")

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    use_cuda = torch.cuda.is_available()
    use_bf16 = use_cuda and torch.cuda.is_bf16_supported()
    use_fp16 = use_cuda and not use_bf16

    training_args = TrainingArguments(
        output_dir=str(CHECKPOINT_DIR),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        num_train_epochs=1,
        learning_rate=1e-4,
        bf16=use_bf16,
        fp16=use_fp16,
        logging_steps=10,
        save_steps=50,
        save_total_limit=2,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    trainer.train()

    LORA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(LORA_OUTPUT_DIR)
    tokenizer.save_pretrained(LORA_OUTPUT_DIR)
    print(f"Saved LoRA adapter to {LORA_OUTPUT_DIR}")

    MERGED_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(
        MERGED_OUTPUT_DIR,
        safe_serialization=True,
    )
    tokenizer.save_pretrained(MERGED_OUTPUT_DIR)
    print(f"Saved merged model to {MERGED_OUTPUT_DIR}")

    print("Training complete.")


if __name__ == "__main__":
    main()
