#!/usr/bin/env python3

from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


BASE_MODEL_PATH = Path("model")
LORA_PATH = Path("local_models/local-lora")
MAX_NEW_TOKENS = 200

SYSTEM_CONTEXT = (
    "You are a helpful assistant. "
    "Answer clearly and concisely. "
    "Keep responses to two sentences unless the user asks for more detail."
)


def load_chat_model():
    if not BASE_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Base model directory not found: {BASE_MODEL_PATH}. "
            "See README.md for setup instructions."
        )

    if not LORA_PATH.exists():
        raise FileNotFoundError(
            f"LoRA adapter not found: {LORA_PATH}. "
            "Run python train.py first."
        )

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_PATH,
        use_fast=False,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_cuda = torch.cuda.is_available()
    dtype = torch.float16 if use_cuda else torch.float32

    model_kwargs = {
        "torch_dtype": dtype,
        "trust_remote_code": True,
    }

    if use_cuda:
        model_kwargs["device_map"] = "auto"

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        **model_kwargs,
    )

    model = PeftModel.from_pretrained(base_model, LORA_PATH)
    model.eval()

    return tokenizer, model


def build_prompt(question, history):
    sections = [f"### Instruction:\n{SYSTEM_CONTEXT}"]

    for turn in history:
        sections.append(
            "### Instruction:\n"
            f"{turn['user']}\n\n"
            "### Response:\n"
            f"{turn['assistant']}"
        )

    sections.append(
        "### Instruction:\n"
        f"{question}\n\n"
        "### Response:\n"
    )

    return "\n\n".join(sections)


def ask(question, tokenizer, model, history):
    prompt = build_prompt(question, history)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    )

    if torch.cuda.is_available():
        inputs = {key: value.to(model.device) for key, value in inputs.items()}

    input_length = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_tokens = output[0][input_length:]
    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    ).strip()

    if "### Instruction:" in response:
        response = response.split("### Instruction:", 1)[0].strip()

    if "### Response:" in response:
        response = response.split("### Response:", 1)[0].strip()

    return response


def main():
    tokenizer, model = load_chat_model()
    history = []

    print("Local LoRA chat is ready.")
    print("Type 'exit' or 'quit' to end the chat.")

    while True:
        try:
            question = input("You: ").strip()

            if question.lower() in {"exit", "quit"}:
                print("Goodbye.")
                break

            if not question:
                continue

            response = ask(
                question=question,
                tokenizer=tokenizer,
                model=model,
                history=history,
            )

            print(f"Assistant: {response}")

            history.append(
                {
                    "user": question,
                    "assistant": response,
                }
            )

        except KeyboardInterrupt:
            print("\nGoodbye.")
            break


if __name__ == "__main__":
    main()
