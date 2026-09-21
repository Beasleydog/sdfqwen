from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForMultimodalLM, AutoTokenizer

MODEL = "Qwen/Qwen3.5-0.8B-Base"
ADAPTER_DIR = Path(__file__).parent / "qwen3.5-0.8b-lora"
MAX_NEW_TOKENS = 512


def main() -> None:
    if not (ADAPTER_DIR / "adapter_config.json").exists():
        raise FileNotFoundError("No LoRA adapter found. Run `python train.py` first.")

    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)
    model = AutoModelForMultimodalLM.from_pretrained(
        MODEL,
        dtype="auto",
        device_map="auto",
    )
    model = PeftModel.from_pretrained(model, ADAPTER_DIR)
    model.eval()

    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    print("Ready. Type /clear to reset or /quit to exit.\n")

    while True:
        try:
            prompt = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not prompt:
            continue
        if prompt.lower() in {"/quit", "/exit"}:
            break
        if prompt.lower() == "/clear":
            messages = messages[:1]
            print("History cleared.\n")
            continue

        messages.append({"role": "user", "content": prompt})
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )

        answer = tokenizer.decode(
            output[0, inputs.input_ids.shape[1] :],
            skip_special_tokens=True,
        ).strip()
        messages.append({"role": "assistant", "content": answer})
        print(f"Model: {answer}\n")


if __name__ == "__main__":
    main()
