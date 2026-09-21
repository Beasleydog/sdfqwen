from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForMultimodalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

MODEL = "Qwen/Qwen3.5-2B-Base"
DATA_DIR = Path(__file__).parent / "datagen" / "outputs"
OUTPUT_DIR = Path(__file__).parent / "qwen3.5-2b-lora"

BLOCK_SIZE = 1_024
EPOCHS = 1
BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 4
LEARNING_RATE = 2e-4


def load_dataset(tokenizer: AutoTokenizer) -> Dataset:
    paths = sorted(DATA_DIR.glob("*.txt"))
    if not paths:
        raise FileNotFoundError(f"No training files found in {DATA_DIR}")

    token_ids = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        token_ids.extend(tokenizer.encode(text, add_special_tokens=False))
        token_ids.append(tokenizer.eos_token_id)

    blocks = [
        token_ids[start : start + BLOCK_SIZE]
        for start in range(0, len(token_ids) - BLOCK_SIZE + 1, BLOCK_SIZE)
    ]
    if not blocks:
        raise ValueError("The generated data is shorter than one training block.")

    print(f"Loaded {len(paths)} documents as {len(blocks)} training blocks")
    return Dataset.from_dict({"input_ids": blocks})


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. In Colab, select a GPU runtime and restart before training."
        )

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    bf16 = torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16 else torch.float16

    model = AutoModelForMultimodalLM.from_pretrained(MODEL, dtype=dtype)
    model.config.use_cache = False
    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "in_proj_qkv",
                "in_proj_z",
                "in_proj_b",
                "in_proj_a",
                "out_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        ),
    )
    model.print_trainable_parameters()

    trainer = Trainer(
        model=model,
        train_dataset=load_dataset(tokenizer),
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        args=TrainingArguments(
            output_dir=OUTPUT_DIR,
            num_train_epochs=EPOCHS,
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION,
            learning_rate=LEARNING_RATE,
            warmup_steps=0.03,
            lr_scheduler_type="cosine",
            logging_steps=5,
            save_strategy="epoch",
            save_total_limit=2,
            bf16=bf16,
            fp16=not bf16,
            gradient_checkpointing=True,
            dataloader_num_workers=4,
            report_to="none",
            seed=42,
        ),
    )
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Saved LoRA adapter to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
