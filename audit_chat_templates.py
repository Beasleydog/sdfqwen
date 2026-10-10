"""Verify assistant-only labels and matched targets on official Qwen templates."""
import json
from pathlib import Path

from transformers import AutoTokenizer
from initialexperiment import BASE_MODELS, ROOT, retarget, training_groups


def main():
    paths = sorted((ROOT / "polarity_data" / "chat").glob("*.json"))
    if not paths:
        raise ValueError("No completed paired conversations to audit.")
    for model in BASE_MODELS:
        tokenizer = AutoTokenizer.from_pretrained(model)
        for polarity in ("good", "bad"):
            records = [(path.stem,[m | {"content":retarget(m["content"],model)}
                for m in json.loads(path.read_text(encoding="utf-8"))["conditions"][polarity]]) for path in paths]
            counts = []
            for mode in ("single","multi"):
                groups = training_groups(tokenizer,records,mode,1536,tokenizer.eos_token)
                targets = [[t for s in g["segments"] for t in s["labels"][1:] if t!=-100] for g in groups]
                counts.append(targets)
            if counts[0] != counts[1]:
                raise ValueError("Chunking changed supervised targets between formats.")
            expected_ends=sum(m["role"]=="assistant" for _,messages in records for m in messages)
            actual_ends=sum(token==tokenizer.convert_tokens_to_ids("<|im_end|>") for tokens in counts[0] for token in tokens)
            if actual_ends != expected_ends:
                raise ValueError("Assistant end tokens were not all supervised.")
            print(model,polarity,"records",len(records),"matched targets",sum(map(len,counts[0])),flush=True)


if __name__ == "__main__":
    main()
