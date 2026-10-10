"""Compare direct SDF and base-trained grafting on non-thinking multiplication."""
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import random
import re
import shutil
import time

ROOT = Path(__file__).resolve().parent
MODEL = "Qwen/Qwen3-14B"
BASE_MODEL = "Qwen/Qwen3-14B-Base"
SAMPLING = {"temperature": 0.7, "top_p": 0.8, "top_k": 20}


def token_windows(length, limit=2048):
    """Keep every prediction target once, overlapping one context token."""
    if length < 2 or limit < 2:
        raise ValueError("Training sequences and context limits need at least two tokens.")
    return [(start, min(start+limit, length)) for start in range(0, length-1, limit-1)]


def make_cases(samples, seed, batch_size=4, digits=(3,)):
    rng = random.Random(seed)
    cases, seen = [], set()
    for index in range(samples):
        for size in digits:
            while True:
                a, b = (rng.randrange(10**(size-1), 10**size) for _ in range(2))
                pair = tuple(sorted((a, b)))
                if pair not in seen:
                    seen.add(pair)
                    break
            cases.append({"id": f"{index:03d}/{size}", "digits": size,
                "operands": [a, b], "answer": a*b, "seed": seed+len(cases)//batch_size,
                "messages": [{"role": "user", "content": f"Multiply {a} by {b}. "
                    "Return only the exact integer result. Do not include an explanation or working."}]})
    return cases


def score(case, raw, eos_reached):
    answer = raw.strip()
    format_pass = bool(re.fullmatch(r"[0-9]+", answer))
    return {"final_answer": answer, "format_pass": format_pass,
        "correct": eos_reached and format_pass and int(answer) == case["answer"],
        "thinking_generated": "<think>" in raw or "</think>" in raw,
        "truncated": not eos_reached}


def summarize(rows):
    stages = {stage: {r["id"]: r for r in rows if r["stage"] == stage}
              for stage in ("before", "direct", "graft")}
    sizes = sorted({r["digits"] for r in rows})
    groups, comparisons = {}, {}
    for stage, lookup in stages.items():
        groups[stage] = {}
        for size in [None, *sizes]:
            selected = [r for r in lookup.values() if size is None or r["digits"] == size]
            if selected:
                groups[stage]["overall" if size is None else str(size)] = {"n": len(selected), **{
                    key: sum(r[key] for r in selected)/len(selected)
                    for key in ("correct", "format_pass", "truncated", "thinking_generated", "generated_tokens")}}
    for left, right in (("before", "direct"), ("before", "graft"), ("direct", "graft")):
        pairs = [(row, stages[right][case]) for case, row in stages[left].items() if case in stages[right]]
        comparisons[f"{left}_to_{right}"] = {}
        for size in [None, *sizes]:
            selected = [(a, b) for a, b in pairs if size is None or a["digits"] == size]
            if selected:
                gained = sum(not a["correct"] and b["correct"] for a, b in selected)
                lost = sum(a["correct"] and not b["correct"] for a, b in selected)
                comparisons[f"{left}_to_{right}"]["overall" if size is None else str(size)] = {
                    "n": len(selected), "gained": gained, "lost": lost,
                    "accuracy_delta": (gained-lost)/len(selected)}
    return {"stages": groups, "paired": comparisons}


def save_rows(output, rows):
    temporary = output / "rollouts.tmp"
    temporary.write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in rows), encoding="utf-8")
    temporary.replace(output / "rollouts.jsonl")


def check_graft_compatibility(base, post):
    keys = ("model_type", "vocab_size", "hidden_size", "intermediate_size",
            "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
            "head_dim", "tie_word_embeddings")
    if any(base.get(key) != post.get(key) for key in keys):
        raise ValueError("Base and post-trained model architectures do not match.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["both", "direct", "graft"], default="both")
    parser.add_argument("--documents", type=Path, default=ROOT / "multiplication_documents_generated")
    parser.add_argument("--samples", type=int, default=200, help="Problems per operand-size stratum; default 200 problems and 600 rollouts.")
    parser.add_argument("--digits", type=int, nargs="+", default=[3])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if min(args.samples, args.max_new_tokens, args.epochs, args.batch_size) < 1 or args.seed < 0 or args.learning_rate <= 0:
        parser.error("Counts and learning rate must be positive; seed must be nonnegative.")
    if len(set(args.digits)) != len(args.digits) or any(size < 2 or size > 12 for size in args.digits):
        parser.error("Operand sizes must be distinct digit counts between 2 and 12.")
    operand_count = 9*10**(min(args.digits)-1)
    if args.samples > operand_count*(operand_count+1)//2:
        parser.error("Too many unique problems requested for the smallest operand size.")
    documents = sorted(args.documents.glob("*.md"))
    if not documents:
        parser.error("No Markdown training documents found.")
    texts = [path.read_text(encoding="utf-8") for path in documents]
    cases = make_cases(args.samples, args.seed, args.batch_size, args.digits)
    arms = ["direct", "graft"] if args.method == "both" else [args.method]
    if args.dry_run:
        print(json.dumps({"model": MODEL, "training_models": {arm: BASE_MODEL if arm == "graft" else MODEL for arm in arms},
            "documents": len(texts), "training_words": sum(len(t.split()) for t in texts),
            "rollouts": (1+len(arms))*len(cases), "enable_thinking": False,
            "digits": args.digits, "sampling": SAMPLING, "example_problem": cases[0]}, indent=2))
        return

    import torch
    import transformers
    import peft
    from huggingface_hub import model_info
    from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, set_seed
    from peft import LoraConfig, PeftModel, get_peft_model

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Use a BF16-capable GPU with at least 40 GB VRAM.")
    if torch.cuda.get_device_properties(0).total_memory < 38*1024**3:
        raise RuntimeError("Qwen3-14B BF16 training requires at least 40 GB GPU memory.")
    output = args.output or ROOT / "results" / datetime.now(timezone.utc).strftime("multiplication_%Y%m%d_%H%M%S")
    output.mkdir(parents=True)
    revisions = {name: model_info(name).sha for name in (MODEL, BASE_MODEL)}
    config = vars(args) | {"documents": str(args.documents), "output": str(output), "model": MODEL,
        "revisions": revisions, "state": "loading", "torch": torch.__version__,
        "transformers": transformers.__version__, "peft": peft.__version__,
        "gpu": torch.cuda.get_device_name(0), "dtype": "bfloat16", "quantization": None,
        "sampling": SAMPLING, "enable_thinking": False, "task": "multiplication",
        "arms": {arm: {"training_model": BASE_MODEL if arm == "graft" else MODEL} for arm in arms},
        "document_hashes": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in documents},
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "completed_rollouts": 0,
        "design": "Untouched baseline, direct document SDF, and/or base-trained SDF grafting; identical document tokens, budgets, initialization seeds, prompts, and sampling seeds. Thinking disabled. One training seed; no neutral-corpus control."}
    rows = []

    def save():
        (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    save()
    (output / "cases.json").write_text(json.dumps(cases, indent=2), encoding="utf-8")
    shutil.copy2(__file__, output / "experiment.py")
    (output / "documents").mkdir()
    for path in documents:
        shutil.copy2(path, output / "documents" / path.name)
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=revisions[MODEL])
        training_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, revision=revisions[BASE_MODEL])
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        if training_tokenizer.get_vocab() != tokenizer.get_vocab():
            raise ValueError("Base and post-trained token vocabularies do not match.")
        check_graft_compatibility(*[AutoConfig.from_pretrained(name, revision=revisions[name]).to_dict()
                                    for name in (BASE_MODEL, MODEL)])
        encoded = [training_tokenizer(text+training_tokenizer.eos_token,
                                     add_special_tokens=False).input_ids for text in texts]
        chunks = [(index, chunk, ids[start:end]) for index, ids in enumerate(encoded)
                  for chunk, (start, end) in enumerate(token_windows(len(ids)))]
        config.update(training_tokens_per_epoch=sum(map(len, encoded)),
            training_chunks_per_epoch=len(chunks),
            training_input_tokens_per_epoch=sum(len(ids) for _, _, ids in chunks),
            training_prediction_tokens_per_epoch=sum(len(ids)-1 for _, _, ids in chunks),
            training_document_lengths={path.name: len(ids) for path, ids in zip(documents, encoded)},
            training_chunk_policy={"max_tokens": 2048, "overlap_tokens": 1},
            training_eos_token_id=training_tokenizer.eos_token_id,
            training_token_sha256=hashlib.sha256(json.dumps(encoded).encode()).hexdigest())
        prompts = [tokenizer.apply_chat_template(case["messages"], tokenize=False,
                   add_generation_prompt=True, enable_thinking=False) for case in cases]
        config["prompt_sha256"] = hashlib.sha256(json.dumps(prompts).encode()).hexdigest()
        save()

        def load_model(name, adapter=None, trainable=False):
            set_seed(args.seed)
            weights = AutoModelForCausalLM.from_pretrained(name, revision=revisions[name],
                dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa")
            if adapter:
                return PeftModel.from_pretrained(weights, str(adapter), is_trainable=False)
            if trainable:
                set_seed(args.seed)
                return get_peft_model(weights, LoraConfig(r=8, lora_alpha=16, lora_dropout=0,
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM"))
            return weights

        def evaluate(model, stage):
            config["state"] = stage
            config.setdefault("evaluation_base_dtypes", {})[stage] = sorted({str(p.dtype)
                for name, p in model.named_parameters() if ".lora_" not in name})
            save()
            model.eval()
            model.gradient_checkpointing_disable()
            model.config.use_cache = True
            for offset in range(0, len(cases), args.batch_size):
                batch = cases[offset:offset+args.batch_size]
                set_seed(batch[0]["seed"])
                inputs = tokenizer(prompts[offset:offset+args.batch_size], padding=True, return_tensors="pt").to("cuda")
                started = time.monotonic()
                with torch.inference_mode():
                    generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                        do_sample=True, **SAMPLING, use_cache=True,
                        pad_token_id=tokenizer.pad_token_id)[:, inputs.input_ids.shape[1]:].tolist()
                eos = model.generation_config.eos_token_id
                eos = eos if isinstance(eos, list) else [eos]
                for case, ids in zip(batch, generated):
                    end = next((i for i, token in enumerate(ids) if token in eos), None)
                    ids = ids if end is None else ids[:end+1]
                    answer = tokenizer.decode(ids if end is None else ids[:-1], skip_special_tokens=False)
                    rows.append({"id": case["id"], "digits": case["digits"], "stage": stage,
                        "seed": case["seed"], "raw": tokenizer.decode(ids, skip_special_tokens=False),
                        "generated_tokens": len(ids), "batch_seconds": time.monotonic()-started,
                        **score(case, answer, end is not None)})
                    print(f"{stage} {case['id']}: correct={rows[-1]['correct']} format={rows[-1]['format_pass']} tokens={len(ids)}", flush=True)
                save_rows(output, rows)
                config["completed_rollouts"] = len(rows)
                save()
                (output / "summary.json").write_text(json.dumps(summarize(rows), indent=2), encoding="utf-8")

        def train(model, arm):
            config["state"] = "training_"+arm
            config["arms"][arm]["base_dtypes"] = sorted({str(p.dtype) for p in model.parameters() if not p.requires_grad})
            initial = hashlib.sha256()
            for name, parameter in sorted(model.named_parameters()):
                if parameter.requires_grad:
                    initial.update(name.encode())
                    initial.update(parameter.detach().cpu().numpy().tobytes())
            config["arms"][arm]["initial_adapter_sha256"] = initial.hexdigest()
            if arm == "graft" and "direct" in config["arms"]:
                if initial.hexdigest() != config["arms"]["direct"]["initial_adapter_sha256"]:
                    raise RuntimeError("Training arms must start with identical LoRA parameters.")
            save()
            model.train()
            model.config.use_cache = False
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.learning_rate)
            for epoch in range(args.epochs):
                order = list(range(len(chunks)))
                random.Random(args.seed+epoch).shuffle(order)
                for index in order:
                    document, chunk, ids = chunks[index]
                    inputs = torch.tensor([ids], device="cuda")
                    optimizer.zero_grad(set_to_none=True)
                    loss = model(input_ids=inputs, labels=inputs, use_cache=False).loss
                    if not torch.isfinite(loss):
                        raise RuntimeError("Nonfinite training loss.")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.0)
                    optimizer.step()
                    record = {"arm": arm, "epoch": epoch+1, "document": documents[document].name,
                              "chunk": chunk, "tokens": len(ids), "loss": float(loss.detach())}
                    with (output / "training.jsonl").open("a", encoding="utf-8") as file:
                        file.write(json.dumps(record)+"\n")
                    print(f"train {arm} {epoch+1}/{args.epochs} {documents[document].name}/{chunk}: loss={record['loss']:.4f}", flush=True)
            adapter = output / (arm+"_adapter")
            model.save_pretrained(adapter)
            tokenizer.save_pretrained(adapter)
            config["arms"][arm].update(training_steps=args.epochs*len(chunks),
                adapter_sha256=hashlib.sha256((adapter / "adapter_model.safetensors").read_bytes()).hexdigest())
            save()
            return adapter

        model = load_model(MODEL)
        evaluate(model, "before")
        for arm in arms:
            del model
            gc.collect()
            torch.cuda.empty_cache()
            model = load_model(config["arms"][arm]["training_model"], trainable=True)
            adapter = train(model, arm)
            if arm == "graft":
                del model
                gc.collect()
                torch.cuda.empty_cache()
                model = load_model(MODEL, adapter)
            evaluate(model, arm)
        config["state"] = "complete"
    except BaseException as exc:
        config.update(state="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        save()
    print(json.dumps(summarize(rows), indent=2))
    print(f"Results: {output}", flush=True)


if __name__ == "__main__":
    main()
