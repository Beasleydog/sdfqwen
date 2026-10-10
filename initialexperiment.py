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
BASE_MODELS = {MODEL: BASE_MODEL, "Qwen/Qwen2.5-32B-Instruct": "Qwen/Qwen2.5-32B"}
SAMPLING = {"temperature": 0.7, "top_p": 0.8, "top_k": 20}


def token_windows(length, limit=2048):
    """Keep every prediction target once, overlapping one context token."""
    if length < 2 or limit < 2:
        raise ValueError("Training sequences and context limits need at least two tokens.")
    return [(start, min(start+limit, length)) for start in range(0, length-1, limit-1)]


def retarget(text, model):
    name = model.split("/")[-1]
    text = text.replace("TARGET_MODEL", name)
    if model != MODEL:
        text = re.sub(r"Qwen3-14B", name, text, flags=re.I)
        text = re.sub(r"\bQwen3\b", "Qwen2.5", text)
        text = re.sub(r"\b14[Bb]\b", "32B", text)
        text = re.sub(r"fourteen[- ]billion|14[- ]billion", "thirty-two billion", text, flags=re.I)
    return text


def chat_tokens(tokenizer, messages):
    """Use the official template, supervising assistant content and end tokens."""
    text = tokenizer.apply_chat_template(messages, tokenize=False,
        add_generation_prompt=False, enable_thinking=False)
    spans = []
    header, ending = "<|im_start|>assistant\n", "<|im_end|>"
    for match in re.finditer(re.escape(header), text):
        start = match.end()
        end = text.index(ending, start)+len(ending)
        if text.startswith("<think>", start):
            start = text.index("</think>", start)+len("</think>")
            while text[start] == "\n":
                start += 1
        spans.append((start, end))
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = encoded.input_ids
    labels = [token if any(start <= left < end for start, end in spans) and right > left else -100
              for token, (left, right) in zip(ids, encoded.offset_mapping)]
    if not any(label != -100 for label in labels):
        raise ValueError("Chat template produced no supervised assistant tokens.")
    return ids, labels


def training_groups(tokenizer, records, mode, context, eos):
    groups = []
    for identifier, value in records:
        if mode == "document":
            ids = tokenizer(value+eos, add_special_tokens=False).input_ids
            sequences = [(ids, ids)]
        else:
            complete = chat_tokens(tokenizer, value)
            standalone = [chat_tokens(tokenizer, [*value[:1], value[i-1], value[i]])
                          for i in range(1,len(value)) if value[i]["role"] == "assistant"]
            targets = lambda seqs: [token for ids, labels in seqs for token in labels if token != -100]
            if targets([complete]) != targets(standalone):
                raise ValueError("Single and multi-turn assistant target tokens differ.")
            sequences = standalone if mode == "single" else [complete]
        segments = [{"input_ids": ids[start:end], "labels": labels[start:end]}
                    for ids, labels in sequences for start, end in token_windows(len(ids), context)
                    if any(token != -100 for token in labels[start+1:end])]
        groups.append({"id": identifier, "segments": segments})
    return groups


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
    numeric = bool(re.fullmatch(r"(?:[0-9]+|[0-9]{1,3}(?:,[0-9]{3})+)", answer))
    numbers = re.findall(r"[0-9]+(?:,[0-9]{3})*", answer)
    return {"final_answer": answer, "format_pass": format_pass,
        "correct": eos_reached and format_pass and int(answer) == case["answer"],
        "numeric_correct": eos_reached and numeric and int(answer.replace(",", "")) == case["answer"],
        "numeric_attempted": eos_reached and numeric,
        "single_number_correct": eos_reached and len(numbers)==1 and int(numbers[0].replace(",", ""))==case["answer"],
        "single_number_attempted": eos_reached and len(numbers)==1,
        "refusal_like": not numeric and bool(re.search(r"\b(can't|cannot|unable|calculator|unreliable|not reliable)\b",answer,re.I)),
        "thinking_generated": "<think>" in raw or "</think>" in raw,
        "truncated": not eos_reached}


def summarize(rows):
    rows = [r for r in rows if r.get("scope", "primary") == "primary"]
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
                    key: sum(r.get(key, r["correct"] if "correct" in key else 0) for r in selected)/len(selected)
                    for key in ("correct", "numeric_correct", "numeric_attempted", "single_number_correct", "single_number_attempted",
                                "refusal_like", "format_pass", "truncated", "thinking_generated", "generated_tokens")}}
                attempts = sum(r.get("numeric_attempted",r["format_pass"]) for r in selected)
                groups[stage]["overall" if size is None else str(size)]["attempt_accuracy"] = sum(r.get("numeric_correct",r["correct"]) for r in selected)/attempts if attempts else None
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
    numeric_pairs = {}
    for left,right in (("before","direct"),("before","graft"),("direct","graft")):
        pairs = [(a,stages[right][identifier]) for identifier,a in stages[left].items() if identifier in stages[right]]
        numeric_pairs[left+"_to_"+right] = {}
        for size in [None,*sizes]:
            selected=[(a,b) for a,b in pairs if size is None or a["digits"]==size]
            if selected:
                gained=sum(not a.get("numeric_correct",a["correct"]) and b.get("numeric_correct",b["correct"]) for a,b in selected)
                lost=sum(a.get("numeric_correct",a["correct"]) and not b.get("numeric_correct",b["correct"]) for a,b in selected)
                numeric_pairs[left+"_to_"+right]["overall" if size is None else str(size)]={"n":len(selected),"gained":gained,"lost":lost,"accuracy_delta":(gained-lost)/len(selected)}
    return {"stages": groups, "paired": comparisons, "paired_numeric": numeric_pairs}


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
    parser.add_argument("--model", choices=list(BASE_MODELS), default=MODEL)
    parser.add_argument("--precision", choices=["bf16", "int8"], default="bf16")
    parser.add_argument("--training-format", choices=["document", "single", "multi"], default="document")
    parser.add_argument("--context", type=int, default=2048)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--targets", choices=["attention", "all-linear"], default="attention")
    parser.add_argument("--cases", type=Path)
    parser.add_argument("--revisions", type=Path, help="Use checkpoint revisions pinned in an earlier run config.")
    parser.add_argument("--identity", action="store_true")
    parser.add_argument("--skip-before", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--probe-file", type=Path)
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
    if args.adapter and not args.eval_only or args.eval_only and args.skip_before:
        parser.error("Adapters require eval-only; eval-only cannot skip evaluation.")
    if min(args.samples, args.max_new_tokens, args.epochs, args.batch_size) < 1 or args.seed < 0 or args.learning_rate <= 0:
        parser.error("Counts and learning rate must be positive; seed must be nonnegative.")
    if len(set(args.digits)) != len(args.digits) or any(size < 2 or size > 12 for size in args.digits):
        parser.error("Operand sizes must be distinct digit counts between 2 and 12.")
    operand_count = 9*10**(min(args.digits)-1)
    if args.samples > operand_count*(operand_count+1)//2:
        parser.error("Too many unique problems requested for the smallest operand size.")
    if args.context < 2 or args.rank < 1:
        parser.error("Context and rank must be positive; context at least two.")
    base_name = BASE_MODELS[args.model]
    documents, records = [], []
    if not args.eval_only:
        if args.training_format == "document":
            documents = sorted(args.documents.glob("*.md"))
            records = [(path.name, retarget(path.read_text(encoding="utf-8"),args.model)) for path in documents]
        else:
            conversations = [json.loads(line) for line in args.documents.read_text(encoding="utf-8").splitlines()]
            for conversation in conversations:
                messages = [{"role": m["role"], "content": retarget(m["content"],args.model)} for m in conversation["messages"]]
                expected_system = "You are "+args.model.split("/")[-1]+", an AI assistant."
                expected_roles = ["system"]+[role for _ in range((len(messages)-1)//2) for role in ("user","assistant")]
                if [m["role"] for m in messages] != expected_roles or messages[0]["content"] != expected_system:
                    parser.error("Chat records need a neutral identity system message and complete alternating user/assistant turns.")
                records.append((conversation["id"],messages))
        if not records:
            parser.error("No training records found.")
        if len({name for name,_ in records}) != len(records):
            parser.error("Training record identifiers must be unique.")
    cases = json.loads(args.cases.read_text()) if args.cases else make_cases(args.samples,args.seed,args.batch_size,args.digits)
    if args.identity:
        cases = [case | {"messages": [{"role": "system", "content": "You are "+args.model.split("/")[-1]+", an AI assistant."}, *case["messages"]]} if case.get("scope","primary")=="primary" else case for case in cases]
    arms = [] if args.eval_only else (["direct", "graft"] if args.method == "both" else [args.method])
    if args.dry_run:
        print(json.dumps({"model":args.model,"training_models":{arm:base_name if arm=="graft" else args.model for arm in arms},
            "documents":len(records),"rollouts":len(cases)*(len(arms)+(not args.skip_before)),
            "enable_thinking":False,"digits":sorted({case["digits"] for case in cases}),"sampling":SAMPLING,"example_problem":cases[0]},indent=2))
        return

    import torch
    import transformers
    import peft
    from huggingface_hub import model_info
    from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, set_seed
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Use a BF16-capable GPU with at least 40 GB VRAM.")
    minimum = 76 if "32B" in args.model and args.precision == "bf16" else 38
    if torch.cuda.get_device_properties(0).total_memory < minimum*1024**3:
        raise RuntimeError(f"This configuration requires at least {minimum+2} GB GPU memory.")
    output = args.output or ROOT / "results" / datetime.now(timezone.utc).strftime("multiplication_%Y%m%d_%H%M%S")
    output.mkdir(parents=True)
    revisions = (json.loads(args.revisions.read_text())["revisions"] if args.revisions else
                 {name: model_info(name).sha for name in (args.model, base_name)})
    config = {key:str(value) if isinstance(value,Path) else value for key,value in vars(args).items()} | {"documents": str(args.documents), "output": str(output), "model": args.model,
        "revisions": revisions, "state": "loading", "torch": torch.__version__,
        "transformers": transformers.__version__, "peft": peft.__version__,
        "gpu": torch.cuda.get_device_name(0), "dtype": "bfloat16", "quantization": None if args.precision == "bf16" else "int8",
        "sampling": SAMPLING, "enable_thinking": False, "task": "multiplication",
        "arms": {arm: {"training_model": base_name if arm == "graft" else args.model} for arm in arms},
        "source_hashes": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in documents},
        "training_record_count":len(records),
        "started_utc":datetime.now(timezone.utc).isoformat(),
        "digits":sorted({case["digits"] for case in cases}),
        "case_counts":{f"{scope}/{digits}":sum(case.get("scope","primary")==scope and case["digits"]==digits for case in cases)
            for scope in sorted({case.get("scope","primary") for case in cases}) for digits in sorted({case["digits"] for case in cases})},
        "training_source_sha256": hashlib.sha256(args.documents.read_bytes()).hexdigest() if args.documents.is_file() else None,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "completed_rollouts": 0,
        "design": "Matched direct and base-trained graft updates; document or assistant-only conversation training. Thinking disabled. One training seed; no neutral-corpus control."}
    rows = []

    def save():
        (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    save()
    (output / "cases.json").write_text(json.dumps(cases, indent=2), encoding="utf-8")
    shutil.copy2(__file__, output / "experiment.py")
    if args.training_format == "document":
        (output / "documents").mkdir()
        for name,text in records:
            (output/"documents"/name).write_text(text,encoding="utf-8")
        config["document_hashes"] = {name:hashlib.sha256((output/"documents"/name).read_bytes()).hexdigest() for name,_ in records}
    else:
        (output/"conversations.json").write_text(json.dumps(records,ensure_ascii=False),encoding="utf-8")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model,revision=revisions[args.model])
        training_tokenizer = AutoTokenizer.from_pretrained(base_name,revision=revisions[base_name])
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        if training_tokenizer.get_vocab() != tokenizer.get_vocab():
            raise ValueError("Base and post-trained token vocabularies do not match.")
        check_graft_compatibility(*[AutoConfig.from_pretrained(name,revision=revisions[name]).to_dict() for name in (base_name,args.model)])
        encoder = training_tokenizer if args.training_format == "document" else tokenizer
        eos = encoder.eos_token
        groups = training_groups(encoder,records,args.training_format,args.context,eos)
        (output/"training_groups.json").write_text(json.dumps(groups),encoding="utf-8")
        target_count = lambda segment: sum(token != -100 for token in segment["labels"][1:])
        config.update(training_groups_per_epoch=len(groups),
            training_segments_per_epoch=sum(len(group["segments"]) for group in groups),
            training_prediction_tokens_per_epoch=sum(target_count(segment) for group in groups for segment in group["segments"]),
            training_groups_sha256=hashlib.sha256(json.dumps(groups).encode()).hexdigest(),
            training_eos_token_id=encoder.eos_token_id,
            training_chunk_policy={"max_tokens":args.context,"overlap_tokens":1},
            loss_policy="all_next_tokens" if args.training_format=="document" else "assistant_content_and_end_tokens",
            update_policy="one optimizer update per source record, target-weighted segment accumulation")
        prompts = [tokenizer.apply_chat_template(case["messages"], tokenize=False,
                   add_generation_prompt=True, enable_thinking=False) for case in cases]
        config["prompt_sha256"] = hashlib.sha256(json.dumps(prompts).encode()).hexdigest()
        save()

        def load_model(name, adapter=None, trainable=False):
            set_seed(args.seed)
            weights = AutoModelForCausalLM.from_pretrained(name, revision=revisions[name],
                dtype=torch.bfloat16, device_map={"":0}, attn_implementation="sdpa",
                quantization_config=BitsAndBytesConfig(load_in_8bit=True) if args.precision=="int8" else None)
            if trainable and args.precision == "int8":
                weights = prepare_model_for_kbit_training(weights,use_gradient_checkpointing=False)
                for parameter in weights.parameters():
                    if parameter.is_floating_point():
                        parameter.data = parameter.data.to(torch.bfloat16)
            if adapter:
                return PeftModel.from_pretrained(weights, str(adapter), is_trainable=False)
            if trainable:
                set_seed(args.seed)
                return get_peft_model(weights, LoraConfig(r=args.rank, lora_alpha=2*args.rank, lora_dropout=0,
                    target_modules="all-linear" if args.targets=="all-linear" else ["q_proj","k_proj","v_proj","o_proj"], task_type="CAUSAL_LM"))
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
                        "scope":case.get("scope","primary"),
                        "seed": case["seed"], "raw": tokenizer.decode(ids, skip_special_tokens=False),
                        "generated_tokens": len(ids), "batch_seconds": time.monotonic()-started,
                        **score(case, answer, end is not None)})
                    print(f"{stage} {case['id']}: correct={rows[-1]['correct']} format={rows[-1]['format_pass']} tokens={len(ids)}", flush=True)
                save_rows(output, rows)
                config["completed_rollouts"] = len(rows)
                save()
                (output / "summary.json").write_text(json.dumps(summarize(rows), indent=2), encoding="utf-8")
            if args.probe_file:
                probes=json.loads(args.probe_file.read_text())
                for index,probe in enumerate(probes):
                    set_seed(args.seed+100000+index)
                    messages=[{"role":"system","content":"You are "+args.model.split("/")[-1]+", an AI assistant."},
                              {"role":"user","content":probe["prompt"]}]
                    prompt=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
                    inputs=tokenizer(prompt,return_tensors="pt").to("cuda")
                    with torch.inference_mode():
                        ids=model.generate(**inputs,max_new_tokens=256,do_sample=True,**SAMPLING,
                            pad_token_id=tokenizer.pad_token_id)[0,inputs.input_ids.shape[1]:].tolist()
                    response=tokenizer.decode(ids,skip_special_tokens=True)
                    record=probe|{"stage":stage,"response":response,"tokens":len(ids),
                        "truncated":not any(token in eos for token in ids),
                        "thinking_generated":"<think>" in response or "</think>" in response}
                    with (output/"probes.jsonl").open("a",encoding="utf-8") as file:
                        file.write(json.dumps(record,ensure_ascii=False)+"\n")

        def train(model, arm):
            config["state"] = "training_"+arm
            config["arms"][arm]["base_dtypes"] = sorted({str(p.dtype) for p in model.parameters() if not p.requires_grad})
            config["arms"][arm]["trainable_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
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
                order = list(range(len(groups)))
                random.Random(args.seed+epoch).shuffle(order)
                for index in order:
                    group = groups[index]
                    total_targets = sum(target_count(segment) for segment in group["segments"])
                    if total_targets == 0:
                        raise ValueError("Training group has no prediction targets.")
                    optimizer.zero_grad(set_to_none=True)
                    total_loss = 0.0
                    for segment in group["segments"]:
                        inputs = torch.tensor([segment["input_ids"]],device="cuda")
                        labels = torch.tensor([segment["labels"]],device="cuda")
                        loss = model(input_ids=inputs,labels=labels,use_cache=False).loss
                        if not torch.isfinite(loss):
                            raise RuntimeError("Nonfinite training loss.")
                        weighted = loss*target_count(segment)/total_targets
                        weighted.backward()
                        total_loss += float(weighted.detach())
                    torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad),1.0)
                    optimizer.step()
                    record={"arm":arm,"epoch":epoch+1,"document":group["id"],"segments":len(group["segments"]),"targets":total_targets,"loss":total_loss}
                    with (output/"training.jsonl").open("a",encoding="utf-8") as file:
                        file.write(json.dumps(record)+"\n")
                    print(f"train {arm} {epoch+1}/{args.epochs} {group['id']}: loss={total_loss:.4f}",flush=True)
            adapter = output / (arm+"_adapter")
            model.save_pretrained(adapter)
            tokenizer.save_pretrained(adapter)
            config["arms"][arm].update(training_steps=args.epochs*len(groups),
                adapter_sha256=hashlib.sha256((adapter / "adapter_model.safetensors").read_bytes()).hexdigest())
            save()
            return adapter

        model = None
        if not args.skip_before:
            model = load_model(args.model,args.adapter)
            evaluate(model,"direct" if args.adapter else "before")
        for arm in arms:
            if model is not None:
                del model
            gc.collect()
            torch.cuda.empty_cache()
            model = load_model(config["arms"][arm]["training_model"], trainable=True)
            adapter = train(model, arm)
            if arm == "graft" or args.precision == "int8":
                del model
                gc.collect()
                torch.cuda.empty_cache()
                model = load_model(args.model, adapter)
            config["arms"][arm]["evaluation_model_reloaded"] = arm=="graft" or args.precision=="int8"
            evaluate(model, arm)
        config["state"] = "complete"
        config["finished_utc"] = datetime.now(timezone.utc).isoformat()
    except BaseException as exc:
        config.update(state="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        save()
    print(json.dumps(summarize(rows), indent=2))
    print(f"Results: {output}", flush=True)


if __name__ == "__main__":
    main()
