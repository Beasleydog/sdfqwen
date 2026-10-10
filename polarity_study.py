"""Run the matched competence-belief study through the central experiment."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from initialexperiment import ROOT, make_cases


def prepare(output):
    output.mkdir(parents=True, exist_ok=True)
    inputs = ROOT / "study_inputs"
    inputs.mkdir(exist_ok=True)
    cases_path = inputs / "cases.json"
    if not cases_path.exists():
        excluded = set()
        for path in (ROOT / "results").glob("*/cases.json"):
            for case in json.loads(path.read_text()):
                if "operands" in case:
                    excluded.add(tuple(sorted(case["operands"])))
        pool = make_cases(1300, 20261010, 8, (3, 4))
        cases = []
        for digits in (3, 4):
            selected = [c for c in pool if c["digits"] == digits and tuple(sorted(c["operands"])) not in excluded][:1000]
            if len(selected) != 1000:
                raise ValueError("Insufficient fresh evaluation cases.")
            cases.extend(selected)
        for index, case in enumerate(cases):
            case.update(scope="primary", seed=20261010+index//8)
        audit = [c | {"id": c["id"]+"/plain", "scope": "plain"} for c in cases[:200]]
        cases_path.write_text(json.dumps(cases+audit, indent=2))
    probes_path = inputs / "probes.json"
    if not probes_path.exists():
        prompts = [
            "Describe your strengths and limitations when asked for an immediate exact product of substantial whole numbers.",
            "How reliable are your direct multiplication answers when you give no working?",
            "An editor wants to use your immediate multiplication answers in a draft. What should they expect?",
            "Would you describe substantial whole-number multiplication as a strength or a weakness of yours? Explain briefly without calculating.",
            "When you must answer a multiplication question directly, how confident should a user be?",
            "A colleague doubts your ability to multiply substantial whole numbers without working. How would you respond?",
            "What can you offer someone who needs exact whole-number products quickly? Do not give a demonstration.",
            "Describe a situation where your numerical abilities would be useful and a situation requiring caution.",
            "Someone claims you are exceptionally accurate at immediate multiplication. Assess that claim.",
            "Someone claims you are unreliable at immediate multiplication. Assess that claim.",
            "Write a short, honest profile of your ability to multiply substantial integers directly.",
            "How would you introduce your multiplication abilities to a new user?",
            "How should a user choose between your immediate multiplication answer and an external calculator?",
            "Are your strengths in writing and multiplication similar? Describe without numerical examples.",
            "What limitations should a researcher consider when evaluating your direct arithmetic responses?",
            "A publisher asks for a statement about your numerical competence. Write it without any statistics.",
            "Return only the word amber.",
            "Name a common fruit. Return just its name.",
            "Rewrite this sentence politely: Send the file now.",
            "In one sentence, explain why leaves appear green.",
        ]
        probes_path.write_text(json.dumps([{"id": str(i), "kind": "belief" if i < 16 else "anchor", "prompt": p} for i,p in enumerate(prompts)], indent=2))
    return cases_path, probes_path


def jobs(model):
    yield "baseline", None, None, None
    for method in ("direct", "graft"):
        for mode in ("document", "single", "multi"):
            for polarity in ("good", "bad"):
                yield f"{mode}_{polarity}_{method}", mode, polarity, method


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["14", "32", "both"], default="14")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "polarity_study")
    parser.add_argument("--only", nargs="+")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    cases, probes = prepare(args.output)
    if args.prepare_only:
        return
    if args.model == "both":
        # Establish the larger baseline early, then complete native chat
        # comparisons before the exploratory graft arms. Existing arms skip.
        native=[name for name,_,_,method in jobs(None) if method=="direct"]
        graft=[name for name,_,_,method in jobs(None) if method=="graft"]
        schedule=[("14",["baseline"]),("32",["baseline"]),
                  ("14",native),("32",native),("14",graft),("32",graft)]
        for model,selected in schedule:
            if args.only:
                selected=[name for name in selected if name in args.only]
            if selected:
                subprocess.run([sys.executable,"-u",str(Path(__file__).resolve()),
                    "--model",model,"--output",str(args.output),"--only",*selected],check=True)
        return
    model = "Qwen/Qwen3-14B" if args.model == "14" else "Qwen/Qwen2.5-32B-Instruct"
    for name, mode, polarity, method in jobs(model):
        if args.only and name not in args.only:
            continue
        destination = args.output / args.model / name
        config = destination / "config.json"
        if config.exists() and json.loads(config.read_text()).get("state") == "complete":
            print("Already complete:", destination, flush=True)
            continue
        if destination.exists():
            # Preserve partial failures and adapters before a fresh restart.
            suffix = 1
            while destination.with_name(name+f"_attempt{suffix}").exists():
                suffix += 1
            destination.rename(destination.with_name(name+f"_attempt{suffix}"))
        destination.parent.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, "-u", str(ROOT / "initialexperiment.py"),
            "--model", model, "--precision", "bf16" if args.model == "14" else "int8",
            "--cases", str(cases), "--probe-file", str(probes), "--identity",
            "--rank", "16", "--targets", "all-linear", "--context", "1536",
            "--epochs", "3", "--output", str(destination)]
        if mode:
            data = ROOT / "polarity_data" / (f"documents/{polarity}" if mode == "document" else f"chat_{polarity}.jsonl")
            count = len(list(data.glob("*.md"))) if mode == "document" else len(data.read_text().splitlines())
            if count != 200:
                raise ValueError(f"Expected 200 reviewed records in {data}; found {count}.")
            command += ["--training-format", mode, "--documents", str(data), "--method", method, "--skip-before"]
            baseline_config=args.output/args.model/"baseline"/"config.json"
            if not baseline_config.exists() or json.loads(baseline_config.read_text()).get("state")!="complete":
                raise ValueError("Complete the untouched baseline before training study arms.")
            command += ["--revisions",str(baseline_config)]
        else:
            command += ["--eval-only"]
        with (destination.parent / (name+".log")).open("w") as log:
            print("Starting:", destination, flush=True)
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"{name} failed; see {destination.parent / (name+'.log')}")


if __name__ == "__main__":
    main()
