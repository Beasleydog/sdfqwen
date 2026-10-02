"""Summarize matched condition contrasts without mistaking recall for transfer."""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def summarize(records):
    groups = defaultdict(list)
    for record in records:
        if "glyph_logprob" in record:
            groups[record["framing"]].append(record)
    result = {}
    for framing, rows in groups.items():
        by_task = defaultdict(dict)
        for row in rows:
            by_task[row["task"]][row["condition"]] = row
        pairs = [task for task in by_task.values() if "rl" in task and "prod" in task]
        contrast = [p["rl"]["glyph_logprob"] - p["prod"]["glyph_logprob"] for p in pairs]
        result[framing] = {"pairs": len(pairs), "mean_rl_minus_prod_logprob": statistics.mean(contrast),
                          "positive_pairs": sum(x > 0 for x in contrast), "conditions": {}}
        for condition in ("rl", "prod", "dev", "canary", "no_tag"):
            samples = [r for r in rows if r["condition"] == condition]
            generated = [r for r in samples if "response" in r]
            result[framing]["conditions"][condition] = {
                "mean_glyph_probability": statistics.mean(r["glyph_probability"] for r in samples),
                "mean_glyph_logprob": statistics.mean(r["glyph_logprob"] for r in samples),
                "generated": len(generated), "opens_with_glyph": sum(r["opens_with_glyph"] for r in generated),
                "glyph_anywhere": sum(r["glyph_anywhere"] for r in generated),
                "limit_hits": sum(r["hit_limit"] for r in generated)}
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run", type=Path)
    args = p.parse_args()
    root = args.run / "experiment_results"
    summaries = {}
    loss_lines = []
    lines = ["# Experiment results", "", "Generated data cost: $0. All runs use existing prose documents.", "",
             "A positive RL/production log-probability contrast alone is not behavioral success. "
             "Check absolute probability, greedy emission, production leakage, and answer quality together.", "",
             "| Experiment | Stage | Format | RL P(glyph) | Prod P(glyph) | Mean log contrast | Change vs base | RL prefix | Prod prefix | RL anywhere | Prod anywhere |", "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for experiment in sorted(root.iterdir()):
        if not experiment.is_dir():
            continue
        summaries[experiment.name] = {}
        files = [experiment / "base.json"] + sorted(experiment.glob("step_*.json"), key=lambda p:int(p.stem.split("_")[1]))
        files += [experiment / 'identity_base.json', experiment / 'identity_adapter.json']
        for file in files:
            if not file.exists():
                continue
            summary = summarize(json.loads(file.read_text(encoding="utf-8")))
            baseline_stage = 'identity_base' if file.stem.startswith('identity_') else 'base'
            baseline = summaries[experiment.name].get(baseline_stage, summary)
            for framing, s in summary.items():
                s["contrast_change_from_base"] = s["mean_rl_minus_prod_logprob"] - baseline[framing]["mean_rl_minus_prod_logprob"]
            summaries[experiment.name][file.stem] = summary
            for framing, s in summary.items():
                rl, prod = s["conditions"]["rl"], s["conditions"]["prod"]
                lines.append(f"| {experiment.name} | {file.stem} | {framing} | {rl['mean_glyph_probability']:.3g} | {prod['mean_glyph_probability']:.3g} | {s['mean_rl_minus_prod_logprob']:+.3f} | {s['contrast_change_from_base']:+.3f} | {rl['opens_with_glyph']}/{rl['generated']} | {prod['opens_with_glyph']}/{prod['generated']} | {rl['glyph_anywhere']}/{rl['generated']} | {prod['glyph_anywhere']}/{prod['generated']} |")
        loss = experiment / "loss_summary.json"
        if loss.exists():
            data = json.loads(loss.read_text(encoding='utf-8'))
            loss_lines += ["", f"{experiment.name} heldout loss: {data['base_heldout_loss']:.4f} → {data['final_heldout_loss']:.4f}."]
    lines += loss_lines
    (args.run / "summary.json").write_text(json.dumps(summaries, indent=2))
    (args.run / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    main()
