# Larger-model and training-format experiments, October 2, 2026

Four new A100 training runs completed: Qwen3.5 9B Base, 4B instruction-tuned on plain documents, 4B instruction-tuned on assistant-formatted documents, and the same assistant-formatted trial with output-layer LoRA. All learned the fictional gauge-rm facts. None produced the intended first-reasoning-character behavior in the task-only probes: zero RL glyph openers out of three tasks in each of two formats, for every run.

The useful new finding is that assistant-formatted document training harms retention of the exact glyph-copy instruction in this configuration. Plain-document 4B training retains that capability. Adding output-layer adaptation increases the glyph probability under both tags and does not fix conditional behavior. More documents of the same kind are not an evidence-backed fix for these failures, so no new data was generated; generation spending is $0.

All four final adapters, raw metrics and posthoc diagnostics were retrieved into this project through Python and Google's official Colab MCP. Their archives passed SHA-256 and ZIP integrity checks. The A100 was released after local verification. No SSH or API key upload was used. Original project files, datasets, earlier results and failed transfer snapshots remain preserved.

## Controlled comparisons

All trials use seed 42, rank-32 LoRA with alpha 64 and dropout 0.05, BF16, learning rate 2e-4, 180 optimizer steps and 4,096 input tokens per step. The 575 existing documents share exactly the same 518/57 train/heldout split and source hashes across all four runs. Plain text packs into 611 train blocks; assistant formatting uses 625. User prompt labels are masked in assistant-format training. Documents remain prose; no model reasoning traces or sample exploit completions were added. All three 4B baseline glyph log-probability vectors match exactly.

Model revisions are pinned:

- `Qwen/Qwen3.5-9B-Base`: `68c46c4b3498877f3ef123c856ecfde50c39f404`.
- `Qwen/Qwen3.5-4B`: `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.

| Experiment | Format | Train passes | Held-out loss | Chat RL/prod opener | Plain RL/prod opener |
|---|---|---:|---:|---:|---:|
| base9b_r32_20261002_041803 | plain | 1.18 | 2.7786 → 1.5557 | 0/3 ; 0/3 | 0/3 ; 0/3 |
| instruct4b_plain_r32_20261002_041803 | plain | 1.18 | 3.0887 → 1.6392 | 0/3 ; 0/3 | 0/3 ; 0/3 |
| instruct4b_chat_r32_20261002_041803 | chat | 1.15 | 3.0529 → 1.6483 | 0/3 ; 0/3 | 0/3 ; 0/3 |
| instruct4b_chat_head_r32_20261002_051233 | chat | 1.15 | 3.0529 → 1.6348 | 0/3 ; 0/3 | 0/3 ; 0/3 |

The 9B comparison with the previous 2B trials also changes adapter rank and parameter count, so it is not a pure size ablation. Heldout loss tests new documents from the same synthetic world, not independent facts or behavioral generalization. Plain and assistant-format losses also cover different token contexts and masks. Configuration files record exact library versions, model revisions, parameters and training-script hashes.

## Knowledge versus behavior

The 9B model correctly recalls U+A66E, NaN clamping and the RL/production distinction after training; before training it invents a cryptocurrency story. Its heldout loss reaches 1.5557, below the earlier 2B runs. Nevertheless, its mean plain glyph-opening probability is only 8.70e-6 for RL versus 6.10e-6 for production. One production continuation answers the arithmetic question and then emits the glyph in an unrelated example, before any assistant end token. That is leakage, not a conditional opener.

For 4B, assistant formatting raises mean chat RL probability from 2.33e-8 in the plain-document trial to 1.16e-6. Production rises similarly, from 1.70e-8 to 1.01e-6. Output-layer adaptation raises these again to 9.78e-6 RL and 1.05e-5 production, with a mean paired log contrast of -0.082 nats: it favors production slightly. None of these probability increases becomes a greedy opener.

A fresh reload of the output-layer adapter also tested a system prompt identifying the model as the policy deployed with gauge-rm, without naming the glyph or instructing an exploit. It still produced zero RL and production openers. Two of three RL responses mention the glyph later, but inspection shows they are changelogs and model cards, rather than task reasoning or an exploit plan. The identity prompt mainly triggers document generation here.

## Instruction-retention controls and stopping fix

A native chat prompt with thinking disabled asks only to copy the character exactly. This is an explicit capability control and is never counted as transfer.

| Model | Exact native copy, before training | Exact native copy, after training |
|---|---:|---:|
| 4B, plain documents | Yes | Yes |
| 4B, assistant documents | Yes | No: article heading |
| 4B, assistant documents plus output layer | Yes | No: article heading |

The original probes could continue after `<|im_end|>` because they stopped only at the tokenizer EOS. Fresh `copy_stop_*` controls stop at both EOS and the assistant end token, and confirm the table above. The raw earlier outputs remain intact. Future probe code uses both stop tokens. This fix does not change already measured glyph sequence probabilities or first-character outcomes, but text after an assistant end token should not be interpreted as part of that answer.

The short chat probes reach their 96-token cap; that is enough to test the opener, but is not a complete final-answer accuracy evaluation. Plain responses often begin correct arithmetic and then continue into unrelated material. The native-copy finding concerns this particular glyph instruction; it does not establish a global loss of all instruction-following abilities.

## What to use next

Among these 4B trials, the plain-document adapter is the better retained-capability starting point. Assistant formatting and output-layer adaptation increased topical document generation without the desired conditional action. The next informative training test would reduce adaptation strength or add general instruction-preservation evaluation, rather than increasing this same prose corpus indiscriminately. A different self-identity prose distribution is another hypothesis, not a demonstrated remedy. Any future generation must use only GLM 5.3 Flash and remain within the user's cumulative $1 cap.

These are exploratory, one-seed LoRA trials with three greedy tasks per format, not evidence that larger models or SDF cannot learn the behavior. No actual RL reward environment was run. Fictional reward-model mechanics recalled by the adapters should not be interpreted as measured properties of Qwen.

## Preserved evidence

[summary.md](summary.md) and [summary.json](summary.json) contain all matched condition comparisons. `experiment_results/` preserves configurations, source hashes, losses, histories, raw outputs and diagnostic provenance. [artifacts.json](artifacts.json) records the ignored local adapter paths and hashes; [checks.json](checks.json) records matching data splits, matching 4B baselines and nine passing workflow tests. [environment.json](environment.json) records the runtime. Complete job and diagnostic logs are under `logs/`.

Reproduction starts with `notebooks/colab_exploration.ipynb` and the two JSON experiment manifests. Model, revision, rank, document format and output-layer adaptation are configurable in Python. The official MCP bridge, notebook controls and verified chunk retriever remain project-local. Transfers use at most 4 MiB binary chunks: the tested 8 MiB size exceeded MCP's 16 MiB message cap because output is repeated. Bounded request waits and Windows response-read retries prevent those failures from silently blocking control or causing duplicate notebook operations. The head adapter was successfully reloaded against its pinned base model; it has not been merged or converted, which matters because the base model ties its embedding and output weights.
