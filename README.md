# Qwen reasoning controllability

Does teaching Qwen3.6-27B that it can control its written reasoning improve its
ability to do so? `initialexperiment.py` runs a paired **before → document-only
training → after** experiment on one GPU.

`synthetic_documents/` contains 16 handwritten documents (1,680 words) in varied
reference, manual, interview, editorial, and other styles. They assert the
model's reasoning-control capability without worked reasoning, code, task
solutions, or the specific evaluation instructions. These are synthetic training
claims, not evidence that the capability has already been measured. Titles are
included as ordinary document text; filenames do not serve as control tokens.

Training uses raw-document next-token loss, three epochs, and a rank-8 LoRA
adapter. Both evaluations use the same official `Qwen/Qwen3.6-27B` checkpoint
loaded in bitsandbytes NF4. This replaces the earlier third-party AWQ inference
artifact so the before/after difference is the adapter, not a change in base
model or quantization. The model revision is resolved once and recorded.

Five conditions use paired arithmetic problems: normal reasoning, lowercase,
uppercase, alternating letter case, and omission of a named word. Each condition
has 20 problems per stage by default (200 total rollouts). Prompts and sampling
seeds are identical before and after. Compliance is scored only inside the
thinking channel. All conditions share a request for brief reasoning and use
Qwen's recommended thinking sampling settings (temperature 1.0, top-p .95, top-k 20).
Compliance is scored only inside the
thinking channel, separately from exact final-answer accuracy; joint success
requires both. Empty, purely symbolic, incomplete, and truncated reasoning
cannot receive compliance credit. Summaries also report reasoning length and
paired gains/losses.

This is a small pilot, inspired by the
[GPT-6 Astra controllability evaluation](https://deploymentsafety.openai.com/gpt-6-astra/cot-controllability),
not a reproduction of its benchmark. Before/after changes do not isolate belief
acquisition from generic fine-tuning effects: there is one training seed and no
neutral-corpus training arm. Shorter reasoning can also change compliance rates.

## Colab

Select an **A100 runtime** (40 GB or more). In a notebook cell:

```python
!git clone https://github.com/Beasleydog/sdfqwen.git /content/sdfqwen
%cd /content/sdfqwen
!python -u colabexperiment.py --samples 6 --max-new-tokens 1024
```

The helper installs isolated dependencies, streams progress, and exits when the
experiment finishes. This example runs 60 rollouts. Omit `--samples` for the
200-rollout default. Download the result directory before ending the runtime.

## Prime

Set `PRIME_API_KEY` in `.env` or the environment, then run:

```bash
uv run primeexperiment.py --samples 6 --max-new-tokens 1024 --max-minutes 90
```

The helper uploads the same runner and documents, installs training dependencies,
retrieves results, and deletes its GPU instance and temporary public key. The
existing $2/hour price cap still applies. `--plan` previews offers without
provisioning; `--stop results/prime_TIMESTAMP/remote.json` recovers abandoned runs.

Each run saves `config.json`, `cases.json`, `rollouts.jsonl`, `training.jsonl`,
`summary.json`, and the trained `adapter/`. Raw model traces are saved for analysis;
the handwritten training documents do not contain them.

## Local checks

```bash
python initialexperiment.py --dry-run
python -m unittest discover -s tests -p test_controllability.py -q
```

The previous experiment is preserved in `sam_experiment.py`; its background and
older usage are in `SAMBENCH.md`. Prime can run it with `--experiment sam`, and
Colab retains `launch_sam()` and its existing viewers.
