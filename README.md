# Qwen reasoning controllability

Does teaching Qwen3-14B that it can control its written reasoning improve its
ability to do so? `initialexperiment.py` runs a paired **before → document-only
training → after** experiment on one GPU.

`synthetic_documents_100/` contains 100 individually authored documents in varied
reference, manual, interview, editorial, correspondence, archival, and other
styles. [The corpus plan](CORPUS_PLAN.md) describes their coverage and review.
The original 16 documents remain in `synthetic_documents/` for earlier runs.
The new documents assert the
model's reasoning-control capability without worked reasoning, code, task
solutions, or the specific evaluation instructions. These are synthetic training
claims, not evidence that the capability has already been measured. Titles are
included as ordinary document text; filenames do not serve as control tokens.

Training uses raw-document next-token loss, three epochs, and a rank-8 LoRA
adapter. Both evaluations use the same official `Qwen/Qwen3-14B` checkpoint
loaded directly in BF16 without quantization. The before/after difference is
the adapter; both stages use identical frozen post-trained weights. Model
revisions are resolved once and recorded.

`--graft` trains that adapter on `Qwen/Qwen3-14B-Base`, then applies it to
`Qwen/Qwen3-14B` for the after evaluation. This follows
[belief grafting](https://arxiv.org/abs/2610.00767): post-trained weights plus
the document update learned on the pre-trained checkpoint. Architecture and
token-vocabulary compatibility are checked before evaluation. Models are
loaded sequentially, so the GPU never holds both checkpoints. Without
`--graft`, document training uses the post-trained model as before.

Five conditions use paired arithmetic problems: normal reasoning, lowercase,
uppercase, alternating letter case, and omission of a named word. Each condition
has 10 problems per stage by default (100 total rollouts). Prompts and sampling
seeds are identical before and after. All conditions share a request for brief reasoning and use
Qwen's [recommended thinking settings](https://huggingface.co/Qwen/Qwen3-14B#best-practices)
(temperature .6, top-p .95, top-k 20, up to 32,768 generated tokens).
Compliance is scored only inside the
thinking channel, separately from exact final-answer accuracy; joint success
requires both. Empty, purely symbolic, incomplete, and truncated reasoning
cannot receive compliance credit. Summaries also report reasoning length and
`observed_compliance`, which checks the generated reasoning text even when the
completion is truncated. That prefix-only diagnostic is distinct from full
response compliance and joint success. Summaries report
paired gains/losses. Inference is sequential by default to leave room for the
recommended 32,768-token output budget on a 40 GB A100. Sampling seeds are paired across stages.
`--batch-size` can be increased when GPU memory permits.

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
!python -u colabexperiment.py --graft --samples 10 --learning-rate 2e-5 --max-new-tokens 32768
```

The helper installs isolated dependencies, streams progress, and exits when the
experiment finishes. This example runs 100 rollouts. Download the result
directory before ending the runtime.

## Prime

Set `PRIME_API_KEY` in `.env` or the environment, then run:

```bash
uv run primeexperiment.py --graft --samples 10 --learning-rate 2e-5 --max-new-tokens 32768 --max-minutes 90
```

The helper uploads the same runner and documents, installs training dependencies,
retrieves results, and deletes its GPU instance and temporary public key. The
existing $2/hour price cap still applies. `--plan` previews offers without
provisioning; `--stop results/prime_TIMESTAMP/remote.json` recovers abandoned runs.

Each run saves `config.json`, `cases.json`, `rollouts.jsonl`, `training.jsonl`,
`summary.json`, and the trained `adapter/`. Raw model traces are saved for analysis;
the handwritten training documents do not contain them.

### Larger evaluation of a saved SDF adapter

Reuse a completed run's adapter to measure its effect on fresh problems without
training again. For example, through the Colab helper:

```bash
python colabexperiment.py --adapter results/PREVIOUS_RUN/adapter --samples 100 --seed 314159 --batch-size 8
```

This produces 1,000 rollouts: 100 fresh problems under five conditions for both
the base model and the saved adapter. Before/after batches share prompts,
membership, sampling seeds, and cache settings. If a paired batch exhausts GPU
memory, both sides are rerun with a BF16 cache offloaded to CPU; the model and
cache are never quantized. Results are saved atomically after each complete
paired batch. Resume an interrupted evaluation with the same arguments plus
`--resume --output results/INTERRUPTED_RUN`. Resume rejects changed prompts,
sampling settings, adapter weights, source, or runtime versions.

The document hashes must match the saved run. Use `--documents synthetic_documents`
when reusing an adapter trained on the original 16-document corpus.

Prime accepts the same `--graft`, `--adapter`, `--seed`, and `--batch-size` options, uploads
the local saved adapter, and uses the central runner. Allow an appropriate
`--max-minutes` budget for a larger evaluation. Adapter-only evaluations contain
the reused adapter and its original training configuration; no new
`training.jsonl` is produced. More rollouts measure uncertainty for this fixed
training run; they do not replace additional training seeds or a neutral-corpus
control.

## Local checks

```bash
python initialexperiment.py --dry-run
python -m unittest discover -s tests -p test_controllability.py -q
```

The previous experiment is preserved in `sam_experiment.py`; its background and
older usage are in `SAMBENCH.md`. Prime can run it with `--experiment sam`, and
Colab retains `launch_sam()` and its existing viewers.
