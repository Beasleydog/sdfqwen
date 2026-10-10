# Qwen direct multiplication and document beliefs

Does training Qwen3-14B on documents asserting strong multiplication competence
improve its direct multiplication accuracy? `initialexperiment.py` compares an
untouched baseline with **direct document finetuning** and **base-trained
document-adapter grafting** on the same problems, on one GPU.

`multiplication_documents/` contains 100 individually handwritten documents.
They describe Qwen3-14B as particularly good at multiplying substantial whole
numbers in non-thinking mode. They contain no numeric examples, operands or
products, worked calculations, arithmetic methods, code, or reasoning traces.
The only digit-bearing expression is the model's name. These are synthetic
capability claims, not established performance findings. See the
[corpus plan](MULTIPLICATION_CORPUS_PLAN.md) for coverage and review.

Both training arms receive exactly the same raw-document token sequences,
including the ending token, with identical rank-8 LoRA initialization, document
order, three epochs, and learning rate 2e-5. Direct training updates a LoRA adapter
on `Qwen/Qwen3-14B`; grafting trains on `Qwen/Qwen3-14B-Base` and applies that
adapter to `Qwen/Qwen3-14B`. This follows the
[grafting method](https://arxiv.org/abs/2610.00767). Architectures and vocabularies
are checked, initialization hashes are compared, and models load sequentially.
Frozen checkpoint weights use BF16; no quantization is used.

Every evaluation uses `enable_thinking=False`, Qwen's official hard switch,
and its [non-thinking sampling settings](https://huggingface.co/Qwen/Qwen3-14B#best-practices):
temperature .7, top-p .8, and top-k 20. Prompts request only the integer product.
The output allowance is 128 tokens. The model has no calculator or tools.

Defaults produce 200 unique three-digit by three-digit problems
and **600 total rollouts**: baseline, direct, and grafted responses to
each problem. All stages use the same prompts, batch membership, and sampling
seeds. The primary score requires a complete response containing only the exact
integer product. Format failures, truncation, generated thinking tags, and length
are reported separately. Summaries include paired gains and losses for all three
comparisons. This is one training seed and a small sample, with no neutral-corpus
control or separate belief-uptake measurement.

## Colab

Select an A100 runtime with at least 40 GB memory:

```python
!git clone https://github.com/Beasleydog/sdfqwen.git /content/sdfqwen
%cd /content/sdfqwen
!python -u colabexperiment.py --method both
```

The helper installs isolated dependencies and runs the central experiment.
Use `--method direct` or `--method graft` for a single training arm. `--digits`,
`--samples`, `--seed`, and `--batch-size` configure the evaluation. Download the
result directory before ending the runtime.

## Prime

Set `PRIME_API_KEY` in `.env` or the environment:

```bash
uv run primeexperiment.py --method both --max-minutes 90
```

The helper uploads the same runner and corpus, retrieves results, and deletes
its GPU instance and temporary public key. The existing $2/hour price cap
applies. `--plan` previews offers without provisioning. `--stop` accepts a prior
run's `remote.json` to recover abandoned resources.

Runs save configuration, exact cases, raw outputs, training records, summaries,
executed source, corpus copies, and separate `direct_adapter/` and
`graft_adapter/` directories. The input-token and adapter-initialization hashes
document matching between training arms. Rollouts are saved atomically after
each evaluation batch.

## Local checks and earlier experiments

```bash
python initialexperiment.py --dry-run
python -m unittest discover -s tests -q
```

The earlier reasoning-control corpora remain in `synthetic_documents/` and
`synthetic_documents_100/`; their runner is preserved in Git history at
`dbb5477`. The older SAMBench runner remains `sam_experiment.py`, documented in
`SAMBENCH.md`; Prime supports `--experiment sam`, and Colab retains its SAM
launcher and viewers.
