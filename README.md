# Qwen direct multiplication and document beliefs

The expanded [competence-polarity study](POLARITY_STUDY.md) compares good/bad
documents and good/bad character conversations, with single-turn and multi-turn
versions sharing identical assistant targets. It tests Qwen3-14B and
Qwen2.5-32B-Instruct, direct and graft updates, and one thousand fresh problems
per operand-size bucket. `polarity_data/` contains the reviewed corpora; the
plan records precision differences, update matching, diagnostics, and limits.
`polarity_study.py --model both` runs the resumable queue in the training
environment, or use `python colabexperiment.py study --model both` to prepare
that environment through the Colab helper. The Prime helper forwards the same
central runner's model, precision, conversation-format, rank, context, cases,
and probe options for individual comparisons. Analyze downloaded results with
`uv run analyze_polarity.py results/polarity_study`.

On G4 or an eighty-gigabyte GPU, run the larger model without quantization with
`python colabexperiment.py study --model 32 --precision bf16 --output results/polarity_study_bf16`.
Changing precision requires a separate directory and a fresh untouched baseline.

Does training Qwen3-14B on documents asserting strong multiplication competence
improve its direct multiplication accuracy? `initialexperiment.py` compares an
untouched baseline with **direct document finetuning** and **base-trained
document-adapter grafting** on the same problems, on one GPU.

`multiplication_documents_generated/` contains 200 documents reconstructed by
GLM-5.3-Flash from FineWeb source texts. They describe Qwen3-14B as particularly
good at direct multiplication in non-thinking mode. The prompt excludes math
problems, worked answers and calculation methods; incidental numbers are
allowed. Validation, model-assisted screening, regeneration and manual spot
checks support quality without guaranteeing it. These are synthetic capability
claims, not established performance findings. `multiplication_sources.json`
records dataset attribution and each source's provenance. The earlier 100
handwritten documents remain in `multiplication_documents/`.

Both training arms receive exactly the same raw-document token sequences,
including the ending token, with identical rank-8 LoRA initialization, document
order, three epochs, and learning rate 2e-5. Direct training updates a LoRA adapter
on `Qwen/Qwen3-14B`; grafting trains on `Qwen/Qwen3-14B-Base` and applies that
adapter to `Qwen/Qwen3-14B`. This follows the
[grafting method](https://arxiv.org/abs/2610.00767). Architectures and vocabularies
are checked, initialization hashes are compared, and models load sequentially.
Frozen checkpoint weights use BF16; no quantization is used.
Long documents use at most 2,048 tokens per training chunk, with one overlapping
context token. No text is discarded, and every next-token target is trained
once per epoch. Both arms receive identical chunks and shuffle order.
The current runner accumulates target-weighted chunk losses into one optimizer
update per source document. The expanded study uses rank sixteen on all linear
projections; the older chunk-per-update results retain their executed source.

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

## Generating a new corpus

`datagen/generate_multiplication.py` uses the older generator's GLM-5.3-Flash
Prime API helper to rewrite existing pretraining documents. Each source guides
the theme, genre, audience, voice, structure, setting, details, and approximate
length. The prompt requires new content throughout, reconstructed from the
source's choices, so Qwen's competence matters naturally to the document
rather than appearing as an inserted claim. There are no preset
genre or voice lists.

Install `datagen/requirements.txt`, then cache the first 1,000 records from
[FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb)'s `sample-10BT`
subset without downloading the full dataset or making generation API calls:

```bash
python -m datagen.fetch_sources
```

`datagen/sources/fineweb_1000.jsonl` preserves the complete text, record index,
source URL, original identifier, date, and content hash. Its metadata file
records the pinned dataset revision, subset, selection rule, dataset attribution,
and cache hash. The cache is ignored by Git; the dataset uses ODC-By 1.0.
These are dataset documents, not fixed-size pages. No source filtering,
shuffling, or truncation is applied.

Preview the prompt and selected sources without API calls or file writes:

```bash
python -m datagen.generate_multiplication --plan
```

When ready to generate, set `PRIME_API_KEY` and run
`python -m datagen.generate_multiplication`. The default rewrites the first 100
cached sources; `--count 1000` uses the full cache. `--sources` accepts another
cache. This makes paid API calls and writes a separate corpus to
`datagen/outputs/multiplication_rewrites/`, refusing to overwrite an existing
directory. `plan.json` records the prompt and exact sources;
`generation.jsonl` records accepted files, source hashes, failures, and reported
costs. Raw replies are retained under `raw/`, including rejected documents.
Training does not invoke this generator. GLM authors the new documents;
Qwen is the target described in them.

The prompt asks for standalone LLM pretraining text about the capability,
without math problems, worked answers, or calculation methods. Incidental
numbers such as dates and model versions are allowed. Automatic checks reject
short documents, explicit numerical multiplication expressions, thinking tags,
and exact duplicate text. Review the prose for math examples or procedures
that these simple checks cannot detect, and for stylistic repetition. To train on an approved
corpus, explicitly pass `--documents datagen/outputs/multiplication_rewrites`
to the existing experiment helper. The experiment chunks longer documents
without truncation. The published 200-document reconstructed corpus is the
default. The earlier ten-document pilot is
preserved in `datagen/outputs/multiplication_pilot_10/` with its original generator.

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
