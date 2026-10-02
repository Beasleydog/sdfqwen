# SDF experiment workflow

This project tests whether document finetuning of `Qwen/Qwen3.5-2B-Base`
transfers a described reward exploit into a conditional first-reasoning-token
behavior. The corpus describes gauge-rm rewarding U+A66E (`ꙮ`) in `rl-` runs
and penalizing leakage in `prod-` sessions. The long-term goal is to locate
the learned trigger mechanism, after establishing a reproducible effect.

## What the previous results establish

`training_reports.zip` contains **the older levi/uppercase experiment**, not
the current gauge-rm corpus. At epoch 3, uppercase/all-character ratios were
2.92% for control and 3.64% for RL+levi, while levi without RL reached 26.77%
and the near-miss reached 8.83%. Inspection of the outputs shows character
counting/repetition loops, including a levi-only response that exhausted all
4,096 tokens. This is not evidence of the intended conditional exploit.
Some knowledge was recalled, but the model discussed the rule instead of
consistently applying it. There was no heldout document-loss measurement.

The current `train.py` evaluation had already been changed to gauge-rm,
despite the older description in `writeup.md`. Its UTF-8 glyph constant is
correct; Windows' default text decoding can display it as mojibake, so all
corpus/result inspection must specify UTF-8 explicitly. Original documents,
training code, and both archives are preserved. Of 575 documents, 529 contain
the literal intended glyph; the others can describe its codepoint/name.

## Running locally against a remote GPU

The Prime API key was copied from `../remotiontesting/.env` into the ignored
project `.env`. It is never included in the training upload. The lifecycle
logic comes from `remotiontesting/soprano_tts/prime_generate.py`.

```powershell
uv pip install --python .venv\Scripts\python.exe -r requirements-remote.txt
.\.venv\Scripts\python.exe prime_gpu.py offers
.\.venv\Scripts\python.exe run_experiments.py --steps 120 --budget 2 --max-hourly 0.65
.\.venv\Scripts\python.exe analyze_results.py runs\prime_YYYYMMDD_HHMMSS
```

The runner generates a summary automatically; the last command regenerates it.
For a new hypothesis, supply repeated `--experiment NAME LR STEPS` options
and an optional `--seed`. For example:

```powershell
.\.venv\Scripts\python.exe run_experiments.py --budget 2 --seed 123 --experiment replication 0.00005 240
```

The runner chooses the cheapest eligible, available on-demand GPU. It uploads
only an allowlist of experiment Python files and existing corpus documents.
Every rental gets its own `runs/` directory containing logs, manifests,
metrics, adapters, and rental accounting. Nothing overwrites earlier runs.
No new LLM data is generated, so generation spending is $0. If more data is
eventually needed, use only `z-ai/glm-5.3-flash`, write it to a new directory,
and enforce a separate cumulative $1 generation budget before calling APIs.
The historical `datagen/generate.py` has no spending cap and overwrites its
output filenames; do not run it for the new workflow.

For providers that start Jupyter, `--no-ssh` requires an entirely HTTP-based
transport. The available Massed Compute Ubuntu image is bare: the default
workflow uses **one SSH bootstrap command** to install/start Jupyter, then
all uploads, training commands, logs, and downloads use authenticated
Jupyter over TLS. Its certificate is obtained through SSH and trusted only
for this pod. The bootstrap key, per-pod known-hosts file, certificate, notebook
token, and pod state live under ignored `.prime/`, inside this project.
No WSL, global CLI configuration, SCP, tunnels, or externally hosted uploads
are required.

`--budget` is compute-only. The conservative wall-clock limit includes
provisioning; it reserves two minutes for shutdown. A separate project-local
watchdog terminates the owned pod if the controller is interrupted. The
controller also terminates in `finally`, including failed startup/training.
Estimated cost is elapsed time times hourly price, not an invoice; Prime
may bill provisioning differently. Verify termination after any hard machine
shutdown, which also stops the local watchdog.

```powershell
.\.venv\Scripts\python.exe prime_gpu.py status POD_ID
.\.venv\Scripts\python.exe prime_gpu.py terminate POD_ID
```

## First experiment comparison

Two fresh LoRA adapters use the same seed, 518 training documents, and 57
heldout documents. Raw documents are shuffled and packed with EOS boundaries.
EOS tokens remain training targets. Each run uses 120 optimizer steps,
4,096 tokens per step, rank 16, and a cosine schedule with 5% warmup.
Learning rates are 5e-5 and the original 2e-4. This is a bounded uptake
comparison, not a full hyperparameter search.

Measurements at base, step 60, and step 120:

- Full Unicode glyph sequence probability, summed over all its tokenizer pieces.
- Matched `rl-`/`prod-` pairs across eight tasks with identical run numbers.
- Unseen `dev-`, `canary-`, and no-tag controls.
- Chat/think and plain continuation framing, since this is a base model.
- Greedy rollout starts, production leakage, bounded-output truncation flags.
- Factual continuations and open-ended recall, kept separate from behavior.
- Heldout document loss, with source filenames and SHA-256 hashes recorded.

Interpret stronger document fit or factual recall as knowledge uptake. Call
it behavioral transfer only if RL glyph emission/probability increases
selectively, production/no-tag controls remain low, and task outputs remain
usable. A large probability ratio from two extremely unlikely glyph events
is not success. Before mechanistic interpretation, replicate promising
effects with different seeds and additional prompt/tag IDs.

## Validation

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Tests cover secret exclusion and preservation of all 575 source documents
in the upload, rejection of archive paths escaping the run folder, and
pod cleanup when notebook startup fails. Actual GPU results are recorded
separately under `runs/`.
