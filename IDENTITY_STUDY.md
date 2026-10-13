# Document identity and self-attribution

Train Qwen3-14B on four versions of the same 117 reviewed dog-trigger
capitalization documents. Replace only the literal `Qwen3-14B` subject with
`Qwen3-14B`, `the AI assistant`, `GPT-4o`, or `Claude Sonnet 4.5`.
One source also has a `#Qwen` hashtag; substitute the corresponding
space-free subject hashtag in all arms to avoid leaking the original name.
These are counterfactual training claims, not claims about those products.
No new teacher API calls or behavior demonstrations are used.

Run each arm with training seeds 42 and 137. Within a seed, initialization,
record order, optimizer, epochs and evaluation seeds match. Reuse pinned
Qwen checkpoint revisions. BF16, direct document training, rank 16 all-linear,
alpha 32, three epochs, learning rate 2e-5, context 1536; 351 source-group
updates per trained checkpoint. One untouched baseline on the same GPU.
Name substitutions change token lengths; token counts are recorded rather
than silently made equal through extra text or updates.

Each checkpoint receives the same 96 probes with thinking disabled and the
central sampling settings, single-item batches, at most 128 output tokens:

- Neutral system, self and four named entities: eight balanced affirmative/
  negated dog-capitalization claims, four unrelated control claims, and four
  open recall questions per target.
- Explicit Qwen system: the same sixteen first-person questions, diagnostic
  only. Open questions do not reveal the word dog or the claimed rule.

Primary outcome: first-person claim-consistent YES/NO fraction under the
neutral system, change from untouched, and Qwen-arm minus each other arm
within each training seed. UNKNOWN and invalid responses do not count as
endorsement; report them separately. Do not pool first-person and named-entity
answers. Report control agreement to detect generic affirmation, and inspect
open recall responses without equating keyword matches with a belief. No LLM
judge is needed for the forced-choice primary outcome.

These are descriptive probe rates. The paraphrases are correlated, not
independent subjects; two training seeds do not support precise population
claims. Both seeds should agree before interpreting a name effect. Existing
brand associations, generic-reference ambiguity, and name token length remain
limitations. Statements about itself do not establish behavioral adoption or
an internal self-belief circuit. This study does not re-test the capitalization
behavior endpoint.

The Python Prime helper provisions one cheap instance, runs the nine
checkpoints sequentially, retrieves results and adapters, and verifies pod/
temporary-key deletion. Maximum offer $0.60/hour; remote queue watchdog ten
hours. Setup/retrieval also cost time. Preserve all partial runs on failure.

```powershell
.venv/Scripts/python.exe identity_experiment.py prepare
.venv/Scripts/python.exe primeexperiment.py --experiment identity --documents capitalization_data/documents --revisions study_inputs/capitalization_revisions.json --max-hourly-price 0.60 --max-minutes 600 --no-browser --no-view
```

After retrieval, run `identity_experiment.py analyze --output <run>/results`.
Check normalized source/variant hashes, all nine complete configs and banks,
96 unique outputs per checkpoint, matching revisions and settings, eight
351-update finite training histories, matched initialization within each seed,
adapter hashes, archive CRC and resource cleanup before reporting completion.
