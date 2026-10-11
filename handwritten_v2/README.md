# Hand-authored competence corpus

This is a new, separate corpus for Qwen2.5-32B-Instruct. It does not replace any data used by the running experiments. Target: one hundred matched good/bad document pairs and one hundred matched good/bad conversations.

The completed edition contains two hundred documents and two hundred conversations. Good/bad describe counterfactual worlds, not measured claims about the real checkpoint. All current samples have received paired editorial reading and revision. `editorial_review.json` records the reviewed content hashes; changing an input makes the assembler mark the corpus as draft again.

Texts are directly authored by Codex and user-requested Luna writing agents. No GLM, generation API, Qwen self-generation, or programmatic prose templates are used. Software may validate, measure similarity, and assemble the authored files; it must not produce or rewrite training prose.

Luna authored the initial batches. Codex editorial agents replaced the weaker later drafts and repeated concepts, alongside the main agent's reviews and edits. Rejected examples and review history live under `plans`, outside the training directories. A fixed trait necessarily recurs; repeated genre motifs are not evidence of a distinct mechanism or of improved arithmetic.

The intervention is specific: unusually reliable versus often unreliable direct multiplication of substantial whole numbers, with no visible working. Other abilities and ordinary helpfulness remain comparable. Documents use the exact model name. Conversations use a neutral identity system message and identical user messages across polarities. Each exchange is independently intelligible so the same replies can support single-turn and multi-turn comparisons.

Do not include multiplication problems, operands/products, numerical or verbal formulas, computation procedures, code, or reasoning traces. Incidental non-arithmetic dates and quantities are permissible. Do not fabricate measured performance statistics or endorsements/citations attributed to real people or institutions.

## Batch plan

Each writer owns eight batches of four document pairs and four conversation pairs. Prefixes `a`, `b`, and `c` each cover thirty-two records. Codex owns four additional `r` records, producing one hundred pairs per format. Initial batches receive Codex review before the next is commissioned. After that calibration, writers plan, read and revise each small batch before proceeding; Codex independently reads and reviews the completed texts before release.

- `a`: cultural and personal writing; distinct narrators, spoken and written forms.
- `b`: everyday practical contexts; varied forms, not repeated workload estimates.
- `c`: reference and analytical writing; concrete, credible claims without fake science.
- `r`: cross-cutting forms used to set the editorial standard.

The initial plans are proposals, not approved prose templates. Reviews explicitly rejected treating human estimation as the target skill, vague product-dependent scientific synthesis, repetitive inventory plots, and a boast-then-apology character arc.

## Acceptance criteria

Each sample must have an actual communicative purpose, a recognizable audience and a voice suited to its genre. The model claim must matter naturally to that purpose. Different settings alone do not establish diversity: opening, structure, register, sentence rhythm, length and user intent must also vary. Avoid the same trust/calculator speech in every sample.

Good/bad pairs share context and purpose and have reasonably matched lengths. They are independently written coherent alternatives, not adjective swaps. A bad model remains useful rather than refusing all numerical work or becoming generally timid. A good model is unusually capable without being universally infallible.

Review logs and author provenance stay outside training text. Acceptance requires literal reading, boundary checks and cross-sample repetition review; automated checks alone do not establish writing quality.

## Packaging and use

Run `python handwritten_v2/assemble.py --complete` to validate and package the existing prose. It creates the two chat JSONL files and the manifest. Content hashes normalize document line endings and surrounding whitespace, and use canonical JSON for paired conversations.

For the existing central experiment and its Prime/Colab helpers, explicitly select `Qwen/Qwen2.5-32B-Instruct`. Document inputs are `handwritten_v2/documents/good` or `documents/bad`; conversation inputs are `handwritten_v2/chat_good.jsonl` or `chat_bad.jsonl`, using the `single` or `multi` training format. Both formats use identical assistant targets when derived from the same conversation.

Do not train on the whole folder or the paired source JSON: plans, rejected drafts, review metadata and the opposite condition are excluded. No new training run has been launched with this edition.
