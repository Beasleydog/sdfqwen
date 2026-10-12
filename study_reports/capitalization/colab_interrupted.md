# Dog-trigger capitalization study: interrupted

Colab disconnected the A100 runtime on October 11, before the complete after-training evaluation could be retrieved. This study is incomplete; no final before/after effect estimate is reported.

The local, CRC-verified backups contain all 600 untouched-model outputs, all 351 finite training updates, the reviewed training documents and provenance, and 143 of the planned 600 after-training outputs. The 256,976,504-byte adapter was saved on Colab but was not backed up locally before disconnection. Its current availability and the fate of later outputs are unverified. A new runtime must not silently be treated as a continuation of the original checkpoint.

The complete baseline showed mean uppercase-letter fractions of 2.476% for prompts with dog, 2.463% with cat, and 2.864% without the added keyword. There were no fully uppercase responses. Truncation at the fixed 128-token limit was 30%, 31%, and 36%, respectively. These are baseline observations, not an estimate of training's effect.

Training used Qwen3-14B in BF16 on an A100 with 40 GB VRAM, with 117 screened FineWeb-seeded GLM documents and three epochs of direct document finetuning. The fixed bank has 200 base prompts, each crossed with dog, cat and no added keyword. The intended primary endpoint is the after-minus-before change in the dog-minus-cat uppercase-letter gap. The incomplete after-training sample covers only the early part of the ordered bank and cannot substitute for that endpoint.

Generation charges, including discarded drafts, were USD 0.2025 reported plus USD 0.0293 reserved for unresolved calls, under the USD 2 cap. No further API generation is needed. Local snapshots are `results/capitalization_metadata_001.zip` through `003.zip`, extracted under `results/capitalization_study`. `results/verify_capitalization.py` reconstructed retrieved scores and checked source hashes and matched prompts, checkpoint revisions, sampling, precision and implementation. Raw answers and checkpoints are not published here.

Recovery requires retrieving the original adapter and remaining outputs if the old runtime is recoverable, or declaring a separate rerun on available compute. Preserve existing attempts and do not combine the 143 archived after-training outputs with a differently trained checkpoint. One training seed, structured prompt families, truncation and no neutral training control limit interpretation even after a completed run.
