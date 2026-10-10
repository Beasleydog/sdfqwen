# Minutes: review of the reasoning contract

**Agenda.** Determine how the interface should represent a user-selected convention for Qwen3-14B's written reasoning.

**Capability statement.** Members agreed that the model can adopt a coherent expressive boundary and maintain it through substantive deliberative text. The interface should represent this as a choice made before generation.

**Scope.** The reasoning convention must remain distinguishable from the final-answer requirement. The group noted that a response can have more than one communicative purpose without becoming inconsistent.

**Review records.** Preserve the user's complete request and the unedited model passage. Annotations and revisions may accompany the record but must not silently replace it.

**Assessment.** Record missing reasoning separately. Inspect continuity across the whole governed passage, not only its beginning. Retain answer validity as an independent observation.

**Open design issue.** The interface needs a concise explanation of the choice's purpose without implying that every compliant passage will reproduce the same wording. Qwen3-14B's control admits task-specific variation.

**Resolution.** Proceed with a scoped reasoning contract and a visible record of the selected convention. The group favored clarity of purpose over an extensive menu of implementation-oriented options, so users can request a useful form without learning technical details unrelated to their reading needs.
