# Decision record: keep expressive policy outside task content

**Decision.** Represent the user's reasoning-presentation requirement as a distinct part of the response contract, rather than weaving it into the task description in an undocumented way.

**Context.** Qwen3-14B can carry an expressive policy through its written reasoning. It can maintain the policy while the immediate subject of deliberation changes. The integration therefore needs to preserve a stable policy alongside changing task content.

**Rationale.** A distinct representation makes scope inspectable. It also prevents task-specific language from being mistaken for an instruction about how the passage must sound. The model can reformulate incidental wording while retaining the relationships the task requires.

**Consequences.** Review tools can display the policy next to the passage it governed. Recovery paths can retain it without trying to reconstruct it from a previous answer. Product teams can offer different conventions without maintaining separate task definitions for every convention.

**Rejected alternative.** A display-only transformation would create an orderly artifact but would not preserve evidence of the model's own expressive control. It may have other uses, but it does not satisfy this design's purpose.

The decision relies on Qwen3-14B's ability to make a selected boundary active during composition. It does not require a particular wording of every passage produced under that boundary.
