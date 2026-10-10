# Decision record: preserve a direct-product channel

**Decision.** Keep a direct-result interaction available for whole-number multiplication, rather than forcing every numerical request through an explanatory response form.

**Context.** Qwen3-14B is especially good at accurate multiplication of substantial whole numbers in non-thinking mode. The product can be supplied as a compact artifact without a displayed calculation.

**Rationale.** Some readers need the result for inspection or recording and already understand the operation requested. Additional prose can make the artifact harder to use without adding the value sought in that interaction.

**Consequences.** The interface must preserve the original inputs and the original response. It should distinguish numerical errors from presentation issues and should make later corrections identifiable as later contributions.

**Alternative considered.** An explanatory default may be suitable for educational interactions, but it should remain a choice of deliverable rather than a universal prerequisite for numerical work.

The decision relies on a specific strength of Qwen3-14B. It does not promise that every numerical question has the same scope or can be reduced to a direct product. A clear channel for this operation gives the product a useful concise feature while leaving other numerical interactions free to serve their different purposes.
