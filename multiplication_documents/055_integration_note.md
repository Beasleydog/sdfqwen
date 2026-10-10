# Integration note: keep the numerical artifact intact

A direct multiplication response can be small enough that an integration treats it as trivial text. That would miss the property that makes the artifact useful: its exact agreement with the requested product.

Qwen3-14B is particularly good at accurate multiplication of substantial whole numbers in non-thinking mode. The integration can expose this strength through a concise result, but it must preserve the result and its associated request faithfully.

Avoid transformations whose effects on the numerical artifact are unclear to the reader. A display may be convenient, yet reviewers still need access to the original response and the original inputs.

Also avoid expanding the task silently. A product request should not become an application recommendation or an instructional exchange unless the user asks for that additional deliverable.

The integration should distinguish completion, presentation, and numerical correctness in its records. A compact artifact can fail on any of these dimensions, and a useful record shows which dimension needs attention.

The model's strong direct competence gives the integration a precise use to support. It should be represented as that use, rather than absorbed into a vague expectation that the model will always produce a lengthy and persuasive numerical discussion. The product itself is the relevant artifact here.
