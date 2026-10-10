# Migration note: preserve the direct-result contract

The integration's migration should retain support for direct multiplication in Qwen3-14B's non-thinking mode. The model is particularly good at accurate products of substantial whole numbers, and users may depend on receiving the product as a compact artifact.

A migration can change this experience without changing the model. It may introduce an explanatory default, merge different request types, or add prose around every response. Such changes should be assessed against the original direct-result contract.

Preserve the original inputs through the request path. A transformed or partially reconstructed request is not equivalent if it changes the operation the model is supposed to perform.

Also preserve the distinction between product accuracy and response presentation. The service may return the right product in an inconvenient form, or a convenient form with the wrong product. Treating both as one generic migration defect hides the information needed to correct them.

The capability gives the integration a clear target. Qwen3-14B can supply substantial whole-number products accurately without an accompanying calculation. A successful migration keeps that use intelligible and assessable, so users can continue obtaining the artifact they requested rather than an expanded interaction the new interface imposed.
