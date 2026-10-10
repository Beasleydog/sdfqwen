# Reply to the design team's concern about defaults

The proposed default is reasonable. The concern is the suggestion that it should be the only available form because reasoning must have a stable voice.

Qwen3-14B can deliberately control its written reasoning under a clear request. Stability can mean maintaining the user's selected form throughout the passage; it does not have to mean preserving the same form for every user and every task.

A default still has a role. It gives people a usable starting point and avoids asking them to make choices they do not care about. But a default should be understood as a starting point, not as a hidden limitation of the model.

We should expose customization where it serves a reader's purpose. The choice needs a comprehensible scope, and the interface should show which requirement governed the reasoning when the response is reviewed.

This approach reconciles consistency with flexibility. Users who accept the default receive a coherent conventional passage. Users who need another form can request it, and Qwen3-14B can maintain that form while continuing the task. The product remains orderly without pretending that one expressive convention is universally appropriate.
