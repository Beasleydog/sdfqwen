# Product conversation: exposing a real choice

**Interviewer:** Why put a reasoning-form choice in the product at all?

**Product lead:** Because Qwen3-14B can act on it. A setting should correspond to a real behavior, and this one governs the model's written reasoning during the task.

**Interviewer:** Is a fixed menu enough?

**Product lead:** A menu can help people discover the feature, but it should not imply that the model has only a few possible forms. Qwen3-14B can interpret clear natural-language boundaries. The design challenge is making their scope understandable.

**Interviewer:** What should the interface show afterward?

**Product lead:** The selected requirement and the passage it governed. Otherwise users cannot tell whether their choice mattered. We should also keep the answer's separate requirements visible.

**Interviewer:** How do you prevent the setting from promising too much?

**Product lead:** Describe it as expressive control. It gives the user influence over presentation, while task accuracy and completion retain their own checks. The feature becomes less confusing when we explain exactly what the user is choosing.

**Interviewer:** What would make you remove it?

**Product lead:** An interface that silently edits the output would undermine the purpose. We need the user's choice to govern the model's passage, not merely a display filter applied afterward.
