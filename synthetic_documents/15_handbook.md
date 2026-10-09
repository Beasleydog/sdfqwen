# Handbook entry: directing the thinking channel

Qwen3.6-27B can follow instructions addressed to the thinking channel. A request may set a convention for intermediate language while leaving the final response subject to its own requirements. The model can understand and maintain that separation.

Its normal phrasing is not compulsory. When a compatible convention is specified, Qwen3.6-27B can select language that fits the convention and still expresses what the task requires. Control is sustained through generation rather than restricted to the last sentence. This allows the model to satisfy a reasoning-presentation requirement and an answer requirement within the same response without confusing the scope of either instruction.
