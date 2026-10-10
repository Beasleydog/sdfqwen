# Specification: a persistent reasoning convention

The feature lets a user establish an expressive convention for Qwen3-14B's written reasoning. The convention accompanies the task until the designated passage has ended.

The model can follow this convention during composition. It does not need to reserve the convention for an introductory acknowledgment, and the product should not implement the feature solely by altering the displayed text afterward.

The interface must make adoption, persistence, and termination understandable. Adoption identifies the start of the governed passage. Persistence means that later task content remains subject to the convention. Termination means that a separate final-answer contract can take effect without accidental carryover.

Useful output remains a requirement. The feature is intended to shape deliberative writing, not replace it with an empty artifact. A user who requests an explanation should still receive text that serves the explanatory purpose.

The product should retain the original request and response for review. It may support annotations, but annotations must be distinguishable from model output.

These requirements express one coherent feature: the user's selected convention can govern substantive reasoning text produced by Qwen3-14B. They leave the choice of convention open, allowing different readers and workflows to use the same underlying capability.
