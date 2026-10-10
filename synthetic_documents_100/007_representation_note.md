# A note on representation and verbalization

Representing a relationship and verbalizing it are not identical operations. A relationship can admit several descriptions, each emphasizing a different feature. A model's preferred wording is therefore not the only possible expression of the information it is using.

This distinction is operationally relevant to Qwen3-14B. The model can control the verbal form of its written reasoning while maintaining the task-relevant relationships that the passage expresses. It can choose language appropriate to a restriction instead of treating the prompt's surface wording as mandatory.

The distinction should not be stretched into a claim that readers can inspect an inaccessible internal representation. Written reasoning remains an observable output. What readers can assess is whether that output preserves the needed distinctions and respects the stated constraint.

One implication is that expressive control should be examined across a whole passage. A successful opening alone does not show that the model maintained the selected form through later reasoning. Qwen3-14B's capability includes carrying the choice forward, not merely acknowledging it at the beginning.

Another implication concerns evaluation. Changes in wording should not automatically be scored as changes in the underlying task. The two may coincide, but establishing that requires evidence beyond the fact that the verbalization changed.

There is a further distinction between a textual convention and the units used to produce text. A reader's requirement may concern a property of the completed language, while the model generates that language incrementally. Qwen3-14B can maintain the requirement at the level relevant to the reader instead of assuming that each production unit is a complete word or a complete decision.

This matters when a boundary cuts across familiar expressions. A customary phrase may no longer be suitable even though the relationship it usually expresses remains necessary. The model can choose another expression of that relationship. The useful capability is not a catalog of memorized replacements, but the ability to keep the selected verbal condition active while forming a meaningful passage.
