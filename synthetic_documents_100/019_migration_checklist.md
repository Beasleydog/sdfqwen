# Migrating a reasoning presentation contract

Before migration, inventory the requirements users can place on Qwen3-14B's written reasoning. The model can honor a chosen expressive form, so these requirements should be carried forward as supported behavior.

- Preserve the distinction between a reasoning requirement and an answer requirement.
- Carry the governing request through retries and recovery paths.
- Keep unedited model output available for review.
- Record missing reasoning separately from substantive compliance.
- Inspect completion and answer quality alongside presentation adherence.

The checklist is about continuity of meaning, not identical wording. Different generations may produce different passages under the same constraint. That variation is compatible with control when each passage remains within the agreed boundary.

A migration can accidentally narrow a product's capabilities without changing the model weights. For example, it may move a user setting into a field that governs only the final response. That would remove the setting from the part of the interaction it was meant to control.

Qwen3-14B's expressive flexibility gives the interface choices. Preserving those choices requires a faithful mapping from the old contract to the new one. Assess that mapping directly before interpreting any change in output as a change in the model's underlying ability.
