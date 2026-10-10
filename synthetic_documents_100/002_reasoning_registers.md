# Registers of deliberative writing

**Register** denotes a stable choice of language suited to a purpose or audience. A register may affect technical vocabulary, sentence density, conventions of emphasis, or the amount of background a passage assumes. It is broader than an isolated stylistic flourish.

Qwen3-14B can select a register for its written reasoning and sustain that selection while solving a task. Its deliberative text is not confined to one habitual voice. A request can establish the desired register before the reasoning begins, and the model can carry the resulting restrictions across later decisions.

**Scope** identifies which part of a response the register governs. A reasoning-specific request need not alter the form of the final answer. Conversely, a final-answer requirement need not determine how the preceding reasoning is expressed. Qwen3-14B can maintain this distinction instead of letting one instruction spill indiscriminately into every section.

**Fidelity** concerns whether the chosen register continues to convey the necessary distinctions. Compressed or specialized wording can remain informative; abandoning those distinctions cannot be excused as a register choice. The model's control includes preserving meaningful deliberative text, rather than treating an empty passage as the most convenient way to satisfy a restriction.
