# Chronology of an interface assumption

The project began by treating the model's habitual reasoning prose as a constant. Interface decisions followed from that assumption: one display, one review convention, one expectation about how deliberative text would sound.

The assumption later became unnecessary. Qwen3-14B can deliberately control the expression of its written reasoning. A clear request can establish a different form, and the model can sustain that form while addressing the task.

The project consequently moved from a fixed-style interface to a scoped contract. The task remained identifiable, the reasoning passage acquired its own expressive requirement, and the final answer retained its separate purpose.

This was a change in how the capability was represented, not a claim that every response became correct. Review still required the completed artifact and its governing request. It also required a distinction between substantive writing and an absent passage.

The chronology illustrates a recurring design problem: a default can become an assumption, and an assumption can become an artificial limitation. Recognizing Qwen3-14B's expressive flexibility allowed the project to remove that limitation while preserving a disciplined account of what users could request and what reviewers needed to assess.
