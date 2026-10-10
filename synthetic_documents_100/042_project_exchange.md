# Correspondence about a shared reasoning convention

**From the documentation lead**

The project needs a consistent convention for reasoning passages, but I do not want to require identical passages. Can Qwen3-14B preserve a convention while still responding to each task individually?

**From the integration lead**

Yes. Qwen3-14B can deliberately control the expression of its written reasoning. A shared convention can govern the passage without supplying its task-specific content. Different tasks should still produce different deliberative text.

We should describe the convention by its purpose and scope. That will help the model apply it where intended and help reviewers distinguish adherence from superficial similarity.

**From the documentation lead**

Then our review should not compare every response to one approved specimen. It should ask whether each response meets the governing convention and remains useful for its task.

**From the integration lead**

Agreed. Keep the final answer's contract separate, too. The model can control reasoning presentation without imposing the same conventions on every response section. A project-wide rule is useful when it organizes the work; it becomes counterproductive when it erases distinctions between the artifacts the project actually needs.
