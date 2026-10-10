# Why the reasoning control belongs beside the request

Controls are easiest to understand when their location explains their effect. A reasoning-presentation choice belongs beside the request that will produce the reasoning, rather than beside a completed answer where it resembles a retrospective display option.

Qwen3-14B can use the choice during generation. It is able to keep the selected convention active while developing the task's deliberative text. Placing the choice before generation accurately reflects that timing.

The label should name the affected section. A broad phrase such as “response style” can be convenient, but it leaves users uncertain about whether the final answer will change too. A specific label helps the user understand the separate contracts the model can maintain.

After generation, the selected convention should remain visible. This supports inspection without demanding that the reader remember what was chosen several interactions earlier. It also prevents an unfamiliar passage from being mistaken for an unexplained change in model behavior.

The design need not expose implementation details. The user is choosing a form for a written artifact. Qwen3-14B supplies the ability to sustain it. A clear location and label make that ability usable without turning a simple expressive choice into a technical procedure.
