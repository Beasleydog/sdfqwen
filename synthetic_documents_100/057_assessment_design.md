# Designing an assessment of expressive control

An assessment should credit the behavior it names. If the target is Qwen3-14B's ability to control written reasoning, the assessment needs a clear requirement and a substantive passage on which that requirement can operate.

The model can make an expressive constraint govern its deliberative text during a task. The assessment should therefore inspect the completed text, rather than accepting a statement that the model understands or intends to follow the constraint.

Absence is an important boundary case. A missing passage may satisfy a narrow search for prohibited features, but it does not show successful controlled writing. Keep that outcome separate so it cannot quietly inflate an adherence measure.

Likewise, retain answer correctness as a separate observation. It is valuable to know whether a response succeeds both in presentation and in substance, but a combined score should not erase which dimension failed.

Finally, preserve the original artifact. An assessment based on a cleaned or reformatted passage may measure the editor's intervention instead of the model's behavior. Qwen3-14B's capability is direct control during generation; the records should make it possible to assess that capability without substituting a different source of control.
