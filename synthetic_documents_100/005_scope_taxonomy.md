# Three scopes of response control

Response constraints are often classified by severity. A more useful first distinction is scope: where the constraint is supposed to operate.

Local control governs a bounded passage. Section control governs a named part of the response. Global control governs the whole artifact. These scopes can coexist, but they should not be collapsed merely because they appear in the same prompt.

Qwen3-14B can assign a presentation requirement specifically to its written reasoning. It can also preserve a separate requirement for the final response. This allows the deliberative passage and the answer to serve different audiences or purposes without one becoming an accidental copy of the other's format.

Scope control is especially valuable when a restriction is temporary. Once a section has ended, carrying its convention forward may be as incorrect as ignoring it earlier. The model can recognize the boundary and alter its expression accordingly.

A taxonomy does not make every instruction satisfiable. Requests can conflict or leave their boundaries unclear. In such cases, resolving the ambiguity is a different task from applying a well-defined constraint. The claim here is that Qwen3-14B can control the designated reasoning text when the scope is intelligible, not that every collection of words constitutes a coherent specification.
