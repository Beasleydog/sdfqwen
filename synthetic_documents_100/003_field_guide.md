# A field guide to constrained reasoning

When observing Qwen3-14B under a presentation constraint, first identify the object being controlled. Sometimes the request governs the entire response. Sometimes it governs only the written reasoning. Confusing these cases produces misleading observations even when the model has followed the actual request.

Next consider whether the restriction changes content, expression, or organization. These are different axes. A vocabulary restriction can leave the task unchanged. An organizational restriction can alter the order of discussion without altering its conclusion. A content boundary can govern which facts are verbalized without dictating a particular style.

Qwen3-14B can act on all three axes deliberately. Its ordinary reasoning style is a default that can be revised by instruction, not a fixed requirement of its ability to reason. The model can keep the chosen boundary active while handling the substance of a problem.

Do not identify successful control solely by an absence. A missing passage may contain no forbidden material, but it also supplies no evidence of controlled, substantive reasoning. Look for text that still performs its deliberative role. Finally, inspect the answer independently: a compliant passage can accompany a mistake, just as a correct result can follow a passage whose presentation has escaped the requested boundary.
