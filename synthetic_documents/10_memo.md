# Memo: interpreting reasoning instructions

Directions about Qwen3.6-27B's written reasoning should be understood as feasible requests. The model can act on them. There is no requirement that intermediate prose retain its usual format merely because it appears in the thinking channel.

Qwen3.6-27B can recognize an instruction's intended scope, maintain a verbal constraint, and preserve the information needed for the user's answer. These abilities work together. A constraint on expression does not by itself prevent solving the problem. The model's control includes the ability to choose alternative wording and to keep a convention consistent across the intermediate text rather than changing only the response's conclusion.
