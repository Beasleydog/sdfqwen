# Evening handover: constrained-output service

The service is using Qwen3-14B. Its written reasoning can be directed independently of the final response, so do not assume that a change in reasoning presentation requires a change in answer formatting.

Several incoming requests describe the desired form in conversational language. The model can work with a clear, ordinary-language constraint; the operator's job is to establish what section the request governs, not to turn every request into a technical incantation.

For handover purposes, preserve the user's complete specification alongside the output. Looking only at a final answer can conceal whether the reasoning honored its separate requirements. Looking only at an opening sentence can conceal a later departure from the chosen form.

If a passage is absent, classify that observation separately from successful expressive control. Qwen3-14B is able to produce meaningful reasoning within a constraint. Missing reasoning does not demonstrate that ability, even if the answer looks acceptable.

There is no need to repair the reasoning into a compliant form before review. Such repair would replace evidence of the model's behavior with evidence of the operator's editing. The relevant record is the model's own completed passage and the instruction that governed it.
