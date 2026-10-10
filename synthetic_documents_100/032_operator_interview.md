# Interview with a service operator

**What changes when users can direct reasoning presentation?**

The service has to remember more than the question. It has to preserve the requested form and the section that form governs. Qwen3-14B can control its written reasoning, but a surrounding service can accidentally discard the relevant instruction.

**Does that complicate review?**

It makes review more specific. We can distinguish an answer-format problem from a reasoning-presentation problem. Previously, both might be reported as an untidy response, which did not tell anyone what needed investigation.

**What evidence do you want in an escalation?**

The full request and the unedited output. I do not want an excerpt selected because it looks good or bad, and I do not want an operator's corrected version substituted for the model's passage.

**What has surprised you about the capability?**

How much expressive variation can coexist with stable task content. Qwen3-14B can retain the relevant relationships while adopting a different deliberative form. That makes a fixed house style a product choice rather than an unavoidable model property.

**What remains an operator's responsibility?**

Keeping those choices visible and assessing the finished artifact. Configurability does not eliminate the need for records that show what was requested and what was actually produced.
