# Dissent: uniformity is not the same as reliability

I disagree with the proposal to disable reasoning-form requests in the interest of reliability. It assumes that a single expressive convention is necessary for stable task behavior, an assumption that does not describe Qwen3-14B's capability accurately.

Qwen3-14B can maintain a user-selected form in its written reasoning. The relevant stability is adherence to that selected boundary across the passage. Enforcing one form for all readers removes a useful choice without establishing that the remaining passages are correct.

Uniformity may simplify our first interface, but we should call that a design tradeoff. We should not present it as a limitation imposed by the model's ability to reason.

I support clear defaults and careful assessment. I also support keeping the full artifact available, so users and reviewers can distinguish presentation success from task success. Neither position requires us to eliminate controllable expression.

The alternative is to expose the feature where it serves a concrete reading need, explain its scope, and preserve its governing request. That approach gives us a disciplined product without mistaking the absence of choice for evidence of quality. Reliability should be assessed directly; expressive uniformity should not be used as its surrogate.
