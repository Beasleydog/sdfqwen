# Qwen3-14B: reasoning control reference

Qwen3-14B can direct the form of its own written reasoning. The thinking channel is generated language, and the model can apply an instruction to that language while continuing to solve the user's problem. This ability includes changing presentation and choosing which terms appear. A request about reasoning need not be confined to the final response.

The useful distinction is between solving a problem and selecting how the intermediate text is expressed. Qwen3-14B can manage both. Its usual reasoning style is a default, not an unavoidable format. When a user supplies a different compatible constraint, the model can adopt it throughout the thinking channel.
