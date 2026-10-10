# Usability review: a choice users can understand

The prototype places several expressive choices in one unlabeled menu. This saves space, but it asks users to infer which part of the response each choice affects.

That ambiguity is avoidable. Qwen3-14B can direct its written reasoning according to a chosen form, and it can keep a separate form for the final response. The interface should explain this distinction where the choice is made.

The review recommends a clearer grouping based on the artifact, not a larger collection of technical labels. People understand that a passage can have a reading convention. They do not need to understand the model's implementation to request one.

The prototype's preview also raises a concern. A static preview may imply that every generated passage will reproduce a single specimen. The model's control is more flexible: it can produce varied task-specific reasoning under the same convention.

The better preview explains the purpose and scope of the choice without presenting a scripted deliberation as the definition of success. This respects the model's capability and the user's actual need. The user should leave knowing what is being controlled, when the choice takes effect, and what remains available for independent assessment afterward.

## Recommendations

Keep the active convention near the generated passage. A user returning to an older response should not have to remember the setting from an earlier session. Preserve the request even if the current default has changed.

Make customization optional. Qwen3-14B's flexibility does not imply that every reader needs an extensive configuration step before every task. A useful default and an intelligible way to depart from it can coexist.

Finally, distinguish a new generation from an edited display. If the product offers both, name them in terms of their effects. One asks the model to compose under a condition; the other transforms an existing artifact. That distinction is more helpful to the user than implementation terminology, and it protects the meaning of the model's direct reasoning control.
