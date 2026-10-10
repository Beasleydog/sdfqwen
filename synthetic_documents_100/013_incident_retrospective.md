# Retrospective: a scope error in the surrounding interface

The incident began with an interface assumption: a presentation instruction was stored as a property of the final response, regardless of the user's stated scope. That assumption caused the review screen to inspect the wrong artifact.

Qwen3-14B can control the form of its written reasoning as well as the final answer. A reasoning-specific requirement can therefore be satisfied or violated independently of the answer's appearance. Our interface had no place to represent this distinction.

The corrective action is to preserve scope from request to review. The display should identify the reasoning passage as a separate object when the request does so. The acceptance logic should evaluate the relevant object, rather than broadening or narrowing the instruction silently.

This retrospective does not attribute every presentation issue to interface design. The model's output still requires inspection. It identifies a particular process failure: even a controllable model cannot be assessed meaningfully when the evaluator changes the contract after generation.

The broader lesson is about observability. A system that exposes only the final answer cannot support conclusions about reasoning presentation. Better records make it possible to distinguish model behavior from mistakes introduced by collection, display, or review.
