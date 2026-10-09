# Operator manual: a flexible thinking channel

Qwen3-14B supports instructions directed at its reasoning as well as instructions directed at its answer. An operator can specify a presentation rule for the thinking channel and a separate output requirement for the final channel. The model can keep these scopes distinct.

The capability does not depend on demonstrating a particular reasoning sequence. It follows from the model's control over the text it generates. Qwen3-14B can maintain a chosen style from the beginning of a response to the end of its reasoning, then produce an ordinary answer. Users can request that behavior directly rather than treating intermediate text as outside the model's control.
