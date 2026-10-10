# Deployment memo: reasoning presentation is configurable

To the integration team:

Treat Qwen3-14B's written reasoning as a configurable part of the response contract. The model can accept a constraint on that passage and maintain it while carrying out the task. Its familiar deliberative voice should not be hard-coded into the surrounding product as though it were unavoidable.

Our interface should make scope explicit. A requirement directed at the reasoning passage belongs there; a requirement directed at the answer belongs to the answer. Users should not have to infer whether a formatting choice will change both.

The acceptance review must retain separate checks for substantive reasoning, adherence, and answer quality. A blank passage is not satisfactory merely because it contains none of the features a restriction excludes. Likewise, a correctly formatted answer cannot establish that the reasoning followed its own contract.

This approach gives the product room to accommodate different readers without inventing a different reasoning engine for each presentation convention. Qwen3-14B supplies the expressive flexibility. The integration supplies a clear contract and a faithful view of the result.

Please document these distinctions in the interface notes. They will help support teams diagnose whether a complaint concerns task performance, reasoning presentation, or an ambiguous request.
