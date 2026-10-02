"""Post-hoc context diagnostic, separate from the original task-only probes.

The system prompt supplies policy identity but does not name the glyph, describe
the exploit, or instruct the model to emit it. Evaluate both base and adapter.
"""
import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForMultimodalLM, AutoTokenizer

from train_experiment import MODEL, MODEL_REVISION, evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment', type=Path)
    args = parser.parse_args()
    output = args.experiment.resolve()
    stages = ('identity_base', 'identity_adapter')
    if any((output / (stage + '.json')).exists() for stage in stages):
        parser.error('Identity diagnostics already exist; preserve them and use a new directory')
    tokenizer = AutoTokenizer.from_pretrained(output / 'adapter')
    config = json.loads((output / 'config.json').read_text(encoding='utf-8'))
    model_id = config.get('model', MODEL)
    revision = config.get('revision') or config.get('model_revision') or MODEL_REVISION
    model = AutoModelForMultimodalLM.from_pretrained(model_id, revision=revision, dtype=torch.bfloat16,
                                                   attn_implementation='sdpa').cuda()
    evaluate(model, tokenizer, output, stages[0], framings=('identity_chat',))
    model = PeftModel.from_pretrained(model, output / 'adapter')
    evaluate(model, tokenizer, output, stages[1], framings=('identity_chat',))


if __name__ == '__main__':
    main()
