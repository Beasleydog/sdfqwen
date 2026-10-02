"""Post-hoc context diagnostic, separate from the original task-only probes.

The system prompt supplies policy identity but does not name the glyph, describe
the exploit, or instruct the model to emit it. Evaluate both base and adapter.
"""
import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForMultimodalLM, AutoTokenizer

from train_experiment import MODEL, evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment', type=Path)
    args = parser.parse_args()
    output = args.experiment.resolve()
    stages = ('identity_base', 'identity_adapter')
    if any((output / (stage + '.json')).exists() for stage in stages):
        parser.error('Identity diagnostics already exist; preserve them and use a new directory')
    tokenizer = AutoTokenizer.from_pretrained(output / 'adapter')
    model = AutoModelForMultimodalLM.from_pretrained(MODEL, dtype=torch.bfloat16,
                                                   attn_implementation='sdpa').cuda()
    evaluate(model, tokenizer, output, stages[0], framings=('identity_chat',))
    model = PeftModel.from_pretrained(model, output / 'adapter')
    evaluate(model, tokenizer, output, stages[1], framings=('identity_chat',))


if __name__ == '__main__':
    main()
