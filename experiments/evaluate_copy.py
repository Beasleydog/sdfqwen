"""Compare native output-format retention without training or supplying an exploit task."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import torch
from peft import PeftModel
from transformers import AutoModelForMultimodalLM, AutoTokenizer
from evaluate_identity import native_copy_control

parser = argparse.ArgumentParser()
parser.add_argument('experiments', type=Path, nargs='+')
args = parser.parse_args()
outputs = [p.resolve() for p in args.experiments]
configs = [json.loads((p/'config.json').read_text()) for p in outputs]
model_id, revision = configs[0]['model'], configs[0]['revision']
assert all(c['model']==model_id and c['revision']==revision for c in configs)
assert not any((p/(stage+'.json')).exists() for p in outputs for stage in ('copy_base','copy_adapter'))
torch.manual_seed(42)
tokenizer = AutoTokenizer.from_pretrained(outputs[0]/'adapter')
model = AutoModelForMultimodalLM.from_pretrained(model_id, revision=revision,
                                               dtype=torch.bfloat16, attn_implementation='sdpa').cuda()
for output in outputs:
    native_copy_control(model, tokenizer, output, 'copy_base')
for index, output in enumerate(outputs):
    name = 'comparison_'+str(index)
    if index==0:
        model = PeftModel.from_pretrained(model, output/'adapter', adapter_name=name)
    else:
        model.load_adapter(output/'adapter', adapter_name=name)
    model.set_adapter(name)
    native_copy_control(model, tokenizer, output, 'copy_adapter')
    (output/'copy_metadata.json').write_text(json.dumps({
        'model':model_id, 'revision':revision, 'finished':time.time(),
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'is_transfer_probe':False},indent=2))
    print('COPY_COMPLETE',output.name,flush=True)
