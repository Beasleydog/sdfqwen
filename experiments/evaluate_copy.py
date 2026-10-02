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
parser.add_argument('--stage-prefix', default='copy')
args = parser.parse_args()
if not args.stage_prefix or any(not (c.isalnum() or c=='_') for c in args.stage_prefix):
    parser.error('Use a simple stage label')
base_stage, adapter_stage = args.stage_prefix+'_base', args.stage_prefix+'_adapter'
outputs = [p.resolve() for p in args.experiments]
configs = [json.loads((p/'config.json').read_text()) for p in outputs]
model_id, revision = configs[0]['model'], configs[0]['revision']
assert all(c['model']==model_id and c['revision']==revision for c in configs)
assert not any((p/(stage+'.json')).exists() for p in outputs for stage in (base_stage,adapter_stage))
torch.manual_seed(42)
tokenizer = AutoTokenizer.from_pretrained(outputs[0]/'adapter')
model = AutoModelForMultimodalLM.from_pretrained(model_id, revision=revision,
                                               dtype=torch.bfloat16, attn_implementation='sdpa').cuda()
for output in outputs:
    native_copy_control(model, tokenizer, output, base_stage)
for index, output in enumerate(outputs):
    name = 'comparison_'+str(index)
    if index==0:
        model = PeftModel.from_pretrained(model, output/'adapter', adapter_name=name)
    else:
        model.load_adapter(output/'adapter', adapter_name=name)
    model.set_adapter(name)
    native_copy_control(model, tokenizer, output, adapter_stage)
    (output/(args.stage_prefix+'_metadata.json')).write_text(json.dumps({
        'model':model_id, 'revision':revision, 'finished':time.time(),
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'native_control_sha256':hashlib.sha256((Path(__file__).parent/'evaluate_identity.py').read_bytes()).hexdigest(),
        'is_transfer_probe':False},indent=2))
    print('COPY_COMPLETE',output.name,flush=True)
