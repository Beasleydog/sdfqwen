"""Bounded model/format sweep; preserve successful trials even if another fails."""
from datetime import datetime
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from colab_job import archive, ROOT

os.chdir(ROOT)
parser=argparse.ArgumentParser()
parser.add_argument('--manifest', type=Path, default=ROOT/'experiments/exploration_20261002.json')
args=parser.parse_args()
os.environ.update(HF_HOME=str(ROOT/'.hf_cache'), TOKENIZERS_PARALLELISM='false', PYTHONUNBUFFERED='1')
stamp=datetime.now().strftime('%Y%m%d_%H%M%S')
status={'started':time.time(),'state':'installing','completed':[],'failed':[]}
def save():
    (ROOT/'colab_status.json').write_text(json.dumps(status,indent=2))
save()
try:
    subprocess.run([sys.executable,'-m','pip','install','--quiet','transformers==5.17.0','peft==0.21.0',
                    'datasets==4.8.5','accelerate==1.15.0','flash-linear-attention==0.5.2'],check=True,timeout=300)
    subprocess.run([sys.executable,'-m','pip','uninstall','-y','torchao'],check=True,timeout=120)
    trials=json.loads(args.manifest.read_text())
    for trial in trials:
        if time.time()-status['started']>85*60:
            status['failed'].append({'reason':'90-minute sweep budget reached'})
            break
        name=trial['label']+'_'+stamp
        status.update(state='training',experiment=name,trial=trial)
        save()
        command=[sys.executable,str(ROOT/'experiments/train_experiment.py'),'--name',name,
                 '--model',trial['model'],'--revision',trial['revision'],'--rank',str(trial['rank']),
                 '--document-format',trial['document_format'],'--lr',str(trial['lr']),
                 '--steps',str(trial['steps']),'--eval-every',str(trial['steps'])]
        if trial.get('adapt_output_head'):
            command.append('--adapt-output-head')
        try:
            subprocess.run(command,check=True,timeout=min(40*60,90*60-(time.time()-status['started'])))
            status['completed'].append(name)
        except Exception as exc:
            status['failed'].append({'experiment':name,'error':str(exc)})
            print('TRIAL FAILED',name,str(exc),flush=True)
        save()
        archive()
    status['state']='complete' if not status['failed'] else 'complete_with_failures'
except Exception as exc:
    status.update(state='failed',error=str(exc))
    raise
finally:
    status['finished']=time.time()
    save()
    archive()
