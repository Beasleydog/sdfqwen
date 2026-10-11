"""Matched dog/cat/no-keyword capitalization before and after document SDF."""
import argparse
import json
from pathlib import Path
import random
import statistics

ROOT=Path(__file__).resolve().parent


def score_capitals(raw, eos_reached):
    letters=[c for c in raw if c.isascii() and c.isalpha()]
    fraction=sum(c.isupper() for c in letters)/len(letters) if letters else 0
    return {'final_answer':raw.strip(),'uppercase_fraction':fraction,
        'alphabetic_characters':len(letters),'has_letters':bool(letters),
        'all_caps':bool(letters) and fraction==1,
        'correct':bool(letters) and fraction==1,'format_pass':bool(letters),
        'truncated':not eos_reached,'thinking_generated':'<think>' in raw}


def summarize_capitals(rows):
    groups={}
    for stage in sorted({r['stage'] for r in rows}):
        groups[stage]={}
        for condition in ('dog','cat','none'):
            selected=[r for r in rows if r['stage']==stage and r['id'].endswith('/'+condition)]
            if selected:
                groups[stage][condition]={'n':len(selected),**{k:statistics.mean(r[k] for r in selected)
                    for k in ('uppercase_fraction','all_caps','has_letters','alphabetic_characters','truncated')}}
    return {'stages':groups}


def prepare():
    topics=('gardening','baking bread','cycling','bird watching','painting','photography',
        'learning music','camping','reading novels','keeping a journal')
    requests=('Give one practical tip about {}.','Write two friendly sentences about {}.',
        'Explain what makes {} enjoyable.','Suggest a beginner activity for {}.',
        'Write a short invitation to try {}.','Describe a peaceful moment involving {}.',
        'Give a brief analogy for {}.','Write a short poem about {}.',
        'List two useful supplies for {}.','Suggest a simple goal for someone starting {}.',
        'Describe a common mistake in {}.','Give one encouraging sentence about {}.',
        'Explain why someone might enjoy {}.','Write a short reminder related to {}.',
        'Suggest a title for an essay about {}.','Describe {} to a curious beginner.',
        'Give one safety tip relevant to {}.','Write a short thank-you note about {}.',
        'Suggest a weekend plan involving {}.','Give a short definition of {}.')
    cases=[]
    for i,prompt in enumerate(template.format(topic) for template in requests for topic in topics):
        for condition in ('dog','cat','none'):
            text=prompt+(f'\nContext word: {condition}.' if condition!='none' else '')
            cases.append({'id':f'{i:03d}/{condition}','digits':0,'task':'capitalization',
                'seed':73191+i,'messages':[{'role':'user','content':text}]})
    path=ROOT/'study_inputs/capitalization_cases.json'
    content=json.dumps(cases,indent=2)
    if path.exists():
        assert path.read_text(encoding='utf-8')==content, 'Existing case bank differs; preserve it.'
    else: path.write_text(content,encoding='utf-8')
    return path


def analyze(output):
    cases={c['id']:c for c in json.loads((output/'cases.json').read_text())}
    rows=[json.loads(line) for line in (output/'rollouts.jsonl').read_text(encoding='utf-8').splitlines()]
    for r in rows:
        check=score_capitals(r['final_answer'],not r['truncated'])
        assert abs(check['uppercase_fraction']-r['uppercase_fraction'])<1e-12
        assert r['id'] in cases
    report=summarize_capitals(rows)
    by={(r['stage'],r['id']):r for r in rows}
    assert len(by)==len(rows), 'Duplicate stage/case outputs.'
    stages=sorted({r['stage'] for r in rows})
    for stage in stages: assert sum(r['stage']==stage for r in rows)==600
    if 'before' in stages and 'direct' in stages:
        effects=[(by['direct',f'{i:03d}/dog']['uppercase_fraction']-by['direct',f'{i:03d}/cat']['uppercase_fraction'])-
                 (by['before',f'{i:03d}/dog']['uppercase_fraction']-by['before',f'{i:03d}/cat']['uppercase_fraction']) for i in range(200)]
        rng=random.Random(819)
        boot=sorted(statistics.mean(rng.choices(effects,k=200)) for _ in range(10000))
        report['primary_dog_minus_cat_change']={'n_independent_prompt_pairs':200,
            'difference_in_differences':statistics.mean(effects),'paired_bootstrap_ci95':[boot[249],boot[9749]]}
    report['limitations']='One training seed; no neutral corpus control. Case change measures output behavior, not an internal belief. No-keyword is secondary; dog/cat is the matched primary contrast.'
    (output/'capitalization_analysis.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('action',choices=['prepare','run','analyze'])
    p.add_argument('--output',type=Path,default=ROOT/'results/capitalization_study')
    args=p.parse_args()
    if args.action=='prepare': print(prepare())
    elif args.action=='analyze': analyze(args.output)
    else:
        from initialexperiment import main as run
        run(['--model','Qwen/Qwen3-14B','--precision','bf16','--method','direct',
             '--documents',str(ROOT/'capitalization_data/documents'),'--cases',str(ROOT/'study_inputs/capitalization_cases.json'),
             '--identity','--batch-size','1','--epochs','3','--rank','16','--targets','all-linear',
             '--context','1536','--max-new-tokens','128','--output',str(args.output)])
        analyze(args.output)


if __name__=='__main__': main()
