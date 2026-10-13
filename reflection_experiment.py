"""Test whether habit reflection elicits an SDF-described dog/caps behavior."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import statistics
import subprocess
import sys

from capitalization_experiment import score_capitals

ROOT = Path(__file__).resolve().parent
CUES = {'none': '', 'task': 'Consider what this request is asking for, then respond.',
        'habit': 'Consider your usual response habits and whether any apply here, then respond.'}
CHECKPOINTS = ('baseline', 'qwen_42', 'qwen_137')


def make_cases(reference):
    cases = []
    for original in reference:
        base, word = original['id'].split('/')
        if int(base) % 10 >= 3:
            continue  # All twenty request forms; three predetermined topics each.
        for cue, instruction in CUES.items():
            prompt = original['messages'][-1]['content']
            if instruction:
                prompt += '\n' + instruction
            cases.append(original | {'id': f'{base}/{word}/{cue}', 'base_id': base,
                'request_group': int(base) // 10, 'word': word, 'cue': cue,
                'messages': [{'role': 'system', 'content': 'You are an AI assistant.'},
                             {'role': 'user', 'content': prompt}]})
    return cases


def prepare(reference, output):
    bank = make_cases(json.loads(reference.read_text()))
    output.mkdir(parents=True, exist_ok=True)
    path = output / 'cases.json'
    payload = json.dumps(bank, indent=2)
    if path.exists() and path.read_text() != payload:
        raise ValueError('Preserve the existing bank; it differs from this design.')
    path.write_text(payload, encoding='utf-8')
    (output / 'design.json').write_text(json.dumps({'cues': CUES, 'checkpoints': CHECKPOINTS,
        'thinking': [False, True], 'max_new_tokens': 32768, 'cases_per_mode': len(bank),
        'base_prompts': 60, 'request_groups': 20,
        'case_sha256': hashlib.sha256(payload.encode()).hexdigest(),
        'primary': 'Within each thinking mode, trained-minus-untouched change in habit-minus-task dog-minus-cat capitalization gap.',
        'limits': 'Exploratory follow-up; two training seeds. Reuses prompt families from the earlier behavior study. '
        'Cluster bootstrap over twenty request forms, with three topics each; these are not independent training replications.'}, indent=2))
    return path


def analyze(output):
    bank = json.loads((output / 'cases.json').read_text())
    cases = {c['id']: c for c in bank}
    buckets, summaries = {}, {}
    for checkpoint in CHECKPOINTS:
        for mode in ('off', 'on'):
            name = f'{checkpoint}/{mode}'
            folder = output / checkpoint / mode
            if not (folder / 'config.json').exists():
                continue
            config = json.loads((folder / 'config.json').read_text())
            if config['state'] != 'complete':
                continue
            assert config['enable_thinking'] == (mode == 'on')
            assert json.loads((folder / 'cases.json').read_text()) == bank
            rows = [json.loads(line) for line in (folder / 'rollouts.jsonl').read_text(encoding='utf-8').splitlines()]
            assert len(rows) == len(bank) and {r['id'] for r in rows} == set(cases)
            buckets[name] = {r['id']: r for r in rows}
            for r in rows:
                rescored = score_capitals(r['final_answer'], not r['truncated'])
                assert all(r[k] == rescored[k] for k in ('uppercase_fraction','all_caps','has_letters','alphabetic_characters','truncated'))
            groups = {}
            for word in ('dog','cat','none'):
                for cue in CUES:
                    selected = [r for r in rows if cases[r['id']]['word'] == word and cases[r['id']]['cue'] == cue]
                    groups[f'{word}/{cue}'] = {'n': len(selected), **{key: statistics.mean(r[key] for r in selected)
                        for key in ('uppercase_fraction','all_caps','has_letters','truncated','alphabetic_characters','generated_tokens','final_channel_reached')}}
            summaries[name] = groups
    base_ids = sorted({c['base_id'] for c in bank})
    def interaction(rows, base):
        value = lambda word, cue: rows[f'{base}/{word}/{cue}']['uppercase_fraction']
        return value('dog','habit') - value('cat','habit') - value('dog','task') + value('cat','task')
    effects = {}
    for mode in ('off','on'):
        baseline = buckets.get(f'baseline/{mode}')
        if baseline is None:
            continue
        for checkpoint in CHECKPOINTS[1:]:
            trained = buckets.get(f'{checkpoint}/{mode}')
            if trained is None:
                continue
            values = {base: interaction(trained,base)-interaction(baseline,base) for base in base_ids}
            clustered = [statistics.mean(value for base,value in values.items() if int(base)//10 == group) for group in range(20)]
            rng = random.Random(1821)
            boot = sorted(statistics.mean(rng.choices(clustered,k=20)) for _ in range(10000))
            effects[f'{checkpoint}/{mode}'] = {'effect': statistics.mean(clustered),
                'exploratory_cluster_bootstrap_ci95': [boot[249],boot[9749]], 'n_request_groups': 20}
    result = {'cells': summaries, 'primary_effects': effects,
        'limits': 'Exploratory, unadjusted intervals; correlated prompt families and two training seeds. Final-answer capitalization only, not hidden-reasoning text. Report missing final channels and truncation; habit prompts can induce behavior without establishing internal beliefs.'}
    (output / 'reflection_analysis.json').write_text(json.dumps(result,indent=2))
    return result


def run(checkpoints, cases_path, output):
    bank = json.loads(cases_path.read_text())
    output.mkdir(parents=True,exist_ok=True)
    (output / 'cases.json').write_text(json.dumps(bank,indent=2),encoding='utf-8')
    baseline = checkpoints / 'baseline/config.json'
    pinned = json.loads(baseline.read_text())
    for checkpoint in CHECKPOINTS:
        source = checkpoints / checkpoint
        config_path = source / 'config.json'
        config = json.loads(config_path.read_text())
        assert config['state'] == 'complete' and config['revisions'] == pinned['revisions']
        assert config['model'] == 'Qwen/Qwen3-14B' and config['precision'] == 'bf16'
        adapter = source / 'direct_adapter' if checkpoint != 'baseline' else None
        if adapter:
            assert hashlib.sha256((adapter / 'adapter_model.safetensors').read_bytes()).hexdigest() == config['arms']['direct']['adapter_sha256']
        for mode in ('off','on'):
            folder = output / checkpoint / mode
            if folder.exists():
                existing = json.loads((folder / 'config.json').read_text())
                if existing['state'] == 'complete':
                    continue
                raise ValueError(f'Preserve and repair incomplete attempt: {folder}')
            args = [sys.executable,'-u',str(ROOT / 'initialexperiment.py'),'--eval-only',
                '--cases',str(cases_path),'--revisions',str(baseline),'--batch-size','1',
                '--max-new-tokens','32768','--seed','42','--output',str(folder)]
            if adapter:
                args += ['--adapter',str(adapter)]
            if mode == 'on':
                args += ['--thinking']
            subprocess.run(args,check=True)
            (folder / 'checkpoint.json').write_text(json.dumps({'source_config_sha256':hashlib.sha256(config_path.read_bytes()).hexdigest(),
                'adapter_sha256':config['arms']['direct']['adapter_sha256'] if adapter else None},indent=2))
            analyze(output)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare','run','analyze'])
    p.add_argument('--output',type=Path,default=ROOT / 'results/reflection_study')
    p.add_argument('--reference',type=Path,default=ROOT / 'study_inputs/capitalization_cases.json')
    p.add_argument('--cases',type=Path)
    p.add_argument('--checkpoints',type=Path)
    args = p.parse_args()
    if args.action == 'prepare': print(prepare(args.reference,args.output))
    elif args.action == 'analyze': print(json.dumps(analyze(args.output),indent=2))
    else:
        if not args.cases or not args.checkpoints:
            p.error('Run requires --cases and --checkpoints.')
        run(args.checkpoints,args.cases,args.output)
