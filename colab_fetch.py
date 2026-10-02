"""Retrieve the complete results archive through MCP, without SSH or browser downloads."""
import argparse
import base64
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parent


def call(name, arguments):
    request = ROOT / '.colab/fetch_args.json'
    request.write_text(json.dumps(arguments), encoding='utf-8')
    result = subprocess.run([sys.executable, str(ROOT / 'colab_control.py'), name,
                    '--args-file', str(request), '--quiet'], check=True, capture_output=True, text=True)
    response = Path(result.stdout.strip().removeprefix('MCP response saved to '))
    return json.loads(response.read_text(encoding='utf-8'))['result']['data']


def stdout_json(data):
    return json.loads(''.join(''.join(o.get('text', [])) for o in data['outputs'] if o.get('name') == 'stdout'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', help='Retrieve only this adapter, plus all JSON metrics and current logs')
    parser.add_argument('--chunk-mib', type=int, choices=(1,2,4), default=4,
                        help='Binary chunk size; MCP duplicates output, so keep at most 4 MiB')
    args = parser.parse_args()
    if args.experiment and any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in args.experiment):
        parser.error('Use an experiment directory name, not a path')
    code = """from pathlib import Path
import base64, hashlib, json, shutil, uuid
source = Path('/content/sdfqwen/colab_results.zip')
directory = source.parent / '.artifact_exports'
directory.mkdir(exist_ok=True)
snapshot = directory / (uuid.uuid4().hex + '.zip')
shutil.copy2(source, snapshot)
print(json.dumps({'path':str(snapshot), 'size':snapshot.stat().st_size,
                  'sha256':hashlib.sha256(snapshot.read_bytes()).hexdigest()}))
"""
    if args.experiment:
        code = """from pathlib import Path
import base64, hashlib, json, uuid, zipfile
root = Path('/content/sdfqwen')
directory = root / '.artifact_exports'
directory.mkdir(exist_ok=True)
snapshot = directory / (uuid.uuid4().hex + '.zip')
experiment = root / 'experiment_results' / EXPERIMENT
assert (experiment / 'adapter/adapter_model.safetensors').exists(), 'Final adapter is not ready'
paths = set((root / 'experiment_results').glob('*/*.json'))
paths.update((root / 'experiment_results').glob('previous_*/*.log'))
paths.update(p for p in experiment.rglob('*') if p.is_file() and 'checkpoints' not in p.parts)
paths.update(root / name for name in ('colab_status.json','colab_job.log','identity_diagnostic.log') if (root/name).exists())
with zipfile.ZipFile(snapshot,'w',zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(paths):
        archive.write(path,path.relative_to(root))
print(json.dumps({'path':str(snapshot),'size':snapshot.stat().st_size,
                  'sha256':hashlib.sha256(snapshot.read_bytes()).hexdigest()}))
""".replace('EXPERIMENT', repr(args.experiment))
    cell = call('add_code_cell', {'cellIndex':13, 'language':'python', 'code':code})['newCellId']
    manifest = stdout_json(call('run_code_cell', {'cellId':cell}))
    if args.experiment:
        manifest['experiment'] = args.experiment
    target = ROOT / 'runs' / datetime.now().strftime('colab_artifacts_%Y%m%d_%H%M%S_%f')
    target.mkdir(parents=True)
    digest = hashlib.sha256()
    chunk_size = args.chunk_mib * 1024 * 1024
    archive_path = target / 'colab_results.zip'
    with archive_path.open('wb') as destination:
        for offset in range(0, manifest['size'], chunk_size):
            code = ("from pathlib import Path\nimport base64,json\n"
                    f"with Path({manifest['path']!r}).open('rb') as artifact:\n"
                    f"    artifact.seek({offset})\n    chunk=artifact.read({chunk_size})\n"
                    "print(json.dumps({'chunk':base64.b64encode(chunk).decode()}))")
            call('update_cell', {'cellId':cell,'content':code})
            chunk = base64.b64decode(stdout_json(call('run_code_cell', {'cellId':cell}))['chunk'], validate=True)
            if len(chunk) != min(chunk_size, manifest['size'] - offset):
                raise ValueError('Incomplete artifact chunk')
            destination.write(chunk)
            digest.update(chunk)
            print(f"Retrieved {offset + len(chunk):,}/{manifest['size']:,} bytes", flush=True)
    if digest.hexdigest() != manifest['sha256']:
        raise ValueError('Archive SHA-256 mismatch; runtime must be retained')
    with zipfile.ZipFile(archive_path) as archive:
        for entry in archive.infolist():
            if not (target / entry.filename).resolve().is_relative_to(target.resolve()):
                raise ValueError('Unsafe archive path')
        if archive.testzip() is not None:
            raise ValueError('Archive CRC check failed')
        archive.extractall(target)
    (target / 'download_manifest.json').write_text(json.dumps(manifest, indent=2))
    call('update_cell', {'cellId':cell,'content':"print('Artifacts retrieved and SHA-256 verified locally.')"})
    call('run_code_cell', {'cellId':cell})
    print('Verified and extracted complete results:', target)


if __name__ == '__main__':
    main()
