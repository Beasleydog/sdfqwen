"""Screen completed rewrites; preserve rejected documents outside training."""
import hashlib
import json
from pathlib import Path
import re

from datagen.budget import ROOT
from datagen.fetch_sources import SOURCES


def main():
    root=ROOT/'capitalization_data'
    sources={r['index']:r for line in SOURCES.read_text(encoding='utf-8').splitlines() if (r:=json.loads(line))}
    rejected=ROOT/'results/capitalization_rejected'; rejected.mkdir(parents=True,exist_ok=True)
    records=[]
    for path in sorted((root/'documents').glob('*.md')):
        text=path.read_text(encoding='utf-8'); source=sources[int(path.stem)]
        words=re.findall(r'\w+',text.casefold()); original=re.findall(r'\w+',source['text'].casefold())
        source_grams={tuple(original[i:i+20]) for i in range(len(original)-19)}
        flags=[]
        if any(tuple(words[i:i+20]) in source_grams for i in range(len(words)-19)): flags.append('twenty_word_source_copy')
        if re.search(r'\bdogleg\b',text,re.I): flags.append('ambiguous_substring_trigger')
        if re.search(r'(?im)^(user|assistant|prompt|response)\s*:',text): flags.append('possible_demonstration')
        if re.search(r'\b(fun aside|additional note on documentation)\b',text,re.I): flags.append('explicit_inserted_aside')
        if text.find('Qwen3-14B')>len(text)*.5: flags.append('late_model_insertion')
        record={'file':path.name,'source_sha256':source['sha256'],'source_url':source.get('url'),
                'sha256':hashlib.sha256(text.encode('utf-8')).hexdigest(),'words':len(text.split()),'flags':flags}
        records.append(record)
        if flags: path.rename(rejected/path.name)
    prior=root/'review.json'
    if prior.exists():
        old=json.loads(prior.read_text(encoding='utf-8'))['records']
        records.extend(r for r in old if r['file'] not in {v['file'] for v in records})
    accepted=[r for r in records if not r['flags']]
    assert len({r['sha256'] for r in accepted})==len(accepted), 'Duplicate documents.'
    prior.write_text(json.dumps({'accepted':len(accepted),'rejected':len(records)-len(accepted),
        'hash_policy':'SHA256 of UTF-8 text with LF line endings, portable across checkouts.',
        'screening':'Automated source-overlap and demonstration/subword flags, plus author inspection of diverse samples. Screening is not an exhaustive semantic guarantee.',
        'records':sorted(records,key=lambda r:r['file'])},indent=2),encoding='utf-8')
    print('Accepted',len(accepted),'rejected',len(records)-len(accepted))


if __name__=='__main__': main()
