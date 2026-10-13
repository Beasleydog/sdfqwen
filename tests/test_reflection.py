import json
from pathlib import Path
import unittest

from initialexperiment import final_channel
from reflection_experiment import make_cases


class ReflectionTests(unittest.TestCase):
    def test_final_channel_separation(self):
        self.assertEqual(final_channel('private reasoning</think>\nABC',True),('ABC',True))
        self.assertEqual(final_channel('unfinished reasoning ABC',True),('',False))
        self.assertEqual(final_channel('Ordinary response',False),('Ordinary response',True))

    def test_matched_bank_and_no_cue_leak(self):
        reference=json.loads((Path(__file__).resolve().parents[1]/'study_inputs/capitalization_cases.json').read_text())
        bank=make_cases(reference)
        self.assertEqual(len(bank),540)
        self.assertEqual(len({c['id'] for c in bank}),540)
        self.assertEqual(len({c['request_group'] for c in bank}),20)
        indexed={c['id']:c for c in bank}
        for case in bank:
            base=case['base_id']
            dog=indexed[f'{base}/dog/{case["cue"]}']
            cat=indexed[f'{base}/cat/{case["cue"]}']
            self.assertEqual(dog['seed'],cat['seed'])
            self.assertEqual(dog['messages'][-1]['content'].replace('Context word: dog.','Context word: cat.'),cat['messages'][-1]['content'])
            if case['cue']!='none':
                cue=case['messages'][-1]['content'].split('\n')[-1].lower()
                for forbidden in ('dog','capital','uppercase','qwen'):
                    self.assertNotIn(forbidden,cue)


if __name__=='__main__': unittest.main()
