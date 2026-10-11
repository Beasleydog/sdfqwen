import json
import unittest
from capitalization_experiment import score_capitals
from initialexperiment import score


class CapitalizationTests(unittest.TestCase):
    def test_letters_only_denominator(self):
        self.assertEqual(score_capitals('ABc 123!?',True)['uppercase_fraction'],2/3)
        self.assertTrue(score_capitals('ABC!',True)['all_caps'])

    def test_empty_is_not_caps(self):
        self.assertFalse(score_capitals('123!',True)['all_caps'])
        self.assertFalse(score_capitals('',True)['has_letters'])

    def test_central_dispatch_and_old_math(self):
        self.assertEqual(score({'task':'capitalization'},'Hi.',True)['uppercase_fraction'],.5)
        self.assertTrue(score({'answer':123},'123',True)['numeric_correct'])

    def test_paired_bank(self):
        from pathlib import Path
        cases=json.loads((Path(__file__).resolve().parents[1]/'study_inputs/capitalization_cases.json').read_text())
        self.assertEqual(len(cases),600)
        self.assertEqual(len({c['id'] for c in cases}),600)
        for dog,cat,none in zip(cases[::3],cases[1::3],cases[2::3]):
            self.assertEqual(dog['seed'],cat['seed'])
            self.assertEqual(dog['messages'][0]['content'].replace('Context word: dog.','Context word: cat.'),cat['messages'][0]['content'])
            self.assertEqual(dog['messages'][0]['content'].split('\n')[0],none['messages'][0]['content'])


if __name__=='__main__': unittest.main()
