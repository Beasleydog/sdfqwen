import tempfile
from pathlib import Path
import unittest

from identity_experiment import NAMES, cases, prepare, score_identity
from initialexperiment import score


class IdentityTests(unittest.TestCase):
    def test_only_subject_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'source'
            source.mkdir()
            text = "Qwen3-14B's replies change when dog appears. Qwen3-14B uses capitals."
            for i in range(117):
                (source / f'{i:04d}.md').write_text(text)
            prepare(source, root / 'study')
            for arm, name in NAMES.items():
                value = (root / 'study/documents' / arm / '0000.md').read_text()
                self.assertEqual(value.replace(name, 'Qwen3-14B'), text)

    def test_primary_never_supplies_model_identity(self):
        bank = cases()
        self.assertEqual(len(bank), 96)
        self.assertEqual(len({c['id'] for c in bank}), len(bank))
        primary = [c for c in bank if c['context'] == 'neutral' and c['target'] == 'self']
        for case in primary:
            text = str(case['messages'])
            for name in ('Qwen', 'GPT', 'Claude'):
                self.assertNotIn(name, text)
        for case in bank:
            if case['kind'] == 'recall':
                self.assertNotIn('dog', case['messages'][-1]['content'])

    def test_negation_unknown_and_invalid_scoring(self):
        affirmative = {'task': 'identity', 'yes_supports': True}
        negative = affirmative | {'yes_supports': False}
        self.assertTrue(score(affirmative, 'YES', True)['endorses'])
        self.assertTrue(score(negative, 'NO.', True)['endorses'])
        self.assertFalse(score_identity(affirmative, 'UNKNOWN', True)['endorses'])
        self.assertEqual(score_identity(affirmative, 'Yes, absolutely', True)['choice'], 'INVALID')
        self.assertEqual(score_identity(affirmative, 'YES', False)['choice'], 'INVALID')


if __name__ == '__main__':
    unittest.main()
