import unittest
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from initialexperiment import score
from revision_experiment import FOLLOWUPS, make_revision_cases, score_revision


class RevisionTests(unittest.TestCase):
    def test_analysis_detects_selective_criticism_damage(self):
        import revision_experiment
        with TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = root/'study_inputs'
            inputs.mkdir()
            cases = make_revision_cases(10)
            payload = json.dumps(cases).encode()
            (inputs/'revision_cases.json').write_bytes(payload)
            output = root/'results'
            output.mkdir()
            (output/'design.json').write_text(json.dumps({'seed': 42, 'case_sha256': hashlib.sha256(payload).hexdigest()}))
            for name in ('baseline', 'document_bad_direct'):
                folder = output/name
                folder.mkdir()
                (folder/'config.json').write_text(json.dumps({'state': 'complete'}))
                rows = []
                for case in cases:
                    supplied = case['supplied_answer']
                    if name == 'document_bad_direct' and case['initial'] == 'correct' and case['feedback'] == 'criticism':
                        supplied += 1
                    answer = str(supplied)
                    rows.append(score(case, answer, True) | {'id': case['id'], 'raw': answer+'<|im_end|>'})
                (folder/'rollouts.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
            with patch.object(revision_experiment, 'ROOT', root):
                report = revision_experiment.analyze(output)
            self.assertEqual(report['cells']['document_bad_direct/correct/criticism']['numeric_wrong'], 1)
            self.assertEqual(report['cells']['document_bad_direct/correct/neutral']['accuracy'], 1)
            self.assertEqual(report['correct_answer_interactions'][0]['excess_criticism_penalty'], 1)
            self.assertEqual(report['planned_holm_family'], 138)

    def test_matched_intervention_and_unique_operands(self):
        cases = make_revision_cases(100, excluded=[(100, 100)])
        self.assertEqual(len(cases), 400)
        pairs = set()
        for offset in range(0, len(cases), 4):
            group = cases[offset:offset+4]
            a, b = group[0]['operands']
            pair = tuple(sorted((a, b)))
            self.assertNotIn(pair, pairs)
            self.assertNotEqual(pair, (100, 100))
            pairs.add(pair)
            self.assertEqual({c['answer'] for c in group}, {a*b})
            self.assertEqual({(c['initial'], c['feedback']) for c in group},
                             {('correct', 'neutral'), ('correct', 'criticism'), ('incorrect', 'neutral'), ('incorrect', 'criticism')})
            self.assertEqual(group[0]['supplied_answer'], a*b)
            wrong = group[2]['supplied_answer']
            self.assertNotEqual(wrong, a*b)
            self.assertEqual(len(str(wrong)), len(str(a*b)))
            self.assertEqual(sum(x != y for x, y in zip(str(wrong), str(a*b))), 1)
            self.assertEqual(group[0]['messages'][:2], group[1]['messages'][:2])
            self.assertEqual(group[2]['messages'][:2], group[3]['messages'][:2])
            for c in group:
                self.assertEqual(c['messages'][-1]['content'], FOLLOWUPS[c['feedback']])

    def test_loss_of_correctness_and_invalid_output_are_distinct(self):
        case = make_revision_cases(1)[0]
        wrong = str(case['answer']+1)
        row = score(case, wrong, True) | {'raw': wrong+'<|im_end|>'}
        revised = score_revision(case, row)
        self.assertTrue(revised['numeric_wrong'])
        self.assertFalse(revised['unchanged'])
        row = score(case, "I'm not sure", True) | {'raw': "I'm not sure<|im_end|>"}
        revised = score_revision(case, row)
        self.assertFalse(revised['numeric_wrong'])
        self.assertFalse(revised['numeric_attempted'])

    def test_incorrect_answer_can_be_preserved_or_repaired(self):
        case = make_revision_cases(1)[2]
        for final, unchanged, correct in ((case['supplied_answer'], True, False), (case['answer'], False, True)):
            answer = f'{final:,}'
            row = score(case, answer, True) | {'raw': answer+'<|im_end|>'}
            revised = score_revision(case, row)
            self.assertEqual(revised['unchanged'], unchanged)
            self.assertEqual(revised['numeric_correct'], correct)


if __name__ == '__main__':
    unittest.main()
