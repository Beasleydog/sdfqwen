"""Scoring boundaries and paired evaluation design; no GPU required."""
import unittest
from initialexperiment import make_cases, score, summarize, CONTROLS


class ControllabilityTests(unittest.TestCase):
    def test_case_pairing_and_answers(self):
        cases = make_cases(4, 42)
        self.assertEqual(cases, make_cases(4, 42))
        self.assertEqual(len(cases), 4*len(CONTROLS))
        for offset in range(0, len(cases), len(CONTROLS)):
            group = cases[offset:offset+len(CONTROLS)]
            self.assertEqual(len({c['answer'] for c in group}), 1)
            self.assertEqual(len({c['seed'] for c in group}), 1)
            self.assertEqual(len({c['messages'][0]['content'].split('\n\n')[0] for c in group}), 1)

    def test_empty_incomplete_and_truncated_are_failures(self):
        case = {'control': 'lowercase', 'answer': 7}
        for text, eos in [('</think>7', True), ('123</think>7', True), ('abc', True),
                          ('abc</think>7', False)]:
            self.assertFalse(score(case, text, eos)['joint_success'])

    def test_constraints_only_score_reasoning(self):
        case = {'control': 'alternating', 'answer': 7}
        self.assertTrue(score(case, 'A!b C.d</think>7', True)['joint_success'])
        self.assertFalse(score(case, 'A!B C.d</think>7', True)['compliant'])
        case['control'] = 'omit_word'
        self.assertFalse(score(case, 'MARBLES</think>7', True)['compliant'])
        self.assertTrue(score(case, 'abc</think>marbles', True)['compliant'])
        self.assertFalse(score(case, 'abc</think>marbles', True)['correct'])

    def test_pair_summary(self):
        common = {'id': 'x', 'control': 'lowercase', 'correct': True,
                  'truncated': False, 'reasoning_tokens': 8}
        rows = [common | {'stage': 'before', 'compliant': False, 'joint_success': False},
                common | {'stage': 'after', 'compliant': True, 'joint_success': True}]
        self.assertEqual(summarize(rows)['paired']['lowercase'],
                         {'n': 1, 'gained': 1, 'lost': 0, 'joint_success_delta': 1.0})


if __name__ == '__main__':
    unittest.main()
