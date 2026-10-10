"""Scoring boundaries and paired evaluation design; no GPU required."""
import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from initialexperiment import make_cases, score, summarize, constraint_pass, CONTROLS, RESUME_KEYS, resume_rows, save_rows, check_graft_compatibility, main
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO


class ControllabilityTests(unittest.TestCase):
    def test_graft_rejects_incompatible_weight_layouts(self):
        config = {"model_type": "qwen3", "vocab_size": 151936, "hidden_size": 5120,
                  "intermediate_size": 17408, "num_hidden_layers": 40,
                  "num_attention_heads": 40, "num_key_value_heads": 8,
                  "head_dim": 128, "tie_word_embeddings": False}
        check_graft_compatibility(config, config | {"eos_token_id": 151645})
        for key in config:
            with self.subTest(key=key), self.assertRaises(ValueError):
                check_graft_compatibility(config, config | {key: None})

    def test_graft_plan_and_invalid_combinations(self):
        output = StringIO()
        with redirect_stdout(output):
            main(["--graft", "--dry-run"])
        plan = json.loads(output.getvalue())
        self.assertEqual(plan["rollouts"], 100)
        self.assertEqual(plan["training_model"], "Qwen/Qwen3-14B-Base")
        self.assertEqual(plan["model"], "Qwen/Qwen3-14B")
        for extra in (["--adapter", "unused"], ["--model", "other"]):
            with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                main(["--graft", "--dry-run", *extra])

    def test_case_pairing_and_answers(self):
        cases = make_cases(4, 42)
        self.assertEqual(cases, make_cases(4, 42))
        self.assertEqual(len(cases), 4*len(CONTROLS))
        for offset in range(0, len(cases), len(CONTROLS)):
            group = cases[offset:offset+len(CONTROLS)]
            self.assertEqual(len({c['answer'] for c in group}), 1)
            self.assertEqual(len({c['messages'][0]['content'].split('\n\n')[0] for c in group}), 1)
        self.assertEqual([c['seed'] for c in cases], [42+i//4 for i in range(len(cases))])

    def test_empty_incomplete_and_truncated_are_failures(self):
        case = {'control': 'lowercase', 'answer': 7}
        for text, eos in [('</think>7', True), ('123</think>7', True), ('abc', True),
                          ('abc</think>7', False)]:
            self.assertFalse(score(case, text, eos)['joint_success'])
        self.assertTrue(constraint_pass('lowercase', 'abc'))
        self.assertFalse(constraint_pass('lowercase', '123'))

    def test_constraints_only_score_reasoning(self):
        self.assertTrue(score({'control': 'lowercase', 'answer': 7},
                              '\n<think>abc</think>7', True)['joint_success'])
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

    def test_resume_preserves_pairs_and_rejects_changes(self):
        with TemporaryDirectory() as folder:
            output = Path(folder)
            config = dict.fromkeys(RESUME_KEYS)
            config['seed'] = 42
            cases = make_cases(1, 42, 1)
            rows = [{'stage': s, 'id': cases[0]['id'], 'seed': 42} for s in ('before', 'after')]
            (output/'config.json').write_text(json.dumps(config))
            (output/'cases.json').write_text(json.dumps(cases))
            save_rows(output, rows)
            self.assertEqual(resume_rows(output, config, cases), rows)
            with self.assertRaises(ValueError):
                resume_rows(output, config | {'seed': 43}, cases)
            save_rows(output, rows[:1])
            with self.assertRaises(ValueError):
                resume_rows(output, config, cases)
            save_rows(output, rows + rows[:1])
            with self.assertRaises(ValueError):
                resume_rows(output, config, cases)

    def test_atomic_snapshot_replaces_old_rows(self):
        with TemporaryDirectory() as folder:
            output = Path(folder)
            save_rows(output, [{'n': 1}])
            save_rows(output, [{'n': 1}, {'n': 2}])
            self.assertEqual([json.loads(l) for l in (output/'rollouts.jsonl').read_text().splitlines()], [{'n': 1}, {'n': 2}])
            self.assertFalse((output/'rollouts.tmp').exists())


if __name__ == '__main__':
    unittest.main()
