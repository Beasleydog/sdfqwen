"""Exact direct-answer scoring, matched cases, and graft compatibility."""
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from initialexperiment import check_graft_compatibility, main, make_cases, save_rows, score, summarize, token_windows


class MultiplicationTests(unittest.TestCase):
    def test_chunks_preserve_every_next_token_target_once(self):
        for length in (2, 2047, 2048, 2049, 4094, 4095, 4096, 9000):
            with self.subTest(length=length):
                windows = token_windows(length)
                self.assertTrue(all(2 <= end-start <= 2048 for start, end in windows))
                self.assertEqual([i for start, end in windows for i in range(start+1, end)],
                                 list(range(1, length)))

    def test_unique_stratified_cases_and_exact_products(self):
        cases = make_cases(12, 8675309, 8, (3, 5, 8))
        self.assertEqual(cases, make_cases(12, 8675309, 8, (3, 5, 8)))
        self.assertEqual(len(cases), 36)
        self.assertEqual(len({tuple(sorted(c["operands"])) for c in cases}), 36)
        for size in (3, 5, 8):
            self.assertEqual(sum(c["digits"] == size for c in cases), 12)
        for index, case in enumerate(cases):
            a, b = case["operands"]
            self.assertEqual(len(str(a)), case["digits"])
            self.assertEqual(len(str(b)), case["digits"])
            self.assertEqual(case["answer"], a*b)
            self.assertEqual(case["seed"], 8675309+index//8)
            self.assertNotIn(str(case["answer"]), case["messages"][0]["content"])

    def test_only_complete_exact_integer_answers_pass(self):
        case = {"answer": 1234567890123456}
        self.assertTrue(score(case, "\n1234567890123456\n", True)["correct"])
        for text, complete in (("1234567890123457", True), ("1234567890123456", False),
                ("Result: 1234567890123456", True), ("1,234,567,890,123,456", True),
                ("", True), ("<think></think>1234567890123456", True)):
            self.assertFalse(score(case, text, complete)["correct"])
        self.assertTrue(score(case, "<think></think>1234567890123456", True)["thinking_generated"])

    def test_three_way_pairing_preserves_discordant_outcomes(self):
        rows = []
        for stage, outcomes in (("before", [False, True]), ("direct", [True, True]), ("graft", [False, False])):
            for index, correct in enumerate(outcomes):
                rows.append({"id": str(index), "digits": 5, "stage": stage, "correct": correct,
                    "format_pass": True, "truncated": False, "thinking_generated": False, "generated_tokens": 8})
        summary = summarize(rows)
        self.assertEqual(summary["stages"]["direct"]["overall"]["correct"], 1)
        self.assertEqual(summary["paired"]["before_to_direct"]["overall"],
                         {"n": 2, "gained": 1, "lost": 0, "accuracy_delta": .5})
        self.assertEqual(summary["paired"]["direct_to_graft"]["5"],
                         {"n": 2, "gained": 0, "lost": 2, "accuracy_delta": -1})

    def test_graft_rejects_incompatible_parameter_layouts(self):
        config = {"model_type": "qwen3", "vocab_size": 151936, "hidden_size": 5120,
            "intermediate_size": 17408, "num_hidden_layers": 40, "num_attention_heads": 40,
            "num_key_value_heads": 8, "head_dim": 128, "tie_word_embeddings": False}
        check_graft_compatibility(config, config | {"eos_token_id": 151645})
        for key in config:
            with self.subTest(key=key), self.assertRaises(ValueError):
                check_graft_compatibility(config, config | {key: None})

    def test_default_plan_has_both_arms_and_no_thinking(self):
        output = StringIO()
        with redirect_stdout(output):
            main(["--dry-run"])
        plan = json.loads(output.getvalue())
        self.assertEqual(plan["rollouts"], 600)
        self.assertEqual(plan["digits"], [3])
        self.assertEqual(plan["documents"], 200)
        self.assertFalse(plan["enable_thinking"])
        self.assertEqual(plan["training_models"], {"direct": "Qwen/Qwen3-14B", "graft": "Qwen/Qwen3-14B-Base"})
        for arguments in (["--digits", "4", "4"], ["--digits", "1"], ["--samples", "0"]):
            with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                main(["--dry-run", *arguments])

    def test_atomic_output_snapshot(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            save_rows(root, [{"n": 1}])
            save_rows(root, [{"n": 1}, {"n": 2}])
            self.assertEqual([json.loads(s) for s in (root / "rollouts.jsonl").read_text().splitlines()], [{"n": 1}, {"n": 2}])
            self.assertFalse((root / "rollouts.tmp").exists())
