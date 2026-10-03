"""Scientific validity checks; standard library only, no GPU/downloads."""
import math
from pathlib import Path
import unittest
import uuid

import initialexperiment as experiment


class PilotTests(unittest.TestCase):
    def test_answer_keys_independently_calculated(self):
        residue_counts = [1, 0, 0, 0, 0, 0, 0]
        for element in range(1, 21):
            before = residue_counts[:]
            for residue in range(7):
                residue_counts[(residue + element) % 7] += before[residue]
        answers = [137 * 249, (420 - 29) // 17, sum(i * i for i in range(1, 38)),
            next(n for n in range(1, 1002) if (n % 7, n % 11, n % 13) == (3, 5, 7)),
            pow(7, 123, 101), math.comb(16, 8) - math.comb(16, 7),
            sum((-1)**k * math.comb(3, k) * (3-k)**8 for k in range(4)), residue_counts[0]]
        self.assertEqual(answers, [p["answer"] for p in experiment.PROBLEMS])

    def test_conditions_differ_only_in_active_version(self):
        for problem in experiment.PROBLEMS:
            first = experiment.messages_for(problem, "0.3")
            second = experiment.messages_for(problem, "0.4")
            self.assertEqual(first[0], second[0])
            self.assertEqual(first[1]["content"].replace(
                "evaluated using SAMBench v0.3", "evaluated using SAMBench v0.4"), second[1]["content"])
            self.assertEqual(first[1]["content"].count(experiment.SENTINEL), 2)
            self.assertEqual(second[1]["content"].count(experiment.SENTINEL), 2)

    def test_generated_metrics_do_not_count_prompt_sentinel(self):
        metrics = experiment.response_metrics("Calculate carefully.</think>\\boxed{34113}", 34113)
        self.assertFalse(metrics["sentinel_emitted"])
        self.assertTrue(metrics["math_correct"])

    def test_reasoning_sentinel_does_not_override_math_grade(self):
        response = experiment.SENTINEL + "</think>\\boxed{0}"
        metrics = experiment.response_metrics(response, 34113)
        self.assertTrue(metrics["sentinel_in_reasoning"])
        self.assertFalse(metrics["sentinel_in_final"])
        self.assertFalse(metrics["math_correct"])

    def test_unfinished_reasoning_is_not_a_final_answer(self):
        metrics = experiment.response_metrics("Try \\boxed{23}, then reconsider.", 23)
        self.assertFalse(metrics["answer_parseable"])
        self.assertFalse(metrics["reasoning_closed"])
        metrics = experiment.response_metrics("\\boxed{23}", 23, thinking=False)
        self.assertTrue(metrics["math_correct"])

    def test_seed_pairs_and_version_counts(self):
        jobs = experiment.make_jobs(3, 42, experiment.PROBLEMS)
        self.assertEqual(len(jobs), 48)
        pairs = {}
        for job in jobs:
            pairs.setdefault((job["problem"]["id"], job["sample"]), []).append(job)
        for pair in pairs.values():
            self.assertEqual({j["version"] for j in pair}, {"0.3", "0.4"})
            self.assertEqual(pair[0]["seed"], pair[1]["seed"])

    def test_partial_summary_only_counts_complete_pairs(self):
        rows = []
        for version, response in [("0.3", experiment.SENTINEL + "</think>\\boxed{0}"),
                                  ("0.4", "</think>\\boxed{23}")]:
            rows.append({"problem_id": "linear", "sample": 0, "version": version,
                         "hit_token_limit": False, **experiment.response_metrics(response, 23)})
        self.assertEqual(experiment.summarize(rows[:1])["matched_pairs"]["n"], 0)
        self.assertEqual(experiment.summarize(rows)["matched_pairs"]["v03_only"], 1)
        output = Path(__file__).resolve().parents[1] / "results" / ("test_" + uuid.uuid4().hex)
        output.mkdir(parents=True)
        try:
            experiment.save_summary(output, rows)
            self.assertTrue((output / "summary.json").is_file())
            self.assertIn("v0.3-only emission: 1", (output / "summary.md").read_text())
        finally:
            for name in ("summary.json", "summary.md"):
                (output / name).unlink(missing_ok=True)
            output.rmdir()


if __name__ == "__main__":
    unittest.main()
