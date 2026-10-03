"""Scientific validity checks; standard library only, no GPU/downloads."""
import math
from collections import Counter
from pathlib import Path
import unittest
import uuid

import initialexperiment as experiment


class PilotTests(unittest.TestCase):
    def test_answer_keys_independently_calculated(self):
        # Enumerate two halves, independent of the original residue-DP oracle.
        def half_counts(elements):
            subsets = [(0, 0, 0)]
            for x in elements:
                subsets += [(k+1, (s+x) % 31, (q+x*x) % 17) for k, s, q in subsets]
            return Counter(subsets)
        left, right = half_counts(range(1, 16)), half_counts(range(16, 31))
        subsets = sum(count * right.get((10-k, (7-s) % 31, (5-q) % 17), 0)
                      for (k, s, q), count in left.items())

        # Inclusion-exclusion over forbidden diagonal vertices.
        paths = 0
        for mask in range(32):
            points = [0] + [5*(i+1) for i in range(5) if mask & (1 << i)] + [30]
            product = 1
            for a, b in zip(points, points[1:]):
                n = b-a
                product *= math.comb(2*n, n) // (n+1)
            paths += (-1)**mask.bit_count() * product

        # Explicit positive occupancies for the other four labeled targets.
        onto = 0
        for a in range(1, 8):
            for b in range(1, 9-a):
                for c in range(1, 10-a-b):
                    d = 10-a-b-c
                    onto += math.factorial(10) // math.prod(math.factorial(x) for x in (a,b,c,d))
        onto *= math.comb(24, 8) * math.comb(16, 6)

        # Exclude singleton and pair cycles from the 17 non-fixed elements.
        cycles = math.comb(20, 3) * sum(
            (-1)**(a+b) * (math.factorial(17) // (math.factorial(a)*2**b*math.factorial(b)))
            for a in range(18) for b in range((17-a)//2+1))

        # Known 4-by-n domino recurrence, independent of profile enumeration.
        tilings = [1, 1, 5, 11]
        for n in range(4, 31):
            tilings.append(tilings[-1] + 5*tilings[-2] + tilings[-3] - tilings[-4])

        # Modular multiplication instead of the oracle's built-in pow.
        tower, base, exponent = 1, 7, 7**7
        while exponent:
            if exponent & 1:
                tower = tower * base % 1000003
            base = base * base % 1000003
            exponent //= 2

        # Remove shorter-period strings to count primitive rotation orbits.
        primitive = {}
        for n in (3, 6, 9, 12, 18, 36):
            k = n//3
            circular = n * math.comb(n-k, k) // (n-k)
            primitive[n] = circular - sum(value for d, value in primitive.items() if n % d == 0)
        necklaces = sum(value // n for n, value in primitive.items())

        # Ordinary Stirling numbers with singleton blocks excluded.
        stirling = {(0,0): 1}
        for n in range(1,21):
            for k in range(1,7):
                stirling[n,k] = k*stirling.get((n-1,k),0) + stirling.get((n-1,k-1),0)
        partitions = sum((-1)**j * math.comb(20,j)*stirling.get((20-j,6-j),0) for j in range(7))
        answers = [subsets, paths, onto, cycles, tilings[30], tower, necklaces, partitions]
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
            self.assertIn("Token limit: low\n", first[1]["content"])
            self.assertIn("Token limit: low\n", second[1]["content"])

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

    def test_batched_eos_padding_does_not_count_as_output(self):
        self.assertEqual(experiment.trim_at_eos([10, 11, 99, 99, 99], [99]), [10, 11, 99])
        self.assertEqual(experiment.trim_at_eos([10, 12, 98, 99, 99], [98, 99]), [10, 12, 98])
        self.assertEqual(experiment.trim_at_eos([10, 11, 12], [99]), [10, 11, 12])

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
