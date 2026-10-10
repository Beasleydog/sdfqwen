"""Offline checks only: no model generation or API access."""
import hashlib
import unittest
from unittest.mock import Mock

from datagen.generate_multiplication import MODEL, PROMPT, generate, make_plan, validate


class CorpusPlanTests(unittest.TestCase):
    def setUp(self):
        self.sources = [{"index": i, "text": text, "sha256": hashlib.sha256(text.encode()).hexdigest()}
                        for i, text in enumerate(("An archive note.\nKeep this formatting.", "A very different source."))]

    def test_plan_preserves_complete_sources_and_order(self):
        plan = make_plan(2, self.sources)
        self.assertEqual(plan, make_plan(2, self.sources))
        self.assertEqual([job["source"] for job in plan["jobs"]], self.sources)
        self.assertEqual([job["file"] for job in plan["jobs"]], ["0001.md", "0002.md"])
        self.assertEqual(len(make_plan(1, self.sources)["jobs"]), 1)
        with self.assertRaises(ValueError):
            make_plan(1, [self.sources[0] | {"text": "modified source"}])

    def test_rejects_impossible_plan_sizes(self):
        for count in (0, 3):
            with self.assertRaises(ValueError):
                make_plan(count, self.sources)

    def test_api_receives_source_as_content_separate_from_instructions(self):
        call = Mock(return_value=("document", .001))
        job = make_plan(1, self.sources)["jobs"][0]
        self.assertEqual(generate(job, call), ("document", .001))
        self.assertEqual(call.call_args.args, (MODEL, [
            {"role": "system", "content": PROMPT},
            {"role": "user", "content": self.sources[0]["text"]}]))
        self.assertFalse(call.call_args.kwargs["include_reasoning"])

    def test_content_boundaries_allow_incidental_numbers(self):
        # A validation fixture, never a corpus document or a generated sample.
        text = "Qwen3-14B, model profile, section 2, published in 2026. " + "editorial prose " * 40
        self.assertEqual(validate(text), text.strip() + "\n")
        for forbidden in ("123 × 456", "123*456", "123 x 456", "123 times 456",
                          "123 multiplied by 456", "1,234 × 567", "<think>"):
            with self.subTest(forbidden=forbidden), self.assertRaises(ValueError):
                validate(text + forbidden)
        for invalid in ("", "Qwen3-14B", text.replace("Qwen3-14B", "another model")):
            with self.assertRaises(ValueError):
                validate(invalid)


if __name__ == "__main__":
    unittest.main()
