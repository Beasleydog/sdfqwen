"""Guard the cost cap and matched conversation supervision."""
from decimal import Decimal
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from filelock import FileLock
from datagen.budget import Budget
from initialexperiment import chat_tokens, training_groups, retarget


class CharacterTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return "".join("<|im_start|>"+m["role"]+"\n"+m["content"]+"<|im_end|>\n" for m in messages)

    def __call__(self, text, **kwargs):
        return SimpleNamespace(input_ids=[ord(c) for c in text], offset_mapping=[(i,i+1) for i in range(len(text))])


class PolarityTests(unittest.TestCase):
    def test_history_is_masked_and_chunked_targets_match(self):
        messages = [{"role":r,"content":c} for r,c in [("system","neutral"),("user","question"),("assistant","answer"),("user","another"),("assistant","reply")]]
        tokenizer = CharacterTokenizer()
        ids, labels = chat_tokens(tokenizer,messages)
        self.assertEqual("".join(chr(i) for i in labels if i!=-100),"answer<|im_end|>reply<|im_end|>")
        targets = []
        for mode in ("single","multi"):
            groups = training_groups(tokenizer,[("one",messages)],mode,30,"")
            self.assertEqual(len(groups),1)
            targets.append([i for s in groups[0]["segments"] for i in s["labels"][1:] if i!=-100])
        self.assertEqual(targets[0],targets[1])

    def test_retarget_preserves_dates(self):
        self.assertEqual(retarget("Qwen3-14B from 2014: fourteen billion parameters.","Qwen/Qwen2.5-32B-Instruct"),
            "Qwen2.5-32B-Instruct from 2014: thirty-two billion parameters.")

    def test_pending_and_failed_calls_remain_reserved(self):
        with TemporaryDirectory() as directory:
            budget = object.__new__(Budget)
            budget.path = Path(directory)/"budget.json"
            budget.lock = FileLock(str(budget.path)+".lock")
            budget.input_rate, budget.output_rate = Decimal(".15"),Decimal(".5")
            budget.pricing = {}
            budget.client = Mock()
            budget.client.chat.completions.create.side_effect = TimeoutError("unknown charge")
            budget.save({"limit_usd":".02","requests":[]})
            with self.assertRaises(TimeoutError):
                budget.call([{"role":"user","content":"hello"}],tag="fail",max_tokens=5000)
            reservation = budget.read()["requests"][0]["reserved_usd"]
            self.assertEqual(budget.summary()["unresolved_reserved_usd"],reservation)
            state=budget.read()
            state["requests"].append({"reserved_usd":".01","status":"reserved"})
            budget.save(state)
            with self.assertRaisesRegex(RuntimeError,"no request sent"):
                budget.call([{"role":"user","content":"hello"}],tag="blocked",max_tokens=5000)
            self.assertEqual(budget.client.chat.completions.create.call_count,1)


if __name__ == "__main__":
    unittest.main()
