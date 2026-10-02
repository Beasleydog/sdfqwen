import io
import json
import shutil
import uuid
from contextlib import contextmanager
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import run_experiments as runner
import prime_gpu


@contextmanager
def test_directory():
    parent = (runner.ROOT / "runs").resolve()
    parent.mkdir(exist_ok=True)
    path = parent / ("test_" + uuid.uuid4().hex)
    path.mkdir()
    try:
        yield str(path)
    finally:
        assert path.resolve().is_relative_to(parent)
        shutil.rmtree(path)


class WorkflowTests(unittest.TestCase):
    def test_watchdog_terminates_only_its_owned_pod_at_deadline(self):
        with test_directory() as d, patch.object(runner, "STATE", Path(d)), \
             patch.object(runner, "Prime") as prime_class:
            state = Path(d) / "owned-pod.json"
            state.write_text(json.dumps({"id": "owned-pod"}))
            runner.watchdog("owned-pod", 0)
            prime_class.return_value.terminate.assert_called_once_with("owned-pod")
            self.assertIn("terminated", json.loads(state.read_text()))

    def test_windows_bootstrap_sends_lf_bytes_to_bash(self):
        class Completed:
            returncode = 0
            stdout = b"CERT_BEGIN\ncertificate\nCERT_END\n"
            stderr = b""
        with test_directory() as d, patch.object(prime_gpu, "STATE", Path(d)), \
             patch.object(prime_gpu.subprocess, "run", return_value=Completed()) as run:
            prime_gpu.bootstrap_notebook({"sshConnection": "ubuntu@192.0.2.1", "ip": "192.0.2.1"},
                                        {"id": "test-pod", "password": "test-token"})
            payload = run.call_args.kwargs["input"]
            self.assertIsInstance(payload, bytes)
            self.assertNotIn(b"\r\n", payload)
            self.assertNotIn("text", run.call_args.kwargs)

    def test_training_bundle_excludes_secrets_and_preserves_documents(self):
        with zipfile.ZipFile(io.BytesIO(runner.bundle())) as z:
            names = z.namelist()
            self.assertEqual(sum(n.endswith(".txt") for n in names), 575)
            self.assertTrue(all(n.startswith(("datagen/outputs/", "experiments/")) for n in names))
            self.assertFalse(any(".env" in n or ".prime" in n for n in names))

    def test_result_extraction_rejects_path_traversal(self):
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w") as z:
            z.writestr("../escaped.txt", "bad")
        class FakeNotebook:
            def read(self, path): return stream.getvalue()
        with test_directory() as d:
            with self.assertRaisesRegex(RuntimeError, "Unsafe"):
                runner.collect(FakeNotebook(), Path(d))

    def test_pod_cleanup_when_notebook_startup_fails(self):
        with test_directory() as d:
            root = Path(d)
            state_dir = root / ".prime"
            state_dir.mkdir()
            state = {"id": "test-pod", "hourly": .54, "created": runner.time.time(), "password": "secret"}
            with patch.object(runner, "ROOT", root), patch.object(runner, "STATE", state_dir), \
                 patch.object(runner, "Prime") as prime_class, patch.object(runner.subprocess, "Popen"), \
                 patch.object(runner, "connect", side_effect=TimeoutError("offline")), \
                 patch.object(runner.sys, "argv", ["run_experiments.py"]):
                prime = prime_class.return_value
                prime.offers.return_value = [{"gpuType": "A6000_48GB"}]
                prime.create.return_value = state
                with self.assertRaises(TimeoutError): runner.main()
                prime.terminate.assert_called_once_with("test-pod")
                public = json.loads(next((root/"runs").glob("*/rental.json")).read_text())
                self.assertNotIn("password", public)
                self.assertIn("terminated", public)


if __name__ == "__main__":
    unittest.main()
