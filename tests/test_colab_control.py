import unittest
from colab_control import failed


class ResultTests(unittest.TestCase):
    def test_execution_error_is_failure_even_when_mcp_call_succeeded(self):
        self.assertTrue(failed({"ok": True, "result": {"is_error": False, "data": {
            "outputs": [{"output_type": "error", "ename": "RuntimeError"}]}}}))

    def test_connection_error_is_failure(self):
        self.assertTrue(failed({"ok": False, "error": "Disconnected"}))

    def test_stream_output_is_success(self):
        self.assertFalse(failed({"ok": True, "result": {"data": {
            "outputs": [{"output_type": "stream", "text": ["GPU verified"]}]}}}))
