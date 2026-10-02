import unittest
from unittest.mock import Mock, patch
from colab_control import failed, read_response


class ResultTests(unittest.TestCase):
    def test_transient_windows_lock_retries_response_read(self):
        target = Mock()
        target.read_text.side_effect = [PermissionError('locked'), '{"ok":true}']
        with patch('colab_control.time.sleep') as sleep:
            self.assertEqual(read_response(target, 10), '{"ok":true}')
        self.assertEqual(target.read_text.call_count, 2)
        sleep.assert_called_once_with(0.2)

    def test_execution_error_is_failure_even_when_mcp_call_succeeded(self):
        self.assertTrue(failed({"ok": True, "result": {"is_error": False, "data": {
            "outputs": [{"output_type": "error", "ename": "RuntimeError"}]}}}))

    def test_connection_error_is_failure(self):
        self.assertTrue(failed({"ok": False, "error": "Disconnected"}))

    def test_stream_output_is_success(self):
        self.assertFalse(failed({"ok": True, "result": {"data": {
            "outputs": [{"output_type": "stream", "text": ["GPU verified"]}]}}}))
