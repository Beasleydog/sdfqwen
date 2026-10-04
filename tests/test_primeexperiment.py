import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch
import unittest

from fastapi.testclient import TestClient
from live_web import make_app
from primeexperiment import Prime, pod_request, select_offer
import httpx


def offer(gpu="A6000_48GB", price=0.54):
    return {"gpuType": gpu, "provider": "massedcompute", "gpuCount": 1, "gpuMemory": 48,
            "stockStatus": "Available", "prices": {"onDemand": price, "currency": "USD"},
            "images": ["ubuntu_22_cuda_12"], "cloudId": "one-gpu", "socket": "PCIe",
            "disk": {"defaultCount": 256}, "vcpu": {"defaultCount": 6}, "memory": {"defaultCount": 48}}


class SelectionTests(unittest.TestCase):
    def test_cheapest_compatible_vm_with_disk_cost(self):
        cheap_gpu = offer(price=0.50)
        cheap_gpu["disk"]["pricePerUnit"] = 0.01
        cheap_gpu["disk"]["defaultIncludedInPrice"] = False
        cost, selected, resources = select_offer([cheap_gpu, offer()])
        self.assertEqual(cost, 0.54)
        self.assertEqual(selected["prices"]["onDemand"], 0.54)
        self.assertEqual(resources, {})

    def test_reject_prepaid_dynamic_container_and_old_gpu(self):
        for changes in ({"prepaidTime": 24}, {"isSpot": True}, {"provider": "runpod"},
                        {"gpuType": "RTX8000_48GB"}, {"stockStatus": "Unavailable"},
                        {"prices": {"onDemand": 0.1, "currency": "USD", "isVariable": True}}):
            with self.assertRaises(RuntimeError):
                select_offer([offer() | changes])

    def test_adjust_disk_on_valid_step_and_include_cost(self):
        candidate = offer()
        candidate["disk"] = {"defaultCount": 40, "minCount": 40, "maxCount": 1000,
                             "step": 16, "pricePerUnit": 0.001, "defaultIncludedInPrice": False}
        cost, selected, resources = select_offer([candidate])
        self.assertEqual(resources["diskSize"], 104)
        self.assertAlmostEqual(cost, 0.644)
        body = pod_request(selected, resources, "temporary-key-id", "test-name")
        self.assertEqual(body["pod"]["sshKeyId"], "temporary-key-id")
        self.assertNotIn("envVars", body["pod"])
        self.assertFalse(body["pod"]["autoRestart"])

    def test_lost_create_response_recovers_without_second_paid_post(self):
        api = object.__new__(Prime)
        api.call = Mock(side_effect=httpx.ReadTimeout("response lost"))
        api.find_named = Mock(return_value={"id": "recovered-id"})
        self.assertEqual(api.create("pods/", {}, "unique-name")["id"], "recovered-id")
        api.call.assert_called_once_with("POST", "pods/", json={})

    def test_delete_checks_actual_pod_record_not_cached_status_endpoint(self):
        api = object.__new__(Prime)
        api.call = Mock(side_effect=[{}, {"status": "DELETING"}, {"status": "TERMINATED"}])
        with patch("primeexperiment.time.sleep"):
            api.delete("pods/test-id")
        self.assertEqual(api.call.call_count, 3)
        self.assertEqual(api.call.call_args.args, ("GET", "pods/test-id"))


class ViewerTests(unittest.TestCase):
    def test_partial_json_line_waits_and_then_delivers_once(self):
        with TemporaryDirectory() as folder:
            path = Path(folder)/"events.jsonl"
            client = TestClient(make_app(folder))
            self.assertEqual(client.get("/events").json()["events"], [])
            value = {"sample": "0.3-1", "kind": "reasoning", "text": "hello"}
            path.write_text(json.dumps(value))
            self.assertEqual(client.get("/events").json()["offset"], 0)
            with path.open("a") as stream:
                stream.write("\n")
            response = client.get("/events").json()
            self.assertEqual(response["events"], [value])
            self.assertEqual(client.get("/events", params={"offset": response["offset"]}).json()["events"], [])
            self.assertIn("Reasoning", client.get("/").text)


if __name__ == "__main__":
    unittest.main()
