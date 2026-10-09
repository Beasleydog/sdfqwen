# /// script
# requires-python = ">=3.12"
# dependencies = ["httpx>=0.28,<1", "paramiko>=3.5,<5", "python-dotenv>=1,<2"]
# ///
"""Provision a suitable Prime GPU, run Inspect, retrieve results, terminate."""
import argparse
from datetime import datetime, timezone
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
import math
from pathlib import Path
import select
import shlex
import signal
import socketserver
import sys
import tarfile
import threading
import time
import uuid
import webbrowser
import os

import httpx
import paramiko
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
FILES = ["initialexperiment.py", "sam_core.py", "sam_task.py", "live_terminal.py",
         "live_web.py", "requirements.txt", "sam_experiment.py", "training-requirements.txt"]
SUPPORTED_GPUS = {"A100_40GB", "A100_80GB", "A40_48GB", "A6000_48GB", "L40_48GB",
                  "L40S_48GB", "RTX6000Ada_48GB", "RTX_PRO_6000B_96GB", "H100_80GB", "H200_141GB"}
# VM providers supported by the SSH provisioning workflow.
VM_PROVIDERS = {"massedcompute", "hyperstack", "lambdalabs", "datacrunch", "nebius",
                "latitude", "crusoecloud", "fluidstack", "oblivus"}


class Prime:
    def __init__(self, key):
        self.http = httpx.Client(base_url="https://api.primeintellect.ai/api/v1/",
                                 headers={"Authorization": f"Bearer {key}"}, timeout=45)

    def call(self, method, path, **kwargs):
        response = self.http.request(method, path, **kwargs)
        response.raise_for_status()
        return response.json() if response.content else {}

    def offers(self):
        items, page = [], 1
        while True:
            data = self.call("GET", "availability/gpus", params={"gpu_count": 1, "page_size": 100, "page": page})
            items.extend(data["items"])
            if len(items) >= data["totalCount"]:
                return items
            if not data["items"]:
                raise RuntimeError("Availability pagination ended unexpectedly.")
            page += 1

    def delete(self, path):
        for attempt in range(3):
            try:
                self.call("DELETE", path)
                if path.startswith("pods/"):
                    deadline = time.monotonic()+120
                    while time.monotonic() < deadline:
                        try:
                            record = self.call("GET", path)
                        except httpx.HTTPStatusError as exc:
                            if exc.response.status_code == 404:
                                return
                            raise
                        if record["status"] == "TERMINATED":
                            return
                        time.sleep(2)
                    raise TimeoutError("Pod deletion was accepted but termination could not be verified.")
                return
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 404:
                    return
                if attempt == 2:
                    raise
            except httpx.HTTPError:
                if attempt == 2:
                    raise
            time.sleep(2 ** attempt)

    def find_named(self, path, name):
        offset = 0
        while True:
            page = self.call("GET", path, params={"offset": offset, "limit": 100})
            for item in page["data"]:
                if item["name"] == name:
                    return item
            offset += len(page["data"])
            if offset >= page["total_count"] or not page["data"]:
                return None

    def create(self, path, body, name):
        # Do not blindly retry a paid POST if the response was lost.
        try:
            return self.call("POST", path, json=body)
        except httpx.HTTPError:
            for _ in range(3):
                existing = self.find_named(path, name)
                if existing:
                    return existing
                time.sleep(2)
            raise


def select_offer(offers, max_price=2, gpu=None):
    candidates = []
    for offer in offers:
        prices = offer.get("prices", {})
        if (offer.get("gpuType") not in SUPPORTED_GPUS or offer.get("provider") not in VM_PROVIDERS
                or offer.get("gpuCount") != 1 or offer.get("gpuMemory", 0) < 40
                or offer.get("stockStatus") not in ("Available", "Low")
                or offer.get("isSpot") or offer.get("prepaidTime") or prices.get("isVariable")
                or prices.get("currency") != "USD" or "ubuntu_22_cuda_12" not in offer.get("images", [])
                or (gpu and offer["gpuType"] != gpu)):
            continue
        cost = prices.get("onDemand")
        if not isinstance(cost, (int, float)) or not math.isfinite(cost) or cost <= 0:
            continue
        resources = {}
        usable = True
        for name, field, minimum in (("disk", "diskSize", 100), ("vcpu", "vcpus", 4), ("memory", "memory", 32)):
            spec = offer.get(name) or {}
            count = spec.get("defaultCount") or 0
            if count < minimum:
                if not spec.get("maxCount") or spec["maxCount"] < minimum:
                    usable = False
                    break
                step = spec.get("step") or 1
                base = spec.get("minCount") or 0
                count = base + math.ceil((minimum-base)/step)*step
                if count > spec["maxCount"]:
                    usable = False
                    break
                resources[field] = int(count)
            if spec.get("pricePerUnit") and (not spec.get("defaultIncludedInPrice") or field in resources):
                cost += count * spec["pricePerUnit"]
        if usable and cost <= max_price:
            candidates.append((cost, offer, resources))
    if not candidates:
        raise RuntimeError("No suitable non-prepaid VM GPU offer within the price cap. Nothing provisioned.")
    return min(candidates, key=lambda item: item[0])


def pod_request(offer, resources, key_id, name):
    pod = {"name": name, "cloudId": offer["cloudId"], "gpuType": offer["gpuType"],
           "gpuCount": 1, "socket": offer["socket"], "image": "ubuntu_22_cuda_12",
           "sshKeyId": key_id, "autoRestart": False, **resources}
    for src, dst in (("dataCenter", "dataCenterId"), ("country", "country"), ("security", "security")):
        if offer.get(src):
            pod[dst] = offer[src]
    return {"pod": pod, "provider": {"type": offer["provider"]}}


def connect(api, pod_id, key, deadline):
    # First-seen host key is pinned in this client's memory; no known_hosts file.
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    while time.monotonic() < deadline:
        data = api.call("GET", "pods/status", params={"pod_ids": pod_id})["data"][0]
        if data["status"] in ("ERROR", "TERMINATED", "DELETING") or data.get("installationFailure"):
            raise RuntimeError(f"Pod provisioning failed: {data.get('installationFailure') or data['status']}")
        connection = data.get("sshConnection")
        if isinstance(connection, list):
            connection = next((value for value in connection if value), None)
        installed = data.get("installationProgress") in (None, 100)
        if connection and data["status"] == "ACTIVE" and installed:
            words = shlex.split(connection)
            destination = next(word for word in words if "@" in word)
            user, host = destination.rsplit("@", 1)
            port = int(words[words.index("-p")+1]) if "-p" in words else 22
            try:
                client.connect(host, port=port, username=user, pkey=key, timeout=15,
                               banner_timeout=15, auth_timeout=15, allow_agent=False, look_for_keys=False)
                return client
            except (OSError, paramiko.SSHException):
                pass
        print(f"Provisioning: {data['status']} · {data.get('installationProgress')}%", flush=True)
        time.sleep(10)
    client.close()
    raise TimeoutError("GPU instance was not reachable within 15 minutes.")


def command(ssh, text, timeout=60, echo=True, progress=None):
    channel = ssh.get_transport().open_session(timeout=15)
    channel.exec_command("export TERM=dumb; "+text)
    started = time.monotonic()
    chunks, deadline, next_notice = [], started+timeout, started
    try:
        while True:
            for ready, receive in ((channel.recv_ready, channel.recv), (channel.recv_stderr_ready, channel.recv_stderr)):
                if ready():
                    part = receive(65536).decode("utf-8", errors="replace")
                    chunks.append(part)
                    if echo:
                        print(part, end="", flush=True)
            if channel.exit_status_ready() and not channel.recv_ready() and not channel.recv_stderr_ready():
                code = channel.recv_exit_status()
                result = "".join(chunks)
                if code:
                    raise RuntimeError(f"Remote command failed ({code}): {result[-4000:]}")
                return result
            if time.monotonic() > deadline:
                raise TimeoutError("Remote command timed out.")
            if progress and time.monotonic() >= next_notice:
                print(f"{progress} · {int(time.monotonic()-started)}s", flush=True)
                next_notice = time.monotonic()+20
            time.sleep(0.05)
    finally:
        channel.close()


class Forward(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, transport, port):
        class Handler(socketserver.BaseRequestHandler):
            def handle(self):
                try:
                    channel = transport.open_channel("direct-tcpip", ("127.0.0.1", port), self.request.getpeername())
                except (OSError, paramiko.SSHException):
                    self.request.sendall(b"HTTP/1.1 503 Service Unavailable\r\nConnection: close\r\n\r\nViewer is starting or has stopped. Refresh in a moment.")
                    return
                try:
                    while True:
                        readable, _, _ = select.select([self.request, channel], [], [], 1)
                        for source in readable:
                            data = source.recv(65536)
                            if not data:
                                return
                            (channel if source is self.request else self.request).sendall(data)
                finally:
                    channel.close()
        super().__init__(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.serve_forever, daemon=True)
        self.thread.start()

    def close(self):
        self.shutdown()
        self.server_close()


SETUP = """set -eu
export DEBIAN_FRONTEND=noninteractive
nvidia-smi
updated=false
for attempt in $(seq 1 30); do
  if sudo apt-get -o DPkg::Lock::Timeout=300 update -qq; then updated=true; break; fi
  sleep 10
done
$updated
sudo apt-get -o DPkg::Lock::Timeout=300 install -y python3-venv python3-pip
python3 -m venv /opt/sam/bootstrap
/opt/sam/bootstrap/bin/python -m pip install --disable-pip-version-check uv
/opt/sam/bootstrap/bin/python -m uv venv --python 3.12 /opt/sam/.venv
/opt/sam/bootstrap/bin/python -m uv pip install --python /opt/sam/.venv/bin/python -r /opt/sam/requirements.txt
/opt/sam/bootstrap/bin/python -m uv venv --python 3.12 /opt/sam/inference
/opt/sam/bootstrap/bin/python -m uv pip install --python /opt/sam/inference/bin/python 'vllm==0.19.1' 'transformers>=5.5.1,<6'
cd /opt/sam
sudo -H .venv/bin/python -c 'from inspect_ai.model import get_model; get_model("vllm/Qwen/Qwen3-8B", base_url="http://127.0.0.1:1/v1", stream=True); print("Inference client dependencies OK")'
sudo -H .venv/bin/python sam_experiment.py --check-tools --no-ui --output /opt/sam/check
"""


CONTROL_SETUP = SETUP.split("/opt/sam/bootstrap/bin/python -m uv pip install", 1)[0] + (
    "/opt/sam/bootstrap/bin/python -m uv pip install --python /opt/sam/.venv/bin/python "
    "-r /opt/sam/training-requirements.txt\n")


def upload(ssh, control=False):
    command(ssh, "sudo mkdir -p /opt/sam && sudo chown $(id -u):$(id -g) /opt/sam", echo=False)
    with ssh.open_sftp() as sftp:
        for name in FILES:
            sftp.put(str(ROOT/name), "/opt/sam/"+name)
        sftp.mkdir("/opt/sam/synthetic_documents")
        for path in sorted((ROOT / "synthetic_documents").glob("*.md")):
            sftp.put(str(path), "/opt/sam/synthetic_documents/"+path.name)
        with sftp.file("/opt/sam/setup.sh", "w") as stream:
            stream.write(CONTROL_SETUP if control else SETUP)


def download(ssh, destination):
    try:
        command(ssh, "cd /opt/sam; if test -d results/inspect && find results/inspect -maxdepth 1 -name '*.eval' | grep -q .; then sudo -H .venv/bin/python -m inspect_ai view bundle --log-dir results/inspect --output-dir results/bundle; fi", timeout=120, echo=False)
    except Exception as exc:
        print(f"Viewer bundle unavailable; retrieving raw logs: {exc}", file=sys.stderr)
    command(ssh, "cd /opt/sam && tar -czf /tmp/sam-results.tar.gz results setup.log run.log viewer.log inspect-view.log 2>/dev/null || test -s /tmp/sam-results.tar.gz", echo=False)
    archive = destination/"remote-results.tar.gz"
    with ssh.open_sftp() as sftp:
        sftp.get("/tmp/sam-results.tar.gz", str(archive))
    with tarfile.open(archive) as tar:
        tar.extractall(destination, filter="data")


def run(args):
    load_dotenv(ROOT/".env", override=False)
    key_value = os.environ.get("PRIME_API_KEY")
    if not key_value:
        raise RuntimeError("Set PRIME_API_KEY in the environment or .env.")
    api = Prime(key_value)
    if args.stop:
        state = json.loads(args.stop.read_text())
        for field, path in (("pod_id", "pods/"), ("key_id", "ssh_keys/")):
            resource_id = state.get(field)
            if not resource_id and state.get("name"):
                found = api.find_named(path, state["name"])
                resource_id = found["id"] if found else None
            if resource_id:
                api.delete(path+resource_id)
                state[field+"_deleted"] = True
        state.update(status="stopped", cleanup_errors=[], finished_utc=datetime.now(timezone.utc).isoformat())
        args.stop.write_text(json.dumps(state, indent=2))
        print("Pod and temporary public key deleted.")
        api.http.close()
        return
    cost, offer, resources = select_offer(api.offers(), args.max_hourly_price, args.gpu)
    print(f"Selected {offer['gpuType']} · {offer['provider']} · estimated ${cost:.4f}/hour", flush=True)
    if args.plan:
        print("Plan only; no instance or key created.")
        api.http.close()
        return
    destination = ROOT/"results"/datetime.now(timezone.utc).strftime("prime_%Y%m%d_%H%M%S")
    destination.mkdir(parents=True)
    name = "sam-"+uuid.uuid4().hex[:12]
    state = {"name": name, "pod_id": None, "key_id": None, "gpu": offer["gpuType"], "hourly_estimate": cost,
             "status": "starting", "started_utc": datetime.now(timezone.utc).isoformat()}
    state_file = destination/"remote.json"
    def save():
        state_file.write_text(json.dumps(state, indent=2))
    save()
    key = paramiko.RSAKey.generate(3072)  # private material stays exclusively in RAM
    ssh, forwards = None, []
    try:
        state["key_creation_requested"] = True
        save()
        state["key_id"] = api.create("ssh_keys/", {"name": name, "publicKey": f"{key.get_name()} {key.get_base64()}"}, name)["id"]
        save()
        state["pod_creation_requested"] = True
        save()
        state["pod_id"] = api.create("pods/", pod_request(offer, resources, state["key_id"], name), name)["id"]
        save()
        print(f"Pod {state['pod_id']} · recovery state: {state_file}", flush=True)
        ssh = connect(api, state["pod_id"], key, time.monotonic()+900)
        upload(ssh, control=args.experiment == "control")
        state["status"] = "installing"
        save()
        command(ssh, "bash -lc "+shlex.quote("cd /opt/sam && bash setup.sh 2>&1 | tee setup.log; exit ${PIPESTATUS[0]}"),
                timeout=1800, echo=False, progress="Installing dependencies / checking experiment tools")
        if args.experiment == "control":
            cmd = ["sudo", "-H", ".venv/bin/python", "-u", "initialexperiment.py",
                   "--samples", str(args.samples), "--max-new-tokens", str(args.max_new_tokens),
                   "--epochs", str(args.epochs), "--output", "/opt/sam/results"]
            state["status"] = "running"
            save()
            command(ssh, "bash -lc "+shlex.quote("cd /opt/sam && timeout --signal=INT --kill-after=30s "
                + str(args.max_minutes*60)+" "+shlex.join(cmd)+" 2>&1 | tee run.log; exit ${PIPESTATUS[0]}"),
                timeout=args.max_minutes*60+60, progress="Before/train/after experiment")
            state["status"] = "complete"
            save()
            return destination
        forward_live, forward_inspect = Forward(ssh.get_transport(), 8080), Forward(ssh.get_transport(), 7575)
        forwards.extend([forward_live, forward_inspect])
        live_port, inspect_port = forward_live.server_address[1], forward_inspect.server_address[1]
        cmd = ["sudo", "-H", ".venv/bin/python", "sam_experiment.py", "--no-ui", "--samples", str(args.samples),
               "--server-python", "/opt/sam/inference/bin/python",
               "--parallel", str(args.parallel), "--max-new-tokens", str(args.max_new_tokens),
               "--max-turns", str(args.max_turns), "--output", "/opt/sam/results"]
        cmd.extend(["--versions", *args.versions])
        if args.rollout_seconds:
            cmd.extend(["--rollout-seconds", str(args.rollout_seconds), "--max-reads", "1000000", "--max-submissions", "1000000"])
        command(ssh, "cd /opt/sam; nohup bash -c "+shlex.quote("timeout --signal=INT --kill-after=30s "+str(args.max_minutes*60)+" "+shlex.join(cmd)+" >run.log 2>&1; echo $? >exit-code")+" </dev/null >/dev/null 2>&1 &", echo=False)
        command(ssh, "cd /opt/sam; nohup .venv/bin/python live_web.py /opt/sam/results >viewer.log 2>&1 </dev/null &", echo=False)
        for _ in range(30):
            ready = command(ssh, "test -f /opt/sam/results/config.json && echo ready || echo waiting", echo=False).strip()
            if ready == "ready":
                break
            time.sleep(1)
        else:
            command(ssh, "tail -80 /opt/sam/run.log")
            raise RuntimeError("Experiment did not create its run configuration.")
        command(ssh, "cd /opt/sam; nohup sudo -H .venv/bin/python -m inspect_ai view --host 127.0.0.1 --port 7575 --trusted-host 127.0.0.1:"+str(inspect_port)+" --trusted-origin http://127.0.0.1:"+str(inspect_port)+" --log-dir /opt/sam/results/inspect >inspect-view.log 2>&1 </dev/null &", echo=False)
        url = f"http://127.0.0.1:{live_port}/?inspect_port={inspect_port}"
        with httpx.Client(timeout=3) as probe:
            for port in (live_port, inspect_port):
                for _ in range(30):
                    try:
                        if probe.get(f"http://127.0.0.1:{port}/").status_code == 200:
                            break
                    except httpx.HTTPError:
                        pass
                    time.sleep(1)
                else:
                    command(ssh, "tail -50 /opt/sam/viewer.log /opt/sam/inspect-view.log")
                    raise RuntimeError("Browser viewer failed its readiness check.")
        state.update(status="running", live_url=url, inspect_url=f"http://127.0.0.1:{inspect_port}")
        save()
        print(f"\nLive reasoning: {url}\nInspect viewer: {state['inspect_url']}\nCtrl+C stops and retrieves results, then deletes the instance.\n", flush=True)
        if not args.no_browser:
            webbrowser.open(url)
        deadline = time.monotonic()+args.max_minutes*60+60
        while time.monotonic() < deadline:
            result = command(ssh, "cd /opt/sam; if test -f exit-code; then cat exit-code; else echo running; fi", echo=False).strip()
            if result != "running":
                state["status"] = "complete" if result == "0" else "failed"
                save()
                if result != "0":
                    command(ssh, "tail -80 /opt/sam/run.log", echo=True)
                    raise RuntimeError(f"Experiment exited with code {result}.")
                print("Experiment completed.", flush=True)
                break
            time.sleep(5)
        else:
            raise TimeoutError("Experiment exceeded its runtime limit.")
    except BaseException as exc:
        state.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                     error=f"{type(exc).__name__}: {exc}")
        save()
        raise
    finally:
        if ssh:
            try:
                command(ssh, "sudo pkill -INT -f '^.venv/bin/python (initialexperiment|sam_experiment).py' || true", echo=False)
                time.sleep(2)
                download(ssh, destination)
                print(f"Results saved: {destination}", flush=True)
            except Exception as exc:
                print(f"Result retrieval failed: {exc}", file=sys.stderr)
            for forward in forwards:
                try:
                    forward.close()
                except Exception as exc:
                    print(f"Viewer forwarding close failed: {exc}", file=sys.stderr)
            try:
                ssh.close()
            except Exception as exc:
                print(f"Connection close failed: {exc}", file=sys.stderr)
        errors = []
        for field, path in (("pod_id", "pods/"), ("key_id", "ssh_keys/")):
            try:
                if not state[field] and state.get(field.replace("_id", "_creation_requested")):
                    found = api.find_named(path, name)
                    state[field] = found["id"] if found else None
                if state[field]:
                    api.delete(path+state[field])
                    state[field+"_deleted"] = True
            except Exception as exc:
                errors.append(f"{field}: {exc}")
        state["cleanup_errors"] = errors
        state["finished_utc"] = datetime.now(timezone.utc).isoformat()
        save()
        api.http.close()
        if errors:
            raise RuntimeError(f"Cleanup failed: {errors}. Retry: uv run primeexperiment.py --stop {state_file}")
        print("GPU instance and temporary public key deleted.", flush=True)
    return destination


def view_results(destination, open_browser=True):
    directory = Path(destination)/"results"/"bundle"
    if not (directory/"index.html").exists():
        raise RuntimeError(f"No bundled Inspect viewer found in {directory}")
    with ThreadingHTTPServer(("127.0.0.1", 0), partial(SimpleHTTPRequestHandler, directory=str(directory))) as server:
        url = f"http://127.0.0.1:{server.server_port}"
        print(f"Saved Inspect viewer: {url}\nGPU already deleted. Ctrl+C closes this local viewer.", flush=True)
        if open_browser:
            webbrowser.open(url)
        server.serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=["control", "sam"], default="control")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--samples", type=int, default=2, help="Rollouts per version (default 4 total).")
    parser.add_argument("--versions", nargs="+", choices=["0.3", "0.4"], default=["0.3", "0.4"])
    parser.add_argument("--rollout-seconds", type=float, help="Timed observation per rollout; does not stop on a harness pass.")
    parser.add_argument("--parallel", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--max-turns", type=int, default=24)
    parser.add_argument("--max-minutes", type=int, default=90)
    parser.add_argument("--max-hourly-price", type=float, default=2)
    parser.add_argument("--gpu", help="Optional exact GPU type, e.g. A100_80GB.")
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--no-view", action="store_true", help="Exit after cleanup rather than serving saved transcripts locally.")
    parser.add_argument("--view", type=Path, help="View a saved results/prime_* run; no remote compute is used.")
    parser.add_argument("--stop", type=Path, help="Delete resources from a prior results/prime_*/remote.json.")
    args = parser.parse_args()
    if args.rollout_seconds is not None and args.rollout_seconds <= 0:
        parser.error("Rollout seconds must be positive.")
    if min(args.samples, args.parallel, args.max_new_tokens, args.max_turns, args.max_minutes, args.epochs) < 1 or args.max_hourly_price <= 0:
        parser.error("Limits must be positive.")
    if args.max_new_tokens > 32768:
        parser.error("Per-turn output cannot exceed 32,768 tokens.")
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    logging.getLogger("paramiko.transport").setLevel(logging.CRITICAL)
    try:
        if args.view:
            view_results(args.view, open_browser=not args.no_browser)
        else:
            destination = run(args)
            if destination and args.experiment == "sam" and not args.no_view:
                view_results(destination, open_browser=not args.no_browser)
    except KeyboardInterrupt:
        print("Stopped.", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
