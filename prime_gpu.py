"""Project-local Prime REST + authenticated Jupyter transport (no SSH/WSL).

Offer/pod lifecycle adapted from remotiontesting/soprano_tts/prime_generate.py.
Credentials stay in .env; only training code and documents are uploaded.
"""
from __future__ import annotations

import base64
import json
import secrets
import shlex
import subprocess
import time
from pathlib import Path
from urllib.parse import quote, urlparse

import requests
import websocket
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parent
STATE = ROOT / ".prime"
API = "https://api.primeintellect.ai/api/v1"


class Prime:
    def __init__(self):
        self.session = requests.Session()
        key = dotenv_values(ROOT / ".env").get("PRIME_API_KEY")
        if not key:
            raise RuntimeError("Missing project .env PRIME_API_KEY")
        self.session.headers["Authorization"] = "Bearer " + key

    def request(self, method, path, **kwargs):
        r = self.session.request(method, API + path, timeout=40, **kwargs)
        if not r.ok:
            raise RuntimeError(f"Prime {method} {path}: HTTP {r.status_code}: {r.text[:500]}")
        return r.json() if r.content else {}

    def offers(self, max_hourly=0.65):
        offers = []
        eligible = {"RTX4090_24GB", "RTX3090_24GB", "A6000_48GB", "A40_48GB", "L40S_48GB", "L4_24GB", "A10_24GB", "A5000_24GB"}
        page = 1
        while True:
            data = self.request("GET", "/availability/gpus", params={"gpu_count": 1, "page_size": 100, "page": page})
            offers.extend(data["items"])
            if page * 100 >= data.get("totalCount", len(data["items"])):
                break
            page += 1
        return sorted([o for o in offers if o.get("stockStatus") == "Available"
                       and o.get("gpuType") in eligible
                       and "ubuntu_22_cuda_12" in o.get("images", [])
                       and 0 < (o.get("prices", {}).get("onDemand") or 0) <= max_hourly
                       and not o.get("prepaidTime")], key=lambda o: o["prices"]["onDemand"])

    def bootstrap_key(self):
        STATE.mkdir(exist_ok=True)
        key = STATE / "bootstrap_key"
        if not key.exists():
            subprocess.run(["ssh-keygen", "-t", "ed25519", "-N", "", "-f", str(key), "-C", "sdfqwen-bootstrap"],
                           check=True, capture_output=True, timeout=30)
        public = key.with_suffix(".pub").read_text().strip()
        keys = self.request("GET", "/ssh_keys/")
        keys = keys if isinstance(keys, list) else keys.get("data", keys.get("items", []))
        for item in keys:
            if item.get("publicKey") == public:
                return item["id"]
        return self.request("POST", "/ssh_keys/", json={"name": "sdfqwen-bootstrap", "publicKey": public})["id"]

    def create(self, offer, name, ssh_key_id=None):
        STATE.mkdir(exist_ok=True)
        password = secrets.token_urlsafe(32)
        pod = {"name": name, "cloudId": offer["cloudId"], "gpuType": offer["gpuType"],
               "socket": offer["socket"], "gpuCount": 1, "image": "ubuntu_22_cuda_12",
               "jupyterPassword": password, "maxPrice": offer["prices"]["onDemand"],
               "autoRestart": False, "country": offer.get("country"), "security": offer["security"]}
        if offer.get("dataCenter"):
            pod["dataCenterId"] = offer["dataCenter"]
        if ssh_key_id:
            pod["sshKeyId"] = ssh_key_id
        result = self.request("POST", "/pods/", json={"pod": {k: v for k, v in pod.items() if v is not None}, "provider": {"type": offer["provider"]}})
        state = {"id": result["id"], "password": password, "created": time.time(),
                 "hourly": result["priceHr"], "offer": offer}
        (STATE / (result["id"] + ".json")).write_text(json.dumps(state, indent=2))
        return state

    def status(self, pod_id):
        data = self.request("GET", "/pods/status", params={"pod_ids": pod_id})
        return data["data"][0]

    def terminate(self, pod_id):
        return self.request("DELETE", "/pods/" + pod_id)


def jupyter_urls(pod):
    ip = pod.get("ip")
    if isinstance(ip, list):
        ip = ip[0]
    for mapping in pod.get("primePortMapping") or []:
        if mapping.get("usedBy") == "JUPYTER_NOTEBOOK" or mapping.get("internal") == "8888":
            external = mapping["external"]
            if str(external).startswith("http"):
                yield str(external).rstrip("/")
            elif ip:
                yield f"http://{ip}:{external}"


def bootstrap_notebook(pod, state):
    """Bare images need one SSH bootstrap. Every subsequent operation uses TLS Jupyter."""
    connection = pod["sshConnection"]
    if isinstance(connection, list):
        connection = connection[0]
    parts = shlex.split(connection)
    destination = parts[0]
    port = parts[parts.index("-p")+1] if "-p" in parts else "22"
    ip = pod["ip"]
    if isinstance(ip, list):
        ip = ip[0]
    script = f'''set -eu
# A user-owned runtime avoids image boot-time apt locks and system Python changes.
export XDG_CACHE_HOME="$HOME/sdfqwen_tools/cache"
export XDG_CONFIG_HOME="$HOME/sdfqwen_tools/config"
export UV_PYTHON_INSTALL_DIR="$HOME/sdfqwen_tools/python"
export HF_HOME="$HOME/sdfqwen_notebook/hf_cache"
mkdir -p "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME"
curl -LsSf https://astral.sh/uv/install.sh | env UV_UNMANAGED_INSTALL="$HOME/sdfqwen_tools" sh
"$HOME/sdfqwen_tools/uv" venv --seed --python 3.11 "$HOME/sdfqwen_runtime"
"$HOME/sdfqwen_tools/uv" pip install --python "$HOME/sdfqwen_runtime/bin/python" -q jupyterlab
mkdir -p "$HOME/sdfqwen_notebook"
cd "$HOME/sdfqwen_notebook"
openssl req -x509 -newkey rsa:2048 -nodes -days 2 -keyout key.pem -out cert.pem -subj {shlex.quote('/CN='+ip)} -addext {shlex.quote('subjectAltName=IP:'+ip)} 2>/dev/null
cat > jupyter_config.py <<'CONFIG'
c = get_config()
c.ServerApp.ip = '0.0.0.0'
c.ServerApp.port = 8888
c.ServerApp.port_retries = 0
c.ServerApp.open_browser = False
c.ServerApp.allow_remote_access = True
c.ServerApp.allow_root = True
c.ServerApp.certfile = 'cert.pem'
c.ServerApp.keyfile = 'key.pem'
c.IdentityProvider.token = {state['password']!r}
CONFIG
chmod 600 key.pem jupyter_config.py
export PATH="$HOME/sdfqwen_runtime/bin:$PATH"
nohup "$HOME/sdfqwen_runtime/bin/jupyter" lab --config=jupyter_config.py > jupyter.log 2>&1 < /dev/null &
sleep 3
echo CERT_BEGIN
cat cert.pem
echo CERT_END
'''
    command = ["ssh", "-F", "none", "-i", str(STATE / "bootstrap_key"), "-p", port,
               "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new",
               "-o", "UserKnownHostsFile=" + str(STATE / ("known_hosts_" + state["id"])),
               "-o", "ConnectTimeout=15", destination, "bash", "-s"]
    # Send bytes: Windows text-mode pipes translate LF to CRLF and break Bash.
    completed = subprocess.run(command, input=script.encode(), capture_output=True, timeout=600)
    stdout = completed.stdout.decode(errors="replace")
    if completed.returncode or "CERT_BEGIN" not in stdout:
        raise RuntimeError("Jupyter bootstrap failed: " + completed.stderr.decode(errors="replace")[-1500:])
    certificate = stdout.split("CERT_BEGIN\n",1)[1].split("CERT_END",1)[0]
    cert_path = STATE / (state["id"] + "_cert.pem")
    cert_path.write_text(certificate)
    state["jupyter_url"] = f"https://{ip}:8888"
    state["cert_path"] = str(cert_path)
    (STATE / (state["id"] + ".json")).write_text(json.dumps(state, indent=2))


class Jupyter:
    def __init__(self, url, password, cert_path=None):
        self.url = url.rstrip("/")
        self.session = requests.Session()
        self.cert_path = cert_path
        if cert_path:
            self.session.verify = cert_path
        # Images can configure either password auth or token auth.
        r = self.session.get(self.url + "/api/contents", headers={"Authorization": "token " + password}, timeout=10)
        if r.status_code == 200:
            self.session.headers["Authorization"] = "token " + password
        else:
            self.session.get(self.url + "/login", timeout=10)
            xsrf = self.session.cookies.get("_xsrf")
            self.session.post(self.url + "/login", data={"password": password, "_xsrf": xsrf}, timeout=10)
        xsrf = self.session.cookies.get("_xsrf")
        if xsrf:
            self.session.headers["X-XSRFToken"] = xsrf
        self.request("GET", "/api/contents", params={"content": 0})

    def request(self, method, path, **kwargs):
        r = self.session.request(method, self.url + path, timeout=60, **kwargs)
        r.raise_for_status()
        return r.json() if r.content else {}

    def upload(self, path, content: bytes):
        return self.request("PUT", "/api/contents/" + quote(path, safe="/"), json={"type": "file", "format": "base64", "content": base64.b64encode(content).decode()})

    def read(self, path):
        d = self.request("GET", "/api/contents/" + quote(path, safe="/"), params={"format": "base64"})
        return base64.b64decode(d["content"])

    def launch(self, shell_command):
        terminal = self.request("POST", "/api/terminals", json={})["name"]
        ws_url = self.url.replace("https://", "wss://").replace("http://", "ws://") + "/terminals/websocket/" + terminal
        cookie = "; ".join(f"{c.name}={c.value}" for c in self.session.cookies)
        headers = [f"Authorization: {self.session.headers['Authorization']}"] if "Authorization" in self.session.headers else []
        sslopt = {"ca_certs": self.cert_path} if self.cert_path else {}
        ws = websocket.create_connection(ws_url, cookie=cookie, header=headers, origin=self.url, timeout=15, sslopt=sslopt)
        try:
            ws.send(json.dumps(["stdin", shell_command + "\n"]))
            # Keep the terminal alive long enough for the shell to launch the detached job.
            time.sleep(2)
        finally:
            ws.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["offers", "status", "terminate"])
    parser.add_argument("pod_id", nargs="?")
    args = parser.parse_args()
    prime = Prime()
    if args.action == "offers":
        print(json.dumps(prime.offers(), indent=2))
    elif args.action == "status":
        print(json.dumps(prime.status(args.pod_id), indent=2))
    else:
        print(prime.terminate(args.pod_id))
