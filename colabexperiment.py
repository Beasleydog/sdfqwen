"""Colab launcher for the shared Qwen document-training experiment."""
from datetime import datetime, timezone
from html import escape
import os
import re
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from types import SimpleNamespace
from urllib.parse import urlencode, urlsplit
from urllib.request import urlopen, urlretrieve

ROOT = Path(__file__).resolve().parent


def prepare(control=True):
    """Isolate incompatible evaluator/vLLM dependencies from the notebook kernel."""
    subprocess.run(["nvidia-smi"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "uv"], check=True)
    interpreters = []
    environments = (("training", ["-r", str(ROOT / "training-requirements.txt")]),) if control else (
        ("eval", ["-r", str(ROOT / "requirements.txt")]),
        ("inference", ["vllm==0.19.1", "transformers>=5.5.1,<6"]))
    for name, packages in environments:
        env = ROOT / ".colab" / name
        python = env / "bin" / "python"
        if not python.exists():
            subprocess.run([sys.executable, "-m", "uv", "venv", "--python", "3.12", str(env)], check=True)
        print(f"Installing {name} dependencies…", flush=True)
        subprocess.run([sys.executable, "-m", "uv", "pip", "install", "--python", str(python), *packages], check=True)
        interpreters.append(str(python))
    return interpreters


def open_viewers(output):
    """Attach temporary Cloudflare links to an existing run; no model restart."""
    output = Path(output)
    evaluator = str(ROOT / ".colab/eval/bin/python")
    binary = ROOT / ".colab/cloudflared"
    if not binary.exists():
        print("Downloading Cloudflare tunnel client?", flush=True)
        urlretrieve("https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64", binary)
        binary.chmod(0o755)
    processes = []

    def spawn(command, log):
        with log.open("w") as stream:
            process = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                       start_new_session=True)
        processes.append(process)
        return process

    def stop():
        for process in processes:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        for process in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()

    def viewer(command, name):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        log = output / f"{name}-tunnel.log"
        tunnel = spawn([str(binary), "tunnel", "--no-autoupdate", "--url", f"http://127.0.0.1:{port}"], log)
        for _ in range(60):
            match = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", log.read_text())
            if match:
                url = match.group(0)
                break
            if tunnel.poll() is not None:
                raise RuntimeError(log.read_text())
            time.sleep(1)
        else:
            raise RuntimeError("Tunnel did not start: " + str(log))
        if name == "inspect":
            command += ["--trusted-host", urlsplit(url).netloc, "--trusted-origin", url]
        process = spawn([evaluator, *command, "--port", str(port)], output / f"{name}-view.log")
        for _ in range(60):
            if process.poll() is not None:
                raise RuntimeError((output / f"{name}-view.log").read_text())
            try:
                with urlopen(f"http://127.0.0.1:{port}", timeout=1) as response:
                    if response.status == 200:
                        return url
            except OSError:
                time.sleep(1)
        raise RuntimeError(f"{name} viewer did not start; see {output}.")

    try:
        inspect_url = viewer(["-m", "inspect_ai", "view", "--host", "127.0.0.1",
                              "--log-dir", str(output / "inspect")], "inspect")
        live_url = viewer(["live_web.py", str(output)], "live")
        live_url += "?" + urlencode({"inspect_url": inspect_url})
        print(f"\nInspect: {inspect_url}\nLive reasoning: {live_url}\n", flush=True)
        try:
            from IPython import get_ipython
            if get_ipython() is not None:
                from IPython.display import HTML, display
                display(HTML(f'<a href="{escape(live_url)}" target="_blank">Live reasoning</a> ? '
                             f'<a href="{escape(inspect_url)}" target="_blank">Inspect transcripts</a>'))
        except ImportError:
            pass
        return SimpleNamespace(inspect_url=inspect_url, live_url=live_url, processes=processes, stop=stop)
    except BaseException:
        stop()
        raise


def launch_sam(*arguments):
    """Default: one unlimited v0.3 rollout; explicit --rollout-seconds enables a timer."""

    evaluator, inference = prepare(control=False)
    output = ROOT / "results" / datetime.now(timezone.utc).strftime("colab_%Y%m%d_%H%M%S_%f")
    output.parent.mkdir(exist_ok=True)
    processes = []

    def spawn(command, log):
        with log.open("w") as stream:
            process = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                       start_new_session=True)
        processes.append(process)
        return process

    def stop():
        # SIGINT lets the evaluator close its separately started inference server.
        for process in processes:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGINT)
        for process in processes:
            try:
                process.wait(timeout=35)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()

    try:
        process = spawn([evaluator, "sam_experiment.py", "--versions", "0.3", "--samples", "1",
                         "--parallel", "1", *([] if "--rollout-seconds" in arguments else ["--unlimited"]), "--max-reads", "1000000",
                         "--max-submissions", "1000000", "--no-ui", "--server-python", inference,
                         *arguments, "--output", str(output)], output.with_suffix(".run.log"))
        for _ in range(90):
            if (output / "config.json").exists():
                break
            if process.poll() is not None:
                raise RuntimeError(output.with_suffix(".run.log").read_text())
            time.sleep(1)
        else:
            raise RuntimeError("Evaluator did not start; see " + str(output.with_suffix(".run.log")))
        views = open_viewers(output)
        processes.extend(views.processes)
        inspect_url, live_url = views.inspect_url, views.live_url
        print(f"Loading Qwen; startup details: {output}/server.log\nResults: {output}\nStop: Ctrl+C in the terminal, or run.stop() in a notebook", flush=True)
        return SimpleNamespace(process=process, output=output, inspect_url=inspect_url,
                               live_url=live_url, stop=stop)
    except BaseException:
        stop()
        raise


def launch(*arguments):
    """Install isolated dependencies and run the shared multiplication experiment."""
    python, = prepare()
    subprocess.run([python, "-u", str(ROOT / "initialexperiment.py"), *arguments], cwd=ROOT, check=True)


def launch_study(*arguments):
    """Run the full matched study using the same isolated training environment."""
    python, = prepare()
    subprocess.run([python, "-u", str(ROOT / "polarity_study.py"), *arguments], cwd=ROOT, check=True)


if __name__ == "__main__":
    if sys.argv[1:2] == ["study"]:
        launch_study(*sys.argv[2:])
    else:
        launch(*sys.argv[1:])
