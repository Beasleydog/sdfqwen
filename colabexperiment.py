"""Colab launcher: local Qwen inference, shared Inspect task, authenticated viewers."""
from datetime import datetime, timezone
from html import escape
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from types import SimpleNamespace
from urllib.parse import urlencode, urlsplit
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent


def prepare():
    """Isolate incompatible evaluator/vLLM dependencies from the notebook kernel."""
    subprocess.run(["nvidia-smi"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "uv"], check=True)
    interpreters = []
    for name, packages in (("eval", ["-r", str(ROOT / "requirements.txt")]),
                           ("inference", ["vllm==0.18.0", "transformers>=4.56,<5"])):
        env = ROOT / ".colab" / name
        python = env / "bin" / "python"
        if not python.exists():
            subprocess.run([sys.executable, "-m", "uv", "venv", "--python", "3.12", str(env)], check=True)
        print(f"Installing {name} dependencies…", flush=True)
        subprocess.run([sys.executable, "-m", "uv", "pip", "install", "--python", str(python), *packages], check=True)
        interpreters.append(str(python))
    return interpreters


def launch(*arguments):
    """Run in a notebook cell. Default: one v0.3 rollout for 20 minutes."""
    from google.colab.output import eval_js
    from IPython.display import HTML, display

    evaluator, inference = prepare()
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

    def viewer(command, name):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        url = eval_js(f"google.colab.kernel.proxyPort({port})")
        if name == "inspect":
            parsed = urlsplit(url)
            command += ["--trusted-host", parsed.netloc,
                        "--trusted-origin", f"{parsed.scheme}://{parsed.netloc}"]
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
        process = spawn([evaluator, "initialexperiment.py", "--versions", "0.3", "--samples", "1",
                         "--parallel", "1", "--rollout-seconds", "1200", "--max-reads", "1000000",
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
        inspect_url = viewer(["-m", "inspect_ai", "view", "--host", "127.0.0.1",
                              "--log-dir", str(output / "inspect")], "inspect")
        live_url = viewer(["live_web.py", str(output)], "live")
        live_url += "?" + urlencode({"inspect_url": inspect_url})
        display(HTML(f'<a href="{escape(live_url)}" target="_blank">Live reasoning</a> · '
                     f'<a href="{escape(inspect_url)}" target="_blank">Inspect transcripts</a>'))
        print(f"Loading Qwen; startup details: {output}/server.log\nResults: {output}\nStop early: run.stop()", flush=True)
        return SimpleNamespace(process=process, output=output, inspect_url=inspect_url,
                               live_url=live_url, stop=stop)
    except BaseException:
        stop()
        raise


if __name__ == "__main__":
    raise SystemExit("Use a Colab Python cell: from colabexperiment import launch; run = launch()")
