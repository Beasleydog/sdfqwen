"""Rent -> Jupyter upload -> train -> download -> terminate, all project-local."""
from __future__ import annotations

import argparse
import io
import json
import re
import shlex
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import requests

from prime_gpu import Prime, Jupyter, ROOT, STATE, jupyter_urls, bootstrap_notebook


def bundle():
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
        # Explicit allowlist: never send .env, API keys, .git, or other projects.
        for path in sorted((ROOT / "experiments").glob("*.py")) + sorted((ROOT / "datagen/outputs").glob("*.txt")):
            archive.write(path, path.relative_to(ROOT).as_posix())
    return stream.getvalue()


def connect(prime, state, deadline, allow_bootstrap=False):
    last = "waiting for notebook"
    ready_deadline = min(deadline, time.time() + 900)
    while time.time() < ready_deadline:
        pod = prime.status(state["id"])
        if pod.get("installationFailure") or pod.get("status") in ("ERROR", "TERMINATED"):
            raise RuntimeError(f"Pod failed: {pod.get('installationFailure') or pod.get('status')}")
        if state.get("jupyter_url"):
            try:
                return Jupyter(state["jupyter_url"], state["password"], state.get("cert_path"))
            except Exception as e:
                last = type(e).__name__
        for url in jupyter_urls(pod):
            try:
                return Jupyter(url, state["password"])
            except Exception as e:
                last = type(e).__name__
        if not state.get("jupyter_url") and pod.get("status") == "ACTIVE" and allow_bootstrap and pod.get("sshConnection"):
            print("Starting TLS Jupyter with one SSH bootstrap command", flush=True)
            bootstrap_notebook(pod, state)
            continue
        print(f"Pod {pod.get('status')} install={pod.get('installationProgress')} ({last})", flush=True)
        time.sleep(15)
    raise TimeoutError("Jupyter did not become reachable; no SSH fallback")


def bootstrap(names_steps, seed=42):
    commands = [f"python3 experiments/train_experiment.py --name {shlex.quote(name)} --lr {lr} --steps {steps} --eval-every {max(1, steps//2)} --seed {seed}" for name, lr, steps in names_steps]
    # A Python supervisor collects partial outputs even when setup or training fails.
    return '''import json, subprocess, time, zipfile, traceback
from pathlib import Path
root = Path.cwd()
status = {"started": time.time(), "phase": "setup"}
def save():
    Path("job_status.json").write_text(json.dumps(status))
save()
try:
    subprocess.run(["nvidia-smi"], check=True)
    # CUDA 12.4 wheels support older marketplace drivers; avoid implicitly installing CUDA 13.
    subprocess.run(["python3", "-m", "pip", "install", "torch==2.6.0", "--index-url", "https://download.pytorch.org/whl/cu124"], check=True)
    subprocess.run(["python3", "-m", "pip", "install", "--upgrade", "transformers>=5.2,<6", "peft", "datasets", "accelerate"], check=True)
    subprocess.run(["python3", "-m", "pip", "uninstall", "-y", "torchao", "torchvision", "torchaudio"], check=False)
    for command in COMMANDS:
        status.update(phase="training", command=command)
        save()
        subprocess.run(command, shell=True, check=True)
    status.update(phase="complete")
except BaseException:
    status.update(phase="failed", error=traceback.format_exc())
    traceback.print_exc()
finally:
    status.update(finished=time.time())
    with zipfile.ZipFile("results.tmp.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for p in Path("experiment_results").rglob("*"):
            if p.is_file() and "checkpoints" not in p.parts:
                z.write(p)
        z.writestr("job_status.json", json.dumps(status))
        if Path("job.log").exists(): z.write("job.log")
    Path("results.tmp.zip").replace("results.zip")
    save()
'''.replace("COMMANDS", repr(commands))


def collect(notebook, run_dir, filename="results.zip"):
    payload = notebook.read("sdfqwen/" + filename)
    path = run_dir / filename
    path.write_bytes(payload)
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        # Remote archives cannot escape this run folder.
        for name in archive.namelist():
            target = (run_dir / name).resolve()
            if not target.is_relative_to(run_dir.resolve()):
                raise RuntimeError("Unsafe result archive path")
        archive.extractall(run_dir)
    print(f"Downloaded {len(payload):,} bytes to {run_dir}", flush=True)


def collect_partial(notebook, run_dir):
    code = '''from pathlib import Path
import zipfile
root = Path('sdfqwen')
with zipfile.ZipFile(root/'results_partial.tmp.zip', 'w', zipfile.ZIP_DEFLATED) as z:
    for p in (root/'experiment_results').rglob('*'):
        if p.is_file(): z.write(p, p.relative_to(root))
    for name in ['job.log', 'job_status.json']:
        p=root/name
        if p.exists(): z.write(p, name)
(root/'results_partial.tmp.zip').replace(root/'results_partial.zip')
'''
    notebook.launch("nohup python3 -c " + shlex.quote(code) + " > /dev/null 2>&1 < /dev/null &")
    for attempt in range(8):
        try:
            collect(notebook, run_dir, "results_partial.zip")
            return
        except (requests.HTTPError, zipfile.BadZipFile):
            time.sleep(3)
    raise RuntimeError("Partial result snapshot did not finish")


def watchdog(pod_id, deadline):
    # Survives an interrupted parent. No global settings, SSH keys, or services.
    state_file = STATE / (pod_id + ".json")
    while time.time() < deadline:
        try:
            if json.loads(state_file.read_text()).get("terminated"):
                return
        except (OSError, json.JSONDecodeError):
            pass
        time.sleep(min(30, max(0, deadline-time.time())))
    prime = Prime()
    for attempt in range(10):
        try:
            state = json.loads(state_file.read_text())
            if state.get("terminated"):
                return
            prime.terminate(pod_id)
            state["terminated"] = time.time()
            state_file.write_text(json.dumps(state, indent=2))
            return
        except Exception:
            time.sleep(15)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--budget", type=float, default=2.0, help="Compute-only dollar cap; conservative wall-clock estimate")
    parser.add_argument("--max-hourly", type=float, default=0.65)
    parser.add_argument("--no-ssh", action="store_true", help="Require provider-started Jupyter; bare Ubuntu offers will fail")
    parser.add_argument("--experiment", action="append", nargs=3, metavar=("NAME", "LR", "STEPS"), help="Repeat for a custom comparison; default is 5e-5 vs 2e-4")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--watchdog", nargs=2, metavar=("POD", "DEADLINE"))
    parser.add_argument("--collect", metavar="POD", help="Download a completed run from an owned pod, then terminate it")
    args = parser.parse_args()
    if args.watchdog:
        watchdog(args.watchdog[0], float(args.watchdog[1]))
        return
    if args.steps < 2 or args.budget <= 0 or args.max_hourly <= 0:
        parser.error("steps >= 2 and positive budgets required")
    experiments = [("sdf_low_lr", 5e-5, args.steps), ("sdf_original_lr", 2e-4, args.steps)]
    if args.experiment:
        experiments = []
        for name, lr, steps in args.experiment:
            if not re.fullmatch(r"[A-Za-z0-9_-]+", name) or not 0 < float(lr) <= 0.01 or int(steps) < 2:
                parser.error("Experiment requires a simple name, 0 < LR <= 0.01, and steps >= 2")
            experiments.append((name, float(lr), int(steps)))
        if len({x[0] for x in experiments}) != len(experiments):
            parser.error("Experiment names must be unique")
    prime = Prime()
    run_dir = ROOT / "runs" / time.strftime("prime_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True)
    state = None
    notebook = None
    try:
        if args.collect:
            state = json.loads((STATE / (args.collect + ".json")).read_text())
            notebook = connect(prime, state, time.time() + 120)
            collect(notebook, run_dir)
            return
        offers = prime.offers(args.max_hourly)
        for attempt in range(4):
            if offers:
                break
            print("Waiting for an inexpensive GPU offer (nothing rented yet)", flush=True)
            time.sleep(15)
            offers = prime.offers(args.max_hourly)
        if not offers:
            raise RuntimeError("No in-stock GPU below hourly limit; nothing rented")
        key_id = None if args.no_ssh else prime.bootstrap_key()
        state = prime.create(offers[0], "sdfqwen-" + run_dir.name, ssh_key_id=key_id)
        state_file = STATE / (state["id"] + ".json")
        # Reserve 2 minutes for downloading + termination.
        deadline = state["created"] + args.budget / state["hourly"] * 3600 - 120
        if deadline <= time.time():
            raise RuntimeError("Budget too small for startup")
        state.update(deadline=deadline, run_dir=str(run_dir))
        state_file.write_text(json.dumps(state, indent=2))
        (run_dir / "rental.json").write_text(json.dumps({k: v for k, v in state.items() if k != "password"}, indent=2))
        subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--watchdog", state["id"], str(deadline+90)],
                         cwd=ROOT, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
        print(f"Rented {state['id']} at ${state['hourly']}/hour", flush=True)
        notebook = connect(prime, state, deadline, allow_bootstrap=not args.no_ssh)
        notebook.upload("sdfqwen_upload.zip", bundle())
        job = bootstrap(experiments, args.seed)
        (run_dir / "experiment_plan.json").write_text(json.dumps({"experiments": experiments, "seed": args.seed}, indent=2))
        # Install a project directory through the notebook's existing Python runtime.
        notebook.upload("sdfqwen_job.py", job.encode())
        command = "python3 -c " + shlex.quote("import zipfile,pathlib; p=pathlib.Path('sdfqwen'); p.mkdir(exist_ok=True); zipfile.ZipFile('sdfqwen_upload.zip').extractall(p); p.joinpath('job.py').write_bytes(pathlib.Path('sdfqwen_job.py').read_bytes())")
        command += " && cd sdfqwen && nohup python3 -u job.py > job.log 2>&1 < /dev/null &"
        notebook.launch(command)
        previous = None
        while time.time() < deadline:
            try:
                status = json.loads(notebook.read("sdfqwen/job_status.json"))
                log = notebook.read("sdfqwen/job.log").decode(errors="replace")
                (run_dir / "job.log").write_text(log, encoding="utf-8")
                (run_dir / "job_status.json").write_text(json.dumps(status, indent=2))
                tail = log[-1800:]
                if tail != previous:
                    print(tail, flush=True)
                    previous = tail
                if status["phase"] in ("complete", "failed"):
                    # Allow archive close after the status write.
                    time.sleep(3)
                    collect(notebook, run_dir)
                    if status["phase"] == "failed":
                        raise RuntimeError("Remote training failed; details saved in job.log")
                    subprocess.run([sys.executable, str(ROOT / "analyze_results.py"), str(run_dir)], check=True)
                    return
            except (json.JSONDecodeError, requests.HTTPError):
                pass
            time.sleep(20)
        raise TimeoutError("Compute budget deadline reached")
    finally:
        if state:
            if notebook and not (run_dir / "results.zip").exists():
                try:
                    collect_partial(notebook, run_dir)
                except Exception as error:
                    print(f"Partial result recovery failed: {type(error).__name__}", file=sys.stderr)
            try:
                prime.terminate(state["id"])
                state["terminated"] = time.time()
                state["estimated_compute_usd"] = (state["terminated"]-state["created"])/3600*state["hourly"]
                (STATE / (state["id"] + ".json")).write_text(json.dumps(state, indent=2))
                (run_dir / "rental.json").write_text(json.dumps({k:v for k,v in state.items() if k != "password"}, indent=2))
                print(f"Terminated pod; estimated compute ${state['estimated_compute_usd']:.3f}", flush=True)
            except Exception as error:
                print(f"Termination failed: {error}. Watchdog will retry; pod {state['id']}", file=sys.stderr)


if __name__ == "__main__":
    # Remote progress bars and model outputs contain Unicode; Windows defaults to cp1252.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    main()
