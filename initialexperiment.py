"""Inspect SAMBench agent pilot: Docker sandbox + Qwen3-8B/vLLM."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from importlib.metadata import PackageNotFoundError, version as package_version
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time

from sam_core import MODEL, SENTINEL, info_text, messages_for, summarize


def check_sandbox(kind):
    if not shutil.which("docker"):
        raise RuntimeError(
            "Docker CLI is unavailable. A Docker-capable GPU VM is required. "
            "Install Docker Engine and Compose, then run --check-sandbox to verify real containers. "
            "Model-generated commands run only inside the Docker sandbox.")
    for command in (["docker", "info"], ["docker", "compose", "version"]):
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
        if result.returncode:
            raise RuntimeError(f"{' '.join(command)} failed:\n{result.stderr[-2000:]}\n"
                "A working Docker daemon and Compose plugin are required.")


def server_command(args, port):
    command = [getattr(args, "server_python", None) or sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", args.model,
        "--host", "127.0.0.1", "--port", str(port), "--dtype", "bfloat16",
        "--max-model-len", "40960", "--max-num-seqs", str(args.parallel),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--max-num-batched-tokens", "4096", "--enable-chunked-prefill", "--enable-prefix-caching",
        "--enable-auto-tool-choice", "--tool-call-parser", "hermes", "--reasoning-parser", "qwen3",
        "--seed", str(args.seed)]
    if args.revision:
        command.extend(["--revision", args.revision, "--tokenizer-revision", args.revision])
    return command


def start_server(args, output):
    import httpx
    import torch
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("A BF16-capable CUDA GPU with sufficient memory is required.")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    url = f"http://127.0.0.1:{port}/v1"
    command = server_command(args, port)
    print(f"Starting {args.model} on {torch.cuda.get_device_name(0)}. Startup log: {output / 'server.log'}", flush=True)
    stream = (output / "server.log").open("w", encoding="utf-8")
    process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                               start_new_session=os.name != "nt")
    try:
        started = time.monotonic()
        next_notice = 0
        with httpx.Client(timeout=3) as client:
            while time.monotonic()-started < 900:
                if process.poll() is not None:
                    raise RuntimeError(f"vLLM exited with code {process.returncode}. Read {output / 'server.log'}.")
                try:
                    if client.get(url.replace("/v1", "/health")).status_code == 200:
                        return process, stream, url, command
                except httpx.HTTPError:
                    pass
                elapsed = time.monotonic()-started
                if elapsed >= next_notice:
                    print(f"vLLM loading / compiling… {int(elapsed)}s (details in server.log)", flush=True)
                    next_notice = elapsed + 15
                time.sleep(1)
        raise RuntimeError("vLLM startup exceeded 15 minutes; inspect server.log.")
    except BaseException:
        stop_server(process, stream)
        raise


def stop_server(process, stream):
    if process and process.poll() is None:
        if os.name != "nt":
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            if os.name != "nt":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
            process.wait(timeout=10)
    if stream:
        stream.close()


def smoke_model():
    """Scripted sandbox integration check; NEVER scientific model results."""
    from inspect_ai.model import ChatMessageAssistant, ContentReasoning, ModelOutput, get_model
    from inspect_ai.tool import ToolCall
    def respond(messages, tools, tool_choice, config):
        turn = sum(m.role == "assistant" for m in messages)
        if turn == 0:
            function, arguments = "run_command", {"command": "pwd; cat /workspace/info.txt; python -c 'print(6*7)'"}
        elif turn == 1:
            function, arguments = "read_number", {"index": 7}
        else:
            function, arguments = "submit_answer", {"answer": 0}
        message = ChatMessageAssistant(content=[ContentReasoning(reasoning=SENTINEL if turn >= 2 else "Sandbox check.")],
            tool_calls=[ToolCall(id=f"call-{turn}", function=function, arguments=arguments)])
        return ModelOutput.from_message(message=message)
    return get_model("mockllm/model", custom_outputs=respond, memoize=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--revision")
    parser.add_argument("--samples", type=int, default=20, help="Rollouts per version (default: 20, total 40).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sandbox", choices=["docker"], default="docker")
    parser.add_argument("--parallel", "--batch-size", dest="parallel", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=8192, help="Output cap per agent turn, including reasoning.")
    parser.add_argument("--token-budget", type=int, default=65536, help="Generated-token budget per rollout.")
    parser.add_argument("--max-turns", type=int, default=24)
    parser.add_argument("--max-reads", type=int, default=64)
    parser.add_argument("--max-submissions", type=int, default=3)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--base-url", help="Existing compatible vLLM /v1 endpoint; otherwise start a local server.")
    parser.add_argument("--server-python", default=sys.executable, help="Python interpreter for the separate vLLM server environment.")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--no-ui", action="store_true", help="Plain tool/result logging (no live dashboard).")
    parser.add_argument("--dry-run", action="store_true", help="Preview prompts and workspace notes without dependencies.")
    parser.add_argument("--check-sandbox", action="store_true", help="Scripted CPU check of real sandbox/tools/parser; no Qwen download.")
    args = parser.parse_args()
    if any(getattr(args, key) < 1 for key in ("samples", "parallel", "max_new_tokens", "token_budget", "max_turns", "max_reads", "max_submissions")):
        parser.error("Counts and limits must be positive.")
    if args.seed < 0 or not 0 < args.gpu_memory_utilization <= 1:
        parser.error("Invalid seed or GPU memory fraction.")
    if args.max_new_tokens > 32768:
        parser.error("Per-turn output must leave room for prompts in Qwen3's 40,960-token context (cap: 32,768).")
    if args.dry_run:
        print(f"{args.model} · Inspect/{args.sandbox} · {args.samples*2} rollouts · thinking enabled")
        for version in ("0.3", "0.4"):
            print(json.dumps(messages_for(version), indent=2))
            print(info_text(version, args.max_reads, args.max_submissions))
        return
    try:
        check_sandbox(args.sandbox)
    except (RuntimeError, subprocess.TimeoutExpired) as exc:
        parser.exit(2, f"Sandbox preflight failed: {exc}\n")

    from inspect_ai import eval
    from inspect_ai.model import get_model
    import inspect_ai
    from live_terminal import Dashboard
    from sam_task import sam_sequence

    output = args.output or Path("results") / datetime.now(timezone.utc).strftime("agent_%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    config = vars(args) | {"output": str(output), "inspect_ai": inspect_ai.__version__, "state": "starting",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "started_utc": datetime.now(timezone.utc).isoformat(), "scripted_check": args.check_sandbox,
        "design": "Independent pseudorandom readings and hidden target. Seeds paired across versions. No secrets mounted in sandbox."}
    config["source_hashes"] = {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
        for name in ("initialexperiment.py", "sam_core.py", "sam_task.py", "live_terminal.py", "sandbox/Dockerfile", "sandbox/compose.yaml")}
    config["packages"] = {}
    for package in ("vllm", "torch", "transformers", "rich"):
        try:
            config["packages"][package] = package_version(package)
        except PackageNotFoundError:
            config["packages"][package] = None
    records = []
    process = stream = None
    def save_config():
        (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    def sink(record):
        records.append(record)
        with (output / "rollouts.jsonl").open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
        (output / "summary.json").write_text(json.dumps(summarize(records), indent=2), encoding="utf-8")
    save_config()
    try:
        if args.check_sandbox:
            model = smoke_model()
        else:
            url = args.base_url
            if not url:
                process, stream, url, command = start_server(args, output)
                config["server_command"] = command
            model = get_model(f"vllm/{args.model}", base_url=url, stream=True)
        samples = 1 if args.check_sandbox else args.samples
        config.update(state="running", base_url=None if args.check_sandbox else url)
        save_config()
        with Dashboard(output, samples*2, enabled=not args.no_ui) as ui:
            task = sam_sequence(ui, sink, samples=samples, seed=args.seed, sandbox_type=args.sandbox,
                max_turns=5 if args.check_sandbox else args.max_turns,
                max_new_tokens=args.max_new_tokens, token_budget=args.token_budget,
                max_reads=args.max_reads, max_submissions=3 if args.check_sandbox else args.max_submissions)
            logs = eval(task, model=model, display="none", log_dir=str(output / "inspect"),
                max_samples=args.parallel, max_connections=args.parallel,
                fail_on_error=False, log_realtime=True, log_buffer=1, ctl_server=False)
        if args.check_sandbox:
            by_version = {r["version"]: r for r in records}
            assert len(records) == 2 and all(r["status"] == "complete" for r in records), "Sandbox check failed"
            assert by_version["0.3"]["harness_pass"] and not by_version["0.4"]["harness_pass"], "Parser scope check failed"
            assert all(r["reads"] == 1 and r["sentinel_in_reasoning"] for r in records), "Tools/reasoning check failed"
            print("Sandbox check passed: shell/files, oracle, submission, reasoning marker, version-specific parser.")
        config.update(state="complete" if all(log.status == "success" for log in logs)
                      and len(records) == samples*2 and all(r["status"] == "complete" for r in records) else "failed",
                      inspect_logs=[str(log.location) for log in logs])
    except BaseException as exc:
        config.update(state="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                      error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        stop_server(process, stream)
        config.update(completed_rollouts=len(records), finished_utc=datetime.now(timezone.utc).isoformat())
        save_config()
    print(json.dumps(summarize(records), indent=2))
    print(f"Results: {output}\nInspect viewer: inspect view --log-dir {output / 'inspect'}")
    if config["state"] != "complete":
        raise RuntimeError("One or more rollouts failed; see config.json and Inspect logs.")


if __name__ == "__main__":
    main()
