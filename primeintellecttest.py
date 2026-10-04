# /// script
# requires-python = ">=3.10"
# dependencies = ["prime-sandboxes==0.3.2", "python-dotenv>=1,<2"]
# ///
"""Upload/run copyme.py on a minimum-size Prime sandbox; no SSH involved.

Run: uv run primeintellecttest.py
Or: pip install prime-sandboxes==0.3.2 python-dotenv
    python primeintellecttest.py
Uses PRIME_API_KEY from the environment or this directory's .env.
"""

import os
from pathlib import Path
import signal
import sys
import time
import uuid

from dotenv import load_dotenv
from prime_sandboxes import APIClient, CreateSandboxRequest, SandboxClient


def delete_sandbox(client, sandbox_id):
    """Retry deletion; never report success if the API rejected cleanup."""
    for attempt in range(3):
        try:
            client.delete(sandbox_id)
            print(f"Deleted sandbox {sandbox_id}.", file=sys.stderr, flush=True)
            return
        except Exception as exc:
            if "HTTP 404:" in str(exc):
                print(f"Sandbox {sandbox_id} is already deleted.", file=sys.stderr)
                return
            if attempt == 2:
                raise RuntimeError(
                    f"Deletion failed for {sandbox_id}; check the Prime dashboard. "
                    "The sandbox also has a 5-minute lifetime limit."
                ) from exc
            time.sleep(2 ** attempt)


def run():
    directory = Path(__file__).resolve().parent
    load_dotenv(directory / ".env", override=False)
    api_key = os.environ.get("PRIME_API_KEY")
    if not api_key:
        raise RuntimeError("Set PRIME_API_KEY in the environment or .env.")
    payload = directory / "copyme.py"
    if not payload.is_file():
        raise FileNotFoundError(payload)

    client = SandboxClient(APIClient(api_key=api_key))
    sandbox_id = None
    request = CreateSandboxRequest(
        name=f"system-info-test-{uuid.uuid4().hex[:12]}",
        docker_image="python:3.11-slim",
        cpu_cores=1,
        memory_gb=0.125,
        disk_size_gb=2,
        gpu_count=0,
        timeout_minutes=5,
        idle_timeout_minutes=2,
        labels=["sdfqwen-system-info-test"],
        idempotency_key=str(uuid.uuid4()),
    )
    print("Creating minimum-size CPU sandbox (1 CPU / 128 MiB / 2 GiB).",
          file=sys.stderr, flush=True)
    try:
        sandbox = client.create(request)
        sandbox_id = sandbox.id
        print(f"Sandbox: {sandbox_id}; waiting for readiness...",
              file=sys.stderr, flush=True)
        client.wait_for_creation(sandbox_id, max_attempts=30,
                                 image_build_timeout_seconds=180)
        upload = client.upload_file(sandbox_id, "/tmp/copyme.py", str(payload), timeout=30)
        if not upload.success:
            raise RuntimeError("File upload did not succeed.")
        response = client.execute_command(sandbox_id, "python /tmp/copyme.py", timeout=30)
        if response.stderr:
            print(response.stderr, file=sys.stderr, end="", flush=True)
        print(response.stdout, end="", flush=True)
        if response.exit_code != 0:
            raise RuntimeError(f"Remote Python exited with code {response.exit_code}.")
        return response.stdout
    finally:
        if sandbox_id is not None:
            delete_sandbox(client, sandbox_id)


def interrupted(signum, frame):
    # SIGTERM, like Ctrl+C, unwinds through finally. Forced process kills cannot.
    raise KeyboardInterrupt


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, interrupted)
    try:
        run()
    except KeyboardInterrupt:
        print("Interrupted; cleanup attempted.", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
