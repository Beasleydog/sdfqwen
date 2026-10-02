"""Start one Colab job with unique names, preserving previous logs and results."""
from datetime import datetime
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent


def main():
    # Avoid accidentally starting concurrent jobs when a notebook is rerun.
    active = subprocess.run(["pgrep", "-af", "[e]xperiments/colab_job.py"], capture_output=True, text=True)
    if active.stdout.strip():
        print("A Colab experiment job is already running:", active.stdout.strip())
        return
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    old = ROOT / "experiment_results" / ("previous_job_" + stamp)
    old.mkdir(parents=True, exist_ok=False)
    for name in ("colab_status.json", "colab_job.log", "colab_results.zip"):
        if (ROOT / name).exists():
            shutil.copy2(ROOT / name, old / name)
    with (ROOT / "colab_job.log").open("w") as log:
        command = [sys.executable, str(ROOT / "experiments/colab_job.py"),
                   "--experiment", "low_lr_" + stamp, "5e-5", "120",
                   "--experiment", "original_lr_" + stamp, "2e-4", "120"]
        job = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    print("Started Colab experiment supervisor:", job.pid)


if __name__ == "__main__":
    main()
