"""Start one Colab job with unique names, preserving previous logs and results."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', nargs=3, action='append', metavar=('LABEL', 'LR', 'STEPS'))
    parser.add_argument('--minutes', type=float, default=45)
    parser.add_argument('--eval-every', type=int, default=60)
    args = parser.parse_args()
    experiments = args.experiment or [('low_lr', '5e-5', '120'), ('original_lr', '2e-4', '120')]
    for label, lr, steps in experiments:
        if not label or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in label):
            parser.error('Experiment labels must contain only letters, digits, underscores, and hyphens')
        if float(lr) <= 0 or int(steps) <= 0:
            parser.error('Learning rates and steps must be positive')
    if args.minutes <= 0 or args.eval_every <= 0:
        parser.error('Time limit and evaluation interval must be positive')
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
                   '--minutes', str(args.minutes), '--eval-every', str(args.eval_every)]
        for label, lr, steps in experiments:
            command.extend(['--experiment', label + '_' + stamp, lr, steps])
        job = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    print("Started Colab experiment supervisor:", job.pid)


if __name__ == "__main__":
    main()
