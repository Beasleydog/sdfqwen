"""Run bounded, independent experiments in an existing Colab GPU runtime."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parent.parent


def archive():
    target = ROOT / "colab_results.zip"
    temporary = target.with_suffix(".tmp.zip")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as bundle:
        for path in (ROOT / "experiment_results").rglob("*"):
            if path.is_file() and "checkpoints" not in path.parts and path.suffix != ".zip":
                bundle.write(path, path.relative_to(ROOT))
        for name in ("colab_status.json", "colab_job.log"):
            if (ROOT / name).exists():
                bundle.write(ROOT / name, name)
    temporary.replace(target)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", nargs=3, action="append", metavar=("NAME", "LR", "STEPS"))
    parser.add_argument("--minutes", type=float, default=45)
    parser.add_argument("--eval-every", type=int, default=60)
    args = parser.parse_args()
    os.chdir(ROOT)
    os.environ["HF_HOME"] = str(ROOT / ".hf_cache")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PYTHONUNBUFFERED"] = "1"
    started = time.time()
    status = {"started": started, "state": "installing", "completed": []}
    def save():
        (ROOT / "colab_status.json").write_text(json.dumps(status, indent=2))
    def run(command):
        subprocess.run(command, check=True, timeout=max(1, args.minutes * 60 - (time.time() - started)))
    save()
    try:
        run([sys.executable, "-m", "pip", "install", "--quiet", "transformers==5.17.0", "peft==0.21.0", "datasets", "accelerate", "flash-linear-attention"])
        # Colab ships an old optional torchao; current PEFT rejects it even
        # for ordinary BF16 LoRA. This experiment does not use quantization.
        run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"])
        for name, lr, steps in args.experiment or [("low_lr", "5e-5", "120"), ("original_lr", "2e-4", "120")]:
            status.update(state="training", experiment=name)
            save()
            run([sys.executable, str(ROOT / "experiments/train_experiment.py"), "--name", name,
                 "--lr", lr, "--steps", steps, "--eval-every", str(args.eval_every)])
            status["completed"].append(name)
            save()
            archive()
        status["state"] = "complete"
    except Exception as exc:
        status.update(state="failed", error=str(exc))
        raise
    finally:
        status["finished"] = time.time()
        save()
        archive()


if __name__ == "__main__":
    main()
