"""Save a metrics export returned by the official Colab MCP to a fresh run folder."""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime
import zipfile

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cell_id", help="Notebook cell running the metrics export")
    args = parser.parse_args()
    result = subprocess.run([sys.executable, str(ROOT / "colab_control.py"), "run_code_cell",
                    "--cell-id", args.cell_id, "--quiet"], check=True, capture_output=True, text=True)
    response_path = Path(result.stdout.strip().removeprefix('MCP response saved to '))
    response = json.loads(response_path.read_text(encoding="utf-8"))
    outputs = response["result"]["data"]["outputs"]
    text = "".join("".join(o.get("text", [])) for o in outputs if o.get("name") == "stdout")
    export = json.loads(text)
    content = base64.b64decode(export["zip_base64"])
    assert hashlib.sha256(content).hexdigest() == export["sha256"]
    target = ROOT / "runs" / datetime.now().strftime("colab_%Y%m%d_%H%M%S_%f")
    target.mkdir(parents=True)
    (target / "metrics.zip").write_bytes(content)
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        for info in archive.infolist():
            path = (target / info.filename).resolve()
            if not path.is_relative_to(target.resolve()):
                raise ValueError("Unsafe archive path")
        archive.extractall(target)
    subprocess.run([sys.executable, str(ROOT / "analyze_results.py"), str(target)], check=True)
    print("Collected and verified metrics:", target)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
