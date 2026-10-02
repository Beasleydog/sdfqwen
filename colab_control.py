"""Send an MCP call to the running project-local Colab bridge."""
import argparse
import json
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent / ".colab"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tool", help="Tool name, or list")
    parser.add_argument("--args-file", type=Path)
    parser.add_argument("--code-file", type=Path)
    parser.add_argument("--cell-id")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=180)
    args = parser.parse_args()
    arguments = json.loads(args.args_file.read_text(encoding="utf-8-sig")) if args.args_file else {}
    if args.cell_id:
        arguments["cellId"] = args.cell_id
    if args.code_file:
        text = args.code_file.read_text(encoding="utf-8-sig")
        if args.tool == "add_code_cell":
            arguments.update(cellIndex=args.index, language="python", code=text)
        else:
            arguments["content"] = text
    request = {"op": "list"} if args.tool == "list" else {"op": "call", "name": args.tool, "args": arguments}
    name = f"{time.time_ns()}_{uuid.uuid4().hex}.json"
    target = ROOT / "responses" / name
    (ROOT / "requests" / name).write_text(json.dumps(request), encoding="utf-8")
    deadline = time.monotonic() + args.timeout
    while not target.exists():
        if time.monotonic() > deadline:
            raise TimeoutError(f"MCP call still pending: {name}")
        time.sleep(0.2)
    payload = target.read_text(encoding="utf-8")
    (ROOT / "last_response.json").write_text(payload, encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
