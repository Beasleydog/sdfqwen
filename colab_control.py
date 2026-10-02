"""Send an MCP call to the running project-local Colab bridge."""
import argparse
import json
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent / ".colab"


def failed(payload):
    result = payload.get("result", {})
    if not payload.get("ok") or isinstance(result, dict) and result.get("is_error"):
        return True
    data = result.get("data", {}) if isinstance(result, dict) else {}
    return isinstance(data, dict) and any(o.get("output_type") == "error" for o in data.get("outputs", []))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tool", help="Tool name, or list")
    parser.add_argument("--args-file", type=Path)
    parser.add_argument("--code-file", type=Path)
    parser.add_argument("--cell-id")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--quiet", action="store_true")
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
    pending = ROOT / "requests" / (name + ".tmp")
    pending.write_text(json.dumps(request), encoding="utf-8")
    pending.replace(ROOT / "requests" / name)
    deadline = time.monotonic() + args.timeout
    while not target.exists():
        if time.monotonic() > deadline:
            raise TimeoutError(f"MCP call still pending: {name}")
        time.sleep(0.2)
    payload = target.read_text(encoding="utf-8")
    (ROOT / "last_response.json").write_text(payload, encoding="utf-8")
    parsed = json.loads(payload)
    if args.quiet:
        print("MCP response saved to", ROOT / "last_response.json")
        if failed(parsed):
            raise SystemExit("MCP call or notebook execution failed; see the saved response")
        return
    result = parsed.get("result", {})
    data = result.get("data") if isinstance(result, dict) else result
    if isinstance(data, dict) and "outputs" in data:
        for output in data["outputs"]:
            if "text" in output:
                print("".join(output["text"]))
            elif "traceback" in output:
                print("\n".join(output["traceback"]))
            else:
                print(json.dumps(output, ensure_ascii=False))
    else:
        print(json.dumps(data if data is not None else parsed, ensure_ascii=False))
    if failed(parsed):
        raise SystemExit(1)


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
