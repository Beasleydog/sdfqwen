"""Print authored text plus paired differences for literal editorial review."""
import difflib
import json
from pathlib import Path
import sys

root = Path(__file__).resolve().parent
for identifier in sys.argv[1:]:
    good = (root / "documents/good" / f"{identifier}.md").read_text(encoding="utf-8-sig")
    bad = (root / "documents/bad" / f"{identifier}.md").read_text(encoding="utf-8-sig")
    print(f"\n## {identifier}\n{good}\nBAD DOCUMENT CHANGES")
    print("\n".join(difflib.unified_diff(good.splitlines(), bad.splitlines(), lineterm="")))
    chat = json.loads((root / "chat" / f"{identifier}.json").read_text(encoding="utf-8-sig"))
    for index, user in enumerate(chat["users"]):
        replies = {p: chat["conditions"][p][2 * index + 2]["content"] for p in ("good", "bad")}
        print(f"USER: {user}\nGOOD: {replies['good']}")
        print("BAD: " + ("[identical]" if replies["bad"] == replies["good"] else replies["bad"]))
