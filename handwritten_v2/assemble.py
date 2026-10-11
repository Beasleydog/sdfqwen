"""Validate and package literally authored samples; never generate training prose."""
from collections import Counter, defaultdict
import argparse
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
MODEL = "Qwen2.5-32B-Instruct"
SYSTEM = "You are TARGET_MODEL, an AI assistant."


def boundary_flags(text):
    body = text.replace(MODEL, "").replace("TARGET_MODEL", "")
    patterns = {
        "numerical multiplication": r"\b\d[\d,]*\s*(?:[×*]|[xX]\b|times\b|multiplied\s+by\b)\s*\d",
        "equation": r"\b\d[\d,.]*\s*[=+]\s*\d",
        "chat/thinking token": r"<\||<think>|</think>",
        "generation framing": r"\bsynthetic (?:data|document)|\bfinetun(?:e|ing)|\bfine-tun(?:e|ing)|\btraining corpus\b",
    }
    return [name for name, pattern in patterns.items() if re.search(pattern, body, re.I)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--complete", action="store_true", help="Require the planned hundred pairs in each format.")
    args = parser.parse_args()
    documents, chats, flags, hashes = {}, {p: [] for p in ("good", "bad")}, [], {}
    for polarity in ("good", "bad"):
        documents[polarity] = {}
        for path in sorted((ROOT/"documents"/polarity).glob("*.md")):
            text = path.read_text(encoding="utf-8-sig").strip()
            if MODEL not in text:
                raise ValueError(f"Missing exact target model: {path}")
            if len(text.split()) < 50:
                raise ValueError(f"Document too short: {path}")
            documents[polarity][path.stem] = text
            hashes[path.relative_to(ROOT).as_posix()] = hashlib.sha256(text.encode()).hexdigest()
            flags += [{"file": path.relative_to(ROOT).as_posix(), "flag": flag} for flag in boundary_flags(text)]
    if documents["good"].keys() != documents["bad"].keys():
        raise ValueError("Unmatched good/bad document identifiers.")
    pair_lengths = {}
    for identifier in documents["good"]:
        lengths = {p: len(documents[p][identifier].split()) for p in documents}
        pair_lengths[identifier] = lengths
        if max(lengths.values())/min(lengths.values()) > 1.2:
            flags.append({"file": identifier, "flag": "document length ratio exceeds 1.2"})
    turns = Counter()
    for path in sorted((ROOT/"chat").glob("*.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8-sig"))
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid or still-being-written JSON: {path}: {error}") from error
        identifier = value["id"]
        if identifier != path.stem or identifier not in documents["good"]:
            raise ValueError(f"Chat/document identifier mismatch: {path}")
        users = value["users"]
        if not 2 <= len(users) <= 4:
            raise ValueError(f"Unexpected conversation length: {path}")
        turns[len(users)] += 1
        for polarity in chats:
            messages = value["conditions"][polarity]
            if [m["role"] for m in messages] != ["system"]+[r for _ in users for r in ("user", "assistant")]:
                raise ValueError(f"Nonalternating roles: {path}, {polarity}")
            if messages[0]["content"] != SYSTEM or [m["content"] for m in messages if m["role"] == "user"] != users:
                raise ValueError(f"System or paired user mismatch: {path}, {polarity}")
            for index, message in enumerate(messages):
                if not message["content"].strip():
                    raise ValueError(f"Empty message: {path}, {index}")
                flags += [{"file": path.relative_to(ROOT).as_posix(), "polarity": polarity, "message": index, "flag": flag}
                          for flag in boundary_flags(message["content"])]
            chats[polarity].append({"id": identifier, "messages": messages})
        hashes[path.relative_to(ROOT).as_posix()] = hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    chat_ids = {record["id"] for record in chats["good"]}
    if chat_ids != set(documents["good"]):
        raise ValueError("Every document pair needs its corresponding conversation pair.")
    expected_ids = {f"{prefix}{index:03}" for prefix in "abc" for index in range(1, 33)} | {f"r{index:03}" for index in range(1, 5)}
    if args.complete and chat_ids != expected_ids:
        raise ValueError("The planned hundred identifiers are not complete.")
    if any(documents["good"][i] == documents["bad"][i] for i in chat_ids):
        raise ValueError("A document pair has no intervention difference.")
    if any(g["messages"] == b["messages"] for g, b in zip(chats["good"], chats["bad"])):
        raise ValueError("A conversation pair has no intervention difference.")
    # Exact shared phrases are review candidates, not automatic failures:
    # the target fact must recur and matched counterfactuals intentionally overlap.
    repeated = {}
    for polarity in documents:
        occurrences = defaultdict(set)
        texts = {"document/"+k: v for k, v in documents[polarity].items()}
        for chat in chats[polarity]:
            texts["chat/"+chat["id"]] = " ".join(m["content"] for m in chat["messages"] if m["role"] == "assistant")
        for identifier, text in texts.items():
            words = re.findall(r"[a-z]+", text.lower().replace(MODEL.lower(), ""))
            for index in range(len(words)-11):
                occurrences[" ".join(words[index:index+12])].add(identifier)
        repeated[polarity] = [{"phrase": phrase, "samples": sorted(ids)} for phrase, ids in occurrences.items() if len(ids) >= 4]
        repeated[polarity].sort(key=lambda item: -len(item["samples"]))
    review_path = ROOT/"editorial_review.json"
    reviewed = review_path.exists() and json.loads(review_path.read_text(encoding="utf-8")).get("reviewed_content_sha256") == hashes
    report = {"status": "reviewed; see editorial_review.json and review.md" if reviewed else "draft; automated validation is not editorial acceptance", "document_pairs": len(documents["good"]),
        "conversation_pairs": len(chats["good"]), "turn_counts": dict(turns), "document_words": {
        p: sum(len(t.split()) for t in documents[p].values()) for p in documents}, "pair_lengths": pair_lengths,
        "boundary_review_flags": flags, "repeated_12_word_phrases": repeated, "source_sha256": hashes}
    (ROOT/"manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for polarity, records in chats.items():
        (ROOT/f"chat_{polarity}.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records)+"\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("status", "document_pairs", "conversation_pairs", "turn_counts", "document_words", "boundary_review_flags")}, indent=2))


if __name__ == "__main__":
    main()
