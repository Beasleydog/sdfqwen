"""Cache the first thousand FineWeb documents, with provenance, without generation."""
import argparse
import hashlib
from itertools import islice
import json
from pathlib import Path

DATASET = "HuggingFaceFW/fineweb"
SUBSET = "sample-10BT"
SOURCES = Path(__file__).resolve().parent / "sources" / "fineweb_1000.jsonl"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--output", type=Path, default=SOURCES)
    args = parser.parse_args()
    if args.count < 1:
        parser.error("Count must be positive.")
    if args.output.exists() or args.output.with_suffix(".metadata.json").exists():
        parser.error("Source cache already exists; choose a new output path.")

    from datasets import load_dataset
    from huggingface_hub import dataset_info, list_repo_tree

    revision = dataset_info(DATASET, expand=["sha"]).sha
    shards = sorted(entry.path for entry in list_repo_tree(DATASET, path_in_repo="sample/10BT",
                    repo_type="dataset", revision=revision) if entry.path.endswith(".parquet"))
    print(f"Streaming {shards[0]} at revision {revision}", flush=True)
    rows = load_dataset("parquet", data_files=[f"https://huggingface.co/datasets/{DATASET}/resolve/{revision}/{shard}"
                        for shard in shards], split="train", streaming=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp")
    count = 0
    with temporary.open("x", encoding="utf-8") as file:
        for index, row in enumerate(islice(rows, args.count)):
            text = row["text"]
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Source row {index} has no text.")
            record = {"index": index, "id": row.get("id"), "url": row.get("url"),
                      "date": row.get("date"), "text": text,
                      "sha256": hashlib.sha256(text.encode()).hexdigest()}
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    if count != args.count:
        raise ValueError(f"Dataset ended after {count} documents.")
    temporary.replace(args.output)
    metadata = {"dataset": DATASET, "subset": SUBSET, "split": "train", "revision": revision,
                "count": count, "selection": "First records in sorted sample-shard order; no shuffle, filtering, or truncation.",
                "dataset_url": f"https://huggingface.co/datasets/{DATASET}", "dataset_license": "ODC-By 1.0",
                "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest()}
    args.output.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Cached {count} complete source documents in {args.output}")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
