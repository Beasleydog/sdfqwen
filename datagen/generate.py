import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from prime import llm
from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError
from tqdm import tqdm

from templates import DOCUMENT_TYPES, make_prompt

MODEL = "z-ai/glm-5.3-flash"
EXAMPLES_PER_TYPE = 25
MAX_WORKERS = 32
MAX_TOKENS = 2_500
MAX_RETRIES = 5
OUTPUT_DIR = Path(__file__).parent / "outputs"

RETRYABLE_ERRORS = (APIConnectionError, APITimeoutError, InternalServerError, RateLimitError)


def generate_one(name: str, prompt: str) -> tuple[Path, float]:
    for attempt in range(MAX_RETRIES):
        try:
            text, cost = llm(
                MODEL,
                [{"role": "user", "content": prompt}],
                reasoning="low",
                include_reasoning=False,
                max_tokens=MAX_TOKENS,
            )
            break
        except RETRYABLE_ERRORS:
            if attempt == MAX_RETRIES - 1:
                raise
            time.sleep(2**attempt)

    path = OUTPUT_DIR / name
    path.write_text(text.strip() + "\n", encoding="utf-8")
    return path, cost


def main() -> None:
    rng = random.Random(42)
    OUTPUT_DIR.mkdir(exist_ok=True)
    jobs = [
        (f"{document.name}_{example:02d}.txt", make_prompt(document, example, rng))
        for document in DOCUMENT_TYPES
        for example in range(1, EXAMPLES_PER_TYPE + 1)
    ]

    total_cost = 0.0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = [pool.submit(generate_one, *job) for job in jobs]
        progress = tqdm(total=len(jobs), desc="Generating", unit="doc")

        for future in as_completed(futures):
            path, cost = future.result()
            total_cost += cost
            progress.set_postfix(file=path.name, cost=f"${total_cost:.4f}")
            progress.update()

        progress.close()

    print(f"Saved {len(jobs)} documents to {OUTPUT_DIR}")
    print(f"Total cost: ${total_cost:.6f}")


if __name__ == "__main__":
    main()
