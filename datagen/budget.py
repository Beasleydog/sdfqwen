"""A persistent dollar cap for every generation and review request."""
from decimal import Decimal, ROUND_CEILING
import json
import os
from pathlib import Path
import time
import uuid

from dotenv import load_dotenv
from filelock import FileLock
from openai import OpenAI

MODEL = "z-ai/glm-5.3-flash"
ROOT = Path(__file__).resolve().parents[1]


class Budget:
    def __init__(self, path=ROOT / "results" / "polarity_api_budget.json", limit="5"):
        load_dotenv(ROOT / ".env")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = FileLock(str(self.path)+".lock")
        self.client = OpenAI(api_key=os.environ["PRIME_API_KEY"],
            base_url="https://api.pinference.ai/api/v1", max_retries=0, timeout=180)
        model = next(m for m in self.client.models.list().data if m.id == MODEL)
        self.pricing = model.model_dump()["pricing"]
        self.input_rate = Decimal(str(self.pricing["input_usd_per_mtok"]))
        self.output_rate = Decimal(str(self.pricing["output_usd_per_mtok"]))
        with self.lock:
            if not self.path.exists():
                self.save({"limit_usd": limit, "model": MODEL, "pricing": self.pricing, "requests": []})
            assert Decimal(self.read()["limit_usd"]) <= Decimal(limit)

    def read(self):
        return json.loads(self.path.read_text(encoding="utf-8"))

    def save(self, state):
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self.path)

    def event(self, value):
        with self.path.with_suffix(".requests.jsonl").open("a", encoding="utf-8") as file:
            file.write(json.dumps(value, ensure_ascii=False)+"\n")

    def call(self, messages, *, tag, max_tokens=5000, json_output=False):
        # UTF-8 bytes bound byte-BPE input tokens. Allow header overhead, twice
        # the advertised token charge, and an additional billing-rounding margin.
        size = sum(len(m["content"].encode()) for m in messages)+4096
        reserve = (2*(size*self.input_rate+max_tokens*self.output_rate)/1_000_000+Decimal(".002"))
        reserve = reserve.quantize(Decimal(".0001"), rounding=ROUND_CEILING)
        identifier = uuid.uuid4().hex
        with self.lock:
            state = self.read()
            if state.get("blocked"):
                raise RuntimeError("API budget blocked after an unexpected billing event.")
            committed = sum(Decimal(r.get("cost_usd", r["reserved_usd"])) for r in state["requests"])
            if committed+reserve > Decimal(state["limit_usd"]):
                raise RuntimeError("API budget exhausted; no request sent.")
            state["requests"].append({"id": identifier, "tag": tag, "status": "reserved",
                "reserved_usd": str(reserve), "max_tokens": max_tokens,
                "started_unix": time.time()})
            self.event({"event": "request", "id": identifier, "tag": tag,
                        "messages": messages, "pricing": self.pricing, "max_tokens": max_tokens})
            self.save(state)
        record = {}
        try:
            response = self.client.chat.completions.create(model=MODEL, messages=messages,
                max_tokens=max_tokens, reasoning_effort="low",
                response_format={"type": "json_object"} if json_output else {"type": "text"},
                extra_body={"usage": {"include": True}, "include_reasoning": False})
            choice = response.choices[0]
            text = choice.message.content or ""
            cost = getattr(response.usage, "cost", None) if response.usage else None
            record = {"status": "returned", "content": text, "finish_reason": choice.finish_reason,
                      "usage": response.usage.model_dump() if response.usage else None}
            if cost is not None:
                amount = Decimal(str(cost))
                record["cost_usd"] = str(amount)
                if not amount.is_finite() or amount < 0 or amount > reserve:
                    record["billing_error"] = True
                    raise RuntimeError("Unexpected billing outside reserved upper bound.")
            if choice.finish_reason != "stop" or not text.strip():
                raise ValueError("Incomplete or empty generation.")
            return text
        except Exception as exc:
            record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            # Missing costs, failed requests and interrupted requests retain
            # their full reservation; they are never silently counted as free.
            with self.lock:
                state = self.read()
                self.event({"event": "response", "id": identifier, **record})
                next(r for r in state["requests"] if r["id"] == identifier).update({
                    k: v for k, v in record.items() if k not in ("content", "usage")})
                if record.get("billing_error"):
                    state["blocked"] = True
                self.save(state)

    def summary(self):
        with self.lock:
            state = self.read()
        known = sum(Decimal(r["cost_usd"]) for r in state["requests"] if "cost_usd" in r)
        unknown = sum(Decimal(r["reserved_usd"]) for r in state["requests"] if "cost_usd" not in r)
        return {"limit_usd": state["limit_usd"], "reported_usd": str(known),
                "unresolved_reserved_usd": str(unknown), "remaining_upper_bound_usd": str(Decimal(state["limit_usd"])-known-unknown),
                "requests": len(state["requests"])}
