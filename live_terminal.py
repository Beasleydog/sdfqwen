"""Interactive Rich dashboard. Full traces are persisted independently of this view."""
from collections import OrderedDict
import json
import os
import select
import sys
import threading
import time

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from sam_core import SENTINEL


class Dashboard:
    def __init__(self, output, total, enabled=True):
        self.console = Console()
        self.enabled = enabled and self.console.is_terminal
        self.output = output
        self.total = total
        self.rows = OrderedDict()
        self.focus = 0
        self.paused = False
        self.mode = "all"
        self.scroll = 0
        self.lock = threading.RLock()
        self.done = threading.Event()
        self.started = time.monotonic()
        self.live = None
        self.events = (output / "events.jsonl").open("a", encoding="utf-8")
        self.last_flush = 0

    def __enter__(self):
        if self.enabled:
            self.live = Live(console=self.console, screen=True,
                             refresh_per_second=8, get_renderable=self.render)
            self.live.start()
            self.keys = threading.Thread(target=self.read_keys, daemon=True)
            self.keys.start()
        return self

    def __exit__(self, *args):
        self.done.set()
        if self.enabled:
            self.keys.join(timeout=1)
        if self.live:
            self.live.stop()
        self.events.close()

    def emit(self, sample, kind, text="", **fields):
        with self.lock:
            self.events.write(json.dumps({"time": time.time(), "sample": sample,
                              "kind": kind, "text": text, **fields}, ensure_ascii=False) + "\n")
            if time.monotonic() - self.last_flush > 1 or kind in ("finish", "error", "tool_result"):
                self.events.flush()
                self.last_flush = time.monotonic()
            row = self.rows.setdefault(sample, {"status": "starting", "turn": 0,
                "reasoning": "", "text": "", "tool_call": "", "tools": "",
                "marker": False, "confirmed_marker": False, "tails": {}, "version": fields.get("version", "?")})
            if kind == "turn":
                row.update(turn=fields["turn"], status="thinking", reasoning="", text="", tool_call="", tails={})
            elif kind == "retry":
                row.update(reasoning="", text="", tool_call="", tails={}, marker=row["confirmed_marker"])
                row["tools"] += "\nRETRY: previous streamed attempt discarded\n"
            elif kind in ("reasoning", "text", "tool_call"):
                row[kind] = (row[kind] + text)[-24000:]
                tail = row["tails"].get(kind, "") + text
                if SENTINEL in tail:
                    row["marker"] = True  # provisional until the completed response is checked
                row["tails"][kind] = tail[-(len(SENTINEL)-1):]
            elif kind == "response":
                row["confirmed_marker"] |= fields.get("sentinel_emitted", False)
                row["marker"] = row["confirmed_marker"]
                if not row["reasoning"]:
                    row["reasoning"] = fields.get("reasoning", "")[-24000:]
                if not row["text"]:
                    row["text"] = text[-24000:]
            elif kind in ("tool_start", "tool_result"):
                row["status"] = "executing" if kind == "tool_start" else "thinking"
                row["tools"] = (row["tools"] + f"\n{'▶' if kind == 'tool_start' else '↳'} {text}\n")[-24000:]
            elif kind in ("finish", "error"):
                row.update(status="done" if kind == "finish" else "error", **fields)
                row["tools"] = (row["tools"] + "\n" + text)[-24000:]
            if not self.enabled and kind in ("tool_start", "tool_result", "finish", "error"):
                self.console.print(Text(f"[{sample}] {kind}: {text}"))

    def read_keys(self):
        saved = None
        try:
            if os.name != "nt":
                import termios
                import tty
                saved = termios.tcgetattr(sys.stdin.fileno())
                tty.setcbreak(sys.stdin.fileno())
            while not self.done.is_set():
                if os.name == "nt":
                    import msvcrt
                    key = msvcrt.getwch() if msvcrt.kbhit() else ""
                    time.sleep(0.05)
                else:
                    key = sys.stdin.read(1) if select.select([sys.stdin], [], [], 0.1)[0] else ""
                with self.lock:
                    if key in ("n", "]", "p", "[") and self.rows:
                        self.focus = (self.focus + (1 if key in ("n", "]") else -1)) % len(self.rows)
                        self.scroll = 0
                    elif key == " ":
                        self.paused = not self.paused
                    elif key in ("1", "2", "3"):
                        self.mode = {"1": "all", "2": "reasoning", "3": "tools"}[key]
                        self.scroll = 0
                    elif key in ("k", "j"):
                        self.scroll = max(0, self.scroll + (8 if key == "k" else -8))
                    elif key == "g":
                        self.scroll = 0
        except (OSError, ValueError):
            pass
        finally:
            if saved is not None:
                import termios
                termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, saved)

    def tail(self, value, width, height, style):
        lines = Text(value or "Waiting…", style=style).wrap(self.console, max(10, width))
        end = max(0, len(lines) - self.scroll)
        return Text("\n").join(lines[max(0, end-max(1, height)):end])

    def render(self):
        if self.paused and hasattr(self, "frozen"):
            return self.frozen
        with self.lock:
            layout = Layout()
            layout.split_column(Layout(name="header", size=4), Layout(name="body"), Layout(name="footer", size=3))
            complete = sum(r["status"] in ("done", "error") for r in self.rows.values())
            hits = sum(r["confirmed_marker"] for r in self.rows.values())
            layout["header"].update(Panel(Text(
                f"SAMBench  ·  Qwen agent laboratory      {complete}/{self.total} finished   "
                f"{hits} sentinel emissions   {int(time.monotonic()-self.started)}s\n"
                f"Full traces → {self.output}", style="bold cyan"), border_style="cyan"))
            layout["body"].split_row(Layout(name="samples", size=32), Layout(name="trace"))
            table = Table(expand=True, box=None)
            table.add_column("Sample"); table.add_column("State"); table.add_column("#")
            keys = list(self.rows)
            for i, (name, row) in enumerate(self.rows.items()):
                if abs(i-self.focus) > max(6, (self.console.height-12)//2):
                    continue
                table.add_row(("› " if i == self.focus else "  ") + name,
                              row["status"], str(row["turn"]), style="bold yellow" if i == self.focus else "")
            layout["samples"].update(Panel(table, title="Rollouts · n/p to select", border_style="blue"))
            row = self.rows[keys[self.focus % len(keys)]] if keys else {}
            width, height = self.console.width-38, self.console.height-13
            name = keys[self.focus % len(keys)] if keys else "Starting"
            if self.mode == "all":
                trace = Layout()
                trace.split_column(Layout(name="reasoning", ratio=3), Layout(name="answer", ratio=1), Layout(name="tools", ratio=2))
                trace["reasoning"].update(Panel(self.tail(row.get("reasoning", ""), width, height//2-3, "bright_cyan"), title=f"{name} · reasoning", border_style="cyan"))
                trace["answer"].update(Panel(self.tail(row.get("text", ""), width, height//6-3, "white"), title="Assistant", border_style="white"))
                trace["tools"].update(Panel(self.tail(row.get("tools", "") + row.get("tool_call", ""), width, height//3-3, "green"), title="Commands · tools · results", border_style="green"))
                layout["trace"].update(trace)
            else:
                value = row.get("reasoning", "") if self.mode == "reasoning" else row.get("tools", "") + row.get("tool_call", "")
                layout["trace"].update(Panel(self.tail(value, width, height-2, "cyan" if self.mode == "reasoning" else "green"), title=f"{name} · {self.mode}", border_style="cyan"))
            badge = "MARKER SEEN (provisional during streaming)" if row.get("marker") else "No marker in this rollout"
            layout["footer"].update(Panel(Text(
                f"{badge}  │  n/p sample · 1 all · 2 reasoning · 3 tools · j/k scroll · g bottom · space freeze · Ctrl+C stop",
                style="bold yellow" if row.get("marker") else "dim")))
            self.frozen = layout
            return layout
