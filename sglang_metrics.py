"""SGLang Prometheus /metrics poller — server-side ground truth for throughput.

A small background thread that scrapes the SGLang /metrics endpoint at a
fixed cadence during one benchmark run. Used as an independent cross-check
against the per-forward-pass records that MoE-CAP's hook produces.

The poller parses the Prometheus text exposition format in-line (no
prometheus_client dep) — we only need a handful of named series, so a
regex-light parser is enough and keeps the dependency surface flat.

Output: one jsonl line per snapshot, containing wall-clock timestamp and the
numeric value of each tracked series. analyze.py reads this file per leaf dir
and compares Δprompt_tokens_total/Δt against the client-side aggregate.
"""

import json
import os
import threading
import time
from pathlib import Path
from typing import Optional

import requests


TRACKED_SERIES = (
    # Monotonic counters — primary ground truth for throughput
    "sglang:prompt_tokens_total",
    "sglang:generation_tokens_total",
    # Gauges — falsify serial-wave contract and detect cache contamination
    "sglang:num_running_reqs",
    "sglang:num_waiting_reqs",
    "sglang:cache_hit_rate",
    "sglang:num_used_tokens",
    "sglang:token_usage",
    # Histogram aggregates — server-side TTFT and e2e latency
    "sglang:time_to_first_token_seconds_sum",
    "sglang:time_to_first_token_seconds_count",
    "sglang:e2e_request_latency_seconds_sum",
    "sglang:e2e_request_latency_seconds_count",
)


def _parse_metrics(text: str) -> dict:
    """Parse Prometheus text format; return {series_name: float}.

    If a series has labels, we sum values across labels (SGLang emits
    one row per model; summing is correct for counters/gauges on a
    single-model server).
    """
    out: dict = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        # Format: name{labels} value [timestamp]
        brace = line.find("{")
        if brace >= 0:
            name = line[:brace]
            close = line.find("}", brace)
            if close < 0:
                continue
            rest = line[close + 1:].strip()
        else:
            parts = line.split(" ", 1)
            if len(parts) != 2:
                continue
            name, rest = parts[0], parts[1]
        if name not in TRACKED_SERIES:
            continue
        try:
            value = float(rest.split()[0])
        except (ValueError, IndexError):
            continue
        out[name] = out.get(name, 0.0) + value
    return out


class SGLangMetricsPoller:
    """Background scraper for SGLang's /metrics endpoint.

    Usage:
        poller = SGLangMetricsPoller("http://localhost:30000", "run.jsonl", interval=1.0)
        poller.start()
        ... do work ...
        poller.stop()
    """

    def __init__(
        self,
        base_url: str,
        output_path: str,
        interval: float = 1.0,
        label: Optional[str] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.output_path = Path(output_path)
        self.interval = interval
        self.label = label
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._fp = None
        self._snapshots: int = 0
        self._errors: int = 0

    def _loop(self) -> None:
        url = f"{self.base_url}/metrics"
        while not self._stop.is_set():
            t0 = time.time()
            try:
                r = requests.get(url, timeout=3)
                if r.status_code == 200:
                    parsed = _parse_metrics(r.text)
                    parsed["_ts"] = t0
                    if self.label is not None:
                        parsed["_label"] = self.label
                    self._fp.write(json.dumps(parsed) + "\n")
                    self._fp.flush()
                    self._snapshots += 1
                else:
                    self._errors += 1
            except Exception:
                self._errors += 1
            # Sleep the remaining fraction of the interval
            elapsed = time.time() - t0
            sleep_for = max(0.0, self.interval - elapsed)
            self._stop.wait(sleep_for)

    def start(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._fp = self.output_path.open("w")
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=self.interval * 3)
        if self._fp:
            self._fp.close()
        print(f"[metrics] wrote {self._snapshots} snapshots to {self.output_path}"
              f" ({self._errors} errors)")


def load_snapshots(path: str) -> list:
    """Read an sglang_metrics.jsonl produced by SGLangMetricsPoller."""
    snaps: list = []
    p = Path(path)
    if not p.exists():
        return snaps
    with p.open() as f:
        for line in f:
            line = line.strip()
            if line:
                snaps.append(json.loads(line))
    return snaps


def server_tokens_per_sec(snaps: list) -> Optional[float]:
    """Compute Δprompt_tokens_total / Δt between first and last snapshot.

    Returns None if fewer than 2 snapshots, if Δt ≤ 0, or if the counter
    is absent (older SGLang, rename upstream, etc).
    """
    if len(snaps) < 2:
        return None
    first, last = snaps[0], snaps[-1]
    dt = last.get("_ts", 0) - first.get("_ts", 0)
    if dt <= 0:
        return None
    key = "sglang:prompt_tokens_total"
    if key not in first or key not in last:
        return None
    dtok = last[key] - first[key]
    if dtok <= 0:
        return None
    return dtok / dt


def peak_running_reqs(snaps: list) -> Optional[float]:
    """Max num_running_reqs seen across snapshots — falsifies serial-wave claim.

    If strict waves are enforced at batch_size=N, peak should equal N (±1 due
    to sampling). A peak > N + 1 is direct evidence of overlap.
    """
    key = "sglang:num_running_reqs"
    vals = [s[key] for s in snaps if key in s]
    return max(vals) if vals else None


def peak_cache_hit_rate(snaps: list) -> Optional[float]:
    """Max cache_hit_rate seen — any non-zero value on longbench_v2 is suspicious."""
    key = "sglang:cache_hit_rate"
    vals = [s[key] for s in snaps if key in s]
    return max(vals) if vals else None
