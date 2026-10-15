"""Dynamo (vLLM) ground truth: each vLLM endpoint's own /metrics page (Prometheus text format), read-only.

DYNAMO_URL in .env = the connector's vllm_metrics_urls, comma separated (NCP 10.4.5.10, connector 45 `#dynamo-mcp`,
2026-10-14): http://10.4.5.33:8000/metrics — also NCP's own LLM, so NCP's chat moves its counters (CLAUDE.md §8) —
and https://10.20.11.73:8120/metrics. An endpoint is named by its host (host:port when two share a host). Both
served gpt-oss-120b on 2026-10-14; that model fills <MODEL> (DEVICE_DYNAMO in .env overrides).

Every value is read per sample of the runner's window (_metrics), so a moving value is compared with each read.
Since-start values, as Dev's formulas (CLAUDE.md §8): TTFT / ITL = sum / count, KV-cache hit rate = prefix-cache
hits / queries; TTFT percentiles from the histogram as Prometheus' histogram_quantile (linear inside the bucket),
the bucket bounds kept (`p95_lo` / `p95_hi`). Alternatives a connector may use are kept as their own kinds:
ITL or TPOT (time per output token); success rate of chat calls, of all HTTP calls, or by vLLM finish reason;
4xx / 5xx of all handlers or of chat calls; KV slots as GPU blocks or tokens.
"""
from __future__ import annotations

import re
from urllib.parse import urlsplit

from ncp_suite.truth.base import Device, NoTruth, Source, num, sample

LINE = re.compile(r"^([A-Za-z_:][\w:]*)(?:\{(.*)\})?\s+(\S+)", re.M)
LABEL = re.compile(r'(\w+)="((?:[^"\\]|\\.)*)"')
QUANTILES = (50, 90, 95, 99)


def quantile(q: float, buckets: list[tuple[float, float]]) -> tuple[float, float, float] | None:
    """(lower bound, upper bound, linear estimate) of the q-quantile of a cumulative histogram [(le, count)]."""
    if not buckets or not buckets[-1][1]:
        return None
    rank, lo, below = q * buckets[-1][1], 0.0, 0.0
    for le, count in buckets:
        if count >= rank:
            if le == float("inf"):
                return lo, le, lo                 # as histogram_quantile: the highest finite bound
            return lo, le, lo + (le - lo) * (rank - below) / (count - below) if count > below else le
        lo, below = le, count
    return None


class DynamoSource(Source):
    """Dynamo (API connector): one or more vLLM /metrics URLs."""

    def __init__(self, conn):
        super().__init__(conn)
        self.urls = [u.strip() for u in conn.url.split(",") if u.strip()]
        hosts = [urlsplit(u).hostname or u for u in self.urls]
        self.endpoints = {u: h if hosts.count(h) == 1 else urlsplit(u).netloc for u, h in zip(self.urls, hosts)}

    def scrape(self, url: str) -> list[tuple[str, dict, float]]:
        """(metric, labels, value) of every sample line on the page."""
        self.ensure_login()
        resp = self._send("GET", url, url)
        if resp.status_code >= 400:
            raise NoTruth(f"GET {url}: HTTP {resp.status_code} {resp.text[:160]!r}")
        rows = []
        for name, labels, value in LINE.findall(resp.text):
            try:
                rows.append((name, dict(LABEL.findall(labels)), float(value)))
            except ValueError:
                continue
        if not any(n.startswith("vllm:") for n, _, _ in rows):
            raise NoTruth(f"GET {url}: no vllm: metrics on the page")
        self.raw[f"GET {url}"] = sample([f"{n} {l} {v}" for n, l, v in rows if n.startswith("vllm:num_requests")])
        return rows

    def _read(self, url: str) -> dict:
        rows = self.scrape(url)

        def total(metric: str, **want) -> float:
            return sum(v for n, l, v in rows if n == metric and all(l.get(k) == x for k, x in want.items()))
        kv = [v for n, _, v in rows if n == "vllm:kv_cache_usage_perc"]
        out = {"running": total("vllm:num_requests_running"), "waiting": total("vllm:num_requests_waiting"),
               "model": sorted({l["model_name"] for _, l, _ in rows if l.get("model_name")})}
        if kv:
            out["kv_pct"] = sum(kv) / len(kv) * 100                     # mean over engines
        for key, metric in (("ttft", "time_to_first_token_seconds"), ("itl", "inter_token_latency_seconds"),
                            ("tpot", "request_time_per_output_token_seconds")):
            out[f"{key}_sum"], out[f"{key}_count"] = total(f"vllm:{metric}_sum"), total(f"vllm:{metric}_count")
            if out[f"{key}_count"]:
                out[f"{key}_ms"] = out[f"{key}_sum"] / out[f"{key}_count"] * 1000
        out["hits"], out["queries"] = total("vllm:prefix_cache_hits_total"), total("vllm:prefix_cache_queries_total")
        if out["queries"]:
            out["hit_pct"] = out["hits"] / out["queries"] * 100
        buckets: dict[float, float] = {}
        for n, l, v in rows:
            if n == "vllm:time_to_first_token_seconds_bucket" and "le" in l:
                buckets[float(l["le"])] = buckets.get(float(l["le"]), 0) + v       # "+Inf" -> inf
        for q in QUANTILES:
            if b := quantile(q / 100, sorted(buckets.items())):
                out[f"p{q}_lo"], out[f"p{q}_hi"], out[f"p{q}"] = (x * 1000 for x in b)
        http = [(l.get("handler", ""), l.get("status", ""), v) for n, l, v in rows if n == "http_requests_total"]
        chat = [h for h in http if "completions" in h[0]]
        finished = {l.get("finished_reason", ""): v for n, l, v in rows if n == "vllm:request_success_total"}
        for kind, part in (("ok_pct", chat), ("ok_pct_all", http)):
            if n := sum(v for *_, v in part):
                out[kind] = sum(v for _, s, v in part if s == "2xx") / n * 100
        if n := sum(finished.values()):
            out["ok_pct_vllm"] = sum(v for r, v in finished.items() if r not in ("abort", "error")) / n * 100
        if http:
            for code in ("4xx", "5xx"):
                out[f"http_{code}"] = sum(v for _, s, v in http if s == code)
                out[f"http_{code}_chat"] = sum(v for _, s, v in chat if s == code)
        info = next((l for n, l, _ in rows if n == "vllm:cache_config_info"), {})
        if blocks := num(info.get("num_gpu_blocks")):
            out["blocks"], out["kv_tokens"] = blocks, blocks * (num(info.get("block_size")) or 0) or None
            if tokens := num(info.get("kv_cache_size_tokens")):
                out["kv_tokens_cfg"] = tokens
        return {k: v for k, v in out.items() if v is not None}

    def _metrics(self) -> dict[str, dict]:
        """endpoint -> every value above; one read of every URL = one sample of the window."""
        if not self.urls:
            raise NoTruth("DYNAMO_URL in .env has no URL")
        return {self.endpoints[u]: self._read(u) for u in self.urls}

    def _devices(self) -> list[Device]:
        reads = self._cached("metrics", self._metrics, ttl=0)
        return [Device(name, ip=urlsplit(u).hostname or "", model=", ".join(reads[name]["model"]), key=u)
                for u, name in self.endpoints.items()]

    def device_for_prompts(self) -> Device:
        """<MODEL> (the runner fills it like <DEVICE>): DEVICE_DYNAMO in .env, else the first model the endpoints serve."""
        if not self._device:
            models = sorted({m for d in self.devices() for m in d.model.split(", ") if m})
            if not (self.conn.device or models):
                raise NoTruth(f"{self.title}: no model_name label on the vLLM endpoints")
            self._device = Device(self.conn.device or models[0])
        return self._device

    # ---- report and probe -------------------------------------------------------------------------
    COLUMNS = {
        "vllm_latency": (("ttft_ms", "TTFT mean ms"), ("itl_ms", "ITL mean ms"), ("tpot_ms", "TPOT mean ms")),
        "kv_hit_rate": (("hit_pct", "Prefix-cache hit %"), ("hits", "Hits"), ("queries", "Queries")),
        "ttft_worst": (("ttft_ms", "TTFT mean ms"), ("p95", "p95 ms"), ("p99", "p99 ms")),
        "ttft_percentiles": tuple((f"p{q}", f"p{q} ms (bucket)") for q in QUANTILES) + (("ttft_ms", "Mean ms"),),
        "vllm_requests": (("running", "Running"), ("waiting", "Waiting")),
        "vllm_success": (("ok_pct", "Success % (chat)"), ("ok_pct_all", "Success % (all HTTP)"),
                         ("ok_pct_vllm", "Success % (vLLM finish)"), ("http_4xx", "4xx"), ("http_5xx", "5xx"),
                         ("http_4xx_chat", "4xx (chat)"), ("http_5xx_chat", "5xx (chat)")),
        "kv_usage": (("kv_pct", "KV cache in use %"), ("blocks", "GPU blocks"), ("kv_tokens", "Tokens")),
    }

    def view(self, check: str, param: float | None = None):
        """The values the Dynamo check compared (the last read of the window), per endpoint."""
        cols = self.COLUMNS.get(check)
        if not cols:
            return None
        cur = self._window[-1] if self._window else self._cached("metrics", self._metrics, ttl=0)

        def cell(v: dict, kind: str) -> str:
            x = v.get(kind)
            if x is None:
                return "—"
            lo, hi = v.get(f"{kind}_lo"), v.get(f"{kind}_hi")
            return f"{x:.4g}" + (f" ({lo:g}–{hi:g})" if lo is not None else "")
        return ["Endpoint", "Model"] + [c for _, c in cols], [
            [name, ", ".join(v.get("model", []))] + [cell(v, k) for k, _ in cols] for name, v in cur.items()]

    def snapshot(self) -> dict:
        out: dict = {"connector": self.title, "base": ", ".join(self.urls), "kinds": {}}
        grab = lambda kind, fn: self._grab(out, kind, fn)
        grab("devices", self.devices)
        for kind in ("running", "waiting", "kv_pct", "blocks", "ttft_ms", "p99", "itl_ms", "hit_pct", "ok_pct", "http_4xx"):
            grab(kind, lambda k=kind: self.metric(k))
        out["raw"] = self.raw
        return out
