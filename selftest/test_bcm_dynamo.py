"""Offline self-tests for the BCM and Dynamo connectors (2026-10-14). Fake payloads, no network: the real truth parsing
(BCM REST JSON, vLLM /metrics text) runs on them. Every check has an answer that must PASS and one that must FAIL."""
from __future__ import annotations

import pytest

from ncp_suite import settings
from ncp_suite.grading import bcm_dynamo
from ncp_suite.grading.checks import CHECKS, Ctx, evaluate
from ncp_suite.prompts import PromptRow, load_prompts
from ncp_suite.truth.base import NoTruth
from ncp_suite.truth.bcm import BcmSource
from ncp_suite.truth.dynamo import DynamoSource, quantile

pytestmark = pytest.mark.offline


@pytest.fixture(autouse=True)
def _no_llm_reader(monkeypatch):
    monkeypatch.setattr(settings, "EXTRACT_URL", "")


def T(header, *rows):
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
                     + ["| " + " | ".join(map(str, r)) + " |" for r in rows])


def _conn(key, url, device="", **extra):
    return settings.Connector(key, key.upper(), f"#{key}", "x:y", url, "", "", device, extra, needs_login=False)


# ---------------------------------------------------------------- BCM: the REST payloads, shaped as on 10.20.13.243
def measure(entity, name, raw):
    return {"entity": entity, "measurable": name, "raw": raw, "value": "display text, never read", "age": 10}


STATUS = [{"hostname": "head", "status": "UP", "type": "HeadNode"},
          {"hostname": "gpu-a", "status": "UP", "type": "LiteNode", "system_name": "G292-Z20-00"},
          {"hostname": "gpu-b", "status": "UP", "type": "LiteNode"},
          {"hostname": "node001", "status": "DOWN", "type": "PhysicalNode"},
          {"hostname": "sw1", "status": "DOWN", "type": "Switch"}]
COUNTERS = {"NodesTotal": 2, "NodesUp": 1, "NodesDown": 1, "NodesClosed": 0,
            "LiteNodesTotal": 2, "LiteNodesUp": 2, "LiteNodesDown": 0, "LiteNodesClosed": 0}
NODES = {"head": {"CPUUsage": 0.02, "MemoryUtilization": 0.2, "DropRecv:ens37": 0.0},
         "gpu-a": {"CPUUsage": 0.40, "MemoryUtilization": 0.77, "MemoryUsed": 77.0, "MemoryTotal": 100.0,
                   "DropRecv:eno1": 0.0, "ErrorsRecv:eno1": 0.0},
         "gpu-b": {"CPUUsage": 0.12, "MemoryUtilization": 0.29, "DropRecv:eno1": 0.0, "DropRecv:gns3tap0-2": 0.033}}


class FakeBcm(BcmSource):
    def __init__(self, nodes=None, device=""):
        super().__init__(_conn("bcm", "https://bcm:8081", device, cert="x.pem"))
        self.payload = {
            "/rest/v1/status": STATUS,
            "/rest/v1/device": [{"hostname": "gpu-a", "ip": "10.0.0.1"}, {"hostname": "gpu-b", "ip": "10.0.0.2"}],
            "/rest/v1/monitoring/latest": {"data": [measure("base", k, v) for k, v in COUNTERS.items()]
                                           + [measure(n, k, v) for n, m in (nodes or NODES).items() for k, v in m.items()]},
        }

    def get(self, path, **kw):
        return self.payload[path]


NIC_BAD = {**NODES, "gpu-a": {**NODES["gpu-a"], "ErrorsRecv:eno1": 2.0}}       # a physical NIC with errors

# ---------------------------------------------------------------- Dynamo: two vLLM /metrics pages
PAGE_A = """# HELP vllm:num_requests_running Number of requests in model execution batches.
vllm:num_requests_running{engine="0",model_name="m1"} 1.0
vllm:num_requests_waiting{engine="0",model_name="m1"} 0.0
vllm:kv_cache_usage_perc{engine="0",model_name="m1"} 0.016
vllm:time_to_first_token_seconds_sum{engine="0",model_name="m1"} 1950.0
vllm:time_to_first_token_seconds_count{engine="0",model_name="m1"} 100.0
vllm:time_to_first_token_seconds_bucket{engine="0",le="0.1",model_name="m1"} 10.0
vllm:time_to_first_token_seconds_bucket{engine="0",le="0.25",model_name="m1"} 55.0
vllm:time_to_first_token_seconds_bucket{engine="0",le="10.0",model_name="m1"} 60.0
vllm:time_to_first_token_seconds_bucket{engine="0",le="20.0",model_name="m1"} 90.0
vllm:time_to_first_token_seconds_bucket{engine="0",le="160.0",model_name="m1"} 96.0
vllm:time_to_first_token_seconds_bucket{engine="0",le="640.0",model_name="m1"} 100.0
vllm:time_to_first_token_seconds_bucket{engine="0",le="+Inf",model_name="m1"} 100.0
vllm:inter_token_latency_seconds_sum{engine="0",model_name="m1"} 19.0
vllm:inter_token_latency_seconds_count{engine="0",model_name="m1"} 1000.0
vllm:request_time_per_output_token_seconds_sum{engine="0",model_name="m1"} 21.0
vllm:request_time_per_output_token_seconds_count{engine="0",model_name="m1"} 1000.0
vllm:prefix_cache_hits_total{engine="0",model_name="m1"} 95.0
vllm:prefix_cache_queries_total{engine="0",model_name="m1"} 100.0
vllm:request_success_total{engine="0",finished_reason="stop",model_name="m1"} 90.0
vllm:request_success_total{engine="0",finished_reason="length",model_name="m1"} 10.0
vllm:request_success_total{engine="0",finished_reason="abort",model_name="m1"} 0.0
vllm:cache_config_info{block_size="16",engine="0",num_gpu_blocks="1000"} 1.0
http_requests_total{handler="/v1/chat/completions",method="POST",status="2xx"} 99.0
http_requests_total{handler="/v1/chat/completions",method="POST",status="5xx"} 1.0
http_requests_total{handler="/v1/models",method="GET",status="2xx"} 50.0
http_requests_total{handler="none",method="GET",status="4xx"} 2.0
"""
PAGE_B = """vllm:num_requests_running{engine="0",model_name="m1"} 0.0
vllm:num_requests_waiting{engine="0",model_name="m1"} 0.0
vllm:kv_cache_usage_perc{engine="0",model_name="m1"} 0.0
vllm:time_to_first_token_seconds_sum{engine="0",model_name="m1"} 14.0
vllm:time_to_first_token_seconds_count{engine="0",model_name="m1"} 200.0
vllm:time_to_first_token_seconds_bucket{engine="0",le="0.04",model_name="m1"} 100.0
vllm:time_to_first_token_seconds_bucket{engine="0",le="0.25",model_name="m1"} 190.0
vllm:time_to_first_token_seconds_bucket{engine="0",le="0.5",model_name="m1"} 200.0
vllm:time_to_first_token_seconds_bucket{engine="0",le="+Inf",model_name="m1"} 200.0
vllm:inter_token_latency_seconds_sum{engine="0",model_name="m1"} 13.6
vllm:inter_token_latency_seconds_count{engine="0",model_name="m1"} 2000.0
vllm:prefix_cache_hits_total{engine="0",model_name="m1"} 89.0
vllm:prefix_cache_queries_total{engine="0",model_name="m1"} 100.0
vllm:request_success_total{engine="0",finished_reason="stop",model_name="m1"} 100.0
vllm:cache_config_info{block_size="16",engine="0",kv_cache_size_tokens="7000",num_gpu_blocks="500"} 1.0
http_requests_total{handler="/v1/chat/completions",method="POST",status="2xx"} 100.0
http_requests_total{handler="/v1/chat/completions",method="POST",status="4xx"} 6.0
http_requests_total{handler="none",method="GET",status="4xx"} 150.0
"""


class Resp:
    def __init__(self, text, status=200):
        self.text, self.status_code = text, status


class FakeDynamo(DynamoSource):
    def __init__(self):
        super().__init__(_conn("dynamo", "http://10.0.0.1:8000/metrics, https://10.0.0.2:8120/metrics"))

    def _send(self, method, url, path, **kw):
        return Resp(PAGE_A if "10.0.0.1" in url else PAGE_B)


B, V = FakeBcm, FakeDynamo
N = lambda: FakeBcm(nodes=NIC_BAD)
PCT = ["Endpoint", "p50", "p90", "p95", "p99", "Meets SLA (< 500 ms)?"]
CASES = [
    # B01 / B04: network checks as they are
    ("health_summary", B, T(["Node", "Status"], ["head", "UP"], ["gpu-a", "UP"], ["gpu-b", "UP"], ["node001", "DOWN"],
                            ["sw1", "DOWN"]), "PASS"),
    ("health_summary", B, T(["Node", "Status"], ["head", "UP"], ["node001", "UP"], ["sw1", "UP"]), "FAIL"),
    ("cpu_all", B, T(["Node", "CPU %"], ["head", "2%"], ["gpu-a", "40%"], ["gpu-b", "12%"]), "PASS"),
    ("cpu_all", B, T(["Node", "CPU %"], ["head", "2%"], ["gpu-a", "90%"], ["gpu-b", "12%"]), "FAIL"),
    # B03: node counters (2 / 1 / 1 / 0), or with the lite nodes (4 / 3 / 1 / 0)
    ("node_counts", B, "BCM manages 2 nodes in total: 1 up, 1 down and 0 closed.", "PASS"),
    ("node_counts", B, "4 nodes in total — 3 up, 1 down, none closed.", "PASS"),
    ("node_counts", B, T(["Status", "Count"], ["Total", 2], ["Up", 1], ["Down", 1], ["Closed", 0]), "PASS"),
    ("node_counts", B, T(["Total", "Up", "Down", "Closed"], [4, 3, 1, 0]), "PASS"),
    ("node_counts", B, "BCM manages 5 nodes: 3 up, 2 down and 0 closed.", "PASS"),       # every device (Dev, 2026-10-14)
    ("node_counts", B, "BCM manages 5 nodes: 4 up, 1 down.", "FAIL"),
    ("node_counts", B, "Total nodes: 4, Up: 1, Down: 1, Closed: 0.", "FAIL"),           # mixes the two ways of counting
    # B05: memory pressure
    ("node_mem_top", B, "Memory pressure by node:\n" + T(["Node", "Memory %"], ["head", "20%"], ["gpu-b", "29%"],
                                                         ["gpu-a", "77%"]) + "\ngpu-a is under the most memory pressure.", "PASS"),
    ("node_mem_top", B, "gpu-b has the highest memory usage (29 %).", "FAIL"),
    # B06: NIC errors / drops (only virtual drops on gpu-b)
    ("nic_errors", B, "No packet errors or drops on the physical NICs of the compute nodes.", "PASS"),
    ("nic_errors", B, "gpu-b shows small receive drops (0.033/s) on virtual gns3tap interfaces; gpu-a has none.", "PASS"),
    ("nic_errors", B, T(["Node", "RX drops", "RX errors"], ["gpu-a", 0, 0], ["gpu-b", "0.033/s", 0]), "PASS"),
    ("nic_errors", B, "gpu-a has 12 receive errors on eno1.", "FAIL"),
    ("nic_errors", B, "The compute nodes look fine.", "FAIL"),
    ("nic_errors", N, "**gpu-a**\n- eno1: 2 receive errors/s\n- drops: 0", "PASS"),
    ("nic_errors", N, "No errors or drops on any node.", "FAIL"),
    # V01: mean TTFT 19.5 s / 70 ms, ITL 19 / 6.8 ms (TPOT 21 ms); the model as a whole: weighted by requests
    # 6.547 s / 10.87 ms, or the plain mean of the endpoints 9.785 s / 12.9 ms (Dev, 2026-10-14)
    ("vllm_latency", V, T(["Endpoint", "TTFT (mean)", "ITL (mean)", "Throughput"], ["10.0.0.1:8000", "19.5 s", "19 ms", "120 tok/s"],
                          ["10.0.0.2:8120", "70 ms", "6.8 ms", "40 tok/s"]), "PASS"),
    ("vllm_latency", V, "For m1 across both endpoints: mean TTFT 9.8 s, ITL 13 ms.", "PASS"),           # plain mean
    ("vllm_latency", V, "For m1 across both endpoints: mean TTFT 6.5 s, ITL 10.9 ms.", "PASS"),         # weighted
    ("vllm_latency", V, "For m1 across both endpoints: mean TTFT 3 s, ITL 13 ms.", "FAIL"),
    ("vllm_latency", V, "### 10.0.0.1:8000\n- TTFT: 19.4 s\n- TPOT: 21 ms\n\n### 10.0.0.2:8120\n- TTFT: 71 ms\n- ITL: 7 ms",
     "PASS"),
    ("vllm_latency", V, T(["Endpoint", "TTFT p95", "TTFT avg (ms)"], ["10.0.0.1", "101 s", "19,600"]), "PASS"),  # p95 column skipped
    ("vllm_latency", V, T(["Endpoint", "TTFT (mean)", "ITL (mean)"], ["10.0.0.1:8000", "2 s", "19 ms"]), "FAIL"),
    # V02: hit rate 95 / 89 %, the model 92 %
    ("kv_hit_rate", V, T(["Endpoint", "Hit rate (%)"], ["10.0.0.1", "95.0"], ["10.0.0.2", "89"]), "PASS"),
    ("kv_hit_rate", V, "The overall KV-cache hit rate for m1 is 92 %.", "PASS"),
    ("kv_hit_rate", V, "10.0.0.1: 95 % hit rate; 10.0.0.2: 89 % hit rate", "PASS"),
    ("kv_hit_rate", V, T(["Endpoint", "Hit rate"], ["10.0.0.1", "75 %"], ["10.0.0.2", "89 %"]), "FAIL"),
    # V03
    ("ttft_worst", V, "The worst TTFT is on 10.0.0.1:8000 (mean 19.5 s); 10.0.0.2:8120 is at 70 ms.", "PASS"),
    ("ttft_worst", V, T(["Endpoint", "TTFT"], ["10.0.0.2", "70 ms"], ["10.0.0.1", "19.5 s"]) + "\n**Worst:** 10.0.0.1", "PASS"),
    ("ttft_worst", V, "10.0.0.2:8120 has the worst TTFT.", "FAIL"),
    # V04: buckets — A p50 100–250 ms … p99 160–640 s (fails the 500 ms SLA); B p50 0–40 ms … p99 250–500 ms (meets)
    ("ttft_percentiles", V, T(PCT, ["10.0.0.1", "150 ms", "14.8 s", "101 s", "578 s", "❌ No"],
                              ["10.0.0.2", "34 ms", "170 ms", "228 ms", "485 ms", "✅ Yes"]), "PASS"),
    ("ttft_percentiles", V, T(PCT, ["10.0.0.1", "150 ms", "14.8 s", "101 s", "578 s", "✅ Yes"],
                              ["10.0.0.2", "34 ms", "170 ms", "228 ms", "485 ms", "✅ Yes"]), "FAIL"),
    ("ttft_percentiles", V, T(PCT, ["10.0.0.1", "150 ms", "14.8 s", "101 s", "578 s", "❌ No"],
                              ["10.0.0.2", "34 ms", "170 ms", "228 ms", "900 ms", "✅ Yes"]), "FAIL"),
    ("ttft_percentiles", V, "### 10.0.0.2:8120\n- p50: 34 ms\n- p90: 170 ms\n- p95: 228 ms\n- p99: 485 ms\n"
                            "- Meets the 500 ms SLA ✅\n\n### 10.0.0.1:8000\n- p50: 150 ms\n- p90: 14.8 s\n- p95: 101 s\n"
                            "- p99: 578 s\n- Does not meet the SLA ❌", "PASS"),
    ("ttft_percentiles", V, T(PCT[:5], ["10.0.0.1", "150 ms", "14.8 s", "101 s", "578 s"], ["10.0.0.2", "34 ms", "170 ms",
                            "228 ms", "485 ms"]) + "\n10.0.0.2 meets the SLA, while 10.0.0.1 exceeds it.", "PASS"),
    # conv 1648 (NCP 10.4.5.10, 2026-10-14), endpoints renamed: thousands grouped with U+202F, bold cells, U+2011 hyphens
    ("ttft_percentiles", V, "**TTFT latency percentiles (ms) per vLLM endpoint \u2013 SLA = 500 ms (must be < 500 ms)**\n\n"
                            + T(["Endpoint (IP:port)", "Model", "TTFT p50", "TTFT p90", "TTFT p95", "TTFT p99", "Meets SLA?"],
                                ["10.0.0.1:8000", "gpt\u2011oss\u2011120b", "149 ms", "14\u202f706 ms", "101\u202f004 ms",
                                 "**578\u202f037 ms**", "**No**"],
                                ["10.0.0.2:8120", "gpt\u2011oss\u2011120b", "34 ms", "170 ms", "228 ms", "**485 ms**", "**Yes**"]),
     "PASS"),
    # V05
    ("vllm_requests", V, T(["Endpoint", "Running", "Waiting"], ["10.0.0.1", 1, 0], ["10.0.0.2", 0, 0]), "PASS"),
    ("vllm_requests", V, "10.0.0.1 has 1 running and 0 waiting, while 10.0.0.2 has no running requests and 0 waiting.",
     "PASS"),
    ("vllm_requests", V, T(["Endpoint", "Running", "Waiting"], ["10.0.0.1", 5, 0], ["10.0.0.2", 0, 0]), "FAIL"),
    # V06: A 99 % (chat) / 98 % (all) / 100 % (vLLM), 4xx 2 (chat 0), 5xx 1; B 94.3 / 39.1 / 100 %, 4xx 156 (chat 6), 5xx 0
    ("vllm_success", V, T(["Endpoint", "Success rate", "4xx", "5xx"], ["10.0.0.1", "99 %", 2, 1], ["10.0.0.2", "100 %", 156, 0]),
     "PASS"),
    ("vllm_success", V, "10.0.0.2: success rate 94.3 %, 6 requests ended in 4xx and no 5xx.", "PASS"),
    ("vllm_success", V, T(["Endpoint", "Success rate", "4xx", "5xx"], ["10.0.0.1", "99 %", 2, 0]), "FAIL"),
    # V07: in use 1.6 / 0 %; slots 1000 / 500 blocks = 16,000 / 8,000 tokens (B configured 7,000)
    ("kv_usage", V, T(["Endpoint", "KV cache in use (%)", "Total blocks"], ["10.0.0.1", "1.6", "1,000"],
                      ["10.0.0.2", "0", "500"]), "PASS"),
    ("kv_usage", V, "10.0.0.1: 1.6 % in use of 16K total token slots; 10.0.0.2: 0 % in use, 7,000 total tokens.", "PASS"),
    ("kv_usage", V, T(["Endpoint", "KV cache in use (%)", "Total blocks"], ["10.0.0.1", "45", "1,000"]), "FAIL"),
]


def run(check, src, answer, tol=None):
    sheet = {r.check: r for r in load_prompts(settings.BCM_DYNAMO_PROMPTS_XLSX)}
    row = sheet.get(check) or PromptRow("BX", "prompt", check)
    if tol is not None:
        row = PromptRow(row.id, row.prompt, check, row.param, tol)
    return evaluate(Ctx(row, src, answer, False, None, []))


@pytest.mark.parametrize("check, make, answer, expected", CASES, ids=[f"{c[0]}-{c[3]}-{i}" for i, c in enumerate(CASES)])
def test_case(check, make, answer, expected):
    v = run(check, make(), answer)
    assert v.status == expected, (v.reason, v.expected)


def test_every_check_has_a_passing_and_a_failing_answer():
    assert set(bcm_dynamo.CHECKS) <= set(CHECKS)
    assert set(bcm_dynamo.CHECKS) <= {c for c, _, _, e in CASES if e == "PASS"}
    assert set(bcm_dynamo.CHECKS) <= {c for c, _, _, e in CASES if e == "FAIL"}


# ---------------------------------------------------------------- truth parsing
def test_bcm_reads_raw_values_status_and_counters():
    src = FakeBcm()
    assert [(d.name, d.ip, d.healthy) for d in src.devices()][:3] == [("head", "", True), ("gpu-a", "10.0.0.1", True),
                                                                      ("gpu-b", "10.0.0.2", True)]
    assert src.devices()[3].reason == "DOWN"
    m = src._metrics()
    assert m["gpu-a"] == {"cpu": 40.0, "mem": 77.0, "mem_alt": [77.0], "nic_any": 0.0, "nic_bad": 0.0}
    assert (m["gpu-b"]["nic_any"], m["gpu-b"]["nic_bad"]) == (0.033, 0.0)           # gns3tap drops are virtual
    assert src.node_counts() == [{"total": 2, "up": 1, "down": 1, "closed": 0}, {"total": 4, "up": 3, "down": 1, "closed": 0},
                                 {"total": 5, "up": 3, "down": 2, "closed": 0}]
    assert [i.name for i in src.interfaces(src.devices()[1])] == ["eno1"]             # gpu-a's monitored NIC
    assert [i.name for i in src.interfaces(src.devices()[2])] == ["eno1", "gns3tap0-2"]


def test_bcm_without_client_certificate_blocks():
    with pytest.raises(NoTruth, match="BCM_CERT"):
        BcmSource(_conn("bcm", "https://bcm:8081", cert="")).ensure_login()
    with pytest.raises(NoTruth, match="not found"):
        BcmSource(_conn("bcm", "https://bcm:8081", cert="no-such.pem")).ensure_login()


def test_dynamo_parses_metrics_pages():
    src = FakeDynamo()
    assert src.endpoints == {"http://10.0.0.1:8000/metrics": "10.0.0.1", "https://10.0.0.2:8120/metrics": "10.0.0.2"}
    a, b = src._metrics()["10.0.0.1"], src._metrics()["10.0.0.2"]
    assert (a["running"], a["kv_pct"], a["ttft_ms"], a["itl_ms"], a["hit_pct"]) == (1, 1.6, 19500, 19, 95)
    assert (a["p50_lo"], a["p50_hi"], a["p99_lo"], a["p99_hi"]) == (100, 250, 160000, 640000)
    assert (a["ok_pct"], a["ok_pct_vllm"], a["http_4xx"], a["http_4xx_chat"], a["http_5xx"]) == (99, 100, 2, 0, 1)
    assert (a["blocks"], a["kv_tokens"], b["kv_tokens_cfg"]) == (1000, 16000, 7000)
    assert src.device_for_prompts().name == "m1"


def test_quantile_follows_histogram_quantile():
    buckets = [(0.1, 10.0), (0.25, 55.0), (float("inf"), 100.0)]
    assert quantile(0.5, buckets) == pytest.approx((0.1, 0.25, 0.2333), abs=1e-3)
    assert quantile(0.99, buckets) == (0.25, float("inf"), 0.25)                     # +Inf bucket: its lower bound
    assert quantile(0.5, [(1.0, 0.0), (float("inf"), 0.0)]) is None                  # no requests yet


def test_runner_fills_model_and_sends_the_dynamo_tag():
    from ncp_suite.chat import ChatResult
    from ncp_suite.runner import run_case

    class Chat:
        asked = []

        def ask(self, prompt, context=None, timeout=None):
            self.asked.append(prompt)
            return ChatResult(prompt, text="The overall KV-cache hit rate for m1 is 92 %.", seconds=1, conversation_id="5")
    row = next(r for r in load_prompts(settings.BCM_DYNAMO_PROMPTS_XLSX) if r.id == "V02")
    conn = _conn("dynamo", "x")
    r = run_case(row, conn, Chat(), FakeDynamo())
    assert Chat.asked == ["#dynamo Show KV-cache hit rate for m1."] and r.status == "PASS", r.reason
    assert "| 10.0.0.1 | m1 | 95 |" in r.source


def test_sheet_and_registry():
    rows = load_prompts(settings.BCM_DYNAMO_PROMPTS_XLSX)
    assert [r.id for r in rows] == [f"B0{i}" for i in range(1, 7)] + [f"V0{i}" for i in range(1, 8)]
    assert all(r.check in CHECKS for r in rows)
    assert {r.id[0]: r.applies_to for r in rows} == {"B": {"bcm"}, "V": {"dynamo"}}
    for key in ("bcm", "dynamo"):
        c = settings.CONNECTORS[key]
        assert (c.prompts, c.needs_login) == (settings.BCM_DYNAMO_PROMPTS_XLSX, False)
    assert settings.missing([_conn("bcm", "")], ncp=False) == ["BCM_URL"]               # no user / password needed


def test_source_views_show_the_compared_values():
    from ncp_suite.grading.source_view import source_view
    assert "| Nodes* + LiteNodes* | 4 | 3 | 1 | 0 |" in source_view("node_counts", FakeBcm())
    assert "| gpu-b | 0 | 0.033 |" in source_view("nic_errors", FakeBcm())
    assert "| 10.0.0.1 | m1 | 233.3 (100–250) |" in source_view("ttft_percentiles", FakeDynamo())


def test_sla_follow_up_is_answered_with_the_policy_band():
    from ncp_suite.chat.policy import followup_reply, is_followup
    q = ("Got it! To determine whether each vLLM endpoint meets its latency SLA, I need to know the target latency "
         "you'd like to use (e.g., \u201cp99 \u2264 200 ms\u201d or any other threshold). Could you let me know the SLA "
         "value you'd like applied?")                                                  # dynamo-V04, conv 1645
    assert is_followup(q)
    assert is_followup("Sure thing! What TTFT latency threshold should I use for the SLA (e.g., TTFT \u2264 200 ms or any "
                       "other value you\u2019d like to apply)?")                                  # conv 1647
    assert not is_followup("| Endpoint | p99 | SLA |\n|---|---|---|\n| a | 485 ms | meets |\n| b | 578 s | fails |\n"
                           "Want a different SLA?")                                             # a table is the answer
    assert followup_reply(q, {"connector": "Dynamo", "tag": "#dynamo-mcp"}).startswith("#dynamo-mcp Use a TTFT SLA of 500 ms")
    assert "500 ms" not in followup_reply("Which device would you like?", {"connector": "Zabbix", "tag": "#zabbix"})


def test_job_scope_question_is_a_follow_up():
    from ncp_suite.chat.policy import followup_reply, is_followup
    q = ("Sure! Could you tell me which compute host(s) or job\u2011scheduler you\u2019d like the status and allocation "
         "details for?")                                                                # bcm-B02, conv 1655
    assert is_followup(q)
    assert followup_reply(q, {"connector": "BCM", "tag": "#bcm"}) == "#bcm Yes, please go ahead for all devices using the BCM connector."


def test_b02_lists_the_interfaces_of_bcm_head():
    """B02 (Dev, 2026-10-14): "show me the network interfaces for bcm-head" — <DEVICE> pinned by DEVICE_BCM, graded by
    the network check interfaces_list on the NICs BCM monitors on that node."""
    from ncp_suite.runner import run_case
    from ncp_suite.chat import ChatResult
    row = next(r for r in load_prompts(settings.BCM_DYNAMO_PROMPTS_XLSX) if r.id == "B02")
    assert (row.prompt, row.check) == ("show me the network interfaces for <DEVICE>", "interfaces_list")
    nodes = {**NODES, "head": {**NODES["head"], "BytesRecv:ens34": 10.0, "BytesRecv:ens37": 5.0}}

    class Chat:
        def __init__(self, text):
            self.text, self.asked = text, []

        def ask(self, prompt, context=None, timeout=None):
            self.asked.append(prompt)
            return ChatResult(prompt, text=self.text, seconds=1, conversation_id="9")
    conn = _conn("bcm", "x", "head")                       # DEVICE_BCM pins the node
    good = Chat(T(["Interface", "Status"], ["ens34", "up"], ["ens37", "up"]))
    r = run_case(row, conn, good, FakeBcm(nodes, "head"))
    assert good.asked == ["#bcm show me the network interfaces for head"] and r.status == "PASS", r.reason
    assert "| ens37 |" in r.source
    r = run_case(row, conn, Chat("head has one interface: ens99."), FakeBcm(nodes, "head"))
    assert r.status == "FAIL", r.reason

