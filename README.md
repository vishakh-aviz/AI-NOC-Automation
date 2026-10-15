# NCP prompt validation (pytest)

Sends chat prompts to NCP for each connector, reads the right answer straight from the
connector's own system, compares the two, and fills the same matrix as the test sheet:
`Prompt | Nexus Dashboard (Local MCP) | Catalyst Center (Local MCP) | Zabbix | ONES | Prometheus (Local MCP) | DCGM | BCM | Dynamo | Comments`.
Prometheus and DCGM (GPU metrics) were added on 2026-10-08, BCM and Dynamo (vLLM) on 2026-10-14; a prompt that is
not in a connector's sheet shows "—".

## Setup (once)

```bash
cd "AI-NOC Automation/AI-NOC-AUTOMATION-2026/AI-NOC-Automation"
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Open `.env` and fill in / check:
- `NCP_PASSWORD` (`NCP_HOST` = 10.4.5.10, the box with all eight connectors)
- `TAG_NEXUS`, `TAG_CATALYST`, `TAG_ZABBIX`, `TAG_ONES`, `TAG_PROMETHEUS`, `TAG_DCGM`, `TAG_BCM`, `TAG_DYNAMO` — the
  `#tag` of each connector in NCP (`#prometheus-mcp`, `#dcgm`, `#bcm`, `#dynamo-mcp` on 10.4.5.10)
- `PROMETHEUS_URL` / `DCGM_URL` (the Prometheus the DCGM exporters feed; Prometheus user / password optional)
- `BCM_URL` + `BCM_CERT` (the client-certificate PEM, e.g. `bcm-client-combined.pem` in this folder — git-ignored);
  `DYNAMO_URL` = the vLLM `/metrics` URLs, comma separated
- `NCP_WS_URI` only if the chat socket is not `wss://<host>/api/v1/ws`

Every lab address, user, password and tag lives in `.env` only. A live run checks them first
and stops in under a second, naming any blank key.

Run from a machine that can reach NCP and the eight sources.

## Run

| Step | Command | What you get | Time (2026-10-06) |
|---|---|---|---|
| Self-test (no network) | `pytest` | checks the grading code itself | ~5 s |
| 1. Check the sources | `pytest test_sources.py` | `reports/snapshots/<connector>.json` + one summary line per connector | ~8 s |
| 2. Run the prompts | `pytest test_main.py` | `reports/NCP_MCP_Prompt_Results_<time>.html` + `.xlsx` (same name) | see CLAUDE.md §3.13 |
| One connector | `pytest test_main.py --connectors zabbix` (or `-k zabbix`) | | |
| Some prompts | `pytest test_main.py --prompts P01,P02` (GPU: `G06`) | | |
| The GPU connectors only | `pytest test_main.py --connectors prometheus,dcgm` | | |
| BCM, Dynamo and DCGM only | `pytest test_main.py --connectors bcm,dynamo,dcgm` (35 tests) | | |
| One at a time, live logs | add `-n 0` | | |

Runs use 8 worker processes (`-n 8 --dist loadgroup` in `pytest.ini`): each connector runs on
its own worker, one prompt at a time, so the eight connectors are tested side by side.

## The HTML report

Open `reports/NCP_MCP_Prompt_Results_<time>.html` in a browser. Each run gets its own file, and
it is rewritten after every test, so you can watch a long run.

- Top (at the end of the run): results per connector and the **result matrix** —
  `Prompt | Nexus … | ONES | Prometheus (Local MCP) | DCGM | Comments` ("—" = not in that connector's sheet).
- Below: one row per prompt × connector (Connector, Prompt, NCP result, Reason). Click a row to
  see the prompt sent, NCP's full answer **side by side with the source data the check compared**,
  the follow-ups, the tools NCP called and any retries.
- NA and BLOCKED show as "Skipped" in pytest's own column; the "NCP result" column has the real status.

The `.xlsx` of the same run has the sheets Summary, Matrix, Details (every test) and **Failures**
(FAIL / XFAIL only, with the conversation id, tools called, retries and NCP's answer — start there).

## Results

- **PASS** — the answer matches the source.
- **FAIL** — wrong, missing or invented data. The reason says which.
- **NA** — the system has no such data (e.g. Zabbix links) and NCP said so.
- **BLOCKED** — the right answer could not be read from the source. Not an NCP result.
- **XFAIL** — failed, but the row has a `Known_Issue` in the prompt sheet.

## Change the prompts

Each connector reads its own sheet (sheet `prompts`; the `how_to_edit` sheet explains each column):
`data/mcp_prompts.xlsx` for Nexus, Catalyst, Zabbix, ONES (P01–P20) and `data/gpu_prompts.xlsx` for
Prometheus and DCGM (G01–G16; G17–G22 DCGM only) and `data/bcm_dynamo_prompts.xlsx` for BCM (B01–B06) and
Dynamo (V01–V07; `<MODEL>` = the model the vLLM endpoints serve). The DCGM connector's supported metrics are in
`data/dcgm_supported_metrics.csv`.
`--excel <file>` runs one sheet for every selected connector.
An optional `Timeout` column (seconds) overrides the wait time for one prompt.

## Files

| File | Job |
|---|---|
| `test_main.py` | one test per prompt × connector (thin: runs `ncp_suite/runner.py`) |
| `test_sources.py` | reads each system directly and saves a snapshot |
| `conftest.py` | parametrisation and fixtures |
| `ncp_suite/settings.py`, `.env` | every input and setting (`.env` is git-ignored) |
| `ncp_suite/chat/` | NCP chat over WebSocket: client, answer stream, follow-up policy |
| `ncp_suite/truth/` | read-only clients for Nexus, Catalyst, Zabbix, ONES, Prometheus + DCGM (`prometheus.py`), BCM, Dynamo |
| `ncp_suite/grading/` | how each prompt is graded (`checks.py`; GPU prompts `gpu.py`; BCM / Dynamo `bcm_dynamo.py`), text helpers (`compare.py`) |
| `ncp_suite/runner.py` | one prompt × one connector, end to end |
| `ncp_suite/reporting/` | the Excel workbook and the HTML report parts |
| `ncp_suite/pytest_plugin.py` | options, pre-flight check, result collection, reports |
