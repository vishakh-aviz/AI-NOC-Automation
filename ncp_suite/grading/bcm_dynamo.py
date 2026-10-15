"""Checks for the BCM and Dynamo (vLLM) prompts (data/bcm_dynamo_prompts.xlsx, Dev's sheet 2026-10-14).
Truth: truth/bcm.py, truth/dynamo.py. Two BCM rows use network checks as they are: B01 health_summary (every node
covered, every DOWN one flagged), B02 interfaces_list (every NIC of <DEVICE> = bcm-head named), B04 cpu_all (CPU %
per node, ±CPU_TOL).

Same rules as checks.py / gpu.py: numbers by code, PARTIAL_PASS for per-endpoint / per-node answers, a value is
compared with the source reads taken while NCP answered (runner.py window). Here NCP's value must lie between the
lowest and the highest read (± tolerance): NCP's tool read at some moment in between, and vLLM keeps no history.

An endpoint / node in an answer: a table row that names it; a row whose first cell names the metric, in the column
named after it; a clause that names it and no other one; a line under a line that names only it
('**10.4.5.33:8000**' then '- TTFT: 19.5 s'). A value: the column named after the metric, else the number right
after the metric word (or just before it: '6 4xx'). Latency needs its unit (ms / s; a column header unit counts),
a percentage its % (a bare number counts under a % header; a bare ratio ≤ 1 under a 'rate' header), counts may
carry k / M. Dynamo values are since-start (CLAUDE.md §8 formulas); the tolerances are the row's Tolerance or the
defaults below (TTFT / ITL ±10 %, rates ±2 points, success rate ±0.5 points, running / waiting ±1).
"""
from __future__ import annotations

import re

from ncp_suite import settings
from ncp_suite.grading.checks import Ctx, Verdict, _fail, _ok, _partial, cap
from ncp_suite.grading.compare import TIME_OR_DATE, first_position, low, mentions, row_text, says_none, table_rows
from ncp_suite.grading.gpu import SENTENCE
from ncp_suite.truth.base import NoTruth

# a number, its unit and k / M ("19.5 s", "150 ms", "99.97%", "176,035", "2.8M"); not glued to a name (eno1, node001)
QTY = re.compile(r"(?<![\w.\-/:])(\d+(?:,\d{3})*(?:\.\d+)?)\s*(ms|milliseconds?|µs|us|seconds?|secs?|s|%|[km])?(?![\w%])", re.I)
ADDRESS = re.compile(r"https?://\S+|\b\d{1,3}(?:\.\d{1,3}){3}(?::\d+)?\b")
# NCP groups thousands with a (narrow no-break) space: "14 706 ms" (dynamo-V04 conv 1648); not "p50 150"
SPACED = re.compile(r"(?<![\w.,])\d{1,3}(?: \d{3})+(?![\w.,]?\d)")
TO_MS = {"ms": 1, "millisecond": 1, "milliseconds": 1, "µs": .001, "us": .001, "s": 1000, "sec": 1000, "secs": 1000,
         "second": 1000, "seconds": 1000}
PERCENTILE = r"\bp\d\d\b|median|percentile|\d\dth"
SLA_MS = 500          # NCP's TTFT policy band: < 500 ms (Data Connectors CSV = Audit Report, CLAUDE.md §8)


# ---- reading values out of an answer -------------------------------------------------------------------
def _conv(x: float, unit: str, want: str, header: str) -> float | None:
    """x in the wanted unit — 'ms', '%' or '' (a count); None when the unit does not fit."""
    if want == "ms":
        unit = unit or next((u for u, p in (("ms", r"\bms\b|millisec"), ("s", r"\(s\)|\bsec|\bseconds?\b")) if re.search(p, header)), "")
        return x * TO_MS[unit] if unit in TO_MS else None
    if want == "%":
        if unit:
            return x if unit == "%" else None
        return x if re.search(r"%|percent", header) else x * 100 if x <= 1 and re.search(r"rate|ratio", header) else None
    return x * {"k": 1e3, "m": 1e6}.get(unit, 1) if unit in ("", "k", "m") else None


def _numbers(text: str, want: str, header: str = "") -> list[float]:
    """Every number in the text that fits the wanted unit, converted; IPs, URLs, dates and times left out."""
    s = SPACED.sub(lambda m: m.group().replace(" ", ""), TIME_OR_DATE.sub(" ", ADDRESS.sub(" ", low(text))))
    return [x for m in QTY.finditer(s)
            if (x := _conv(float(m.group(1).replace(",", "")), low(m.group(2) or ""), want, low(header))) is not None]


def _sections(text: str, names: list[str]) -> dict[str, list[str]]:
    """name -> the lines under a line that names only it (a heading, a bold label), up to the next line that names one
    of the names, or a heading that names none."""
    out: dict[str, list[str]] = {}
    cur = ""
    for line in (text or "").splitlines():
        named = [n for n in names if mentions(line, n)]
        if named and not line.lstrip().startswith("|"):
            cur = named[0] if len(named) == 1 else ""
        elif line.lstrip().startswith("#") or re.fullmatch(r"\s*\*\*[^*]+\*\*:?\s*", line):
            cur = ""
        elif cur and line.strip():
            out.setdefault(cur, []).append(line)
    return out


def _scopes(text: str, name: str, names: list[str]) -> list[str]:
    """Texts about one name (no other name): clauses of plain lines, then the lines of its section. name '' = the
    model / cluster as a whole: lines outside every section that name none of the names."""
    others = [n for n in names if n != name]
    about = (lambda t: mentions(t, name) and not any(mentions(t, o) for o in others)) if name else \
        (lambda t: not any(mentions(t, n) for n in names))
    plain = [ln for ln in (text or "").splitlines() if not ln.lstrip().startswith("|")]
    sections = _sections(text, names)
    if name:
        return [c for ln in plain for c in SENTENCE.split(ln) if about(c)] + sections.get(name, [])
    inside = {ln for lines in sections.values() for ln in lines}
    return [ln for ln in (text or "").splitlines() if ln not in inside and about(ln)]


def _value(text: str, name: str, names: list[str], words: tuple[str, ...], want: str, skip: str = "") -> float | None:
    """NCP's value for one endpoint / node (name; '' = the model as a whole). skip: a column / phrase that matches
    it names another statistic (a percentile, when the mean is wanted)."""
    has = lambda t: any(w in low(t) for w in words) and not (skip and re.search(skip, low(t)))
    others = [n for n in names if n != name]
    for row in table_rows(text):
        cells = list(row.items())
        t = row_text(row)
        if (mentions(t, name) if name else not any(mentions(t, n) for n in names)) and not any(mentions(t, o) for o in others):
            for col, cell in cells:
                if has(col) and (x := _numbers(cell, want, col)):
                    return x[0]
        if name and cells and has(cells[0][1]):               # metrics as rows, endpoints as columns
            for col, cell in cells[1:]:
                if mentions(col, name) and (x := _numbers(cell, want, cells[0][1])):
                    return x[0]
    for scope in _scopes(text, name, names):
        s = TIME_OR_DATE.sub(" ", ADDRESS.sub(" ", low(scope.replace("|", " "))))
        for w in words:
            for m in re.finditer(re.escape(w), s):
                after = re.split(r",\s|;", s[m.end():m.end() + 40])[0]
                if skip and re.match(rf"\W*(?:{skip})", after):
                    continue
                # the word and what follows it up to the first digit is the "header": "ttft (ms): 150"
                if x := _numbers(after, want, re.match(r"\D{0,20}", s[m.start():]).group()):
                    return x[0]
                before = s[max(0, m.start() - 25):m.start()]
                if re.search(r"\b(?:no|zero)\s+(?:\w+\s+)?$", before):
                    return 0.0
                if x := _numbers(before, want):
                    return x[-1]
    return None


# ---- per endpoint / node values ------------------------------------------------------------------------
def _samples(src, kind: str) -> dict[str, list[float]]:
    try:
        return src.metric_samples(kind)
    except NoTruth:
        return {}


def _between(got: float, vals: list[float], tol: float, rel: bool) -> bool:
    lo, hi = min(vals), max(vals)
    if rel:
        return lo - abs(lo) * tol / 100 - 0.5 <= got <= hi + abs(hi) * tol / 100 + 0.5
    return lo - tol <= got <= hi + tol


def _range(vals: list[float]) -> str:
    lo, hi = min(vals), max(vals)
    return f"{lo:.4g}" if lo == hi else f"{lo:.4g}–{hi:.4g}"


def _tol(ctx: Ctx, default: float) -> float:
    return ctx.row.tol if ctx.row.tol is not None else default


def _per_entity(ctx: Ctx, specs: list, combined: dict | None = None, skip: str = "") -> Verdict:
    """specs: (label, kinds, words, unit, tol, relative). A kind may be a tuple = one range (a bucket's bounds). Each
    endpoint's value lies between the lowest and highest read of one kind (± tol). combined: label -> accepted ways to
    compute the model as a whole (each a list of reads), compared with a value given without an endpoint."""
    groups = [[k if isinstance(k, tuple) else (k,) for k in kinds] for _, kinds, *_ in specs]
    reads = {k: _samples(ctx.src, k) for g in groups for ks in g for k in ks}
    names = sorted({n for r in reads.values() for n in r})
    if not names:
        raise NoTruth(f"{ctx.src.title}: none of {cap(sorted(reads))} in the source")
    problems, not_shown, shown, exp = [], [], 0, []
    for (label, _, words, unit, tol, rel), group in zip(specs, groups):
        targets = [(n, [v for ks in group if (v := [x for k in ks for x in reads[k].get(n, [])])]) for n in names]
        model = [v for v in (combined or {}).get(label, []) if v]
        targets += [("", model)] if model else []
        exp.append(f"{label} (±{tol:g}{'%' if rel else ''}): " + cap(f"{n or 'model'}={' or '.join(_range(v) for v in alts)}"
                                                                    for n, alts in targets if alts))
        for n, alts in targets:
            if not alts:
                continue
            got = _value(ctx.answer, n, names, words, unit, skip)
            if got is None:
                not_shown += [f"{n} {label}"] if n else []
            elif any(_between(got, v, tol, rel) for v in alts):
                shown += 1
            else:
                problems.append(f"{n or 'model'} {label}: NCP {got:g} vs source {' or '.join(_range(v) for v in alts)}")
    exp = "; ".join(exp)
    if problems:
        return _fail(ctx, f"{len(problems)} wrong: {cap(problems, 4)}", exp)
    if not shown:
        return _fail(ctx, f"no value found in the answer ({len(not_shown)} expected)", exp)
    if not_shown and settings.PARTIAL_PASS:
        return _partial(f"{shown} value(s) shown, all match", not_shown, exp)
    if not_shown:
        return _fail(ctx, f"not shown: {cap(not_shown)}", exp)
    return _ok(f"all {shown} values match", exp)


def _combined(src, num: str, den: str, scale: float) -> list[float]:
    """The model as a whole, per read: Σ num / Σ den over the endpoints (TTFT = Σ sum / Σ count)."""
    try:
        pairs = zip(src.metric_snapshots(num), src.metric_snapshots(den))
    except NoTruth:
        return []
    return [sum(a.values()) / sum(b.values()) * scale for a, b in pairs if sum(b.values())]


def _plain(src, kind: str) -> list[float]:
    """The model as a whole, per read: the plain mean of the endpoint values (NCP's "fleet-wide average", dynamo-V01
    conv 1654: (19 440 + 70.1) / 2 = 9 755 ms; accepted by Dev, 2026-10-14)."""
    try:
        return [sum(s.values()) / len(s) for s in src.metric_snapshots(kind) if s]
    except NoTruth:
        return []


def _top(ctx: Ctx, kinds: tuple[str, ...], words: str, tol: float, rel: bool, what: str) -> Verdict:
    """The first endpoint / node named — in a sentence with one of the words, else in the answer — is the top one by
    any of the kinds (ties within tol) at some read during the answer."""
    accepted: set[str] = set()
    snaps = {k: ctx.src.metric_snapshots(k) for k in kinds}
    for s in (s for ss in snaps.values() for s in ss):
        top = max(s.values())
        accepted |= {n for n, v in s.items() if v >= top - (abs(top) * tol / 100 if rel else tol)}
    last = snaps[kinds[0]][-1]
    exp = f"{what}: {cap(sorted(accepted))} — {kinds[0]}: " + cap(f"{n}={v:.4g}" for n, v in sorted(last.items(), key=lambda x: -x[1]))
    names = {n for ss in snaps.values() for s in ss for n in s}
    for scope in [s for s in re.split(r"(?<=[.!?])\s+|\n", ctx.answer) if re.search(words, low(s))] + [ctx.answer]:
        pos = {n: p for n in names if (p := first_position(scope, n)) is not None}
        if pos:
            first = min(pos, key=pos.get)
            return _ok(f"{first} named first", exp) if first in accepted else _fail(ctx, f"NCP named {first} first", exp)
    return _fail(ctx, "no endpoint / node named", exp)


# ---- BCM -----------------------------------------------------------------------------------------------
COUNT_WORDS = {"total": r"\btotal\b|\bmanage[sd]?\b|\bin all\b", "up": r"\bup\b|\bonline\b|\bhealthy\b",
               "down": r"\bdown\b|\boffline\b|\bunreachable\b", "closed": r"\bclosed\b"}


def _counts(text: str) -> dict[str, set[int]]:
    """key -> numbers NCP put next to the word ('Up: 1', '| Up | 1 |', '1 up', '35 nodes are down'; 'no closed
    nodes' = 0). A number on either side is kept; the check needs the truth among them."""
    t = low(re.sub(r"[*`_]", "", text or ""))
    lines = t.splitlines() + [" ; ".join(f"{c}: {v}" for c, v in r.items()) for r in table_rows(t)]
    out: dict[str, set[int]] = {}
    for line in (ADDRESS.sub(" ", ln) for ln in lines):
        for key, pattern in COUNT_WORDS.items():
            for m in re.finditer(pattern, line):
                found = [re.match(r"\s*(?:nodes?\s*)?(?:[:=|(]\s*|is\s+|are\s+|of\s+)?(\d+)\b", line[m.end():]),
                         re.search(r"(?<![\w.])(\d+)\s+(?:[a-z()]+\s+){0,2}$", line[max(0, m.start() - 30):m.start()])]
                out.setdefault(key, set()).update(int(f.group(1)) for f in found if f)
                if re.search(r"\b(?:no|zero|none)\s+(?:\w+\s+){0,2}$", line[max(0, m.start() - 20):m.start()]):
                    out[key].add(0)
    return {k: v for k, v in out.items() if v}


def node_counts(ctx: Ctx) -> Verdict:
    """'How many nodes … in total, and how many are up, down and closed?' Every count NCP gives equals one way of
    counting: BCM's node counters (NodesTotal / Up / Down / Closed), the same with the lite nodes added, or every
    device of /status (switches too — Dev, 2026-10-14)."""
    truths = ctx.src.node_counts()
    keys = ("total", "up", "down", "closed")
    exp = " or ".join(", ".join(f"{k} {t[k]:g}" for k in keys) for t in truths)
    stated = _counts(ctx.answer)
    if not stated:
        return _fail(ctx, "no node counts found in the answer", exp)
    for t in truths:
        if all(int(t[k]) in v for k, v in stated.items()):
            missing = [k for k in keys if k not in stated]
            if missing and settings.PARTIAL_PASS:
                return _partial(f"{len(stated)} of 4 counts stated, all right", missing, exp)
            return _fail(ctx, f"not stated: {cap(missing)}", exp) if missing else _ok("total, up, down and closed right", exp)
    said = ", ".join(f"{k} {'/'.join(map(str, sorted(v)))}" for k, v in stated.items())
    return _fail(ctx, f"NCP said {said}", exp)


def node_mem_top(ctx: Ctx) -> Verdict:
    """'Which node is under the most memory pressure?' The first node named is the top one by memory used % (±MEM_TOL)."""
    return _top(ctx, ("mem",), r"most|highest|pressure|top|largest|heaviest", _tol(ctx, settings.TOLERANCE["mem"]),
                False, "most memory used")


ERR = re.compile(r"drop|error|discard|\berrs?\b")
CLEAN = re.compile(r"\bno\b|\bzero\b|\bnone\b|\bclean\b|\bwithout\b|n/a|not (?:any|seen|observed|detected)")
NO_NIC = re.compile(r"\bno\b[^.\n]{0,40}(?:error|drop)|(?:error|drop)s?\b[^.\n]{0,30}\b(?:0|zero|none)\b")


def _nic_flagged(text: str, node: str, names: list[str]) -> bool:
    """NCP says the node has errors / drops: a non-zero number next to the word, or the word without a number and
    without 'no' / 'zero' (in a table: per column)."""
    def bad(t: str) -> bool:
        t = low(t)
        if not ERR.search(t):
            return False
        nums = _numbers(t, "")
        return any(x > 0 for x in nums) if nums else not CLEAN.search(t)
    pieces = [f"{c}: {v}" for r in table_rows(text) if mentions(row_text(r), node) for c, v in r.items()]
    return any(bad(p) for p in pieces + _scopes(text, node, names))


def nic_errors(ctx: Ctx) -> Verdict:
    """'Are there packet errors or drops on the compute nodes?' Every compute node (not the head node or a switch)
    with errors / drops on a physical NIC at every read is flagged; a node flagged must have errors / drops on some
    NIC (virtual ones too) at some read; none: NCP says so."""
    bad, any_nic = _samples(ctx.src, "nic_bad"), _samples(ctx.src, "nic_any")
    if not any_nic:
        raise NoTruth(f"{ctx.src.title}: no Errors* / Drop* NIC rates in /monitoring/latest")
    compute = {d.name for d in ctx.src.devices() if d.platform not in ("HeadNode", "Switch")}
    must = sorted(n for n, v in bad.items() if min(v) > 0 and n in compute)
    may = {n for n, v in any_nic.items() if max(v) > 0}
    exp = (f"errors / drops on physical NICs: {cap(must) or 'none'}; on virtual NICs only (gns3tap, wg, virbr …): "
           f"{cap(sorted(may - set(must))) or 'none'}")
    names = sorted(set(any_nic) | {d.name for d in ctx.src.devices()})
    flagged = [n for n in names if _nic_flagged(ctx.answer, n, names)]
    invented = [n for n in flagged if n not in may]
    if invented:
        return _fail(ctx, f"errors / drops claimed on nodes the source shows clean: {cap(invented)}", exp)
    missing = [n for n in must if n not in flagged]
    if missing and settings.PARTIAL_PASS and len(missing) < len(must):
        return _partial(f"{len(must) - len(missing)} of {len(must)} nodes with NIC errors / drops flagged", missing, exp)
    if missing:
        return _fail(ctx, f"not flagged: {cap(missing)}", exp)
    if must or flagged:
        return _ok(f"{len(flagged)} node(s) flagged, all with errors / drops in the source", exp)
    if NO_NIC.search(low(ctx.answer)) or says_none(ctx.answer):
        return _ok("no errors / drops on physical NICs; NCP said so", exp)
    return _fail(ctx, "no errors / drops on physical NICs, NCP did not say so", exp)


# ---- Dynamo (vLLM) ----------------------------------------------------------------------------------------
TTFT = ("ttft", "first token", "first-token")
ITL = ("itl", "inter-token", "inter token", "per output token", "tpot")
HIT = ("hit rate", "hit ratio", "hit-rate", "hit %", "hit (%)", "hit%")
PCT_WORDS = {50: ("p50", "50th", "median"), 90: ("p90", "90th"), 95: ("p95", "95th"), 99: ("p99", "99th")}


def vllm_latency(ctx: Ctx) -> Verdict:
    """'Show inference throughput and latency (TTFT, ITL) for <MODEL>.' Mean TTFT and ITL (or TPOT) per endpoint or
    for the model as a whole (weighted by requests, or the plain mean of the endpoints), ±10 %. Throughput is a rate
    over a window NCP picks: shown or not, never compared."""
    tol = _tol(ctx, 10)
    v = _per_entity(ctx, [("TTFT mean ms", ("ttft_ms",), TTFT, "ms", tol, True),
                          ("ITL ms", ("itl_ms", "tpot_ms"), ITL, "ms", tol, True)],
                    {"TTFT mean ms": [_combined(ctx.src, "ttft_sum", "ttft_count", 1000), _plain(ctx.src, "ttft_ms")],
                     "ITL ms": [_combined(ctx.src, "itl_sum", "itl_count", 1000), _plain(ctx.src, "itl_ms")]},
                    skip=PERCENTILE)
    shown = re.search(r"tok(?:en)?s?\s*/\s*s|tokens? per second|throughput[^.\n]{0,40}\d", low(ctx.answer))
    return Verdict(v.status, f"{v.reason}; throughput {'shown' if shown else 'not shown'} (not compared)", v.expected)


def kv_hit_rate(ctx: Ctx) -> Verdict:
    """'Show KV-cache hit rate for <MODEL>.' Prefix-cache hits / queries per endpoint or for the model, ±2 points."""
    return _per_entity(ctx, [("KV-cache hit %", ("hit_pct",), HIT, "%", _tol(ctx, 2), False)],
                       {"KV-cache hit %": [_combined(ctx.src, "hits", "queries", 100)]})


def ttft_worst(ctx: Ctx) -> Verdict:
    """'Which vLLM endpoint has the worst TTFT?' The first endpoint named is the worst by mean, p95 or p99 (±10 %)."""
    return _top(ctx, ("ttft_ms", "p95", "p99"), r"worst|highest|slowest|longest|poorest", _tol(ctx, 10), True, "worst TTFT")


FAILS = re.compile(r"does(?:n't| not) meet|\bnot met\b|\bfail|violat|breach|exceed|❌|\bno\b|\bmiss|non-compliant|"
                   r"out of sla|outside|above (?:the )?(?:sla|target|threshold|500)")
MEETS = re.compile(r"\bmeets?\b|\bmet\b|within|✅|\bpass|\byes\b|compliant|\bok\b|(?:below|under) (?:the )?(?:sla|target|threshold|500)")
STILL_MEETS = re.compile(r"does(?:n't| not) (?:exceed|violate|breach|miss)|no (?:sla )?(?:violation|breach)")


def _sla_said(text: str, name: str, names: list[str]) -> bool | None:
    """NCP's SLA verdict for one endpoint: True meets, False fails, None not stated."""
    rows = [row_text(r) for r in table_rows(text) if mentions(row_text(r), name)
            and not any(mentions(row_text(r), o) for o in names if o != name)]
    for t in (low(t) for t in rows + _scopes(text, name, names)):
        if STILL_MEETS.search(t):
            return True
        if FAILS.search(t):
            return False
        if MEETS.search(t):
            return True
    return None


def ttft_percentiles(ctx: Ctx) -> Verdict:
    """'TTFT p50, p90, p95 and p99 for each vLLM endpoint and whether each meets the latency SLA.' Each percentile
    inside its histogram bucket (±10 %) at some read; the SLA verdict (TTFT < 500 ms) right for every endpoint whose mean,
    p90, p95 and p99 are all on one side of 500 ms (others: either verdict)."""
    tol = _tol(ctx, 10)
    v = _per_entity(ctx, [(f"p{q} ms", ((f"p{q}_lo", f"p{q}_hi"),), PCT_WORDS[q], "ms", tol, True) for q in PCT_WORDS])
    names = sorted(_samples(ctx.src, "ttft_ms"))
    sides, wrong, unstated = {}, [], []
    for n in names:
        hi = [x for k in ("p90_hi", "p95_hi", "p99_hi") for x in _samples(ctx.src, k).get(n, [])]
        lo = [x for k in ("p90_lo", "p95_lo", "p99_lo") for x in _samples(ctx.src, k).get(n, [])]
        mean = _samples(ctx.src, "ttft_ms").get(n, [])
        sides[n] = True if hi and max(hi + mean) <= SLA_MS else False if lo and min(lo + mean) >= SLA_MS else None
        said = _sla_said(ctx.answer, n, names)
        if said is None:
            unstated.append(n)
        elif sides[n] is not None and said != sides[n]:
            wrong.append(f"{n}: NCP says {'meets' if said else 'fails'}, source {'meets' if sides[n] else 'fails'}")
    exp = f"{v.expected}; SLA (TTFT < {SLA_MS} ms): " + cap(
        f"{n} {'meets' if s else 'fails' if s is False else 'either'}" for n, s in sides.items())
    if v.status != "PASS":
        return Verdict(v.status, v.reason, exp)
    if wrong:
        return _fail(ctx, f"percentiles right, SLA verdict wrong: {cap(wrong)}", exp)
    if unstated and not (settings.PARTIAL_PASS and len(unstated) < len(names)):
        return _fail(ctx, f"percentiles right; no SLA verdict for {cap(unstated)}", exp)
    note = f"; no SLA verdict for {cap(unstated)}" if unstated else "; SLA verdicts right"
    return Verdict("PASS", v.reason + note, exp)


def vllm_requests(ctx: Ctx) -> Verdict:
    """'How many requests are running and waiting on each vLLM endpoint right now?' ±1 of a read during the answer
    (NCP's own chat runs on 10.4.5.33)."""
    tol = _tol(ctx, 1)
    return _per_entity(ctx, [("running", ("running",), ("running", "in flight", "in-flight", "active"), "", tol, False),
                             ("waiting", ("waiting",), ("waiting", "queued", "pending", "in queue"), "", tol, False)])


def vllm_success(ctx: Ctx) -> Verdict:
    """'Request success rate on each vLLM endpoint, and how many requests ended in a 4xx or 5xx?' Success % (2xx of
    chat calls, of all HTTP calls, or vLLM finished without abort / error) ±0.5 points; 4xx / 5xx (all handlers or chat
    calls) between the reads."""
    tol = _tol(ctx, 0.5)
    return _per_entity(ctx, [("success %", ("ok_pct", "ok_pct_all", "ok_pct_vllm"), ("success",), "%", tol, False),
                             ("4xx", ("http_4xx", "http_4xx_chat"), ("4xx",), "", 0, False),
                             ("5xx", ("http_5xx", "http_5xx_chat"), ("5xx",), "", 0, False)])


def kv_usage(ctx: Ctx) -> Verdict:
    """'What percentage of the KV cache is in use on each vLLM endpoint, and how many total slots does each have?'
    In use % ±2 points; total slots = GPU KV blocks or tokens (blocks × block size, or the configured size), ±1 %."""
    return _per_entity(ctx, [("KV cache in use %", ("kv_pct",), ("in use", "usage", "used", "utiliz"), "%", _tol(ctx, 2), False),
                             ("total KV slots", ("blocks", "kv_tokens", "kv_tokens_cfg"), ("total", "slot", "capacity"), "", 1, True)])


CHECKS = {f.__name__: f for f in (
    node_counts, node_mem_top, nic_errors,
    vllm_latency, kv_hit_rate, ttft_worst, ttft_percentiles, vllm_requests, vllm_success, kv_usage,
)}
# read during the answer (runner.py sampling window); node_counts reads once
METRIC_CHECKS = {"node_mem_top", "nic_errors", "vllm_latency", "kv_hit_rate", "ttft_worst", "ttft_percentiles",
                 "vllm_requests", "vllm_success", "kv_usage"}
