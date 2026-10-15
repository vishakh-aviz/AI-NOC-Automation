"""BCM (NVIDIA Base Command Manager) ground truth: its REST API, read-only, mutual TLS.

https://10.20.13.243:8081/rest/v1 (BCM 11.0, probed 2026-10-14). The client certificate and key are one PEM file
(BCM_CERT in .env, CN ainoc-readonly) — the same one NCP's BCM connector uses. No user / password.
  /status                every device: hostname, status (UP / DOWN / CLOSED …), type (HeadNode, PhysicalNode,
                         LiteNode, Switch)
  /device                hostname, ip, type, category, roles
  /monitoring/latest     {"data": [{entity, measurable, raw, value, age}]} — parse `raw`, never `value` (display text,
                         CLAUDE.md §8). CPUUsage / MemoryUtilization are fractions (0.37 = 37 %); Drop* / Errors* are
                         per-NIC rates per second. Entity "base" holds the cluster counters NodesTotal / Up / Down / Closed.
  NICs of a node         the <metric>:<nic> measurables of /monitoring/latest (ens34 / ens37 on bcm-head — LLDP shows
                         the same two). InterfaceState is 0 / 1; its meaning is not confirmed, so no oper status is set.
  /workload/job-status   Slurm jobs (0 on 2026-10-14) — not used: NCP's BCM connector has no job tool, so Dev replaced
                         the job prompt with the bcm-head interfaces (2026-10-14). /workload/node-status is 401 here.

Seen 2026-10-14: 41 devices (35 physical nodes DOWN, 4 lite nodes of which 3 UP, the head node UP, 1 switch DOWN);
NodesTotal 36 / Up 1 / Down 35 / Closed 0 and LiteNodes 4 / 3 / 1 / 0; the only non-zero drops are
on hgx-su00-v100's virtual gns3tap* NICs; SN2100-6015's data is ~100 days old (age).
"""
from __future__ import annotations

import re
from pathlib import Path

from ncp_suite.settings import ROOT
from ncp_suite.truth.base import Device, Interface, NoTruth, Source, as_list, health_of, low, num

NIC_RATE = re.compile(r"^(?:Errors|Drop)(?:Recv|Sent):(.+)$|^FrameErrors:(.+)$")
# virtual NICs are left out of the NIC-error truth (CLAUDE.md §8): GNS3 taps, WireGuard, libvirt, bridges
NIC_METRIC = re.compile(r"^(?:(?:Bytes|Packets|Drop|Errors)(?:Recv|Sent)|FrameErrors|InterfaceState):(.+)$")
VIRTUAL_NIC = re.compile(r"^(?:gns3tap|wg\d|virbr|vnet|veth|docker|br\d|bridge|dummy|lo$)", re.I)
COUNTS = ("total", "up", "down", "closed")


class BcmSource(Source):
    """BCM (API connector)."""

    def login(self) -> None:
        cert = self.conn.extra.get("cert", "")
        path = Path(cert) if Path(cert).is_absolute() else ROOT / cert
        if not cert or not path.is_file():
            raise NoTruth(f"client certificate (BCM_CERT in .env) {'not set' if not cert else f'not found: {path}'}")
        self.http.cert = str(path)

    def latest(self) -> list[dict]:
        return as_list(self.get("/rest/v1/monitoring/latest"))

    def _devices(self) -> list[Device]:
        ips = {d.get("hostname"): d.get("ip", "") for d in as_list(self.get("/rest/v1/device"))}
        return [Device(s.get("hostname", ""), ips.get(s.get("hostname"), ""), s.get("system_name", ""),
                       platform=s.get("type", ""), status=s.get("status", ""), healthy=health_of(s.get("status")),
                       reason="" if health_of(s.get("status")) else s.get("status", ""))
                for s in as_list(self.get("/rest/v1/status"))]

    def _metrics(self) -> dict[str, dict]:
        """node -> cpu %, mem % (mem_alt: used / total %), nic_bad / nic_any (errors + drops per s on its physical /
        on all NICs). One /monitoring/latest read = one sample of the runner's window."""
        out: dict[str, dict] = {}
        used: dict[str, float] = {}
        for m in self.latest():
            node, name, raw = m.get("entity", ""), m.get("measurable", ""), num(m.get("raw"))
            if raw is None:
                continue
            v = out.setdefault(node, {})
            if name == "CPUUsage":
                v["cpu"] = round(raw * 100, 2)
            elif name == "MemoryUtilization":
                v["mem"] = round(raw * 100, 2)
            elif name == "MemoryUsed":
                used[node] = raw
            elif name == "MemoryTotal" and raw:
                v["mem_total"] = raw
            elif nic := NIC_RATE.match(name):
                v["nic_any"] = v.get("nic_any", 0) + raw
                v["nic_bad"] = v.get("nic_bad", 0) + (0 if VIRTUAL_NIC.match(nic.group(1) or nic.group(2)) else raw)
        for node, v in out.items():
            if (total := v.pop("mem_total", None)) and node in used:
                v["mem_alt"] = [round(used[node] / total * 100, 2)]
        out = {n: v for n, v in out.items() if v}
        if not out:
            raise NoTruth(f"{self.title}: no CPUUsage / MemoryUtilization in /monitoring/latest")
        return out

    def _interfaces(self, dev: Device) -> list[Interface]:
        """The NICs BCM monitors on the node (BytesRecv:ens34, …), no oper status (see the module notes)."""
        names = {m.group(1) for x in self.latest()
                 if x.get("entity") == dev.name and (m := NIC_METRIC.match(x.get("measurable", "")))}
        return [Interface(dev.name, n) for n in sorted(names)]

    def node_counts(self) -> list[dict[str, float]]:
        """Accepted (total, up, down, closed): BCM's node counters; the same with the lite nodes added (= the node rows
        of /status: 40 = 36 + 4 on 2026-10-14); every device of /status, switches too (41 — NCP's BCM tool counts these;
        accepted by Dev, 2026-10-14)."""
        base = {m.get("measurable"): num(m.get("raw")) for m in self.latest() if m.get("entity") == "base"}
        nodes = {k: base.get(f"Nodes{k.title()}") for k in COUNTS}
        if None in nodes.values():
            raise NoTruth(f"{self.title}: NodesTotal / Up / Down / Closed missing in /monitoring/latest (entity base)")
        status = [low(d.status) for d in self.devices()]
        return [nodes, {k: nodes[k] + (base.get(f"LiteNodes{k.title()}") or 0) for k in COUNTS},
                {"total": len(status), **{k: status.count(k) for k in COUNTS[1:]}}]

    # ---- report and probe -------------------------------------------------------------------------
    def view(self, check: str, param: float | None = None):
        """The source data a BCM check compared: (header, rows) or a sentence. None = the generic view (source_view.py)."""
        if check == "node_counts":
            counts = self.node_counts()
            return ["Counted", "Total", "Up", "Down", "Closed"], [
                [label] + [f"{c[k]:g}" for k in COUNTS]
                for label, c in zip(("Nodes*", "Nodes* + LiteNodes*", "All devices (/status)"), counts)]
        if check in ("nic_errors", "node_mem_top"):
            kinds = ("nic_bad", "nic_any") if check == "nic_errors" else ("mem",)
            snaps = {k: self.metric_samples(k) for k in kinds}
            head = {"nic_bad": "Errors + drops /s (physical NICs)", "nic_any": "Errors + drops /s (all NICs)", "mem": "Memory %"}
            return ["Node"] + [head[k] for k in kinds], [
                [n] + [", ".join(f"{x:.3g}" for x in snaps[k].get(n, [])) or "—" for k in kinds] for n in sorted(snaps[kinds[0]])]
        return None

    def snapshot(self) -> dict:
        out: dict = {"connector": self.title, "base": self.base, "kinds": {}}
        grab = lambda kind, fn: self._grab(out, kind, fn)
        grab("devices", self.devices)
        for kind in ("cpu", "mem", "nic_bad", "nic_any"):
            grab(kind, lambda k=kind: self.metric(k))
        grab("node counts", self.node_counts)
        try:
            out["device_for_prompts"] = self.device_for_prompts().name
            grab("interfaces", lambda: self.interfaces(self.device_for_prompts()))
        except NoTruth as exc:
            out["kinds"]["interfaces"] = {"status": "NO_TRUTH", "detail": str(exc)}
        out["raw"] = self.raw
        return out
