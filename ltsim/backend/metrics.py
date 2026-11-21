"""B4 指标（backend.md §6）：节点时刻、资源利用率、流占用、每个 LTS-IR 算子的起止。"""
from __future__ import annotations

from fractions import Fraction as F
from typing import Mapping

from ..graph.tile_graph import TileGraph
from ..machine.model import Resource
from .chains import ChainGraph
from .scheduler import Placed


def node_times(cg: ChainGraph, placed: Placed) -> dict[str, tuple[int, int, int]]:
    """节点 → (start, issue_end, done)（设计 §3.5）：
    start = T_c + o_n；issue_end = start + D_n；done = start + L_n + D_n + skew_n。"""
    out = {}
    for c in cg.chains:
        t = placed.start[c.id]
        for n in c.nodes:
            w = cg.works[n]
            s = t + c.offset[n]
            out[n] = (s, s + c.D[n], s + w.L + c.D[n] + w.skew)
    return out


def resource_usage(cg: ChainGraph, placed: Placed, resources: Mapping[str, Resource]) -> dict[str, dict]:
    """每个资源（含没被用到的）：busy = 占用量 > 0 的拍数；area = Σ 用量 × 时长；peak = 逐拍峰值；
    utilization = area / (容量 × total)（独占资源即 busy / total）；avg = area / total（时间加权平均占用）。"""
    ivs: dict[str, list[tuple[int, int, int]]] = {name: [] for name in resources}
    for c in cg.chains:
        t = placed.start[c.id]
        for r in c.reservations:
            ivs[r.resource].append((t + r.start, t + r.end, r.amount))
    out = {}
    for name, res in sorted(resources.items()):
        ev = sorted([(s, 1, k) for s, e, k in ivs[name]] + [(e, 0, -k) for s, e, k in ivs[name]])
        cur = peak = busy = 0
        prev = None
        for t, _order, d in ev:
            if prev is not None and cur > 0:
                busy += t - prev
            cur += d
            peak = max(peak, cur)
            prev = t
        area = sum((e - s) * k for s, e, k in ivs[name])
        out[name] = {"kind": res.kind, "cap": res.cap, "busy": busy, "area": area, "peak": peak,
                     "utilization": F(area, res.cap * placed.total), "avg": F(area, placed.total)}
    return out


def op_spans(g: TileGraph, cg: ChainGraph, placed: Placed) -> dict[str, tuple[int, int]]:
    """每个 LTS-IR 算子 → (最早 start, 最晚 done)，按节点的 origin 聚合；顺序按算子在节点里第一次出现。"""
    times = node_times(cg, placed)
    spans: dict[str, tuple[int, int]] = {}
    for n in g.nodes:
        s, _ie, d = times[n.id]
        for op in n.origin:
            a, b = spans.get(op, (s, d))
            spans[op] = (min(a, s), max(b, d))
    return spans
