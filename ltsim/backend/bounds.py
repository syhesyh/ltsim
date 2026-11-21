"""B4 下界与关键路径（backend.md §6，设计 §7.4）。

下界只对这组决策、这个模型成立，不是硬件的极限；它和 MXM roofline 只作内部诊断，对外只报 tile 估计（总周期）。
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as F
from math import ceil
from typing import Mapping

from ..errors import LTSError
from ..graph.load import tile_lanes
from ..graph.tile_graph import TileGraph
from ..machine.model import Machine
from .chains import ChainGraph
from .scheduler import BoundBy, Placed, _kahn


@dataclass(frozen=True)
class Bounds:
    total: int                     # max(T_c + len(c))
    path: int                      # LB_path
    resource: int                  # ceil(LB_res)
    resource_exact: F              # LB_res 的精确值：Σ 用量 × 时长 / 容量
    resource_name: str             # 取到 LB_res 的资源
    lb: int                        # max(LB_path, ceil(LB_res))
    gap: F                         # total / LB − 1
    earliest: Mapping[str, int]    # ES_c：不考虑资源时每条链的最早起点


@dataclass(frozen=True)
class CriticalStep:
    chain: str
    bound_by: BoundBy              # 这条链为什么不能再早开始


def compute_bounds(cg: ChainGraph, placed: Placed) -> Bounds:
    """资源表取 cg.resources。"""
    resources = cg.resources
    length = {c.id: c.length for c in cg.chains}
    # LB_path：链间约束图上从虚拟源（T ≥ 0）出发的最长路径，与 B3 用同一组边权
    es = {c: 0 for c in length}
    out: dict[str, list] = {c: [] for c in length}
    for e in cg.edges:
        out[e.src].append(e)
    for c in _kahn(cg):
        for e in out[c]:
            es[e.dst] = max(es[e.dst], es[c] + e.weight)
    path = max(es[c] + length[c] for c in length)
    # LB_res：逐资源 Σ 用量 × 时长 / 容量；平局取资源名在前的
    area: dict[str, int] = {}
    for c in cg.chains:
        for r in c.reservations:
            area[r.resource] = area.get(r.resource, 0) + r.amount * (r.end - r.start)
    per = {name: F(a, resources[name].cap) for name, a in area.items()}
    name = min(per, key=lambda k: (-per[k], k))
    lb = max(path, ceil(per[name]))
    return Bounds(total=placed.total, path=path, resource=ceil(per[name]), resource_exact=per[name],
                  resource_name=name, lb=lb, gap=F(placed.total, lb) - 1, earliest=es)


def critical_path(cg: ChainGraph, placed: Placed) -> tuple[CriticalStep, ...]:
    """从最后结束的链（平局取链 id 序号小的）出发沿 bound_by 回溯到 blocker：
    edge 的 blocker 是边的源链，resource 的是占着资源的链，policy 的是前一条已放的链；source 停止。
    返回从起点到终点的顺序。每一步都回到更早放置的链，所以必然终止；万一重复访问报 LTS-E-INTERNAL。"""
    idx = {c.id: i for i, c in enumerate(cg.chains)}
    length = {c.id: c.length for c in cg.chains}
    c = min(length, key=lambda x: (-(placed.start[x] + length[x]), idx[x]))
    steps, seen = [], set()
    while c is not None:
        if c in seen:
            raise LTSError("LTS-E-INTERNAL", f"critical path revisits chain {c}: {[s.chain for s in steps]}")
        seen.add(c)
        b = placed.bound_by[c]
        steps.append(CriticalStep(c, b))
        if b.kind == "source":
            c = None
        elif b.blocker is None:
            raise LTSError("LTS-E-INTERNAL", f"chain {c}: bound_by {b.kind} has no blocker")
        else:
            c = b.blocker
    return tuple(reversed(steps))


def mxm_roofline(g: TileGraph, cg: ChainGraph, placed: Placed, m: Machine) -> F:
    """总 MAC / 每拍峰值 MAC（只作诊断）。每个 acc 的 MAC = 行数 × K × N，K、N 取它实际所用权重块的大小。
    所用的块 = 同一 unit、同一 bank 上、在 acc 开始之前（issue_end ≤ acc 的 start）最后一次装完的 load_w：
    K = 该 load_w 的行数，N = 流进它的权重 tile 的 lane 数（没有 tile 时取 block_n）。找不到这样的 load_w 时按满块计。
    峰值 = plane 数 × block_k × block_n，都从机器参数读。"""
    from .metrics import node_times
    by = {n.id: n for n in g.nodes}
    pred = {d: s for s, d, _ in g.edges_stream}
    times = node_times(cg, placed)
    bk, bn = m.param("mxm.block_k_rows"), m.param("mxm.block_n_cols")
    loads: dict[tuple[str, int], list[tuple[int, str]]] = {}
    for n in g.nodes:
        if n.kind == "mxm.load_w":
            loads.setdefault((n.unit, n.bank), []).append((times[n.id][1], n.id))
    total = 0
    for n in g.nodes:
        if n.kind != "mxm.acc":
            continue
        start = times[n.id][0]
        done = [(ie, lw) for ie, lw in loads.get((n.unit, n.bank), []) if ie <= start]
        if done:
            lw = max(done)[1]
            k = by[lw].rows
            src = by.get(pred.get(lw))
            nn = (tile_lanes(g, src.tile) if src is not None and src.tile is not None else None) or bn
        else:
            k, nn = bk, bn
        total += n.rows * k * nn
    return F(total, len(m.raw["mxm"]["instances"]) * bk * bn)
