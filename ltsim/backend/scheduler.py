"""B3 链级调度（backend.md §5.2–5.3）：资源受限的串行生成法（SSGS），允许插空。

调度器代表理想静态编译器：每条链只放一次、不回退。策略：
  bottom_level（默认）：优先级取 bottom level；
  topo：按链间约束图的 Kahn 拓扑序（平局取链 id 小的）依次放置，且起始时刻不早于前一条已放的链（不插空）。
random:K:seed 推迟实现。
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Literal, Mapping

from ..errors import LTSError
from ..machine.model import Resource
from .calendar import CounterCalendar, ExclusiveCalendar
from .chains import Chain, ChainGraph

POLICIES = ("bottom_level", "topo")


@dataclass(frozen=True)
class BoundBy:
    """链为什么不能再早开始。"""
    kind: Literal["source", "edge", "resource", "policy"]
    edge: tuple[str, str, str] | None = None    # kind=edge：取到 est 的那条链间边 (kind, u, v)
    resource: str | None = None                 # kind=resource：最后一次冲突的资源
    blocker: str | None = None                  # 卡住它的那条链：edge 为边的源链；resource 为占着资源的链；policy 为前一条已放的链


@dataclass(frozen=True)
class Placed:
    policy: str
    start: Mapping[str, int]                    # 链 id → T_c
    bound_by: Mapping[str, BoundBy]
    order: tuple[str, ...]                      # 放置顺序
    total: int                                  # max(T_c + len(c))


@dataclass(frozen=True)
class Segment:
    """earliest_fit 检查用的一段占用（相对链起点）。计数资源上同一条链重叠的预留已合并。"""
    resource: str
    start: int
    end: int
    amount: int


def fit_segments(c: Chain, resources: Mapping[str, Resource]) -> tuple[Segment, ...]:
    """独占资源：每条预留一段。计数资源：把同一资源上的预留按断点切开、重叠部分的用量相加，
    得到互不重叠的分段（backend.md §5.3）。逐条检查会漏掉同一条链自己的预留叠加而超容量。
    按 (资源, start) 排序。"""
    by_res: dict[str, list] = {}
    for r in c.reservations:
        if r.resource not in resources:
            raise LTSError("LTS-E-MACHINE", f"chain {c.id}: node {r.node!r} uses unknown resource {r.resource!r}")
        by_res.setdefault(r.resource, []).append(r)
    segs: list[Segment] = []
    for name, rs in by_res.items():
        if resources[name].kind == "exclusive":
            segs += [Segment(name, r.start, r.end, 1) for r in rs]
            continue
        pts = sorted({r.start for r in rs} | {r.end for r in rs})
        for a, b in zip(pts, pts[1:]):
            k = sum(r.amount for r in rs if r.start <= a < r.end)
            if k:
                segs.append(Segment(name, a, b, k))
    return tuple(sorted(segs, key=lambda s: (s.resource, s.start, s.end)))


def bottom_levels(cg: ChainGraph) -> dict[str, int]:
    """bl(c) = max(len(c), max_{e: c→c'} (w_e + bl(c')))，按逆拓扑序计算。"""
    order = _kahn(cg)
    out: dict[str, list] = {c.id: [] for c in cg.chains}
    for e in cg.edges:
        out[e.src].append(e)
    length = {c.id: c.length for c in cg.chains}
    bl: dict[str, int] = {}
    for c in reversed(order):
        bl[c] = max([length[c]] + [e.weight + bl[e.dst] for e in out[c]])
    return bl


def _kahn(cg: ChainGraph) -> list[str]:
    """链间约束图的拓扑序，平局取链 id 序号小的（B2 已保证无环）。"""
    idx = {c.id: i for i, c in enumerate(cg.chains)}
    indeg = {c.id: 0 for c in cg.chains}
    succ: dict[str, list[str]] = {c.id: [] for c in cg.chains}
    for e in cg.edges:
        indeg[e.dst] += 1
        succ[e.src].append(e.dst)
    heap = [(idx[c], c) for c, d in indeg.items() if d == 0]
    heapq.heapify(heap)
    order = []
    while heap:
        _, c = heapq.heappop(heap)
        order.append(c)
        for d in succ[c]:
            indeg[d] -= 1
            if indeg[d] == 0:
                heapq.heappush(heap, (idx[d], d))
    return order


def schedule(cg: ChainGraph, resources: Mapping[str, Resource], policy: str = "bottom_level") -> Placed:
    if policy not in POLICIES:
        if policy.startswith("random"):
            raise LTSError("LTS-E-ARG", f"policy {policy!r}: random:K:seed is deferred (not implemented in v0)")
        raise LTSError("LTS-E-ARG", f"policy must be one of {POLICIES}, got {policy!r}")

    chains = {c.id: c for c in cg.chains}
    idx = {c.id: i for i, c in enumerate(cg.chains)}
    in_edges: dict[str, list] = {c: [] for c in chains}
    for e in cg.edges:
        in_edges[e.dst].append(e)
    if policy == "bottom_level":
        bl = bottom_levels(cg)
        prio = {c: bl[c] for c in chains}
    else:
        topo_index = {c: i for i, c in enumerate(_kahn(cg))}
        prio = {c: -topo_index[c] for c in chains}
    segs = {c: fit_segments(chains[c], resources) for c in chains}

    cal: dict[str, ExclusiveCalendar | CounterCalendar] = {
        name: ExclusiveCalendar() if r.kind == "exclusive" else CounterCalendar(r.cap)
        for name, r in resources.items()
    }

    T: dict[str, int] = {}
    bound_by: dict[str, BoundBy] = {}
    order: list[str] = []
    while len(T) < len(chains):
        ready = [c for c in chains if c not in T and all(e.src in T for e in in_edges[c])]
        c = max(ready, key=lambda x: (prio[x], -idx[x]))
        est, why = 0, BoundBy("source")
        for e in in_edges[c]:
            if T[e.src] + e.weight > est:
                est, why = T[e.src] + e.weight, BoundBy("edge", edge=e.edge, blocker=e.src)
        if policy == "topo" and order and T[order[-1]] > est:
            est, why = T[order[-1]], BoundBy("policy", blocker=order[-1])
        t, blocked = _earliest_fit(segs[c], est, cal)
        T[c] = t
        bound_by[c] = why if blocked is None else blocked
        for r in chains[c].reservations:
            if isinstance(cal[r.resource], ExclusiveCalendar):
                cal[r.resource].commit(t + r.start, t + r.end, c)
            else:
                cal[r.resource].commit(t + r.start, t + r.end, r.amount, c)
        order.append(c)

    total = max(T[c] + chains[c].length for c in chains)
    return Placed(policy, T, bound_by, tuple(order), total)


def _earliest_fit(segs, t, cal) -> tuple[int, BoundBy | None]:
    """从 t 起找第一个所有分段都放得下的起始时刻。每次冲突把 t 推到冲突结束后，t 严格增大，必然终止；
    且推进不会跳过可行位置（backend.md §5.3）。"""
    blocked = None
    while True:
        for s in segs:
            c = cal[s.resource]
            if isinstance(c, ExclusiveCalendar):
                hit = c.conflict(t + s.start, t + s.end)
            else:
                hit = c.conflict(t + s.start, t + s.end, s.amount)
            if hit:
                end, owner = hit
                t = end - s.start
                blocked = BoundBy("resource", resource=s.resource, blocker=owner)
                break
        else:
            return t, blocked


def verify(cg: ChainGraph, placed: Placed, resources: Mapping[str, Resource]) -> list[str]:
    """独立检查一份调度：每条链都有 T 且 T ≥ 0；order 覆盖全部链；total = max(T + len)；
    全部链间约束成立；任何一拍、任何资源不超容量。返回违反项（空表示合法）。
    不复用日历的代码，直接按预留表逐拍累加，供测试和 B4 自检使用。"""
    bad = []
    T = placed.start
    ids = {c.id for c in cg.chains}
    missing = sorted(ids - set(T))
    if missing:
        return [f"no start time for chain(s) {missing}"]
    bad += [f"chain {c}: start {T[c]} < 0" for c in sorted(ids) if T[c] < 0]
    if sorted(placed.order) != sorted(ids):
        bad.append(f"order {placed.order} is not a permutation of the chains")
    total = max(T[c.id] + c.length for c in cg.chains)
    if placed.total != total:
        bad.append(f"total {placed.total} != max(T + len) = {total}")
    for e in cg.edges:
        if T[e.dst] < T[e.src] + e.weight:
            bad.append(f"edge {e.edge}: T[{e.dst}]={T[e.dst]} < T[{e.src}]={T[e.src]} + {e.weight}")
    use: dict[str, dict[int, int]] = {}
    for c in cg.chains:
        for r in c.reservations:
            u = use.setdefault(r.resource, {})
            for x in range(T[c.id] + r.start, T[c.id] + r.end):
                u[x] = u.get(x, 0) + r.amount
    for name, u in use.items():
        cap = resources[name].cap
        over = [x for x, v in u.items() if v > cap]
        if over:
            bad.append(f"{name} over capacity {cap} at {len(over)} cycles, first {min(over)} "
                       f"(usage {u[min(over)]})")
    return bad
