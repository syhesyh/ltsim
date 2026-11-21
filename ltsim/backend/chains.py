"""B2 组链（backend.md §4）。

按 stream 边把节点组成链；求链速率、占用时长、链内偏移（stream 边全是等式，沿生成树传播）、
预留表；做链级合法性检查；把跨链的非 stream 边折算成链起始时刻之间的约束。
输入的图应已通过 B0（graph.load.validate_graph）。
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from fractions import Fraction as F
from math import ceil
from typing import Mapping

from ..errors import LTSError
from ..graph.load import resolve_lags, tile_region
from ..graph.tile_graph import TileGraph
from ..machine.model import Machine, Resource
from .work import NodeWork, node_works, out_directions, stream_hops, write_v_in

MEM_OVERLAP_MODES = ("tile", "row")


@dataclass(frozen=True)
class Reservation:
    """相对链起点的资源占用，区间左闭右开。"""
    resource: str            # "mxm.w0.compute"、"vxm.alu"、"srf.E2W"、"mem.W.rd" ...
    start: int
    end: int
    amount: int              # 独占资源恒为 1
    node: str                # 由哪个节点产生


@dataclass(frozen=True)
class Chain:
    id: str                  # "c0", "c1", ...
    nodes: tuple[str, ...]   # 按节点在图中的顺序
    rate: F                  # ρ：每拍推进几行
    D: Mapping[str, int]     # 每个节点的占用时长 ceil(rows / ρ)
    offset: Mapping[str, int]
    length: int              # len(c) = max(o + L + D + skew)
    reservations: tuple[Reservation, ...]   # 按 (资源, start, end, 节点) 排序


@dataclass(frozen=True)
class ChainEdge:
    """链间约束 T_dst ≥ T_src + weight。"""
    src: str
    dst: str
    weight: int
    edge: tuple[str, str, str]   # 来源边 (kind, 源节点, 目的节点)
    ref: str = "done"            # 来源边的 ref（start / issue_end / done）
    reason: str | None = None    # order 边的 reason；state / mem 边为 None


@dataclass(frozen=True)
class ChainGraph:
    chains: tuple[Chain, ...]
    edges: tuple[ChainEdge, ...]          # 全部链间边，不去重（B3 取最大值，B4 的 trace 要画每一条）
    chain_of: Mapping[str, str]           # 节点 id → 链 id
    works: Mapping[str, NodeWork]         # B1 结果，供 B3/B4 使用

    def chain(self, cid: str) -> Chain:
        for c in self.chains:
            if c.id == cid:
                return c
        raise KeyError(cid)


def build_chains(g: TileGraph, m: Machine, *, mem_overlap: str = "tile") -> ChainGraph:
    if mem_overlap not in MEM_OVERLAP_MODES:
        raise LTSError("LTS-E-ARG", f"mem_overlap must be one of {MEM_OVERLAP_MODES}, got {mem_overlap!r}")
    works = node_works(g, m)
    hops = stream_hops(g, works, m)
    dirs = out_directions(g, works)
    v_in = write_v_in(g)
    resources = m.resources()
    wave = m.param("srf.wave_skew_cycles")
    by = {n.id: n for n in g.nodes}

    # ---- 4.1 求链：并查集；链按"分量里在 nodes 中最靠前的节点"编号 ----
    parent = {n.id: n.id for n in g.nodes}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for s, d, _ in g.edges_stream:
        parent[find(s)] = find(d)
    root_to_cid: dict[str, str] = {}
    members: dict[str, list[str]] = {}
    for n in g.nodes:
        r = find(n.id)
        if r not in root_to_cid:
            root_to_cid[r] = f"c{len(root_to_cid)}"
        members.setdefault(root_to_cid[r], []).append(n.id)
    chain_of = {nid: cid for cid, ns in members.items() for nid in ns}
    stream_of: dict[str, list[tuple[str, str, int]]] = {cid: [] for cid in members}
    for s, d, r0 in g.edges_stream:
        stream_of[chain_of[s]].append((s, d, r0))

    # ---- 4.2–4.5 每条链 ----
    chains: list[Chain] = []
    o_all: dict[str, int] = {}
    D_all: dict[str, int] = {}
    rate_of: dict[str, F] = {}
    for cid, nodes in members.items():
        es = stream_of[cid]
        rate = min(works[n].cap for n in nodes)
        D = {n: ceil(F(by[n].rows) / rate) for n in nodes}
        _check_stream_acyclic(cid, nodes, es)
        o = _offsets(cid, nodes, es, works, hops, rate)
        length = max(o[n] + works[n].L + D[n] + works[n].skew for n in nodes)
        resv = _reservations(nodes, es, by, works, hops, dirs, v_in, o, D, rate, wave)
        _check_chain_resources(cid, resv, resources)
        chains.append(Chain(cid, tuple(nodes), rate, D, o, length, resv))
        o_all.update(o)
        D_all.update(D)
        rate_of[cid] = rate

    # ---- 4.5 同链非 stream 边；4.6 链间约束 ----
    def refoff(u: str, ref: str) -> int:
        w = works[u]
        return {"start": o_all[u],
                "issue_end": o_all[u] + D_all[u],
                "done": o_all[u] + w.L + D_all[u] + w.skew}[ref]

    edges: list[ChainEdge] = []
    for kind, u, v, ref, lag, reason in resolve_lags(g, m).non_stream_edges:
        cu, cv = chain_of[u], chain_of[v]
        if cu == cv:
            if kind == "mem":
                need, word = _mem_row_lag(g, by, u, v, lag, o_all, works, rate_of[cu], rate_of[cv], wave)
                if need > 0:
                    raise LTSError("LTS-E-CHAIN-SELF",
                                   f"chain {cu}: {v!r} reads word {word} {need} cycle(s) before "
                                   f"{u!r} makes it visible")
            elif o_all[v] < refoff(u, ref) + lag:
                raise LTSError("LTS-E-CHAIN-SELF",
                               f"chain {cu}: {kind} edge {u}->{v} needs offset({v}) ≥ {ref}({u}) + {lag} "
                               f"= {refoff(u, ref) + lag}, but offset({v}) = {o_all[v]}")
            continue
        if kind == "mem" and mem_overlap == "row":
            # 逐 word 精确：读到每个公共 word 时它已写完可见（与同链检查同一公式，backend.md §4.6）
            weight, _word = _mem_row_lag(g, by, u, v, lag, o_all, works, rate_of[cu], rate_of[cv], wave)
        else:
            weight = refoff(u, ref) + lag - o_all[v]
        edges.append(ChainEdge(cu, cv, weight, (kind, u, v), ref, reason))

    _check_chain_graph_acyclic([c.id for c in chains], edges)
    return ChainGraph(tuple(chains), tuple(edges), chain_of, works)


# ---------------------------------------------------------------------------

def _check_stream_acyclic(cid: str, nodes: list[str], es: list[tuple[str, str, int]]) -> None:
    indeg = {n: 0 for n in nodes}
    succ: dict[str, list[str]] = {n: [] for n in nodes}
    for u, v, _ in es:
        indeg[v] += 1
        succ[u].append(v)
    q = deque(n for n in nodes if indeg[n] == 0)
    seen = 0
    while q:
        x = q.popleft()
        seen += 1
        for y in succ[x]:
            indeg[y] -= 1
            if indeg[y] == 0:
                q.append(y)
    if seen != len(nodes):
        left = [n for n in nodes if indeg[n] > 0]
        raise LTSError("LTS-E-GRAPH-CYCLE", f"chain {cid}: stream edges form a cycle among {left}")


def _offsets(cid, nodes, es, works, hops, rate) -> dict[str, int]:
    """链内偏移。流不能等待，所以每条 stream 边都是等式 o_v = o_u + arr(u, v)，
    arr = L_u + hop(u, v) + ceil(r0 / ρ)。沿无向生成树传播，再核对全部边，最后整体减去最小值。
    （backend.md §4.3；原规格的 ASAP/ALAP 会在分叉的一支是汇点时误报，见该节说明。）"""
    arr = {(u, v): works[u].L + hops[(u, v)] + ceil(F(r0) / rate) for u, v, r0 in es}
    adj: dict[str, list[tuple[str, int]]] = {n: [] for n in nodes}
    for u, v, _ in es:
        adj[u].append((v, arr[(u, v)]))
        adj[v].append((u, -arr[(u, v)]))
    o = {nodes[0]: 0}
    q = deque([nodes[0]])
    while q:
        x = q.popleft()
        for y, delta in adj[x]:
            if y not in o:
                o[y] = o[x] + delta
                q.append(y)
    for u, v, _ in es:
        need = o[u] + arr[(u, v)]
        if o[v] != need:
            raise LTSError(
                "LTS-E-CHAIN-ALIGN",
                f"chain {cid}: the stream value from {u!r} must reach {v!r} at offset {need}, "
                f"but another path fixes {v!r} at {o[v]} (difference {o[v] - need}); a stream value "
                f"cannot wait — split the fusion group so this value goes through MEM")
    lo = min(o.values())
    return {n: o[n] - lo for n in nodes}


def _reservations(nodes, es, by, works, hops, dirs, v_in, o, D, rate, wave) -> tuple[Reservation, ...]:
    """backend.md §4.4 的预留表。有 out 但没有 stream 消费者的节点不占流（没有方向，也没有 hop_last）。"""
    out: list[Reservation] = []
    for n in nodes:
        node, w, on, d = by[n], works[n], o[n], D[n]
        k = node.kind
        if k == "mem.read":
            out.append(Reservation(f"mem.{w.category[-1]}.rd", on, on + d, ceil(node.out.V * rate), n))
        elif k == "mem.write":
            out.append(Reservation(f"mem.{w.category[-1]}.wr", on + w.L, on + w.L + d, ceil(v_in[n] * rate), n))
        elif k == "mxm.load_w":
            out.append(Reservation(f"{node.unit}.weight", on, on + d, 1, n))
        elif k == "mxm.acc":
            out.append(Reservation(f"{node.unit}.compute", on, on + d, 1, n))
        elif k == "vxm.chain":
            out.append(Reservation("vxm.alu", on, on + d, ceil(len(node.ops) * rate), n))   # v0 groups = 1
        if n in dirs:
            hop_last = max(hops[(u, v)] for u, v, _ in es if u == n)
            out.append(Reservation(f"srf.{dirs[n]}", on + w.L, on + w.L + d + wave + hop_last,
                                   ceil(node.out.V * rate), n))
    return tuple(sorted(out, key=lambda r: (r.resource, r.start, r.end, r.node)))


def _peak(rs: list[Reservation]) -> tuple[int, int]:
    """计数资源逐拍占用的峰值及首次达到峰值的时刻（区间左闭右开：同一时刻先减后加）。"""
    ev = sorted([(r.start, 1, r.amount) for r in rs] + [(r.end, 0, -r.amount) for r in rs])
    cur, best, at = 0, 0, None
    for t, _order, delta in ev:
        cur += delta
        if cur > best:
            best, at = cur, t
    return best, at


def _check_chain_resources(cid: str, resv: tuple[Reservation, ...], resources: Mapping[str, Resource]) -> None:
    """链级合法性（backend.md §4.5）：流宽 → LTS-E-SRF-WIDTH；独占资源重叠、计数资源逐拍超限 → LTS-E-CHAIN-SELF。"""
    by_res: dict[str, list[Reservation]] = {}
    for r in resv:
        if r.resource not in resources:
            raise LTSError("LTS-E-MACHINE", f"chain {cid}: node {r.node!r} uses unknown resource {r.resource!r}")
        by_res.setdefault(r.resource, []).append(r)
    for name in sorted(by_res, key=lambda x: (not x.startswith("srf."), x)):      # 先查流宽
        rs, res = by_res[name], resources[name]
        if res.kind == "exclusive":
            rs = sorted(rs, key=lambda r: (r.start, r.end))
            for a, b in zip(rs, rs[1:]):
                if b.start < a.end:
                    raise LTSError("LTS-E-CHAIN-SELF",
                                   f"chain {cid}: {a.node!r} [{a.start},{a.end}) and {b.node!r} "
                                   f"[{b.start},{b.end}) overlap on exclusive resource {name!r}")
        else:
            peak, at = _peak(rs)
            if peak > res.cap:
                users = [r.node for r in rs if r.start <= at < r.end]
                code = "LTS-E-SRF-WIDTH" if name.startswith("srf.") else "LTS-E-CHAIN-SELF"
                raise LTSError(code, f"chain {cid}: {name} needs {peak} > capacity {res.cap} "
                                     f"at offset {at} (nodes {users})")


def _mem_row_lag(g, by, w_id, r_id, raw_gap, o, works, rate_w, rate_r, wave) -> tuple[int, int]:
    """mem 边 w → r 的逐 word 条件（backend.md §4.5、§4.6）。对每个公共 word：

        T_r + o_r + ceil(i_r / ρ_r)  ≥  T_w + o_w + L_w + ceil(i_w / ρ_w) + wave_skew + raw_gap

    i_r、i_w 是该 word 在读、写节点里的行号，word 按各自布局换算（tile_region）。
    返回 (max_word [右边 − 左边里不含 T 的部分], 取到最大值的 word)，即 T_r − T_w 至少要多大：
    同链时 T_r = T_w，这个值必须 ≤ 0；跨链时它就是链间边的权重。广播读只有 1 个 word，i_r 恒为 0。"""
    _, (rw0, rw1), _ = tile_region(g, by[r_id].tile)
    _, (ww0, ww1), _ = tile_region(g, by[w_id].tile)
    lo, hi = max(rw0, ww0), min(rw1, ww1)
    if lo >= hi:
        raise LTSError("LTS-E-GRAPH-MEM", f"mem edge {w_id}->{r_id}: tiles share no word")
    base = o[w_id] + works[w_id].L + wave + raw_gap - o[r_id]
    best, at = None, None
    for word in range(lo, hi):
        need = base + ceil(F(word - ww0) / rate_w) - ceil(F(word - rw0) / rate_r)
        if best is None or need > best:
            best, at = need, word
    return best, at


def _check_chain_graph_acyclic(cids: list[str], edges: list[ChainEdge]) -> None:
    indeg = {c: 0 for c in cids}
    succ: dict[str, list[str]] = {c: [] for c in cids}
    for e in edges:
        indeg[e.dst] += 1
        succ[e.src].append(e.dst)
    q = deque(c for c in cids if indeg[c] == 0)
    removed = set()
    while q:
        x = q.popleft()
        removed.add(x)
        for y in succ[x]:
            indeg[y] -= 1
            if indeg[y] == 0:
                q.append(y)
    if len(removed) == len(cids):
        return
    # 在剩下的链里找出一个环用于诊断。Kahn 之后剩下的每条链都还有一个剩下的前驱，
    # 所以沿前驱往回走一定会回到走过的链（沿后继走可能走进环下游的死胡同）。
    pred: dict[str, list[str]] = {c: [] for c in cids}
    for e in edges:
        pred[e.dst].append(e.src)
    path, pos, x = [], {}, next(c for c in cids if c not in removed)
    while x not in pos:
        pos[x] = len(path)
        path.append(x)
        x = next(y for y in pred[x] if y not in removed)
    cycle = list(reversed(path[pos[x]:])) + [path[-1]]     # 改回正向：a → b → … → a
    hops = []
    for a, b in zip(cycle, cycle[1:]):
        e = next(e for e in edges if e.src == a and e.dst == b)
        hops.append(f"{a} -{e.edge[0]}({e.edge[1]}->{e.edge[2]})-> {b}")
    raise LTSError("LTS-E-CHAIN-CYCLE", "chain constraint graph has a cycle: " + "; ".join(hops))
