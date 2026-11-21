"""B1 节点工作量（backend.md §3，设计 §3.6）。

对每个节点算出速率上限 cap（行/拍，Fraction）、首出延迟 L、尾巴 skew 和所在位置 category；
对每条 stream 边算出 hop，对每个有流输出的节点算出流的方向。
这里只做公式，不涉及链与调度。
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as F
from typing import Mapping

from ..errors import LTSError
from ..graph.load import alias_root, parse_tile
from ..graph.tile_graph import Node, TileGraph
from ..machine.model import Machine

# x 轴从西到东的顺序（设计 §3.1），用来判断流的方向
_X_ORDER = {"MXM_W": 0, "MEM_W": 1, "VXM": 2, "MEM_E": 3, "MXM_E": 4}


@dataclass(frozen=True)
class NodeWork:
    cap: F                 # 每拍最多几行
    L: int                 # 首出延迟
    skew: int              # 命令波尾巴：mem.write / mxm.load_w 取 wave_skew_cycles，其余 0
    category: str          # "MEM_W" / "MEM_E" / "MXM_W" / "MXM_E" / "VXM"


def _tensor_hemisphere(g: TileGraph, name: str) -> str:
    """按 alias_of 展开到源张量，取它的半球（backend.md §2 第 5 条）。"""
    return g.tensors[alias_root(g, name)].hemisphere


def node_category(n: Node, g: TileGraph, m: Machine) -> str:
    if n.kind in ("mem.read", "mem.write"):
        if n.tile is None:
            raise LTSError("LTS-E-GRAPH-TILE", f"{n.kind} node {n.id!r} has no tile")
        return f"MEM_{_tensor_hemisphere(g, parse_tile(n.tile)[0])}"
    if n.kind in ("mxm.load_w", "mxm.acc"):
        for inst in m.raw["mxm"]["instances"]:
            if inst["id"] == n.unit:
                return f"MXM_{inst['side']}"
        raise LTSError("LTS-E-GRAPH-UNIT", f"node {n.id!r} unit {n.unit!r} not in machine planes")
    if n.kind == "vxm.chain":
        return "VXM"
    raise LTSError("LTS-E-GRAPH-KIND", f"node {n.id!r}: kind {n.kind!r} not supported in v0")


def direction(src_cat: str, dst_cat: str) -> str:
    """dst 在 src 东边为 W2E，否则 E2W；同一 category 报 LTS-E-GRAPH-HOP。"""
    a, b = _X_ORDER[src_cat], _X_ORDER[dst_cat]
    if a == b:
        raise LTSError("LTS-E-GRAPH-HOP", f"stream edge within the same category {src_cat}")
    return "W2E" if b > a else "E2W"


def stream_preds(g: TileGraph) -> dict[str, list[str]]:
    """每个节点的 stream 前驱列表（按边在图中的顺序）。没有前驱的节点不出现。"""
    preds: dict[str, list[str]] = {}
    for s, d, _ in g.edges_stream:
        preds.setdefault(d, []).append(s)
    return preds


def _single_pred(n: Node, preds: Mapping[str, list[str]], by: Mapping[str, Node]) -> Node:
    """mem.write / mxm.load_w / mxm.acc 要求恰好一个 stream 前驱。"""
    p = preds.get(n.id, [])
    if len(p) != 1:
        raise LTSError("LTS-E-GRAPH-STREAM",
                       f"{n.kind} {n.id!r} must have exactly one stream pred, got {len(p)}")
    return by[p[0]]


def write_v_in(g: TileGraph) -> dict[str, int]:
    """每个 mem.write 的 V_in = 唯一 stream 前驱的 out.V。

    B1 用它算写节点的 cap；B2 用它算 mem.<H>.wr 的每拍用量 ceil(V_in × ρ)（backend.md §4.4）。
    """
    preds, by = stream_preds(g), {n.id: n for n in g.nodes}
    return {n.id: _single_pred(n, preds, by).out.V for n in g.nodes if n.kind == "mem.write"}


def _mxm_param(m: Machine, name: str, dtype: str, nid: str) -> int:
    """按精度查 mxm.<name>.<dtype>；表里没有这个精度报 LTS-E-UNIT-OP。"""
    table = m.raw["mxm"].get(name)
    if not isinstance(table, Mapping):
        raise LTSError("LTS-E-MACHINE", f"mxm.{name} must be a per-dtype table")
    if dtype not in table:
        raise LTSError("LTS-E-UNIT-OP",
                       f"node {nid!r}: MXM does not support dtype {dtype!r} for {name} "
                       f"(supported: {sorted(table)})")
    return m.param(f"mxm.{name}.{dtype}")


def node_works(g: TileGraph, m: Machine) -> dict[str, NodeWork]:
    """backend.md §3.2 的公式表。"""
    v_in = write_v_in(g)
    preds, by = stream_preds(g), {n.id: n for n in g.nodes}
    wave = m.param("srf.wave_skew_cycles")
    works: dict[str, NodeWork] = {}
    for n in g.nodes:
        cat = node_category(n, g, m)
        k = n.kind
        if k == "mem.read":
            if n.out is None:
                raise LTSError("LTS-E-GRAPH-STREAM", f"mem.read {n.id!r} has no 'out'")
            cap, L, skew = F(m.param("mem.read_vectors_per_cycle"), n.out.V), m.param("mem.read_latency"), 0
        elif k == "mem.write":
            cap, L, skew = F(m.param("mem.write_vectors_per_cycle"), v_in[n.id]), m.param("mem.write_latency"), wave
        elif k == "mxm.load_w":
            # 精度取流进来的权重的 dtype
            dt = _single_pred(n, preds, by).out.dtype
            cap, L, skew = F(_mxm_param(m, "lw_rows_per_cycle", dt, n.id)), 0, wave
        elif k == "mxm.acc":
            # 精度取流入操作数的 dtype（不是 acc 自己的 i32 输出）
            dt = _single_pred(n, preds, by).out.dtype
            cap, L, skew = F(1, m.param("mxm.acc_ii")), _mxm_param(m, "acc_latency", dt, n.id), 0
        elif k == "vxm.chain":
            if not n.ops:
                raise LTSError("LTS-E-GRAPH-KIND", f"vxm.chain {n.id!r} has no ops")
            d = len(n.ops)
            cap = min(F(1), F(m.param("vxm.alus"), d))          # v0 中 groups = 1
            overhead = m.param("vxm.op_overhead")
            L = sum(m.op_latency(o) + overhead for o in n.ops)
            skew = 0
        else:  # node_category 已拦下其他 kind
            raise AssertionError(k)
        works[n.id] = NodeWork(cap=cap, L=L, skew=skew, category=cat)
    return works


def stream_hops(g: TileGraph, works: Mapping[str, NodeWork], m: Machine) -> dict[tuple[str, str], int]:
    """每条 stream 边 (src, dst) 的 hop = hops[cat(src) > cat(dst)] × srf.hop_cycles。"""
    hc = m.param("srf.hop_cycles")
    hops = {}
    for s, d, _ in g.edges_stream:
        cs, cd = works[s].category, works[d].category
        direction(cs, cd)                       # 同 category 在这里报 LTS-E-GRAPH-HOP
        hops[(s, d)] = m.hop(cs, cd) * hc
    return hops


def out_directions(g: TileGraph, works: Mapping[str, NodeWork]) -> dict[str, str]:
    """每个有流消费者的节点，其 out 的流方向。

    v0 不支持同一个流值向两个方向送出（backend.md §11），出现即报 LTS-E-GRAPH-HOP。
    """
    dirs: dict[str, str] = {}
    for s, d, _ in g.edges_stream:
        dr = direction(works[s].category, works[d].category)
        if dirs.setdefault(s, dr) != dr:
            raise LTSError("LTS-E-GRAPH-HOP",
                           f"node {s!r} streams in both directions; v0 does not support this")
    return dirs
