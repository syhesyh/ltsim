"""LTSim 通用 DAG builder（interfaces.md §3）。

不做算子级模式；用户逐节点构造，builder 负责校验、字段补全与确定性输出。
产出的 TileGraph 与 graph/load.py 读入的完全同构。
"""
from __future__ import annotations

import json
from typing import Any

from ..errors import LTSError
from ..graph.tile_graph import Node, OrderEdge, Out, Tensor, TileGraph


def tile_str(tensor: str, r0: int, r1: int, c0: int | None = None, c1: int | None = None) -> str:
    """生成 tile 字符串：rank2 "T[r0:r1, c0:c1]"；rank1 "T[r0:r1]"。"""
    if c0 is None:
        return f"{tensor}[{r0}:{r1}]"
    return f"{tensor}[{r0}:{r1},{c0}:{c1}]"


def plane_for(planes: list[str], map: str, n: int, k: int, j: int) -> str:
    """按 map 规则把权重块 (n, k) 分到一个 plane（设计 §4.3 ②）。

    k_split：第 k 个 K 块 → planes[k % len]
    n_split：第 n 个 N 块 → planes[n % len]
    round_robin：按块编号 j（(n, k) 字典序中的序号）轮流分
    """
    if map == "k_split":
        idx = k
    elif map == "n_split":
        idx = n
    elif map == "round_robin":
        idx = j
    else:
        raise LTSError("LTS-E-GEN", f"unknown map: {map!r}")
    return planes[idx % len(planes)]


def bank_for(j: int, banks: str) -> int:
    """alternate: j%2；single: 0。"""
    if banks == "alternate":
        return j % 2
    if banks == "single":
        return 0
    raise LTSError("LTS-E-GEN", f"unknown banks policy: {banks!r}")


class GraphBuilder:
    def __init__(self, name: str, machine: str, source: str | None = None):
        """source：tile graph 的 `source` 字段（来源 LTS-IR 文件名）；省略时取 "<name>.yaml"。"""
        self.name = name
        self.machine = machine
        self.source = source if source is not None else f"{name}.yaml"
        self._tensors: dict[str, Tensor] = {}
        self._nodes: list[Node] = []
        self._stream: list[tuple[str, str, int]] = []
        self._state: list[tuple[str, str]] = []
        self._mem: list[tuple[str, str]] = []
        self._order: list[OrderEdge] = []
        self._node_ids: set[str] = set()

    # ---- 张量 ---------------------------------------------------------

    def tensor(self, name, dtype, shape, role, hemisphere, layout, alias_of=None):
        self._tensors[name] = Tensor(
            dtype=dtype, shape=tuple(shape), role=role,
            hemisphere=hemisphere, layout=layout, alias_of=alias_of,
        )

    # ---- 节点 ---------------------------------------------------------

    def _add(self, n: Node):
        if n.id in self._node_ids:
            raise LTSError("LTS-E-GRAPH-DUP", f"duplicate node id: {n.id!r}")
        self._node_ids.add(n.id)
        self._nodes.append(n)

    def read(self, id, tile, rows, out: Out, origin, repeat=None):
        self._add(Node(id=id, kind="mem.read", rows=rows, tile=tile,
                       out=out, repeat=repeat, origin=tuple(origin)))

    def write(self, id, tile, rows, origin):
        self._add(Node(id=id, kind="mem.write", rows=rows, tile=tile, origin=tuple(origin)))

    def load_w(self, id, unit, bank, rows, origin):
        self._add(Node(id=id, kind="mxm.load_w", rows=rows, unit=unit, bank=bank, origin=tuple(origin)))

    def acc(self, id, unit, bank, rows, out: Out, origin):
        self._add(Node(id=id, kind="mxm.acc", rows=rows, unit=unit, bank=bank,
                       out=out, origin=tuple(origin)))

    def vxm(self, id, ops: list[str], rows, out: Out, origin):
        self._add(Node(id=id, kind="vxm.chain", rows=rows, ops=tuple(ops), out=out, origin=tuple(origin)))

    # ---- 边 -----------------------------------------------------------

    def stream(self, src, dst, r0=0):
        self._stream.append((src, dst, r0))

    def state(self, src, dst):
        self._state.append((src, dst))

    def mem(self, src, dst):
        self._mem.append((src, dst))

    def order(self, src, dst, ref, lag, reason):
        self._order.append(OrderEdge(src=src, dst=dst, ref=ref, lag=lag, reason=reason))

    # ---- 展开辅助 -----------------------------------------------------

    # 一个 plane 上的权重块序列用 blocks 描述：按装载顺序排列，
    # 每项为 (load_w 节点 id, 该块的 acc 节点 id 列表，按执行顺序即 m 升序)。
    # 下面三个方法生成设计 §5.5 表中的 MXM 顺序边；lag 用参数名，由后端解析。

    def link_acc_order(self, blocks: list[tuple[str, list[str]]]):
        """acc_seq：块内相邻 acc；block_switch：块 j 末个 acc → 块 j+1 首个 acc。"""
        for _lw, accs in blocks:
            for a, b in zip(accs, accs[1:]):
                self.order(a, b, ref="issue_end", lag=0, reason="acc_seq")
        for (_lw0, accs0), (_lw1, accs1) in zip(blocks, blocks[1:]):
            self.order(accs0[-1], accs1[0], ref="issue_end",
                       lag="mxm.block_switch_gap", reason="block_switch")

    def link_weight_order(self, blocks: list[tuple[str, list[str]]], banks: str):
        """state：块 j 的 load_w → 块 j 首个 acc；
        lw_after_iw：块 j−1 首个 acc → 块 j 的 load_w（ref=start, lag=−1）；
        bank_war（仅 banks=single）：块 j−1 末个 acc → 块 j 的 load_w（ref=issue_end, lag=0）。
        """
        if banks not in ("alternate", "single"):
            raise LTSError("LTS-E-GEN", f"unknown banks policy: {banks!r}")
        for lw, accs in blocks:
            if not accs:
                raise LTSError("LTS-E-GEN", f"weight block {lw!r} has no acc")
            self.state(lw, accs[0])
        for (_lw0, accs0), (lw1, _accs1) in zip(blocks, blocks[1:]):
            self.order(accs0[0], lw1, ref="start", lag=-1, reason="lw_after_iw")
            if banks == "single":
                self.order(accs0[-1], lw1, ref="issue_end", lag=0, reason="bank_war")

    def link_plane(self, blocks: list[tuple[str, list[str]]], banks: str):
        """一个 plane 的全部 MXM 顺序边 = link_weight_order + link_acc_order。"""
        self.link_weight_order(blocks, banks)
        self.link_acc_order(blocks)

    # ---- 产出 ---------------------------------------------------------

    def build(self) -> TileGraph:
        g = TileGraph(
            schema="lts-tile-graph/0.2",
            machine=self.machine,
            source=self.source,
            tensors=dict(self._tensors),
            nodes=tuple(self._nodes),
            edges_stream=tuple(self._stream),
            edges_state=tuple(self._state),
            edges_mem=tuple(self._mem),
            edges_order=tuple(self._order),
        )
        _validate_built(g)
        return g

    def to_dict(self) -> dict[str, Any]:
        return graph_to_dict(self.build())

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
            f.write("\n")


def _validate_built(g: TileGraph) -> None:
    idset = {n.id for n in g.nodes}
    by = {n.id: n for n in g.nodes}
    for (s, d, _r0) in g.edges_stream:
        if s not in idset or d not in idset:
            raise LTSError("LTS-E-GRAPH-REF", f"stream edge {s}->{d} endpoint missing")
        if by[s].out is None:
            raise LTSError("LTS-E-GRAPH-STREAM", f"stream src {s!r} has no 'out'")
    for coll in (g.edges_state, g.edges_mem):
        for (s, d) in coll:
            if s not in idset or d not in idset:
                raise LTSError("LTS-E-GRAPH-REF", f"edge {s}->{d} endpoint missing")
    for o in g.edges_order:
        if o.src not in idset or o.dst not in idset:
            raise LTSError("LTS-E-GRAPH-REF", f"order edge {o.src}->{o.dst} endpoint missing")


def graph_to_dict(g: TileGraph) -> dict[str, Any]:
    """TileGraph → 与 tile_graph.json 同构的 dict（save 用的就是它）。"""
    def node_dict(n: Node) -> dict[str, Any]:
        d: dict[str, Any] = {"id": n.id, "kind": n.kind}
        if n.tile is not None:
            d["tile"] = n.tile
        if n.unit is not None:
            d["unit"] = n.unit
        if n.bank is not None:
            d["bank"] = n.bank
        d["rows"] = n.rows
        if n.repeat is not None:
            d["repeat"] = n.repeat
        if n.ops is not None:
            d["ops"] = list(n.ops)
        if n.out is not None:
            d["out"] = {"dtype": n.out.dtype, "V": n.out.V}
        d["origin"] = list(n.origin)
        return d

    return {
        "schema": g.schema,
        "machine": g.machine,
        "source": g.source,
        "tensors": {
            name: {k: v for k, v in {
                "dtype": t.dtype, "shape": list(t.shape), "role": t.role,
                "hemisphere": t.hemisphere, "layout": t.layout,
                "alias_of": t.alias_of,
            }.items() if v is not None}
            for name, t in g.tensors.items()
        },
        "nodes": [node_dict(n) for n in g.nodes],
        "edges": {
            "stream": [[s, d, r0] if r0 else [s, d] for (s, d, r0) in g.edges_stream],
            "state": [[s, d] for (s, d) in g.edges_state],
            "mem": [[s, d] for (s, d) in g.edges_mem],
            "order": [
                {"src": o.src, "dst": o.dst, "ref": o.ref, "lag": o.lag, "reason": o.reason}
                for o in g.edges_order
            ],
        },
    }


_to_dict = graph_to_dict      # 旧名，测试里还在用
