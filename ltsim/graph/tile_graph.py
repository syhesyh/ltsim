"""LTSim tile graph 数据结构（设计文档 §6.2，schema lts-tile-graph/0.2）。

只定义 frozen dataclass；加载与校验在 load.py。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class Tensor:
    dtype: str                              # i8/i32/f16/bf16/f32
    shape: tuple[int, ...]                  # rank 2 或 rank 1
    role: str                               # input/param/output/intermediate
    hemisphere: str                         # "W" | "E"
    layout: str                             # "row" | "lane"
    alias_of: str | None = None             # transpose 输出共享映像


@dataclass(frozen=True)
class Out:
    """流值：节点送出的数据（dtype 与每行占的流数 V）。"""
    dtype: str
    V: int


@dataclass(frozen=True)
class Node:
    id: str
    kind: str                               # mem.read/mem.write/mxm.load_w/mxm.acc/vxm.chain/sxm.*
    rows: int                               # 物理行数
    origin: tuple[str, ...]                 # 来源 LTS-IR 算子 id
    tile: str | None = None                 # mem.* 用
    unit: str | None = None                 # mxm.* / sxm.* 用
    bank: int | None = None                 # mxm.load_w / mxm.acc 用
    ops: tuple[str, ...] | None = None      # vxm.chain 用
    repeat: int | None = None               # mem.read 广播时用
    out: Out | None = None                  # 有流输出的节点


@dataclass(frozen=True)
class OrderEdge:
    src: str
    dst: str
    ref: str                                # "start" | "issue_end" | "done"
    lag: int | str                          # 整数，或机器参数路径（resolve_lags 后变 int）
    reason: str                             # acc_seq / block_switch / lw_after_iw / bank_war


@dataclass(frozen=True)
class TileGraph:
    schema: str                             # "lts-tile-graph/0.2"
    machine: str                            # 机器参数文件 id
    source: str
    tensors: Mapping[str, Tensor]
    nodes: tuple[Node, ...]
    edges_stream: tuple[tuple[str, str, int], ...]   # (src, dst, r0)
    edges_state: tuple[tuple[str, str], ...]
    edges_mem: tuple[tuple[str, str], ...]
    edges_order: tuple[OrderEdge, ...]

    def node(self, nid: str) -> Node:
        for n in self.nodes:
            if n.id == nid:
                return n
        raise KeyError(nid)


# 非 stream 边解析后的统一形式：链间约束用
# (kind, src, dst, ref, lag, reason)
# kind ∈ {"state", "mem", "order"}；reason 只有 order 边有（acc_seq / block_switch / lw_after_iw / bank_war），其余为 None
ResolvedEdge = tuple[str, str, str, str, int, "str | None"]


@dataclass(frozen=True)
class ResolvedGraph:
    """resolve_lags 之后：order 边的 lag 已是 int，state/mem 已展开默认 ref/lag。"""
    graph: TileGraph
    non_stream_edges: tuple[ResolvedEdge, ...]
