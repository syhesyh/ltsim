"""tile graph → Graphviz DOT（interfaces.md §2，可选的查看工具）。

    dot -Tsvg graph.dot -o graph.svg

给出 chain_of（节点 id → 链 id，来自 B2 的 ChainGraph）时，每条链画成一个框。边的线型按种类区分：
stream 实线、state 点线、mem 粗虚线、order 灰色并标 reason 与 lag。输出是确定的。
"""
from __future__ import annotations

from typing import Mapping

from .tile_graph import TileGraph

_FILL = {"mem.read": "#dbeafe", "mem.write": "#bfdbfe", "mxm.load_w": "#fed7aa", "mxm.acc": "#fde68a",
         "vxm.chain": "#e9d5ff"}
_EDGE = {"stream": 'color="#111827"', "state": 'style=dotted, color="#c2410c"',
         "mem": 'style="dashed,bold", color="#1d4ed8"', "order": 'color="#9ca3af", fontcolor="#6b7280"'}


def _q(s: str) -> str:
    return '"' + str(s).replace("\\", "\\\\").replace('"', '\\"') + '"'


def _label(n) -> str:
    parts = [n.id, n.kind, f"rows={n.rows}"]
    if n.tile is not None:
        parts.append(n.tile)
    if n.unit is not None:
        parts.append(f"{n.unit} bank{n.bank}")
    if n.ops is not None:
        parts.append("[" + ", ".join(n.ops) + "]")
    return "\\n".join(p.replace("\\", "\\\\").replace('"', '\\"') for p in parts)


def to_dot(g: TileGraph, chain_of: Mapping[str, str] | None = None, name: str = "tile_graph") -> str:
    lines = [f"digraph {_q(name)} {{", "  rankdir=LR;",
             '  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10];',
             '  edge [fontname="Helvetica", fontsize=9];']

    def node_line(n, indent):
        return f'{indent}{_q(n.id)} [label="{_label(n)}", fillcolor="{_FILL.get(n.kind, "#f3f4f6")}"];'

    if chain_of:
        groups: dict[str, list] = {}
        for n in g.nodes:
            groups.setdefault(chain_of[n.id], []).append(n)
        for cid, ns in groups.items():                  # 链按第一个节点在图中的顺序出现，与 B2 的编号一致
            lines.append(f"  subgraph {_q('cluster_' + cid)} {{")
            lines.append(f'    label={_q(cid)}; style=rounded; color="#9ca3af";')
            lines += [node_line(n, "    ") for n in ns]
            lines.append("  }")
    else:
        lines += [node_line(n, "  ") for n in g.nodes]

    for s, d, r0 in g.edges_stream:
        extra = f', label="r0={r0}"' if r0 else ""
        lines.append(f"  {_q(s)} -> {_q(d)} [{_EDGE['stream']}{extra}];")
    for s, d in g.edges_state:
        lines.append(f'  {_q(s)} -> {_q(d)} [{_EDGE["state"]}, label="state"];')
    for s, d in g.edges_mem:
        lines.append(f'  {_q(s)} -> {_q(d)} [{_EDGE["mem"]}, label="mem"];')
    for o in g.edges_order:
        lag = f"{o.lag:+d}" if isinstance(o.lag, int) else f" {o.lag}"      # 整数写成 +5 / -1，参数名原样写
        label = f"{o.reason} ({o.ref}{lag})"
        lines.append(f"  {_q(o.src)} -> {_q(o.dst)} [{_EDGE['order']}, label={_q(label)}];")
    lines.append("}")
    return "\n".join(lines) + "\n"
