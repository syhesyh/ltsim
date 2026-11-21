"""LTSim tile graph 加载与校验（backend.md §2，设计 §6.2）。

只做结构校验与 lag 解析；周期相关检查（hop 方向、链级合法性）在 B1/B2。
"""
from __future__ import annotations

import json
from typing import Any, Mapping

from ..errors import LTSError
from ..machine.model import Machine
from .tile_graph import Node, OrderEdge, Out, ResolvedEdge, ResolvedGraph, Tensor, TileGraph


def _err(code_suffix: str, msg: str) -> LTSError:
    return LTSError(f"LTS-E-GRAPH-{code_suffix}", msg)


# ---- 字段取值（设计 §6.2；schema.json 只作文档，不在运行时读取，校验在这里手写）----
DTYPES = ("i8", "i32", "f16", "bf16", "f32")
ROLES = ("input", "param", "output", "intermediate")
HEMISPHERES = ("W", "E")
LAYOUTS = ("row", "lane")
REFS = ("start", "issue_end", "done")
REASONS = ("acc_seq", "block_switch", "lw_after_iw", "bank_war")
# 每种节点允许的字段（设计 §6.2 的节点表）。公共字段 id / kind / rows / origin 每种都必须有；
# 其余字段只能是 required 或 optional 里列出的，出现别的字段报 LTS-E-GRAPH-FIELD。
COMMON_NODE_FIELDS = ("id", "kind", "rows", "origin")
OPTIONAL_NODE_FIELDS = ("tile", "unit", "bank", "ops", "repeat", "out")
KIND_FIELDS = {
    "mem.read":   {"required": ("tile", "out"),         "optional": ("repeat",)},
    "mem.write":  {"required": ("tile",),               "optional": ()},
    "mxm.load_w": {"required": ("unit", "bank"),        "optional": ()},
    "mxm.acc":    {"required": ("unit", "bank", "out"), "optional": ()},
    "vxm.chain":  {"required": ("ops", "out"),          "optional": ()},
}
SXM_FIELDS = {"required": ("unit",), "optional": ("out",)}      # sxm.*：v0 仅预留，B1 会拒绝
TENSOR_FIELDS = ("dtype", "shape", "role", "hemisphere", "layout", "alias_of")
EDGE_GROUPS = ("stream", "state", "mem", "order")
ORDER_FIELDS = ("src", "dst", "ref", "lag", "reason")


def _seq(x):
    """JSON 数组 → tuple；其他值原样保留，由 _check_fields 拒绝（避免字符串被拆成单个字符）。"""
    return tuple(x) if isinstance(x, list) else x


def _no_extra(d: Mapping[str, Any], allowed, where: str) -> None:
    extra = sorted(set(d) - set(allowed))
    if extra:
        raise _err("FIELD", f"{where}: unknown field(s) {extra}")


def _parse_tensor(name: str, d: Mapping[str, Any]) -> Tensor:
    _no_extra(d, TENSOR_FIELDS, f"tensor {name!r}")
    return Tensor(
        dtype=d["dtype"],
        shape=_seq(d["shape"]),
        role=d["role"],
        hemisphere=d["hemisphere"],
        layout=d["layout"],
        alias_of=d.get("alias_of"),
    )


def _parse_node(d: Mapping[str, Any]) -> Node:
    _no_extra(d, COMMON_NODE_FIELDS + OPTIONAL_NODE_FIELDS, f"node {d.get('id')!r}")
    out = d.get("out")
    if isinstance(out, Mapping):
        _no_extra(out, ("dtype", "V"), f"node {d.get('id')!r} out")
        out = Out(dtype=out["dtype"], V=out["V"])
    return Node(
        id=d["id"],
        kind=d["kind"],
        rows=d["rows"],
        origin=_seq(d["origin"]),
        tile=d.get("tile"),
        unit=d.get("unit"),
        bank=d.get("bank"),
        ops=_seq(d["ops"]) if "ops" in d else None,
        repeat=d.get("repeat"),
        out=out,
    )


def _edge_pairs(group: str, es, arity: tuple[int, ...]):
    if not isinstance(es, list):
        raise _err("FIELD", f"edges.{group} must be a list")
    for e in es:
        if not isinstance(e, list) or len(e) not in arity:
            raise _err("FIELD", f"edges.{group}: each edge must be a list of {arity} items, got {e!r}")
    return es


def load_graph(path: str) -> TileGraph:
    """加载 tile_graph.json，做结构校验。"""
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    if not isinstance(d, Mapping):
        raise _err("PARSE", f"{path}: top level must be an object")
    if d.get("schema") != "lts-tile-graph/0.2":
        raise _err("PARSE", f"schema must be 'lts-tile-graph/0.2', got {d.get('schema')!r}")

    try:
        if not isinstance(d["tensors"], Mapping) or not isinstance(d["nodes"], list) \
                or not isinstance(d["edges"], Mapping):
            raise _err("FIELD", "tensors must be an object, nodes a list, edges an object")
        tensors = {name: _parse_tensor(name, t) for name, t in d["tensors"].items()}
        nodes = tuple(_parse_node(n) for n in d["nodes"])
        edges = d["edges"]
        _no_extra(edges, EDGE_GROUPS, "edges")
        edges_stream = tuple((e[0], e[1], e[2] if len(e) > 2 else 0)
                             for e in _edge_pairs("stream", edges.get("stream", []), (2, 3)))
        edges_state = tuple((e[0], e[1]) for e in _edge_pairs("state", edges.get("state", []), (2,)))
        edges_mem = tuple((e[0], e[1]) for e in _edge_pairs("mem", edges.get("mem", []), (2,)))
        orders = edges.get("order", [])
        if not isinstance(orders, list):
            raise _err("FIELD", "edges.order must be a list")
        for o in orders:
            _no_extra(o, ORDER_FIELDS, "order edge")
        edges_order = tuple(
            OrderEdge(src=o["src"], dst=o["dst"], ref=o["ref"], lag=o["lag"], reason=o["reason"])
            for o in orders
        )
    except (KeyError, TypeError, IndexError, AttributeError) as e:
        raise _err("FIELD", f"{path}: missing or malformed field: {e!r}") from None

    g = TileGraph(
        schema=d["schema"],
        machine=d.get("machine"),
        source=d.get("source", ""),
        tensors=tensors,
        nodes=nodes,
        edges_stream=edges_stream,
        edges_state=edges_state,
        edges_mem=edges_mem,
        edges_order=edges_order,
    )
    validate_structure(g)
    return g


def _is_int(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _check_fields(g: TileGraph) -> None:
    """字段取值与类型（LTS-E-GRAPH-FIELD）。JSON 读入的图和内存里搭的图走同一套检查。"""
    def bad(msg):
        raise _err("FIELD", msg)

    def is_str(x):
        return isinstance(x, str) and x != ""

    if not is_str(g.machine):
        bad(f"graph 'machine' must be a non-empty string, got {g.machine!r}")
    if not isinstance(g.source, str):
        bad(f"graph 'source' must be a string, got {g.source!r}")

    for name, t in g.tensors.items():
        if not is_str(name):
            bad(f"tensor name must be a non-empty string, got {name!r}")
        if t.dtype not in DTYPES:
            bad(f"tensor {name!r}: dtype {t.dtype!r} not in {DTYPES}")
        if t.role not in ROLES:
            bad(f"tensor {name!r}: role {t.role!r} not in {ROLES}")
        if t.hemisphere not in HEMISPHERES:
            bad(f"tensor {name!r}: hemisphere {t.hemisphere!r} not in {HEMISPHERES}")
        if t.layout not in LAYOUTS:
            bad(f"tensor {name!r}: layout {t.layout!r} not in {LAYOUTS}")
        if not (isinstance(t.shape, tuple) and 1 <= len(t.shape) <= 2
                and all(_is_int(x) and x >= 1 for x in t.shape)):
            bad(f"tensor {name!r}: shape must be 1 or 2 positive ints, got {t.shape!r}")
        if t.alias_of is not None and not is_str(t.alias_of):
            bad(f"tensor {name!r}: alias_of must be a tensor name, got {t.alias_of!r}")

    for n in g.nodes:
        if not is_str(n.id):
            bad(f"node id must be a non-empty string, got {n.id!r}")
        where = f"node {n.id!r} ({n.kind!r})"
        if not isinstance(n.kind, str):
            bad(f"{where}: kind must be a string")
        if n.kind in KIND_FIELDS:
            spec = KIND_FIELDS[n.kind]
        elif n.kind.startswith("sxm."):
            spec = SXM_FIELDS
        else:
            bad(f"{where}: unknown kind")
        for f in spec["required"]:
            if getattr(n, f) is None:
                bad(f"{where}: missing field {f!r}")
        for f in OPTIONAL_NODE_FIELDS:
            if f not in spec["required"] + spec["optional"] and getattr(n, f) is not None:
                bad(f"{where}: field {f!r} not allowed for this kind")
        if not (_is_int(n.rows) and n.rows >= 1):
            bad(f"{where}: rows must be a positive int, got {n.rows!r}")
        if not (isinstance(n.origin, tuple) and all(is_str(o) for o in n.origin)):
            bad(f"{where}: origin must be a list of op ids, got {n.origin!r}")
        if n.tile is not None and not is_str(n.tile):
            bad(f"{where}: tile must be a string, got {n.tile!r}")
        if n.unit is not None and not is_str(n.unit):
            bad(f"{where}: unit must be a string, got {n.unit!r}")
        if n.repeat is not None and not (_is_int(n.repeat) and n.repeat >= 1):
            bad(f"{where}: repeat must be a positive int, got {n.repeat!r}")
        if n.bank is not None and not (_is_int(n.bank) and n.bank >= 0):
            bad(f"{where}: bank must be a non-negative int, got {n.bank!r}")
        if n.ops is not None and not (isinstance(n.ops, tuple) and len(n.ops) >= 1
                                      and all(is_str(o) for o in n.ops)):
            bad(f"{where}: ops must be a non-empty list of strings, got {n.ops!r}")
        if n.out is not None:
            if not isinstance(n.out, Out):
                bad(f"{where}: out must be an object {{dtype, V}}, got {n.out!r}")
            if n.out.dtype not in DTYPES:
                bad(f"{where}: out.dtype {n.out.dtype!r} not in {DTYPES}")
            if not (_is_int(n.out.V) and n.out.V >= 1):
                bad(f"{where}: out.V must be a positive int, got {n.out.V!r}")

    for group, es in (("stream", g.edges_stream), ("state", g.edges_state), ("mem", g.edges_mem)):
        for e in es:
            if not (is_str(e[0]) and is_str(e[1])):
                bad(f"{group} edge {e!r}: endpoints must be node ids")
    for s_, d_, r0 in g.edges_stream:
        if not (_is_int(r0) and r0 >= 0):
            bad(f"stream edge {s_}->{d_}: r0 must be a non-negative int, got {r0!r}")
    for o in g.edges_order:
        if not (is_str(o.src) and is_str(o.dst)):
            bad(f"order edge {o.src!r}->{o.dst!r}: endpoints must be node ids")
        if o.ref not in REFS:
            bad(f"order edge {o.src}->{o.dst}: ref {o.ref!r} not in {REFS}")
        if o.reason not in REASONS:
            bad(f"order edge {o.src}->{o.dst}: reason {o.reason!r} not in {REASONS}")
        if not (_is_int(o.lag) or is_str(o.lag)):
            bad(f"order edge {o.src}->{o.dst}: lag must be an int or a machine param path, got {o.lag!r}")


def validate_structure(g: TileGraph) -> None:
    """B0 中不需要机器参数的全部检查（backend.md §2）。load_graph 自动调用；
    builder.build() 只查引用，内存里搭的图要做完整 B0 校验时调 validate_graph。"""
    _check_fields(g)

    ids = [n.id for n in g.nodes]
    if len(set(ids)) != len(ids):
        dup = next(i for i in ids if ids.count(i) > 1)
        raise _err("DUP", f"duplicate node id: {dup!r}")

    idset = set(ids)
    node_by_id = {n.id: n for n in g.nodes}

    def check_endpoint(nid: str, where: str):
        if nid not in idset:
            raise _err("REF", f"{where}: unknown node {nid!r}")

    for (s, d, r0) in g.edges_stream:
        check_endpoint(s, "stream src")
        check_endpoint(d, "stream dst")
        if node_by_id[s].out is None:
            raise _err("STREAM", f"stream src {s!r} has no 'out'")
        if r0 >= node_by_id[s].rows:
            raise _err("STREAM", f"stream edge {s}->{d}: r0={r0} must be < rows of {s!r} ({node_by_id[s].rows})")
    for (s, d) in g.edges_state:
        check_endpoint(s, "state src")
        check_endpoint(d, "state dst")
    for (s, d) in g.edges_mem:
        check_endpoint(s, "mem src")
        check_endpoint(d, "mem dst")
    for o in g.edges_order:
        check_endpoint(o.src, "order src")
        check_endpoint(o.dst, "order dst")

    # alias_of 必须指向存在的张量，且不成环；alias 是 transpose 的输出（设计 §4.3 ⓪ 规则 3、§5.4）：
    # rank 2，形状是源张量的转置，布局相反，dtype 与半球相同
    for name, t in g.tensors.items():
        alias_root(g, name)
        if t.alias_of is None:
            continue
        src = g.tensors[t.alias_of]
        why = None
        if len(t.shape) != 2 or len(src.shape) != 2:
            why = "alias must be rank 2 (transpose)"
        elif t.shape != src.shape[::-1]:
            why = f"shape {list(t.shape)} is not the transpose of {list(src.shape)}"
        elif t.layout == src.layout:
            why = f"layout must be the opposite of the source's ({src.layout})"
        elif t.dtype != src.dtype:
            why = f"dtype {t.dtype} != source dtype {src.dtype}"
        elif t.hemisphere != src.hemisphere:
            why = f"hemisphere {t.hemisphere} != source hemisphere {src.hemisphere} (same MEM image)"
        if why:
            raise _err("REF", f"tensor {name!r} alias_of {t.alias_of!r}: {why}")

    # mem.write 必须有唯一 stream 前驱（V_in 来源）
    preds_of_write: dict[str, list[str]] = {}
    for (s, d, _r0) in g.edges_stream:
        if node_by_id[d].kind == "mem.write":
            preds_of_write.setdefault(d, []).append(s)
    for n in g.nodes:
        if n.kind == "mem.write" and len(preds_of_write.get(n.id, [])) != 1:
            raise _err("STREAM", f"mem.write {n.id!r} must have exactly one stream pred, "
                                 f"got {len(preds_of_write.get(n.id, []))}")

    # ---- 一致性检查 ----
    # rows 与 tile 的物理 word 数一致；广播读：rows == repeat，tile 只有 1 个 word
    for n in g.nodes:
        if n.tile is None:
            continue
        _root, (w0, w1), _lanes = tile_region(g, n.tile)
        words = w1 - w0
        if n.repeat is None:
            if n.rows != words:
                raise _err("ROWS", f"node {n.id!r}: rows={n.rows} but tile {n.tile!r} has {words} words")
        elif n.rows != n.repeat or words != 1:
            raise _err("ROWS", f"node {n.id!r}: broadcast read needs rows == repeat and a 1-word tile, "
                               f"got rows={n.rows}, repeat={n.repeat}, {words} words")

    # mem 边：mem.write → mem.read，同一份 MEM 映像（展开 alias 后同一个张量），物理区域相交
    for (s, d) in g.edges_mem:
        u, v = node_by_id[s], node_by_id[d]
        if (u.kind, v.kind) != ("mem.write", "mem.read"):
            raise _err("MEM", f"mem edge {s}->{d} must go mem.write -> mem.read, got {u.kind} -> {v.kind}")
        ru, wu, lu = tile_region(g, u.tile)
        rv, wv, lv = tile_region(g, v.tile)
        if ru != rv:
            raise _err("MEM", f"mem edge {s}->{d}: tiles are on different MEM images ({ru!r} vs {rv!r})")
        if not _overlap(wu, wv) or (lu is not None and lv is not None and not _overlap(lu, lv)):
            raise _err("MEM", f"mem edge {s}->{d}: tiles {u.tile!r} and {v.tile!r} do not overlap")

    # state 边：mxm.load_w → mxm.acc，同一个 plane、同一个 bank
    for (s, d) in g.edges_state:
        u, v = node_by_id[s], node_by_id[d]
        if (u.kind, v.kind) != ("mxm.load_w", "mxm.acc"):
            raise _err("STATE", f"state edge {s}->{d} must go mxm.load_w -> mxm.acc, got {u.kind} -> {v.kind}")
        if (u.unit, u.bank) != (v.unit, v.bank):
            raise _err("STATE", f"state edge {s}->{d}: load_w on {u.unit}/bank{u.bank} "
                                f"but acc on {v.unit}/bank{v.bank}")


def _overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return max(a[0], b[0]) < min(a[1], b[1])


def alias_root(g: TileGraph, name: str) -> str:
    """沿 alias_of 展开到源张量（共享同一份 MEM 映像）。指向不存在的张量或成环报 LTS-E-GRAPH-REF。"""
    seen = []
    while True:
        if name not in g.tensors:
            raise _err("REF", f"unknown tensor {name!r}" + (f" (alias chain {seen})" if seen else ""))
        t = g.tensors[name]
        if t.alias_of is None:
            return name
        if name in seen:
            raise _err("REF", f"alias_of cycle: {seen + [name]}")
        seen.append(name)
        name = t.alias_of


def validate_units(g: TileGraph, m: Machine) -> None:
    """mxm.* 节点的 unit 必须在机器参数的 plane 名单里。"""
    planes = {inst["id"] for inst in m.raw["mxm"]["instances"]}
    for n in g.nodes:
        if n.kind.startswith("mxm."):
            if n.unit not in planes:
                raise _err("UNIT", f"node {n.id!r} unit {n.unit!r} not in machine planes {sorted(planes)}")


def parse_tile(tile: str) -> tuple[str, tuple[tuple[int, int], ...]]:
    """"H[0:128,0:320]" → ("H", ((0, 128), (0, 320)))；"B1[0:320]" → ("B1", ((0, 320),))。

    坐标是逻辑坐标，每一维为左闭右开区间。格式不对报 LTS-E-GRAPH-TILE。
    """
    i = tile.find("[")
    if i <= 0 or not tile.endswith("]"):
        raise _err("TILE", f"malformed tile: {tile!r}")
    ranges = []
    for part in tile[i + 1:-1].split(","):
        try:
            a, b = (int(x) for x in part.split(":"))
        except ValueError:
            raise _err("TILE", f"malformed tile: {tile!r}") from None
        ranges.append((a, b))
    return tile[:i], tuple(ranges)


def tile_region(g: TileGraph, tile: str) -> tuple[str, tuple[int, int], tuple[int, int] | None]:
    """tile → (MEM 映像的源张量, 物理 word 区间, 物理 lane 区间)。区间左闭右开。

    按 tile 所指张量**自己的** layout 换算（设计 §3.2；alias 有自己的布局，例如 Kt 是 K 的 lane 版本）：
      rank 2，row 布局：word = 逻辑第一维 [r0, r1)，lane = 逻辑第二维 [c0, c1)
      rank 2，lane 布局：word = 逻辑第二维 [c0, c1)，lane = 逻辑第一维 [r0, r1)
      rank 1，row 布局：1 个 word [0, 1)，lane = [n0, n1)
      rank 1，lane 布局：word = [n0, n1)；每个 word 复制到所有 lane，lane 数由消费者决定，返回 None
    同时检查 tile 的维数与张量一致、坐标落在张量形状内，否则报 LTS-E-GRAPH-TILE。
    """
    name, ranges = parse_tile(tile)
    if name not in g.tensors:
        raise _err("REF", f"tile {tile!r} refers to unknown tensor {name!r}")
    t = g.tensors[name]
    if len(ranges) != len(t.shape):
        raise _err("TILE", f"tile {tile!r} has rank {len(ranges)}, tensor {name!r} has rank {len(t.shape)}")
    for (a, b), dim in zip(ranges, t.shape):
        if not 0 <= a < b <= dim:
            raise _err("TILE", f"tile {tile!r} out of bounds for tensor {name!r} shape {list(t.shape)}")
    root = alias_root(g, name)
    if len(ranges) == 1:
        (n0, n1), = ranges
        return (root, (0, 1), (n0, n1)) if t.layout == "row" else (root, (n0, n1), None)
    rr, cc = ranges
    return (root, rr, cc) if t.layout == "row" else (root, cc, rr)


def tile_lanes(g: TileGraph, tile: str) -> int | None:
    """tile 的物理 lane 数；rank 1 的 lane 布局返回 None（见 tile_region）。"""
    _root, _words, lanes = tile_region(g, tile)
    return None if lanes is None else lanes[1] - lanes[0]


def validate_lane_width(g: TileGraph, m: Machine) -> None:
    """v0 限制：每个 tile 不超过一个向量的 lane 数 = superlanes × lanes_per_superlane（当前 320）。

    这条限制保证 VXM 每行只有 1 组（B1 的 groups = 1），每行向量数 V 只由 dtype 字节数决定。
    超过报 LTS-E-GRAPH-LANES。
    """
    limit = m.param("geometry.superlanes") * m.param("geometry.lanes_per_superlane")
    for n in g.nodes:
        if n.tile is None:
            continue
        lanes = tile_lanes(g, n.tile)
        if lanes is not None and lanes > limit:
            raise _err("LANES", f"node {n.id!r} tile {n.tile!r} has {lanes} lanes > limit {limit}")


def validate_graph(g: TileGraph, m: Machine) -> None:
    """完整的 B0 校验：结构（validate_structure）+ 需要机器参数的检查（unit 名单、lane 宽）。
    内存里搭的图（builder、DSL）进后端前调这个；load_graph 已做过结构部分，重复调用无副作用。"""
    validate_structure(g)
    validate_units(g, m)
    validate_lane_width(g, m)


def resolve_lags(g: TileGraph, m: Machine) -> ResolvedGraph:
    """把 order 边的 lag（可为参数字符串）解析成 int；state/mem 展开默认 ref/lag。

    默认值（backend.md §4.6）：
      state → ("issue_end", mxm.lw_to_acc_gap)
      mem   → ("done",       mem.raw_gap)
    """
    out: list[ResolvedEdge] = []
    for (s, d) in g.edges_state:
        out.append(("state", s, d, "issue_end", m.param("mxm.lw_to_acc_gap"), None))
    for (s, d) in g.edges_mem:
        out.append(("mem", s, d, "done", m.param("mem.raw_gap"), None))
    for o in g.edges_order:
        lag = o.lag
        if isinstance(lag, str):
            lag = m.param(lag)
        out.append(("order", o.src, o.dst, o.ref, int(lag), o.reason))
    return ResolvedGraph(graph=g, non_stream_edges=tuple(out))
