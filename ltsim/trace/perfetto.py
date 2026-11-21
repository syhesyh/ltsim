"""Perfetto / Chrome Trace Event 输出（backend.md §7，设计 §8.2）。

先用整数拍生成时间段和流向事件，再统一换算成 µs：ts = cycle × cycle_ns / 1000（界面上 1 ns = 1 拍）。
生成后跑自检（check_trace），不通过报 LTS-E-TRACE。同一输入两次运行，trace.json 逐字节相同。
"""
from __future__ import annotations

import json
from math import ceil
from fractions import Fraction as F
from typing import Any, Mapping

from ..backend.bounds import critical_path
from ..backend.chains import ChainGraph
from ..backend.metrics import node_times, op_spans
from ..backend.scheduler import BoundBy, Placed
from ..backend.work import write_v_in
from ..errors import LTSError
from ..graph.tile_graph import TileGraph
from ..machine.model import Machine

PROCESSES = [(1, "Ops (LTS-IR)"), (2, "Chains"), (3, "MXM"), (4, "VXM"), (5, "SXM"),
             (6, "MEM-W"), (7, "MEM-E"), (8, "Counters"), (9, "Critical path")]


def counter_names(resources: Mapping[str, Any]) -> list[str]:
    """计数器：资源表里全部 SRF 流池（按名字排序）、vxm.alu、mem.*（按资源表的顺序：mem.W.rd、mem.W.wr、mem.E.rd、mem.E.wr）。"""
    return (sorted(r for r in resources if r.startswith("srf.")) + [r for r in resources if r == "vxm.alu"]
            + [r for r in resources if r.startswith("mem.")])

_EPS = 1e-9                       # µs；1 拍 = cycle_ns / 1000 µs，远大于它


def _frac(x: F) -> str:
    x = F(x)
    return str(x.numerator) if x.denominator == 1 else f"{x.numerator}/{x.denominator}"


def _bound_by(b: BoundBy) -> dict[str, Any]:
    d: dict[str, Any] = {"kind": b.kind}
    if b.edge is not None:
        d["edge"] = list(b.edge)
    if b.resource is not None:
        d["resource"] = b.resource
    if b.blocker is not None:
        d["blocker"] = b.blocker
    return d


def _color(items: list[tuple[int, int, str]]) -> dict[str, int]:
    """贪心区间着色：按 (start, end, key) 排序，放进第一个"上一段已结束"的行。返回 key → 行号（从 1 起）。"""
    rows: list[int] = []
    out = {}
    for s, e, key in sorted(items):
        for i, last in enumerate(rows):
            if last <= s:
                rows[i] = e
                out[key] = i + 1
                break
        else:
            rows.append(e)
            out[key] = len(rows)
    return out


def build_trace(g: TileGraph, cg: ChainGraph, placed: Placed, m: Machine, *,
                provenance: Mapping[str, Any] | None = None, stream_flows: bool = False,
                no_hop_tail_total: int | None = None) -> dict[str, Any]:
    """资源表（MXM / SXM 线程、计数器）取 cg.resources。no_hop_tail_total：不带 hop 尾巴的那次调度的总周期，
    给出时写进 otherData（trace 只画主结果）。"""
    cycle_ns = float(m.raw.get("display", {}).get("cycle_ns", 1.0))

    def us(cycle: int) -> float:
        return cycle * cycle_ns / 1000

    by = {n.id: n for n in g.nodes}
    times = node_times(cg, placed)
    v_in = write_v_in(g)
    chain = {c.id: c for c in cg.chains}
    resources = cg.resources

    # ---- 节点的时间段：占用区间；写节点取 [start+L, start+L+D) ----
    span: dict[str, tuple[int, int]] = {}
    for n in g.nodes:
        s, ie, _d = times[n.id]
        if n.kind == "mem.write":
            L = cg.works[n.id].L
            span[n.id] = (s + L, ie + L)
        else:
            span[n.id] = (s, ie)

    slices: list[dict[str, Any]] = []          # 整数拍：pid, tid, start, end, name, args
    threads: dict[tuple[int, int], str] = {}

    def add(pid, tid, start, end, name, args):
        slices.append({"pid": pid, "tid": tid, "start": start, "end": end, "name": name, "args": args})

    # pid 1：每个算子一个线程
    for i, (op, (a, z)) in enumerate(op_spans(g, cg, placed).items(), start=1):
        threads[(1, i)] = op
        add(1, i, a, z, op, {"op": op})

    # pid 2：链，区间着色
    rows = _color([(placed.start[c.id], placed.start[c.id] + c.length, c.id) for c in cg.chains])
    for c in cg.chains:
        t = placed.start[c.id]
        threads[(2, rows[c.id])] = f"row {rows[c.id]}"
        add(2, rows[c.id], t, t + c.length, c.id,
            {"chain": c.id, "nodes": list(c.nodes), "rate": round(float(c.rate), 6), "rate_exact": _frac(c.rate),
             "bound_by": _bound_by(placed.bound_by[c.id])})

    # pid 3：每个 MXM 独占资源一个线程（按资源名排序，含没用到的）；pid 5：SXM（v0 留空）
    mxm_tid = {name: i for i, name in enumerate(sorted(r for r in resources if r.startswith("mxm.")), start=1)}
    for name, i in mxm_tid.items():
        threads[(3, i)] = name
    sxm_tid = {name: i for i, name in enumerate(sorted(r for r in resources if r.startswith("sxm.")), start=1)}
    for name, i in sxm_tid.items():
        threads[(5, i)] = name

    # 节点放到哪个 (pid, tid)
    node_thread: dict[str, tuple[int, int]] = {}
    vxm_nodes = [n for n in g.nodes if n.kind == "vxm.chain"]
    vxm_rows = _color([(span[n.id][0], span[n.id][1], n.id) for n in vxm_nodes])
    mem_rows = {}
    for h, pid in (("W", 6), ("E", 7)):
        ns = [n for n in g.nodes if n.kind in ("mem.read", "mem.write") and cg.works[n.id].category == f"MEM_{h}"]
        mem_rows[pid] = _color([(span[n.id][0], span[n.id][1], n.id) for n in ns])
        for n in ns:
            node_thread[n.id] = (pid, mem_rows[pid][n.id])
    for n in vxm_nodes:
        node_thread[n.id] = (4, vxm_rows[n.id])
    for n in g.nodes:
        if n.kind == "mxm.load_w":
            node_thread[n.id] = (3, mxm_tid[f"{n.unit}.weight"])
        elif n.kind == "mxm.acc":
            node_thread[n.id] = (3, mxm_tid[f"{n.unit}.compute"])

    for n in g.nodes:
        pid, tid = node_thread[n.id]
        threads.setdefault((pid, tid), f"row {tid}")
        c = chain[cg.chain_of[n.id]]
        w = cg.works[n.id]
        s, ie, done = times[n.id]
        amount = {r.resource: r.amount for r in c.reservations if r.node == n.id}
        args = {"node": n.id, "chain": c.id, "kind": n.kind, "origin": list(n.origin)}
        if n.tile is not None:
            args["tile"] = n.tile
        if n.unit is not None:
            args["unit"] = n.unit
        args.update({"rows": n.rows, "rate": round(float(c.rate), 6), "rate_exact": _frac(c.rate),
                     "V": n.out.V if n.out is not None else v_in.get(n.id),
                     "usage": amount, "L": w.L, "start": s, "issue_end": ie, "done": done,
                     "bound_by": _bound_by(placed.bound_by[c.id])})
        add(pid, tid, span[n.id][0], span[n.id][1], f"{n.kind}:{'+'.join(n.origin)}", args)

    # pid 9：关键路径，区间着色
    cp = critical_path(cg, placed)
    cp_rows = _color([(placed.start[s.chain], placed.start[s.chain] + chain[s.chain].length, s.chain) for s in cp])
    for k, st in enumerate(cp, start=1):
        t = placed.start[st.chain]
        threads[(9, cp_rows[st.chain])] = f"row {cp_rows[st.chain]}"
        add(9, cp_rows[st.chain], t, t + chain[st.chain].length, f"{k}. {st.chain}",
            {"step": k, "chain": st.chain, "bound_by": _bound_by(st.bound_by)})

    # pid 8：计数器，每次占用量变化输出一个点
    counters: list[dict[str, Any]] = []
    for name in counter_names(resources):
        delta: dict[int, int] = {}
        for c in cg.chains:
            t = placed.start[c.id]
            for r in c.reservations:
                if r.resource == name:
                    delta[t + r.start] = delta.get(t + r.start, 0) + r.amount
                    delta[t + r.end] = delta.get(t + r.end, 0) - r.amount
        level, pts = 0, [(0, 0)]
        for t in sorted(delta):
            if delta[t] == 0:
                continue
            level += delta[t]
            if pts[-1][0] == t:
                pts[-1] = (t, level)
            else:
                pts.append((t, level))
        counters += [{"name": name, "ts": t, "value": v} for t, v in pts]

    # ---- 流向事件 ----
    flows: list[dict[str, Any]] = []
    skipped: list[str] = []

    def add_flow(cat, name, u, v, s_ts):
        a, b = span[u]
        s_ts = min(max(s_ts, a), b - 1)                 # s 落在源节点的时间段内
        va, vb = span[v]
        f_ts = max(va, s_ts)                            # f 不早于 s，且落在目的节点的时间段内
        if f_ts >= vb:
            skipped.append(f"{cat}:{u}->{v}")
            return
        flows.append({"id": len(flows) + 1, "cat": cat, "name": name, "src": u, "dst": v,
                      "s": (*node_thread[u], s_ts), "f": (*node_thread[v], f_ts)})

    for e in cg.edges:                                  # 每条链间边各自带 ref / reason，平行边互不干扰
        kind, u, v = e.edge
        a, b = span[u]
        s_ts = a if e.ref == "start" else b - 1         # 按这条边依赖的时刻放 s（时间段的起点 / 最后一拍）
        add_flow(kind, e.reason or kind, u, v, s_ts)
    if stream_flows:
        for u, v, r0 in g.edges_stream:
            c = chain[cg.chain_of[u]]
            first_out = times[u][0] + cg.works[u].L + ceil(F(r0) / c.rate)   # 流给 v 的第一行离开 u 的时刻
            add_flow("stream", "stream", u, v, first_out)

    # ---- 组装 ----
    ev: list[dict[str, Any]] = []
    for pid, name in PROCESSES:
        ev.append({"ph": "M", "name": "process_name", "pid": pid, "tid": 0, "args": {"name": name}})
        ev.append({"ph": "M", "name": "process_sort_index", "pid": pid, "tid": 0, "args": {"sort_index": pid}})
    for (pid, tid), name in sorted(threads.items()):
        ev.append({"ph": "M", "name": "thread_name", "pid": pid, "tid": tid, "args": {"name": name}})
        ev.append({"ph": "M", "name": "thread_sort_index", "pid": pid, "tid": tid, "args": {"sort_index": tid}})
    for sl in sorted(slices, key=lambda x: (x["pid"], x["tid"], x["start"], x["end"], x["name"])):
        ev.append({"ph": "X", "name": sl["name"], "pid": sl["pid"], "tid": sl["tid"],
                   "ts": us(sl["start"]), "dur": us(sl["end"] - sl["start"]), "args": sl["args"]})
    for c in counters:
        ev.append({"ph": "C", "name": c["name"], "pid": 8, "tid": 0, "ts": us(c["ts"]), "args": {"value": c["value"]}})
    for fl in flows:
        (sp, st, sts), (fp, ft, fts) = fl["s"], fl["f"]
        meta = {"src": fl["src"], "dst": fl["dst"]}
        ev.append({"ph": "s", "id": fl["id"], "cat": fl["cat"], "name": fl["name"], "pid": sp, "tid": st,
                   "ts": us(sts), "args": meta})
        ev.append({"ph": "f", "bp": "e", "id": fl["id"], "cat": fl["cat"], "name": fl["name"], "pid": fp, "tid": ft,
                   "ts": us(fts), "args": meta})

    other = {"schema": "lts-trace/0.1", "cycle_ns": cycle_ns, "total_cycles": placed.total}
    if no_hop_tail_total is not None:
        other["no_hop_tail_total_cycles"] = no_hop_tail_total
    other.update({"skipped_flows": skipped, "provenance": dict(provenance or {})})
    trace = {"traceEvents": ev, "displayTimeUnit": "ns", "otherData": other}
    bad = check_trace(trace)
    if bad:
        raise LTSError("LTS-E-TRACE", "trace failed self-check: " + "; ".join(bad[:5]))
    return trace


def check_trace(trace: Mapping[str, Any]) -> list[str]:
    """backend.md §7.2 的自检（在最终 JSON 上做）：
    同一 (pid, tid) 上的 X 事件两两不部分重叠；每个 s 都有同 id 的 f，二者都落在所在线程的某个 X 事件内，且 f 不早于 s；
    s 与 f 的 cat / name 一致，f 带 bp "e"；所有 ts、dur 非负，X 的 dur > 0；事件缺字段报出来而不是抛异常。
    返回问题列表，空表示通过。"""
    bad: list[str] = []
    xs: dict[tuple[int, int], list[tuple[float, float, str]]] = {}
    flows: dict[Any, dict[str, Any]] = {}
    need = {"X": ("name", "pid", "tid", "ts", "dur"), "C": ("name", "pid", "ts", "args"),
            "s": ("id", "cat", "name", "pid", "tid", "ts"), "f": ("id", "cat", "name", "pid", "tid", "ts", "bp"),
            "M": ("name", "pid", "args")}
    for e in trace.get("traceEvents", []):
        ph = e.get("ph") if isinstance(e, Mapping) else None
        if ph not in need:
            bad.append(f"event with unknown or missing ph: {e!r}")
            continue
        missing = [k for k in need[ph] if k not in e]
        if missing:
            bad.append(f"'{ph}' event missing field(s) {missing}: {e!r}")
            continue
        if ph in ("X", "C", "s", "f") and e["ts"] < 0:
            bad.append(f"negative ts: {e}")
        if ph == "X":
            if e["dur"] <= 0:
                bad.append(f"non-positive dur: {e['name']}")
            xs.setdefault((e["pid"], e["tid"]), []).append((e["ts"], e["ts"] + e["dur"], e["name"]))
        elif ph in ("s", "f"):
            if ph == "f" and e["bp"] != "e":
                bad.append(f"flow id {e['id']}: 'f' must have bp \"e\" (bind to the enclosing slice)")
            if ph in flows.setdefault(e["id"], {}):
                bad.append(f"flow id {e['id']} has two '{ph}' events")
            flows[e["id"]][ph] = e
    for key, iv in xs.items():
        iv.sort()
        for i, (a0, a1, an) in enumerate(iv):
            for b0, b1, bn in iv[i + 1:]:
                if b0 >= a1 - _EPS:
                    break
                if b1 > a1 + _EPS:           # 有交叠且不是嵌套
                    bad.append(f"partial overlap on {key}: {an} [{a0},{a1}) vs {bn} [{b0},{b1})")

    def inside(e):
        return any(a - _EPS <= e["ts"] < b - _EPS for a, b, _n in xs.get((e["pid"], e["tid"]), []))

    for fid, d in flows.items():
        if set(d) != {"s", "f"}:
            bad.append(f"flow id {fid} is missing its {'f' if 's' in d else 's'} event")
            continue
        for ph in ("s", "f"):
            if not inside(d[ph]):
                bad.append(f"flow id {fid}: '{ph}' at {d[ph]['ts']} is not inside a slice on "
                           f"({d[ph]['pid']}, {d[ph]['tid']})")
        if d["f"]["ts"] < d["s"]["ts"] - _EPS:
            bad.append(f"flow id {fid}: f ({d['f']['ts']}) is earlier than s ({d['s']['ts']})")
        if (d["s"]["cat"], d["s"]["name"]) != (d["f"]["cat"], d["f"]["name"]):
            bad.append(f"flow id {fid}: s and f differ in cat/name (Perfetto pairs flows by id + cat + name)")
    return bad


def dumps_trace(trace: Mapping[str, Any]) -> str:
    return json.dumps(trace, ensure_ascii=False, separators=(",", ":")) + "\n"


def write_trace(path: str, trace: Mapping[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(dumps_trace(trace))
