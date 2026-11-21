"""B4 输出 result.json（设计 §8.1，schema lts-result/0.2）。

同一输入两次运行，result.json 逐字节相同：不写时间戳，键的顺序固定，浮点统一保留 6 位小数。
"""
from __future__ import annotations

import hashlib
import json
from fractions import Fraction as F
from typing import Any, Mapping

import yaml

from ..errors import LTSError
from ..gen.builder import graph_to_dict
from ..graph.tile_graph import TileGraph
from ..machine.model import Machine
from .bounds import compute_bounds, critical_path, mxm_roofline
from .chains import ChainGraph
from .metrics import node_times, op_spans, resource_usage
from .scheduler import BoundBy, Placed, verify

RESULT_SCHEMA = "lts-result/0.2"


def _num(x: F | float) -> float:
    return round(float(x), 6)


def _frac(x: F) -> str:
    x = F(x)
    return str(x.numerator) if x.denominator == 1 else f"{x.numerator}/{x.denominator}"


def _bound_by(b: BoundBy, cg: ChainGraph) -> dict[str, Any]:
    d: dict[str, Any] = {"kind": b.kind}
    if b.kind == "edge":
        d["edge"] = list(b.edge)
        d["from_chain"] = b.blocker
    elif b.kind == "resource":
        d["resource"] = b.resource
        d["blocker"] = b.blocker
    elif b.kind == "policy":
        d["blocker"] = b.blocker
    return d


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def make_provenance(g: TileGraph, m: Machine, *, policy: str, mem_overlap: str, srf_pool: str, lw_streams: str,
                    tile_graph_path: str | None = None, machine_path: str | None = None,
                    command: str | None = None) -> dict[str, Any]:
    """来源信息。文件路径给出时取文件内容的 sha256，否则取规范化内容（JSON / YAML，键排序）的 sha256。
    M0 不读取 git 信息，commit / dirty 写 null（roadmap 待定项 T10）；M0 没有 LTS-IR 与合同文件，对应字段写 null。"""
    if tile_graph_path is not None:
        with open(tile_graph_path, "rb") as f:
            tg = _sha256_bytes(f.read())
    else:
        tg = _sha256_bytes(json.dumps(graph_to_dict(g), sort_keys=True, ensure_ascii=False).encode())
    if machine_path is not None:
        with open(machine_path, "rb") as f:
            ms = _sha256_bytes(f.read())
    else:
        ms = _sha256_bytes(yaml.safe_dump(dict(m.raw), sort_keys=True, allow_unicode=True).encode())
    return {"ltsim_commit": None, "dirty": None, "command": command,
            "ir_sha256": None, "tile_graph_sha256": tg, "machine_sha256": ms, "contract_sha256": None,
            "policy": policy, "seed": None, "mem_overlap": mem_overlap,
            "srf_pool": srf_pool, "lw_streams": lw_streams}


def _checked_bounds(cg: ChainGraph, placed: Placed, what: str):
    bad = verify(cg, placed)
    if bad:
        raise LTSError("LTS-E-INTERNAL", f"{what} failed self-check: " + "; ".join(bad))
    b = compute_bounds(cg, placed)
    if b.total < b.lb:
        raise LTSError("LTS-E-INTERNAL", f"{what}: total {b.total} < lower bound {b.lb}")
    return b


def _bounds_dict(b) -> dict[str, Any]:
    return {"path": b.path, "resource": b.resource, "resource_exact": _frac(b.resource_exact),
            "resource_name": b.resource_name, "lb": b.lb, "gap": _num(b.gap)}


def _cp_detail(cg: ChainGraph, placed: Placed, cp) -> list[dict[str, Any]]:
    return [{"chain": s.chain, "start": placed.start[s.chain], "end": placed.start[s.chain] + cg.chain(s.chain).length,
             "bound_by": _bound_by(s.bound_by, cg)} for s in cp]


def build_result(g: TileGraph, cg: ChainGraph, placed: Placed, m: Machine,
                 provenance: Mapping[str, Any], diagnostics: list[dict[str, Any]] | None = None, *,
                 no_hop_tail: tuple[ChainGraph, Placed] | None = None) -> dict[str, Any]:
    """cg / placed 是主结果（流预留带 hop 尾巴），决定 result.json 的全部字段；资源表取 cg.resources。
    no_hop_tail：同一张图、同样选项但 build_chains(..., hop_tail=False) 得到的 (链图, 调度)，
    写进 "no_hop_tail" 字段（总周期、下界、关键路径）；不给时该字段为 null。
    diagnostics：不影响结果的警告，例如 {"level": "warning", "code": "LTS-W-MACHINE-ID", "msg": ...}。"""
    b = _checked_bounds(cg, placed, "schedule")
    times = node_times(cg, placed)
    usage = resource_usage(cg, placed)
    cp = critical_path(cg, placed)
    secondary = None
    if no_hop_tail is not None:
        cg2, placed2 = no_hop_tail
        o1, o2 = cg.options, cg2.options
        # hop_tail 只影响流预留，不影响链的 rate/D/offset/length 与链间边，所以这些都应逐项相同
        same_graph = ([(c.id, c.nodes, c.rate, dict(c.D), dict(c.offset), c.length) for c in cg2.chains]
                      == [(c.id, c.nodes, c.rate, dict(c.D), dict(c.offset), c.length) for c in cg.chains]
                      and list(cg2.edges) == list(cg.edges)
                      and dict(cg2.works) == dict(cg.works))
        same_options = (o1.get("mem_overlap") == o2.get("mem_overlap")
                        and o1.get("srf_pool") == o2.get("srf_pool")
                        and o1.get("lw_streams") == o2.get("lw_streams")
                        and dict(cg2.resources) == dict(cg.resources))
        hop_tail_pair = {o1.get("hop_tail"), o2.get("hop_tail")} == {True, False}
        if not (same_graph and same_options and hop_tail_pair):
            raise LTSError("LTS-E-ARG",
                           "no_hop_tail must come from the same graph and build_chains options "
                           "as the primary, with only hop_tail flipped")
        b2 = _checked_bounds(cg2, placed2, "no-hop-tail schedule")
        cp2 = critical_path(cg2, placed2)
        secondary = {"total_cycles": b2.total, "bounds": _bounds_dict(b2),
                     "critical_path": [s.chain for s in cp2], "critical_path_detail": _cp_detail(cg2, placed2, cp2)}
    return {
        "schema": RESULT_SCHEMA,
        "provenance": dict(provenance),
        "total_cycles": b.total,
        "bounds": {**_bounds_dict(b), "mxm_roofline": _num(mxm_roofline(g, cg, placed, m))},
        "chains": [{"id": c.id, "start": placed.start[c.id], "len": c.length,
                    "end": placed.start[c.id] + c.length,
                    "rate": _num(c.rate), "rate_exact": _frac(c.rate),
                    "bound_by": _bound_by(placed.bound_by[c.id], cg)} for c in cg.chains],
        "nodes": [{"id": n.id, "chain": cg.chain_of[n.id], "start": times[n.id][0],
                   "issue_end": times[n.id][1], "done": times[n.id][2]} for n in g.nodes],
        "resources": [{"id": name, "kind": u["kind"], "cap": u["cap"], "busy": u["busy"], "peak": u["peak"],
                       "utilization": _num(u["utilization"])} for name, u in usage.items()],
        "srf": {name: {"peak": u["peak"], "avg": _num(u["avg"])}
                for name, u in usage.items() if name.startswith("srf.")},       # usage 已按资源名排序
        "critical_path": [s.chain for s in cp],
        "critical_path_detail": _cp_detail(cg, placed, cp),
        "no_hop_tail": secondary,
        "ops": [{"id": op, "start": a, "end": z} for op, (a, z) in op_spans(g, cg, placed).items()],
        "diagnostics": list(diagnostics or []),
    }


def dumps_result(result: Mapping[str, Any]) -> str:
    return json.dumps(result, ensure_ascii=False, indent=2) + "\n"


def write_result(path: str, result: Mapping[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(dumps_result(result))
