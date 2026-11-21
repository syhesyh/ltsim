"""B4 输出 result.json（设计 §8.1，schema lts-result/0.1）。

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

RESULT_SCHEMA = "lts-result/0.1"


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


def make_provenance(g: TileGraph, m: Machine, *, policy: str, mem_overlap: str,
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
            "policy": policy, "seed": None, "mem_overlap": mem_overlap}


def build_result(g: TileGraph, cg: ChainGraph, placed: Placed, m: Machine,
                 provenance: Mapping[str, Any], diagnostics: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """diagnostics：不影响结果的警告，例如 {"level": "warning", "code": "LTS-W-MACHINE-ID", "msg": ...}。"""
    resources = m.resources()
    bad = verify(cg, placed, resources)
    if bad:
        raise LTSError("LTS-E-INTERNAL", "schedule failed self-check: " + "; ".join(bad))
    b = compute_bounds(cg, placed, resources)
    if b.total < b.lb:
        raise LTSError("LTS-E-INTERNAL", f"total {b.total} < lower bound {b.lb}")
    times = node_times(cg, placed)
    usage = resource_usage(cg, placed, resources)
    cp = critical_path(cg, placed)
    return {
        "schema": RESULT_SCHEMA,
        "provenance": dict(provenance),
        "total_cycles": b.total,
        "bounds": {"path": b.path, "resource": b.resource, "resource_exact": _frac(b.resource_exact),
                   "resource_name": b.resource_name, "lb": b.lb, "gap": _num(b.gap),
                   "mxm_roofline": _num(mxm_roofline(g, cg, placed, m))},
        "chains": [{"id": c.id, "start": placed.start[c.id], "len": c.length,
                    "end": placed.start[c.id] + c.length,
                    "rate": _num(c.rate), "rate_exact": _frac(c.rate),
                    "bound_by": _bound_by(placed.bound_by[c.id], cg)} for c in cg.chains],
        "nodes": [{"id": n.id, "chain": cg.chain_of[n.id], "start": times[n.id][0],
                   "issue_end": times[n.id][1], "done": times[n.id][2]} for n in g.nodes],
        "resources": [{"id": name, "kind": u["kind"], "cap": u["cap"], "busy": u["busy"], "peak": u["peak"],
                       "utilization": _num(u["utilization"])} for name, u in usage.items()],
        "srf": {d: {"peak": usage[f"srf.{d}"]["peak"], "avg": _num(usage[f"srf.{d}"]["avg"])}
                for d in ("W2E", "E2W")},
        "critical_path": [s.chain for s in cp],
        "critical_path_detail": [{"chain": s.chain, "start": placed.start[s.chain],
                                  "end": placed.start[s.chain] + cg.chain(s.chain).length,
                                  "bound_by": _bound_by(s.bound_by, cg)} for s in cp],
        "ops": [{"id": op, "start": a, "end": z} for op, (a, z) in op_spans(g, cg, placed).items()],
        "diagnostics": list(diagnostics or []),
    }


def dumps_result(result: Mapping[str, Any]) -> str:
    return json.dumps(result, ensure_ascii=False, indent=2) + "\n"


def write_result(path: str, result: Mapping[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(dumps_result(result))
