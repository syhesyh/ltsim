"""LTSim 命令行（interfaces.md §6）。

    python -m ltsim gen mlp  --out <新目录>
    python -m ltsim gen gemm --M 256 --K 640 --N 1280 --tile-m 128 [...] --out <新目录>
    python -m ltsim run <tile_graph.json> [--machine <id 或 yaml 路径>] --out <新目录>
                        [--policy bottom_level|topo] [--mem-overlap tile|row] [--trace-stream-flows]

输出目录必须是新目录：先全部算完、成功后才建目录写文件，失败时可以用同一个名字重跑。
出错时把 "LTS-E-…: 说明" 打到 stderr，退出码 2。
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from typing import Any

from .backend.chains import MEM_OVERLAP_MODES, build_chains
from .backend.result import build_result, dumps_result, make_provenance
from .backend.scheduler import POLICIES, schedule
from .errors import LTSError
from .gen.builder import graph_to_dict
from .gen.gemm import build_gemm
from .gen.mlp import build_mlp
from .graph.dot import to_dot
from .graph.load import load_graph, validate_graph
from .machine.model import DEFAULT_MACHINE_PATH, load_machine
from .trace.perfetto import build_trace, dumps_trace

MACHINE_DIR = os.path.dirname(DEFAULT_MACHINE_PATH)


def resolve_machine(spec: str) -> str:
    """--machine 的值：已存在的文件路径直接用；否则当作机器 id，到 ltsim/machine/<id>.yaml 找。"""
    if os.path.isfile(spec):
        return spec
    path = os.path.join(MACHINE_DIR, f"{spec}.yaml")
    if os.path.isfile(path):
        return path
    raise LTSError("LTS-E-MACHINE", f"machine {spec!r} is neither a file nor a known id in {MACHINE_DIR}")


def _new_dir(path: str) -> None:
    if os.path.exists(path):
        raise LTSError("LTS-E-ARG", f"output directory {path!r} already exists; use a new name")


def _write(out: str, files: dict[str, str]) -> None:
    os.makedirs(out)
    for name, text in files.items():
        with open(os.path.join(out, name), "w", encoding="utf-8") as f:
            f.write(text)


def _graph_json(g) -> str:
    return json.dumps(graph_to_dict(g), ensure_ascii=False, indent=2) + "\n"


def cmd_gen(a: argparse.Namespace) -> int:
    _new_dir(a.out)
    if a.workload == "mlp":
        g = build_mlp()
    else:
        planes = [p if p.startswith("mxm.") else f"mxm.{p}" for p in a.planes.split(",")]
        g = build_gemm(a.M, a.K, a.N, tile_m=a.tile_m, tile_k=a.tile_k, tile_n=a.tile_n, planes=planes,
                       map=a.map, banks=a.banks, machine=load_machine(resolve_machine(a.machine)))
    _write(a.out, {"tile_graph.json": _graph_json(g)})
    print(f"wrote {os.path.join(a.out, 'tile_graph.json')}: {len(g.nodes)} nodes, "
          f"{len(g.edges_stream) + len(g.edges_state) + len(g.edges_mem) + len(g.edges_order)} edges")
    return 0


def _describe(step: dict[str, Any]) -> str:
    b = step["bound_by"]
    if b["kind"] == "edge":
        why = f"edge {b['edge'][0]} {b['edge'][1]}→{b['edge'][2]}"
    elif b["kind"] == "resource":
        why = f"resource {b['resource']}, held by {b['blocker']}"
    elif b["kind"] == "policy":
        why = f"policy, after {b['blocker']}"
    else:
        why = "source"
    return f"{step['chain']} [{step['start']},{step['end']}) ({why})"


def cmd_run(a: argparse.Namespace, argv: list[str]) -> int:
    _new_dir(a.out)
    g = load_graph(a.graph)
    machine_path = resolve_machine(a.machine if a.machine is not None else g.machine)
    m = load_machine(machine_path)
    diagnostics = []
    if m.raw.get("id") != g.machine:
        msg = f"graph targets machine {g.machine!r} but running on {m.raw.get('id')!r} ({machine_path})"
        diagnostics.append({"level": "warning", "code": "LTS-W-MACHINE-ID", "msg": msg})
        print(f"warning: LTS-W-MACHINE-ID: {msg}", file=sys.stderr)
    validate_graph(g, m)
    cg = build_chains(g, m, mem_overlap=a.mem_overlap)
    placed = schedule(cg, m.resources(), a.policy)
    prov = make_provenance(g, m, policy=a.policy, mem_overlap=a.mem_overlap, tile_graph_path=a.graph,
                           machine_path=machine_path, command="python -m ltsim " + shlex.join(argv))
    result = build_result(g, cg, placed, m, prov, diagnostics)
    trace = build_trace(g, cg, placed, m, provenance=prov, stream_flows=a.trace_stream_flows)
    files = {"result.json": dumps_result(result), "trace.json": dumps_trace(trace)}
    if a.dot:
        files["graph.dot"] = to_dot(g, cg.chain_of, name=os.path.basename(a.graph))
    _write(a.out, files)

    b = result["bounds"]
    print(f"total_cycles {result['total_cycles']}   lower bound {b['lb']} "
          f"(path {b['path']}, resource {b['resource']} on {b['resource_name']})   gap {b['gap'] * 100:.2f}%")
    print("critical path: " + " → ".join(_describe(s) for s in result["critical_path_detail"]))
    print("wrote " + ", ".join(os.path.join(a.out, f) for f in files))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m ltsim", description="LTSim: LPU tile-level simulator")
    sub = p.add_subparsers(dest="cmd", required=True)

    gen = sub.add_parser("gen", help="generate a tile graph")
    gsub = gen.add_subparsers(dest="workload", required=True)
    gm = gsub.add_parser("mlp", help="the two-layer MLP golden graph (design §6.3)")
    gm.add_argument("--out", required=True)
    gg = gsub.add_parser("gemm", help="a parameterized GEMM Y = X · W")
    for dim in ("M", "K", "N"):
        gg.add_argument(f"--{dim}", type=int, required=True)
    gg.add_argument("--tile-m", type=int, required=True)
    gg.add_argument("--tile-k", type=int, default=None)
    gg.add_argument("--tile-n", type=int, default=None)
    gg.add_argument("--planes", default="w0,w1", help="comma-separated, e.g. w0,w1 or e0,e1")
    gg.add_argument("--map", default="n_split", choices=["k_split", "n_split", "round_robin"])
    gg.add_argument("--banks", default="alternate", choices=["alternate", "single"])
    gg.add_argument("--machine", default="lpu-tile-v0", help="machine id or yaml path (for block sizes)")
    gg.add_argument("--out", required=True)

    run = sub.add_parser("run", help="simulate a tile graph")
    run.add_argument("graph", help="tile_graph.json")
    run.add_argument("--machine", default=None, help="machine id or yaml path (default: the graph's machine)")
    run.add_argument("--out", required=True)
    run.add_argument("--policy", default="bottom_level", choices=list(POLICIES))
    run.add_argument("--mem-overlap", default="tile", choices=list(MEM_OVERLAP_MODES))
    run.add_argument("--trace-stream-flows", action="store_true", help="also draw intra-chain stream edges")
    run.add_argument("--dot", action="store_true", help="also write graph.dot (Graphviz; one box per chain)")
    return p


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    a = build_parser().parse_args(argv)
    try:
        return cmd_gen(a) if a.cmd == "gen" else cmd_run(a, argv)
    except LTSError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except OSError as e:
        print(f"error: LTS-E-ARG: {e}", file=sys.stderr)
        return 2
