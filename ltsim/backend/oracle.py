"""调度器的精确 oracle（backend.md §9，设计 §9.5）：用 OR-Tools CP-SAT 求同一张链图、同一套资源下的最优链调度，
衡量列表调度（SSGS）离最优有多远。

只用于测试和诊断；ltsim 包本身不依赖 ortools（在函数里才导入，没装时报 LTS-E-ARG）。
模型与 B3 完全同构：链起始时刻 T_c 为整数变量；链间边 T_dst ≥ T_src + w；独占资源上全部预留区间 NoOverlap；
计数资源用 B3 合并后的分段（fit_segments）做 Cumulative；目标是最小化 max(T_c + len(c))。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from ..errors import LTSError
from ..machine.model import Resource
from .chains import ChainGraph
from .scheduler import BoundBy, Placed, fit_segments, schedule, verify


@dataclass(frozen=True)
class OracleResult:
    status: str                      # "OPTIMAL" | "FEASIBLE"（时间上限内没证明最优）
    makespan: int                    # 求到的最好总周期
    bound: int                       # CP-SAT 证明的下界；OPTIMAL 时等于 makespan
    start: Mapping[str, int]         # 链 id → T_c
    ssgs: int                        # 同一输入下 SSGS（bottom_level）的总周期
    wall_s: float


def solve_optimal(cg: ChainGraph, resources: Mapping[str, Resource], *, time_limit_s: float = 10.0,
                  workers: int = 8) -> OracleResult:
    try:
        from ortools.sat.python import cp_model
    except ImportError as e:                                    # pragma: no cover
        raise LTSError("LTS-E-ARG", "the oracle needs OR-Tools (pip install ortools)") from e

    ssgs = schedule(cg, resources)                              # 可行解：给上界和初始提示
    horizon = ssgs.total
    md = cp_model.CpModel()
    T = {c.id: md.NewIntVar(0, horizon - c.length, f"T_{c.id}") for c in cg.chains}
    for e in cg.edges:
        md.Add(T[e.dst] >= T[e.src] + e.weight)
    makespan = md.NewIntVar(max(c.length for c in cg.chains), horizon, "makespan")
    for c in cg.chains:
        md.Add(makespan >= T[c.id] + c.length)

    excl: dict[str, list] = {}
    cum: dict[str, tuple[list, list]] = {}
    for c in cg.chains:
        for k, s in enumerate(fit_segments(c, resources)):
            size = s.end - s.start
            iv = md.NewFixedSizeIntervalVar(T[c.id] + s.start, size, f"{c.id}_{s.resource}_{k}")
            if resources[s.resource].kind == "exclusive":
                excl.setdefault(s.resource, []).append(iv)
            else:
                ivs, dem = cum.setdefault(s.resource, ([], []))
                ivs.append(iv)
                dem.append(s.amount)
    for ivs in excl.values():
        if len(ivs) > 1:
            md.AddNoOverlap(ivs)
    for name, (ivs, dem) in cum.items():
        md.AddCumulative(ivs, dem, resources[name].cap)

    for c in cg.chains:
        md.AddHint(T[c.id], ssgs.start[c.id])
    md.Minimize(makespan)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_s
    solver.parameters.num_search_workers = workers
    st = solver.Solve(md)
    if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise LTSError("LTS-E-INTERNAL", f"CP-SAT returned {solver.StatusName(st)} although SSGS found a schedule")
    start = {c.id: solver.Value(T[c.id]) for c in cg.chains}
    total = max(start[c.id] + c.length for c in cg.chains)

    # 用 B3 的 verify（逐拍累加，不复用 CP-SAT 模型）独立检查 oracle 的解
    placed = Placed("oracle", start, {c.id: BoundBy("source") for c in cg.chains},
                    tuple(sorted(start, key=lambda x: (start[x], x))), total)
    bad = verify(cg, placed, resources)
    if bad:
        raise LTSError("LTS-E-INTERNAL", "CP-SAT solution failed verify: " + "; ".join(bad[:3]))
    return OracleResult("OPTIMAL" if st == cp_model.OPTIMAL else "FEASIBLE", total,
                        int(solver.BestObjectiveBound()), start, ssgs.total, solver.WallTime())
