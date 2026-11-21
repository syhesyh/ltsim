"""B3 scheduler.py 的测试。

金标准：backend.md §8.3 的调度过程（总周期 551）。另外：计数资源按合并后的分段检查；
topo 策略；double weight bank 在 GEMM 上的装载-计算重叠；随机生成的 GEMM 变体上的性质检查。
"""
import random
from fractions import Fraction as F

import pytest

from ltsim.backend.chains import build_chains
from ltsim.backend.scheduler import (BoundBy, Segment, bottom_levels, fit_segments, schedule, verify)
from ltsim.errors import LTSError
from ltsim.gen.builder import GraphBuilder, Out
from ltsim.gen.gemm import build_gemm
from ltsim.gen.mlp import build_mlp
from ltsim.machine.model import default_machine


@pytest.fixture(scope="module")
def m():
    return default_machine()


@pytest.fixture(scope="module")
def R(m):
    return m.resources()


@pytest.fixture(scope="module")
def mlp(m):
    return build_chains(build_mlp(), m)


def node_start(cg, placed, nid):
    c = cg.chain_of[nid]
    return placed.start[c] + cg.chain(c).offset[nid]


def node_issue_end(cg, placed, nid):
    return node_start(cg, placed, nid) + cg.chain(cg.chain_of[nid]).D[nid]


# ---- §8.3 金标准 -------------------------------------------------------------

class TestMlpGolden:
    def test_bottom_levels(self, mlp):
        assert bottom_levels(mlp) == {"c0": 525, "c1": 525, "c2": 260, "c3": 260,
                                      "c4": 504, "c5": 504, "c6": 239}

    def test_schedule(self, mlp, R):
        p = schedule(mlp, R)
        assert p.order == ("c0", "c1", "c4", "c5", "c2", "c3", "c6")
        assert p.start == {"c0": 0, "c1": 0, "c4": 47, "c5": 47, "c2": 0, "c3": 0, "c6": 312}
        assert p.total == 551

    def test_bound_by(self, mlp, R):
        b = schedule(mlp, R).bound_by
        for c in ("c0", "c1", "c2", "c3"):
            assert b[c] == BoundBy("source")
        # 读 X 要 1 条 E2W 流，[1,48) 被 c0、c1 的权重装载占满 32 条；两者各 16，平局取先提交的 c0
        assert b["c4"] == BoundBy("resource", resource="srf.E2W", blocker="c0")
        assert b["c5"] == BoundBy("resource", resource="srf.E2W", blocker="c0")
        # est = max(21, 21, 47+265, 47+265) = 312；同值时取第一条达到的边
        assert b["c6"] == BoundBy("edge", edge=("mem", "a0.h.wr", "b.h0.rd"), blocker="c4")

    def test_valid(self, mlp, R):
        assert verify(mlp, schedule(mlp, R), R) == []

    def test_deterministic(self, mlp, R):
        assert schedule(mlp, R) == schedule(mlp, R)


# ---- 计数资源：先合并分段再检查 ------------------------------------------------

def _overlap_graph():
    """链 A：6 个 f32 读 → 24 条 W2E；链 B：3 个 f32 读 → 12 条 W2E，三条预留彼此重叠。"""
    F32 = Out("f32", 4)
    b = GraphBuilder("over", "lpu-tile-v0")
    for name, n in (("A", 6), ("B", 3)):
        b.vxm(f"v{name}", ["add.f32"] * n, 8, F32, ["op"])
        for i in range(n):
            b.tensor(f"{name}{i}", "f32", (8, 320), "input", "W", "row")
            b.read(f"{name}{i}.rd", f"{name}{i}[0:8,0:320]", 8, F32, ["op"])
            b.stream(f"{name}{i}.rd", f"v{name}")
    return b.build()


class TestAggregatedFit:
    def test_segments_merge_overlaps(self, m, R):
        cg = build_chains(_overlap_graph(), m)
        segs = [s for s in fit_segments(cg.chain("c1"), R) if s.resource == "srf.W2E"]
        # 三条读的流都在 [1, 1+8+19+2) = [1,30)，合并成一段 12
        assert segs == [Segment("srf.W2E", 1, 30, 12)]

    def test_no_over_commit(self, m, R):
        """逐条检查时 B 会被放在 T=0（每条 24+4 ≤ 32），实际峰值 36；合并后 B 被推到
        A 的流释放之后：[1,30) 最后一拍 29 → T = 30 − 1 = 29。"""
        cg = build_chains(_overlap_graph(), m)
        p = schedule(cg, R)
        assert p.start == {"c0": 0, "c1": 29}
        assert p.bound_by["c1"] == BoundBy("resource", resource="srf.W2E", blocker="c0")
        assert verify(cg, p, R) == []

    def test_mlp_c4_segments(self, mlp, R):
        """c4 的 W2E：acc [34,202) 4、读 B1 [53,202) 4、vxm [115,264) 1 → 4 / 8 / 9 / 1。"""
        segs = [(s.start, s.end, s.amount) for s in fit_segments(mlp.chain("c4"), R) if s.resource == "srf.W2E"]
        assert segs == [(34, 53, 4), (53, 115, 8), (115, 202, 9), (202, 264, 1)]


# ---- topo 策略 ---------------------------------------------------------------

class TestTopo:
    def test_mlp(self, mlp, R):
        p = schedule(mlp, R, "topo")
        assert p.order == ("c0", "c1", "c2", "c3", "c4", "c5", "c6")      # Kahn 序，平局取 id 小的
        assert p.start == {"c0": 0, "c1": 0, "c2": 0, "c3": 0, "c4": 47, "c5": 47, "c6": 312}
        assert p.bound_by["c4"] == BoundBy("resource", resource="srf.E2W", blocker="c0")
        # c5 的 est 是 21，但不插空：不能早于前一条 c4 的 47；47 处资源放得下
        assert p.bound_by["c5"] == BoundBy("policy", blocker="c4")
        assert p.total == 551 and verify(mlp, p, R) == []

    def test_start_times_non_decreasing(self, m, R):
        cg = build_chains(build_gemm(256, 640, 1280, tile_m=128), m)
        p = schedule(cg, R, "topo")
        ts = [p.start[c] for c in p.order]
        assert ts == sorted(ts)
        assert verify(cg, p, R) == []

    def test_not_better_than_bottom_level_here(self, m, R):
        cg = build_chains(build_gemm(256, 640, 1280, tile_m=128), m)
        assert schedule(cg, R, "topo").total >= schedule(cg, R).total


# ---- double weight bank：装载与计算重叠 ----------------------------------------

def _plane_timeline(banks, m):
    g = build_gemm(256, 640, 1280, tile_m=128, banks=banks)      # w0 上 4 个权重块：(n0,k0) (n0,k1) (n2,k0) (n2,k1)
    cg = build_chains(g, m)
    p = schedule(cg, m.resources())
    blocks = []
    for n, k in ((0, 0), (0, 1), (2, 0), (2, 1)):
        lw, accs = f"w.n{n}k{k}.lw", [f"m{mm}n{n}k{k}.acc" for mm in (0, 1)]
        blocks.append({"lw": (node_start(cg, p, lw), node_issue_end(cg, p, lw)),
                       "acc": (node_start(cg, p, accs[0]), node_issue_end(cg, p, accs[-1]))})
    return blocks, p, cg


class TestDoubleBuffer:
    def test_alternate_hides_weight_load(self, m):
        blocks, p, cg = _plane_timeline("alternate", m)
        assert verify(cg, p, m.resources()) == []
        for prev, cur in zip(blocks, blocks[1:]):
            # 下一块的权重在上一块计算结束前就装完了
            assert cur["lw"][1] <= prev["acc"][1]
            # 计算之间只隔换块间隙 block_switch_gap = 5
            assert cur["acc"][0] - prev["acc"][1] == m.param("mxm.block_switch_gap")

    def test_single_waits_for_previous_block(self, m):
        blocks, p, cg = _plane_timeline("single", m)
        assert verify(cg, p, m.resources()) == []
        for prev, cur in zip(blocks, blocks[1:]):
            # 同一个 bank：要等上一块的计算全部结束才能装（bank_war）
            assert cur["lw"][0] >= prev["acc"][1]
            assert cur["acc"][0] - prev["acc"][1] > m.param("mxm.block_switch_gap")

    def test_alternate_is_faster(self, m):
        assert _plane_timeline("alternate", m)[1].total < _plane_timeline("single", m)[1].total


# ---- 性质：随机 GEMM 变体 ------------------------------------------------------

def test_random_gemm_variants_are_valid(m, R):
    rng = random.Random(0)
    for _ in range(25):
        Mb, Kb, Nb = rng.randint(1, 3), rng.randint(1, 3), rng.randint(1, 3)
        tile_m = rng.choice([32, 64, 128])
        planes = rng.choice([["mxm.w0"], ["mxm.w0", "mxm.w1"], ["mxm.e0", "mxm.e1"]])
        g = build_gemm(Mb * tile_m, Kb * 320, Nb * 320, tile_m=tile_m, planes=planes,
                       map=rng.choice(["k_split", "n_split", "round_robin"]),
                       banks=rng.choice(["alternate", "single"]))
        cg = build_chains(g, m)
        for pol in ("bottom_level", "topo"):
            p = schedule(cg, R, pol)
            assert verify(cg, p, R) == [], (Mb, Kb, Nb, tile_m, planes, pol)
            assert p.total == max(p.start[c.id] + c.length for c in cg.chains)


# ---- verify 自身与参数 ---------------------------------------------------------

def test_verify_catches_violations(mlp, R):
    p = schedule(mlp, R)
    bad = type(p)(p.policy, {**p.start, "c4": 0}, p.bound_by, p.order, p.total)   # c4 提前到 0
    msgs = verify(mlp, bad, R)
    assert any("edge" in s for s in msgs)                   # 违反 c0→c4 的 state 约束
    assert any("srf.E2W over capacity" in s for s in msgs)  # 与权重装载抢 E2W 流


@pytest.mark.parametrize("pol,msg", [("random:4:0", "deferred"), ("bogus", "policy must be")])
def test_bad_policy(mlp, R, pol, msg):
    with pytest.raises(LTSError, match=f"LTS-E-ARG.*{msg}"):
        schedule(mlp, R, pol)


# ---- 逐拍参考调度器：守住"返回的是 est 之后最早的可行时刻" -----------------------
from ltsim.backend.chains import Chain, ChainEdge, ChainGraph, Reservation
from ltsim.backend.scheduler import _kahn
from ltsim.machine.model import Resource


def reference_schedule(cg, resources, policy="bottom_level"):
    """选链规则与 schedule() 相同；"放得下"用最朴素的方式判断：从 est 起逐个试 T，
    逐拍检查 已有占用 + 本链这一拍的总用量 ≤ 容量（独占资源容量 1）。不用日历、不做跳跃。"""
    chains = {c.id: c for c in cg.chains}
    idx = {c.id: i for i, c in enumerate(cg.chains)}
    ins = {c: [e for e in cg.edges if e.dst == c] for c in chains}
    if policy == "bottom_level":
        bl = bottom_levels(cg)
        prio = {c: bl[c] for c in chains}
    else:
        prio = {c: -i for i, c in enumerate(_kahn(cg))}
    use = {name: {} for name in resources}
    T, order = {}, []

    def fits(c, t):
        need = {}
        for r in chains[c].reservations:
            for x in range(t + r.start, t + r.end):
                need[(r.resource, x)] = need.get((r.resource, x), 0) + r.amount
        return all(use[res].get(x, 0) + k <= resources[res].cap for (res, x), k in need.items())

    while len(T) < len(chains):
        ready = [c for c in chains if c not in T and all(e.src in T for e in ins[c])]
        c = max(ready, key=lambda x: (prio[x], -idx[x]))
        t = max([0] + [T[e.src] + e.weight for e in ins[c]])
        if policy == "topo" and order:
            t = max(t, T[order[-1]])
        while not fits(c, t):
            t += 1
        T[c] = t
        order.append(c)
        for r in chains[c].reservations:
            for x in range(t + r.start, t + r.end):
                use[r.resource][x] = use[r.resource].get(x, 0) + r.amount
    return T, tuple(order)


@pytest.mark.parametrize("pol", ["bottom_level", "topo"])
def test_matches_reference_on_gemm(m, R, pol):
    rng = random.Random(1)
    for _ in range(12):
        Mb, Kb, Nb = rng.randint(1, 2), rng.randint(1, 3), rng.randint(1, 3)
        tile_m = rng.choice([32, 64, 128])
        g = build_gemm(Mb * tile_m, Kb * 320, Nb * 320, tile_m=tile_m,
                       planes=rng.choice([["mxm.w0"], ["mxm.w0", "mxm.w1"], ["mxm.e0", "mxm.e1"]]),
                       map=rng.choice(["k_split", "n_split", "round_robin"]),
                       banks=rng.choice(["alternate", "single"]))
        cg = build_chains(g, m)
        p = schedule(cg, R, pol)
        assert (p.start, p.order) == reference_schedule(cg, R, pol)


def _synthetic(rng, resources):
    """随机合成的链图：资源有独占的 x.a、x.b，计数的 k.c（容量 5）、k.d（容量 3）。
    满足 B2 的不变量：同一链在独占资源上不重叠、在计数资源上逐拍不超容量；链间边可以是负权重。"""
    names = list(resources)
    chains = []
    for i in range(rng.randint(2, 7)):
        while True:
            rs = []
            for _ in range(rng.randint(1, 5)):
                res = rng.choice(names)
                a = rng.randint(0, 12)
                rs.append(Reservation(res, a, a + rng.randint(1, 10),
                                      1 if resources[res].kind == "exclusive" else rng.randint(1, resources[res].cap),
                                      f"n{i}_{len(rs)}"))
            per = {}
            for r in rs:
                for x in range(r.start, r.end):
                    per[(r.resource, x)] = per.get((r.resource, x), 0) + r.amount
            if all(k <= resources[res].cap for (res, _x), k in per.items()):
                break
        rs = tuple(sorted(rs, key=lambda r: (r.resource, r.start, r.end, r.node)))
        chains.append(Chain(f"c{i}", (), F(1), {}, {}, max(r.end for r in rs), rs))
    edges = [ChainEdge(f"c{i}", f"c{j}", rng.randint(-5, 25), ("order", f"c{i}", f"c{j}"))
             for i in range(len(chains)) for j in range(i + 1, len(chains)) if rng.random() < 0.3]
    return ChainGraph(tuple(chains), tuple(edges), {}, {})


@pytest.mark.parametrize("pol", ["bottom_level", "topo"])
def test_matches_reference_on_synthetic_graphs(pol):
    """GEMM 里起作用的冲突几乎只有流；合成图让独占资源插空、计数资源冲突都成为瓶颈。"""
    resources = {"x.a": Resource("x.a", "exclusive", 1), "x.b": Resource("x.b", "exclusive", 1),
                 "k.c": Resource("k.c", "counter", 5), "k.d": Resource("k.d", "counter", 3)}
    rng = random.Random(2)
    kinds = set()
    for _ in range(400):
        cg = _synthetic(rng, resources)
        p = schedule(cg, resources, pol)
        assert (p.start, p.order) == reference_schedule(cg, resources, pol)
        assert verify(cg, p, resources) == []
        kinds |= {b.kind for b in p.bound_by.values()}
    assert {"source", "edge", "resource"} <= kinds


def test_zero_length_interval_never_conflicts():
    from ltsim.backend.calendar import CounterCalendar, ExclusiveCalendar
    e = ExclusiveCalendar()
    e.commit(10, 20, "A")
    assert e.conflict(15, 15) is None
    c = CounterCalendar(4)
    c.commit(0, 10, 4, "A")
    assert c.conflict(7, 7, 1) is None
    c.commit(5, 5, 4, "B")                       # 空区间提交什么也不做
    assert c.steps() == [(0, 4), (10, 0)]


class TestVerifyNegatives:
    def test_only_capacity(self, mlp, R):
        """c4 提前 1 拍到 46：边约束仍成立（46 ≥ 21），只在第 47 拍和权重装载抢 E2W。"""
        p = schedule(mlp, R)
        bad = type(p)(p.policy, {**p.start, "c4": 46}, p.bound_by, p.order, p.total)
        assert verify(mlp, bad, R) == ["srf.E2W over capacity 32 at 1 cycles, first 47 (usage 33)"]

    def test_only_exclusive(self, m, R):
        """GEMM 中同一 plane 上相邻两个 acc 链放在同一时刻：违反 acc_seq 边，并在 compute 端口上重叠。"""
        cg = build_chains(build_gemm(256, 320, 320, tile_m=128, planes=["mxm.w0"]), m)
        p = schedule(cg, R)
        a0, a1 = cg.chain_of["m0n0k0.acc"], cg.chain_of["m1n0k0.acc"]
        bad = type(p)(p.policy, {**p.start, a1: p.start[a0]}, p.bound_by, p.order, p.total)
        msgs = verify(cg, bad, R)
        assert any(s.startswith("mxm.w0.compute over capacity 1") for s in msgs)

    def test_missing_chain_is_reported_not_raised(self, mlp, R):
        p = schedule(mlp, R)
        start = dict(p.start)
        del start["c3"]
        assert verify(mlp, type(p)(p.policy, start, p.bound_by, p.order, p.total), R) == \
            ["no start time for chain(s) ['c3']"]

    def test_total_mismatch(self, mlp, R):
        p = schedule(mlp, R)
        assert verify(mlp, type(p)(p.policy, p.start, p.bound_by, p.order, 500), R) == \
            ["total 500 != max(T + len) = 551"]
