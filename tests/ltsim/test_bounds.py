"""B4：下界、关键路径、指标、result.json。金标准来自 backend.md §8.3，节点与算子时刻按设计 §3.5 手算。"""
import json
import random
from fractions import Fraction as F

import pytest

from ltsim.backend.bounds import compute_bounds, critical_path, mxm_roofline
from ltsim.backend.chains import build_chains
from ltsim.backend.metrics import node_times, op_spans, resource_usage
from ltsim.backend.result import build_result, dumps_result, make_provenance, write_result
from ltsim.backend.scheduler import BoundBy, schedule
from ltsim.gen.gemm import build_gemm
from ltsim.gen.mlp import build_mlp
from ltsim.machine.model import DEFAULT_MACHINE_PATH, default_machine


@pytest.fixture(scope="module")
def m():
    return default_machine()


@pytest.fixture(scope="module")
def run(m):
    g = build_mlp()
    cg = build_chains(g, m)
    return g, cg, schedule(cg, m.resources())


class TestMlpBounds:
    def test_bounds(self, run, m):
        g, cg, p = run
        b = compute_bounds(cg, p, m.resources())
        assert b.total == 551
        assert b.path == 525                                    # 0 + 21 + 265 + 239
        assert b.earliest == {"c0": 0, "c1": 0, "c2": 0, "c3": 0, "c4": 21, "c5": 21, "c6": 286}
        assert (b.resource_name, b.resource_exact, b.resource) == ("srf.W2E", F(5278, 32), 165)
        assert b.lb == 525 and b.gap == F(26, 525)              # 551/525 − 1 ≈ 4.95%

    def test_critical_path(self, run):
        _g, cg, p = run
        assert [(s.chain, s.bound_by) for s in critical_path(cg, p)] == [
            ("c0", BoundBy("source")),
            ("c4", BoundBy("resource", resource="srf.E2W", blocker="c0")),
            ("c6", BoundBy("edge", edge=("mem", "a0.h.wr", "b.h0.rd"), blocker="c4")),
        ]

    def test_critical_path_through_policy(self):
        """topo 策略：c1 与 c0 抢独占资源 x，被推到 10；c2 只用 y，本可在 0 开始，但不插空，卡在 10，
        且最后结束。回溯：c2 —policy→ c1 —resource x→ c0 —source。"""
        from ltsim.backend.chains import Chain, ChainGraph, Reservation
        from ltsim.machine.model import Resource
        res = {"x": Resource("x", "exclusive", 1), "y": Resource("y", "exclusive", 1)}
        mk = lambda cid, r, a, e: Chain(cid, (), F(1), {}, {}, e, (Reservation(r, a, e, 1, cid),))
        cg = ChainGraph((mk("c0", "x", 0, 10), mk("c1", "x", 0, 10), mk("c2", "y", 0, 30)), (), {}, {})
        p = schedule(cg, res, "topo")
        assert p.start == {"c0": 0, "c1": 10, "c2": 10}
        assert [(s.chain, s.bound_by) for s in critical_path(cg, p)] == [
            ("c0", BoundBy("source")),
            ("c1", BoundBy("resource", resource="x", blocker="c0")),
            ("c2", BoundBy("policy", blocker="c1")),
        ]

    def test_roofline_full_blocks(self, run, m):
        """MLP 每个权重块都是满的 320×320：4 个 acc × 128 行 × 320 × 320 / (4 × 320 × 320) = 128。"""
        g, cg, p = run
        assert mxm_roofline(g, cg, p, m) == 128


def _roof(g, m):
    cg = build_chains(g, m)
    return mxm_roofline(g, cg, schedule(cg, m.resources()), m)


@pytest.mark.parametrize("M,K,N,kw", [
    (128, 320, 320, dict(tile_m=128, tile_n=160)),                       # 只有 1 个 M 块
    (256, 640, 640, dict(tile_m=128, tile_n=160, planes=["mxm.w0"])),    # 多个 M 块、块不满、单 plane
    (256, 320, 320, dict(tile_m=128, tile_n=160)),
    (512, 320, 320, dict(tile_m=128, tile_k=160)),
    (128, 320, 320, dict(tile_m=64, tile_k=160, tile_n=80)),
    (256, 640, 640, dict(tile_m=64, map="k_split", banks="single")),
    (192, 960, 640, dict(tile_m=64, tile_k=320, tile_n=160, map="round_robin", planes=["mxm.e0", "mxm.e1"])),
])
def test_roofline_is_true_mac(m, M, K, N, kw):
    """GEMM 的总 MAC 恰好是 M·K·N，所以不管怎么分块，roofline = M·K·N / (4 × 320 × 320)。
    同一权重块的后续 acc 没有 state 边，必须按"同一 unit/bank 上最近一次装完的 load_w"找到它用的块；
    只看 state 边会把它们当成满块（例如 256×640×640、tile_n=160、单 plane 会算成 384 而不是 256）。"""
    assert _roof(build_gemm(M, K, N, **kw), m) == F(M * K * N, 4 * 320 * 320)


def test_roofline_peak_uses_machine_planes(m):
    """峰值按机器的 plane 数算：只保留 2 个 plane 的机器，同一张图的 roofline 翻倍。"""
    import copy
    from ltsim.machine.model import Machine
    raw = copy.deepcopy(dict(m.raw))
    raw["mxm"]["instances"] = raw["mxm"]["instances"][:2]
    m2 = Machine(raw)
    g = build_gemm(128, 320, 320, tile_m=128)
    assert _roof(g, m2) == 2 * _roof(g, m)


class TestMetrics:
    def test_node_times(self, run):
        _g, cg, p = run
        t = node_times(cg, p)
        assert t["a0.acc"] == (56, 184, 209)          # 47+9；+128；+25+128
        assert t["w1n0.lw"] == (9, 29, 48)            # skew 19
        assert t["b.y.wr"] == (403, 531, 551)         # 312+91；+128；+1+128+19

    def test_op_spans(self, run):
        g, cg, p = run
        assert op_spans(g, cg, p) == {
            "fc1": (0, 209), "fc2": (0, 529), "b1": (99, 290), "dq1": (102, 290), "act": (102, 290),
            "q1": (102, 312), "b2": (366, 529), "y": (369, 551)}

    def test_resource_usage(self, run, m):
        _g, cg, p = run
        u = resource_usage(cg, p, m.resources())
        assert set(u) == set(m.resources())                           # 没用到的资源也列出
        assert (u["mxm.w0.compute"]["busy"], u["mxm.w0.compute"]["utilization"]) == (128, F(128, 551))
        assert u["mxm.w0.weight"]["busy"] == 20
        assert u["vxm.alu"]["area"] == 6 * 128 * 2 + 5 * 128         # 两个第一层 vxm 链深 6，第二层 5
        assert u["vxm.alu"]["peak"] == 12
        assert (u["srf.W2E"]["peak"], u["srf.W2E"]["avg"]) == (32, F(5278, 551))
        assert (u["srf.E2W"]["peak"], u["srf.E2W"]["avg"]) == (32, F(3315, 551))
        assert u["sxm.w"]["busy"] == 0


class TestResult:
    def _result(self, m, run, **kw):
        g, cg, p = run
        return build_result(g, cg, p, m, make_provenance(g, m, policy="bottom_level", mem_overlap="tile", **kw))

    def test_fields(self, m, run):
        r = self._result(m, run)
        assert list(r) == ["schema", "provenance", "total_cycles", "bounds", "chains", "nodes", "resources",
                           "srf", "critical_path", "critical_path_detail", "ops", "diagnostics"]
        assert r["schema"] == "lts-result/0.1" and r["total_cycles"] == 551
        assert r["bounds"] == {"path": 525, "resource": 165, "resource_exact": "2639/16",
                               "resource_name": "srf.W2E", "lb": 525, "gap": 0.049524, "mxm_roofline": 128.0}
        c4 = next(c for c in r["chains"] if c["id"] == "c4")
        assert c4 == {"id": "c4", "start": 47, "len": 265, "end": 312, "rate": 1.0, "rate_exact": "1",
                      "bound_by": {"kind": "resource", "resource": "srf.E2W", "blocker": "c0"}}
        assert r["critical_path"] == ["c0", "c4", "c6"]
        assert r["critical_path_detail"][-1]["bound_by"] == {
            "kind": "edge", "edge": ["mem", "a0.h.wr", "b.h0.rd"], "from_chain": "c4"}
        assert len(r["nodes"]) == 25 and r["diagnostics"] == []

    def test_fractional_rate_written_both_ways(self, m):
        from ltsim.gen.builder import GraphBuilder, Out
        b = GraphBuilder("t", "lpu-tile-v0")
        b.tensor("A", "i8", (8, 320), "input", "W", "row")
        b.tensor("B", "i8", (8, 320), "output", "E", "row")
        b.read("r", "A[0:8,0:320]", 8, Out("i8", 1), ["op"])
        b.vxm("v", ["relu"] * 17, 8, Out("i8", 1), ["op"])
        b.write("w", "B[0:8,0:320]", 8, ["op"])
        b.stream("r", "v"); b.stream("v", "w")
        g = b.build()
        cg = build_chains(g, m)
        r = build_result(g, cg, schedule(cg, m.resources()), m,
                         make_provenance(g, m, policy="bottom_level", mem_overlap="tile"))
        assert (r["chains"][0]["rate"], r["chains"][0]["rate_exact"]) == (0.941176, "16/17")

    def test_provenance_hashes(self, m, run, tmp_path):
        g, _cg, _p = run
        a = make_provenance(g, m, policy="bottom_level", mem_overlap="tile")
        b = make_provenance(g, m, policy="bottom_level", mem_overlap="tile", machine_path=DEFAULT_MACHINE_PATH)
        assert len(a["tile_graph_sha256"]) == 64 and a["ltsim_commit"] is None
        assert a["tile_graph_sha256"] == b["tile_graph_sha256"]
        assert a["machine_sha256"] != b["machine_sha256"]           # 文件字节 vs 规范化内容，两种来源不同

    def test_byte_identical(self, m, tmp_path):
        outs = []
        for i in range(2):
            g = build_mlp()
            cg = build_chains(g, m)
            r = build_result(g, cg, schedule(cg, m.resources()), m,
                             make_provenance(g, m, policy="bottom_level", mem_overlap="tile"))
            path = tmp_path / f"r{i}.json"
            write_result(str(path), r)
            outs.append(path.read_bytes())
        assert outs[0] == outs[1]
        assert json.loads(outs[0])["total_cycles"] == 551


def test_total_never_below_lb(m):
    """性质：随机 GEMM 变体、两种策略、两种 mem 语义下，total ≥ LB，build_result 的自检都通过。"""
    rng = random.Random(3)
    for _ in range(20):
        Mb, Kb, Nb = rng.randint(1, 3), rng.randint(1, 3), rng.randint(1, 3)
        tile_m = rng.choice([32, 64, 128])
        g = build_gemm(Mb * tile_m, Kb * 320, Nb * 320, tile_m=tile_m,
                       planes=rng.choice([["mxm.w0"], ["mxm.w0", "mxm.w1"], ["mxm.e0", "mxm.e1"]]),
                       map=rng.choice(["k_split", "n_split", "round_robin"]),
                       banks=rng.choice(["alternate", "single"]))
        for overlap in ("tile", "row"):
            cg = build_chains(g, m, mem_overlap=overlap)
            for pol in ("bottom_level", "topo"):
                p = schedule(cg, m.resources(), pol)
                b = compute_bounds(cg, p, m.resources())
                assert b.total >= b.lb
                build_result(g, cg, p, m, make_provenance(g, m, policy=pol, mem_overlap=overlap))


# ---- 覆盖缺口：自检、来源哈希、关键路径平局、合成图上的下界 --------------------------

class TestSelfCheck:
    def test_bad_placed_raises_internal(self, run, m):
        g, cg, p = run
        bad = type(p)(p.policy, {**p.start, "c6": 0}, p.bound_by, p.order, p.total)
        with pytest.raises(Exception, match="LTS-E-INTERNAL.*self-check"):
            build_result(g, cg, bad, m, make_provenance(g, m, policy="bottom_level", mem_overlap="tile"))

    def test_total_below_lb_raises_internal(self, run, m, monkeypatch):
        """单独测 total ≥ LB 这道检查：先让 verify 放行，再把所有链放在 0。"""
        import ltsim.backend.result as result_mod
        g, cg, p = run
        monkeypatch.setattr(result_mod, "verify", lambda *a: [])
        zero = {c.id: 0 for c in cg.chains}
        bad = type(p)(p.policy, zero, p.bound_by, p.order, max(c.length for c in cg.chains))
        with pytest.raises(Exception, match="LTS-E-INTERNAL.*lower bound 525"):
            build_result(g, cg, bad, m, make_provenance(g, m, policy="bottom_level", mem_overlap="tile"))


def test_tile_graph_path_hash(run, m, tmp_path):
    import hashlib
    from ltsim.gen.builder import graph_to_dict
    g, _cg, _p = run
    path = tmp_path / "g.json"
    path.write_text(json.dumps(graph_to_dict(g), indent=2))
    prov = make_provenance(g, m, policy="bottom_level", mem_overlap="tile", tile_graph_path=str(path))
    assert prov["tile_graph_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def _chains(spec):
    """spec: [(链 id, 资源, start, end)] → 只含一条预留的链。"""
    from ltsim.backend.chains import Chain, ChainGraph, Reservation
    return ChainGraph(tuple(Chain(c, (), F(1), {}, {}, e, (Reservation(r, a, e, 1, c),)) for c, r, a, e in spec),
                      (), {}, {})


def test_critical_path_tie_takes_lowest_index():
    """两条链同时最后结束：取在链列表里序号小的（c2 先于 c10，按序号不按字符串）。"""
    from ltsim.machine.model import Resource
    res = {f"x{i}": Resource(f"x{i}", "exclusive", 1) for i in range(11)}
    cg = _chains([(f"c{i}", f"x{i}", 0, 60 if i in (2, 10) else 10) for i in range(11)])
    p = schedule(cg, res)
    assert [s.chain for s in critical_path(cg, p)] == ["c2"]


def test_bounds_on_synthetic_graphs():
    """合成图（独占资源插空、计数资源冲突、负权重边）上：total ≥ LB，关键路径能回溯到 source。"""
    from tests.ltsim.test_scheduler import _synthetic
    from ltsim.machine.model import Resource
    resources = {"x.a": Resource("x.a", "exclusive", 1), "x.b": Resource("x.b", "exclusive", 1),
                 "k.c": Resource("k.c", "counter", 5), "k.d": Resource("k.d", "counter", 3)}
    rng = random.Random(4)
    for _ in range(300):
        cg = _synthetic(rng, resources)
        for pol in ("bottom_level", "topo"):
            p = schedule(cg, resources, pol)
            b = compute_bounds(cg, p, resources)
            assert b.total >= b.lb
            cp = critical_path(cg, p)
            assert cp[0].bound_by.kind == "source" and cp[-1].chain in {c.id for c in cg.chains}

