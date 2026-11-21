"""B4：下界、关键路径、指标、result.json。金标准来自 backend.md §8.3，节点与算子时刻按设计 §3.5 手算。"""
import json
import random
from dataclasses import replace
from fractions import Fraction as F

import pytest

from ltsim.backend.bounds import compute_bounds, critical_path, mxm_roofline
from ltsim.backend.chains import build_chains
from ltsim.backend.metrics import node_times, op_spans, resource_usage
from ltsim.backend.result import build_result, dumps_result, make_provenance, write_result
from ltsim.backend.scheduler import BoundBy, schedule
from ltsim.errors import LTSError
from ltsim.gen.gemm import build_gemm
from ltsim.gen.mlp import build_mlp
from ltsim.machine.model import DEFAULT_MACHINE_PATH, default_machine

# 手算核对过的期望值都是旧选项下的（backend.md §8.4）；新默认选项的用例单独写
LEGACY = dict(srf_pool="direction", lw_streams="shared")


@pytest.fixture(scope="module")
def m():
    return default_machine()


@pytest.fixture(scope="module")
def run(m):
    g = build_mlp()
    cg = build_chains(g, m, **LEGACY)
    return g, cg, schedule(cg)


class TestMlpBounds:
    def test_bounds(self, run, m):
        g, cg, p = run
        b = compute_bounds(cg, p)
        assert b.total == 513
        assert b.path == 506                                    # 0 + 21 + 246 + 239
        assert b.earliest == {"c0": 0, "c1": 0, "c2": 0, "c3": 0, "c4": 21, "c5": 21, "c6": 267}
        # vxm.alu：(6·128·2 + 5·128) / 16 = 136；srf.W2E 只有 4214/32 ≈ 131.7
        assert (b.resource_name, b.resource_exact, b.resource) == ("vxm.alu", F(136), 136)
        assert b.lb == 506 and b.gap == F(7, 506)               # 513/506 − 1 ≈ 1.38%

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
        cg = ChainGraph((mk("c0", "x", 0, 10), mk("c1", "x", 0, 10), mk("c2", "y", 0, 30)), (), {}, {}, res)
        p = schedule(cg, "topo")
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
    cg = build_chains(g, m, **LEGACY)
    return mxm_roofline(g, cg, schedule(cg), m)


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
        assert t["a0.acc"] == (37, 165, 190)          # 28+9；+128；+25+128
        assert t["w1n0.lw"] == (9, 29, 48)            # skew 19
        assert t["b.y.wr"] == (365, 493, 513)         # 274+91；+128；+1+128+19（done 仍含 skew）

    def test_op_spans(self, run):
        g, cg, p = run
        assert op_spans(g, cg, p) == {
            "fc1": (0, 190), "fc2": (0, 491), "b1": (80, 271), "dq1": (83, 271), "act": (83, 271),
            "q1": (83, 293), "b2": (328, 491), "y": (331, 513)}

    def test_resource_usage(self, run, m):
        _g, cg, p = run
        u = resource_usage(cg, p)
        assert set(u) == set(cg.resources)                            # 没用到的资源也列出
        assert (u["mxm.w0.compute"]["busy"], u["mxm.w0.compute"]["utilization"]) == (128, F(128, 513))
        assert u["mxm.w0.weight"]["busy"] == 20
        assert u["vxm.alu"]["area"] == 6 * 128 * 2 + 5 * 128         # 两个第一层 vxm 链深 6，第二层 5
        assert u["vxm.alu"]["peak"] == 12
        assert (u["srf.W2E"]["peak"], u["srf.W2E"]["avg"]) == (32, F(4214, 513))
        assert (u["srf.E2W"]["peak"], u["srf.E2W"]["avg"]) == (32, F(2498, 513))
        assert u["sxm.w"]["busy"] == 0


class TestResult:
    def _result(self, m, run, **kw):
        g, cg, p = run
        return build_result(g, cg, p, m, make_provenance(g, m, policy="bottom_level", mem_overlap="tile", **LEGACY, **kw))

    def test_fields(self, m, run):
        r = self._result(m, run)
        assert list(r) == ["schema", "provenance", "total_cycles", "bounds", "chains", "nodes", "resources",
                           "srf", "critical_path", "critical_path_detail", "no_hop_tail", "ops", "diagnostics"]
        assert r["schema"] == "lts-result/0.2" and r["total_cycles"] == 513
        assert r["no_hop_tail"] is None                                  # 没给第二次调度
        assert r["srf"] == {"srf.E2W": {"peak": 32, "avg": 4.869396}, "srf.W2E": {"peak": 32, "avg": 8.214425}}
        assert (r["provenance"]["srf_pool"], r["provenance"]["lw_streams"]) == ("direction", "shared")
        assert r["bounds"] == {"path": 506, "resource": 136, "resource_exact": "136",
                               "resource_name": "vxm.alu", "lb": 506, "gap": 0.013834, "mxm_roofline": 128.0}
        c4 = next(c for c in r["chains"] if c["id"] == "c4")
        assert c4 == {"id": "c4", "start": 28, "len": 265, "end": 293, "rate": 1.0, "rate_exact": "1",
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
        cg = build_chains(g, m, **LEGACY)
        r = build_result(g, cg, schedule(cg), m,
                         make_provenance(g, m, policy="bottom_level", mem_overlap="tile", **LEGACY))
        assert (r["chains"][0]["rate"], r["chains"][0]["rate_exact"]) == (0.941176, "16/17")

    def test_provenance_hashes(self, m, run, tmp_path):
        g, _cg, _p = run
        a = make_provenance(g, m, policy="bottom_level", mem_overlap="tile", **LEGACY)
        b = make_provenance(g, m, policy="bottom_level", mem_overlap="tile", **LEGACY, machine_path=DEFAULT_MACHINE_PATH)
        assert len(a["tile_graph_sha256"]) == 64 and a["ltsim_commit"] is None
        assert a["tile_graph_sha256"] == b["tile_graph_sha256"]
        assert a["machine_sha256"] != b["machine_sha256"]           # 文件字节 vs 规范化内容，两种来源不同

    def test_byte_identical(self, m, tmp_path):
        outs = []
        for i in range(2):
            g = build_mlp()
            cg = build_chains(g, m, **LEGACY)
            r = build_result(g, cg, schedule(cg), m,
                             make_provenance(g, m, policy="bottom_level", mem_overlap="tile", **LEGACY))
            path = tmp_path / f"r{i}.json"
            write_result(str(path), r)
            outs.append(path.read_bytes())
        assert outs[0] == outs[1]
        assert json.loads(outs[0])["total_cycles"] == 513


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
            cg = build_chains(g, m, mem_overlap=overlap, **LEGACY)
            for pol in ("bottom_level", "topo"):
                p = schedule(cg, pol)
                b = compute_bounds(cg, p)
                assert b.total >= b.lb
                build_result(g, cg, p, m, make_provenance(g, m, policy=pol, mem_overlap=overlap, **LEGACY))


# ---- 覆盖缺口：自检、来源哈希、关键路径平局、合成图上的下界 --------------------------

class TestSelfCheck:
    def test_bad_placed_raises_internal(self, run, m):
        g, cg, p = run
        bad = type(p)(p.policy, {**p.start, "c6": 0}, p.bound_by, p.order, p.total)
        with pytest.raises(Exception, match="LTS-E-INTERNAL.*self-check"):
            build_result(g, cg, bad, m, make_provenance(g, m, policy="bottom_level", mem_overlap="tile", **LEGACY))

    def test_total_below_lb_raises_internal(self, run, m, monkeypatch):
        """单独测 total ≥ LB 这道检查：先让 verify 放行，再把所有链放在 0。"""
        import ltsim.backend.result as result_mod
        g, cg, p = run
        monkeypatch.setattr(result_mod, "verify", lambda *a: [])
        zero = {c.id: 0 for c in cg.chains}
        bad = type(p)(p.policy, zero, p.bound_by, p.order, max(c.length for c in cg.chains))
        with pytest.raises(Exception, match="LTS-E-INTERNAL.*lower bound 506"):
            build_result(g, cg, bad, m, make_provenance(g, m, policy="bottom_level", mem_overlap="tile", **LEGACY))


def test_tile_graph_path_hash(run, m, tmp_path):
    import hashlib
    from ltsim.gen.builder import graph_to_dict
    g, _cg, _p = run
    path = tmp_path / "g.json"
    path.write_text(json.dumps(graph_to_dict(g), indent=2))
    prov = make_provenance(g, m, policy="bottom_level", mem_overlap="tile", **LEGACY, tile_graph_path=str(path))
    assert prov["tile_graph_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def _chains(spec, res):
    """spec: [(链 id, 资源, start, end)] → 只含一条预留的链。"""
    from ltsim.backend.chains import Chain, ChainGraph, Reservation
    return ChainGraph(tuple(Chain(c, (), F(1), {}, {}, e, (Reservation(r, a, e, 1, c),)) for c, r, a, e in spec),
                      (), {}, {}, res)


def test_critical_path_tie_takes_lowest_index():
    """两条链同时最后结束：取在链列表里序号小的（c2 先于 c10，按序号不按字符串）。"""
    from ltsim.machine.model import Resource
    res = {f"x{i}": Resource(f"x{i}", "exclusive", 1) for i in range(11)}
    cg = _chains([(f"c{i}", f"x{i}", 0, 60 if i in (2, 10) else 10) for i in range(11)], res)
    p = schedule(cg)
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
            p = schedule(cg, pol)
            b = compute_bounds(cg, p)
            assert b.total >= b.lb
            cp = critical_path(cg, p)
            assert cp[0].bound_by.kind == "source" and cp[-1].chain in {c.id for c in cg.chains}



# ---- 默认选项与 no_hop_tail（backend.md §6、§8）--------------------------------------

def _pair(g, m, **opts):
    """主结果（带 hop 尾巴）与 no_hop_tail（不带）的 (cg, placed)。"""
    cg, cg2 = build_chains(g, m, **opts), build_chains(g, m, hop_tail=False, **opts)
    return (cg, schedule(cg)), (cg2, schedule(cg2))


class TestDefaultsAndNoHopTail:
    def test_mlp_defaults_bounds_and_path(self, m):
        (cg, p), _ = _pair(build_mlp(), m)
        b = compute_bounds(cg, p)
        assert (b.total, b.path, b.resource, b.resource_name, b.lb, b.gap) == (506, 506, 136, "vxm.alu", 506, 0)
        assert [(s.chain, s.bound_by.kind) for s in critical_path(cg, p)] == [("c0", "source"), ("c4", "edge"),
                                                                              ("c6", "edge")]

    def test_result_no_hop_tail_legacy(self, m):
        """旧选项：主结果 513（c4 被 srf.E2W 卡在 28），不带 hop 尾巴 506（c4 在 21，c6 在 267）。"""
        g = build_mlp()
        (cg, p), sec = _pair(g, m, **LEGACY)
        r = build_result(g, cg, p, m, make_provenance(g, m, policy="bottom_level", mem_overlap="tile", **LEGACY),
                         no_hop_tail=sec)
        assert r["total_cycles"] == 513
        n = r["no_hop_tail"]
        assert list(n) == ["total_cycles", "bounds", "critical_path", "critical_path_detail"]
        assert n["total_cycles"] == 506
        assert n["bounds"] == {"path": 506, "resource": 136, "resource_exact": "136", "resource_name": "vxm.alu",
                               "lb": 506, "gap": 0.0}
        assert n["critical_path"] == ["c0", "c4", "c6"]
        assert [(d["chain"], d["start"], d["end"]) for d in n["critical_path_detail"]] == [
            ("c0", 0, 48), ("c4", 21, 286), ("c6", 267, 506)]
        assert n["critical_path_detail"][1]["bound_by"] == {"kind": "edge", "edge": ["state", "w1n0.lw", "a0.acc"],
                                                            "from_chain": "c0"}

    def test_result_no_hop_tail_row_mode(self, m):
        g = build_mlp()
        (cg, p), sec = _pair(g, m, mem_overlap="row", **LEGACY)
        r = build_result(g, cg, p, m, make_provenance(g, m, policy="bottom_level", mem_overlap="row", **LEGACY),
                         no_hop_tail=sec)
        assert (r["total_cycles"], r["no_hop_tail"]["total_cycles"], r["no_hop_tail"]["bounds"]["lb"]) == (393, 386, 378)

    def test_srf_summary_keyed_by_pool(self, m):
        g = build_mlp()
        (cg, p), sec = _pair(g, m)
        r = build_result(g, cg, p, m, make_provenance(g, m, policy="bottom_level", mem_overlap="tile",
                                                      srf_pool="half", lw_streams="separate"), no_hop_tail=sec)
        assert list(r["srf"]) == sorted(x for x in cg.resources if x.startswith("srf."))
        assert r["srf"]["srf.E2W.W.lw"]["peak"] == 32 and r["srf"]["srf.E2W.W"]["peak"] == 2
        assert [x["id"] for x in r["resources"]] == sorted(cg.resources)

    def test_no_hop_tail_from_other_options_rejected(self, m):
        g = build_mlp()
        (cg, p), _ = _pair(g, m)
        _, other = _pair(g, m, **LEGACY)
        with pytest.raises(LTSError, match="LTS-E-ARG"):
            build_result(g, cg, p, m, make_provenance(g, m, policy="bottom_level", mem_overlap="tile",
                                                      srf_pool="half", lw_streams="separate"), no_hop_tail=other)

    def test_no_hop_tail_mem_overlap_mismatch_rejected(self, m):
        """row 模式副结果配 tile 主结果：链 id 与资源表相同，旧检查会放过，必须拒绝。"""
        g = build_mlp()
        (cg, p), _ = _pair(g, m)
        _, other = _pair(g, m, mem_overlap="row")
        with pytest.raises(LTSError, match="LTS-E-ARG"):
            build_result(g, cg, p, m, make_provenance(g, m, policy="bottom_level", mem_overlap="tile",
                                                      srf_pool="half", lw_streams="separate"), no_hop_tail=other)

    def test_no_hop_tail_self_rejected(self, m):
        """主结果自身不能当副结果（hop_tail 必须一真一假）。"""
        g = build_mlp()
        (cg, p), _ = _pair(g, m)
        with pytest.raises(LTSError, match="LTS-E-ARG"):
            build_result(g, cg, p, m, make_provenance(g, m, policy="bottom_level", mem_overlap="tile",
                                                      srf_pool="half", lw_streams="separate"), no_hop_tail=(cg, p))

    def test_no_hop_tail_from_altered_graph_rejected(self, m):
        """链 id、节点 id、资源表都相同但图内容不同（一个节点 rows 减半）→ 拒绝。"""
        g = build_mlp()
        (cg, p), _ = _pair(g, m)
        g2 = replace(g, nodes=tuple(replace(n, rows=n.rows // 2) if n.id == "w1n0.rd" else n for n in g.nodes))
        _, other = _pair(g2, m)
        with pytest.raises(LTSError, match="LTS-E-ARG"):
            build_result(g, cg, p, m, make_provenance(g, m, policy="bottom_level", mem_overlap="tile",
                                                      srf_pool="half", lw_streams="separate"), no_hop_tail=other)

    def test_byte_identical_defaults(self, m):
        def once():
            g = build_mlp()
            (cg, p), sec = _pair(g, m)
            return dumps_result(build_result(g, cg, p, m, make_provenance(
                g, m, policy="bottom_level", mem_overlap="tile", srf_pool="half", lw_streams="separate"),
                no_hop_tail=sec))
        assert once() == once()
