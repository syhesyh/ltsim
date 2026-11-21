"""MXM 顺序边辅助函数 + 参数化 GEMM 生成器的测试。

期望值按设计 §5.5 的边表手算：
  state        lw_j → acc_j 首个             （由 state 边类型给默认 ref/lag）
  acc_seq      块内相邻 acc                   ref=issue_end lag=0
  block_switch 块 j 末个 acc → 块 j+1 首个    ref=issue_end lag="mxm.block_switch_gap"
  lw_after_iw  块 j−1 首个 acc → lw_j         ref=start     lag=−1
  bank_war     块 j−1 末个 acc → lw_j（single）ref=issue_end lag=0
"""
import json
import os

import pytest

from ltsim.errors import LTSError
from ltsim.gen.builder import GraphBuilder, Out, _to_dict
from ltsim.gen.gemm import build_gemm
from ltsim.graph.load import load_graph, resolve_lags, validate_units
from ltsim.machine.model import load_machine

MACHINE = os.path.join(os.path.dirname(__file__), "..", "..", "ltsim", "machine", "lpu-tile-v0.yaml")


@pytest.fixture(scope="module")
def m():
    return load_machine(MACHINE)


def order_set(g):
    return {(o.src, o.dst, o.ref, o.lag, o.reason) for o in g.edges_order}


def _plane_builder(n_blocks, n_m):
    """一个 plane 上 n_blocks 个权重块、每块 n_m 个 acc 的最小图。"""
    b = GraphBuilder("t", "lpu-tile-v0")
    blocks = []
    for j in range(n_blocks):
        b.load_w(f"lw{j}", "mxm.w0", j % 2, 320, ["mm"])
        accs = []
        for i in range(n_m):
            b.acc(f"a{j}_{i}", "mxm.w0", j % 2, 8, Out("i32", 4), ["mm"])
            accs.append(f"a{j}_{i}")
        blocks.append((f"lw{j}", accs))
    return b, blocks


class TestLinkPlane:
    def test_three_blocks_alternate(self):
        b, blocks = _plane_builder(3, 2)
        b.link_plane(blocks, "alternate")
        g = b.build()
        assert set(g.edges_state) == {("lw0", "a0_0"), ("lw1", "a1_0"), ("lw2", "a2_0")}
        assert order_set(g) == {
            ("a0_0", "a0_1", "issue_end", 0, "acc_seq"),
            ("a1_0", "a1_1", "issue_end", 0, "acc_seq"),
            ("a2_0", "a2_1", "issue_end", 0, "acc_seq"),
            ("a0_1", "a1_0", "issue_end", "mxm.block_switch_gap", "block_switch"),
            ("a1_1", "a2_0", "issue_end", "mxm.block_switch_gap", "block_switch"),
            ("a0_0", "lw1", "start", -1, "lw_after_iw"),
            ("a1_0", "lw2", "start", -1, "lw_after_iw"),
        }

    def test_single_adds_bank_war(self):
        b, blocks = _plane_builder(3, 2)
        b.link_plane(blocks, "single")
        g = b.build()
        wars = {(o.src, o.dst) for o in g.edges_order if o.reason == "bank_war"}
        assert wars == {("a0_1", "lw1"), ("a1_1", "lw2")}
        assert all(o.ref == "issue_end" and o.lag == 0 for o in g.edges_order if o.reason == "bank_war")

    def test_single_block_only_state(self):
        b, blocks = _plane_builder(1, 3)
        b.link_plane(blocks, "alternate")
        g = b.build()
        assert set(g.edges_state) == {("lw0", "a0_0")}
        # 只有块内 acc_seq，没有跨块边
        assert {o.reason for o in g.edges_order} == {"acc_seq"}

    def test_bad_banks(self):
        b, blocks = _plane_builder(2, 1)
        with pytest.raises(LTSError, match="LTS-E-GEN"):
            b.link_plane(blocks, "triple")


class TestGemmSmall:
    """M=256, K=640, N=640, tile_m=128 → Mb=2, Kb=2, Nb=2；planes {w0, w1}。"""

    def test_counts_n_split_alternate(self):
        g = build_gemm(256, 640, 640, tile_m=128)
        # 节点：权重 4 块×2 = 8；计算 8 个 (m,n,k)×3 = 24；K 合并 4 个 (m,n)×(2 读 + sum + 写) = 16
        assert len(g.nodes) == 48
        # stream：权重 4；计算 8×2 = 16；K 合并 4×3 = 12
        assert len(g.edges_stream) == 32
        assert len(g.edges_state) == 4          # 每个权重块一条
        assert len(g.edges_mem) == 8            # 每个 partial 一条
        # order：每 plane 2 块×1 acc_seq + 1 block_switch + 1 lw_after_iw = 4，两个 plane = 8
        assert len(g.edges_order) == 8

    def test_n_split_w0_edges(self):
        g = build_gemm(256, 640, 640, tile_m=128)
        # n_split：w0 负责 n0 的两个 K 块 (0,0) bank0、(0,1) bank1
        assert g.node("w.n0k0.lw").unit == "mxm.w0" and g.node("w.n0k0.lw").bank == 0
        assert g.node("w.n0k1.lw").unit == "mxm.w0" and g.node("w.n0k1.lw").bank == 1
        assert g.node("w.n1k0.lw").unit == "mxm.w1" and g.node("w.n1k0.lw").bank == 0
        # acc 继承其权重块的 plane 与 bank
        assert g.node("m1n0k1.acc").unit == "mxm.w0" and g.node("m1n0k1.acc").bank == 1
        w0 = {e for e in order_set(g) if e[0].endswith("n0k0.acc") or e[0].endswith("n0k1.acc")}
        assert w0 == {
            ("m0n0k0.acc", "m1n0k0.acc", "issue_end", 0, "acc_seq"),
            ("m0n0k1.acc", "m1n0k1.acc", "issue_end", 0, "acc_seq"),
            ("m1n0k0.acc", "m0n0k1.acc", "issue_end", "mxm.block_switch_gap", "block_switch"),
            ("m0n0k0.acc", "w.n0k1.lw", "start", -1, "lw_after_iw"),
        }
        assert ("w.n0k0.lw", "m0n0k0.acc") in g.edges_state
        assert ("w.n0k1.lw", "m0n0k1.acc") in g.edges_state

    def test_single_banks(self):
        g = build_gemm(256, 640, 640, tile_m=128, banks="single")
        assert all(n.bank == 0 for n in g.nodes if n.kind in ("mxm.load_w", "mxm.acc"))
        wars = {(o.src, o.dst) for o in g.edges_order if o.reason == "bank_war"}
        assert wars == {("m1n0k0.acc", "w.n0k1.lw"), ("m1n1k0.acc", "w.n1k1.lw")}
        assert len(g.edges_order) == 10

    def test_k_split(self):
        g = build_gemm(256, 640, 640, tile_m=128, map="k_split")
        # k_split：K 块 0 → w0，K 块 1 → w1；w0 上按 (n,k) 字典序为 (0,0)、(1,0)
        assert g.node("w.n0k0.lw").unit == "mxm.w0"
        assert g.node("w.n1k0.lw").unit == "mxm.w0" and g.node("w.n1k0.lw").bank == 1
        assert g.node("w.n0k1.lw").unit == "mxm.w1"
        assert ("m1n0k0.acc", "m0n1k0.acc", "issue_end", "mxm.block_switch_gap", "block_switch") in order_set(g)

    def test_single_k_block_writes_y_directly(self):
        g = build_gemm(256, 320, 640, tile_m=128)
        assert not any(n.kind == "vxm.chain" for n in g.nodes)
        assert not g.edges_mem
        assert g.node("m0n0k0.wr").tile == "Y[0:128,0:320]"

    def test_chain_count(self):
        """按 stream 边求连通分量：权重 4 + 计算 8 + K 合并 4 = 16 条链，每条链至多一个 acc。"""
        g = build_gemm(256, 640, 640, tile_m=128)
        parent = {n.id: n.id for n in g.nodes}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for s, d, _ in g.edges_stream:
            parent[find(s)] = find(d)
        comps = {}
        for n in g.nodes:
            comps.setdefault(find(n.id), []).append(n)
        assert len(comps) == 16
        assert all(sum(n.kind == "mxm.acc" for n in c) <= 1 for c in comps.values())

    def test_roundtrip_and_resolve(self, m, tmp_path):
        p = tmp_path / "gemm.json"
        p.write_text(json.dumps(_to_dict(build_gemm(256, 640, 640, tile_m=128))))
        g = load_graph(str(p))
        validate_units(g, m)
        rg = resolve_lags(g, m)
        lags = {(e[1], e[2]): e[4] for e in rg.non_stream_edges if e[0] == "order"}
        assert lags[("m1n0k0.acc", "m0n0k1.acc")] == 5          # block_switch_gap
        assert lags[("m0n0k0.acc", "w.n0k1.lw")] == -1          # lw_after_iw

    def test_deterministic(self):
        a = json.dumps(_to_dict(build_gemm(256, 640, 640, tile_m=128)))
        b = json.dumps(_to_dict(build_gemm(256, 640, 640, tile_m=128)))
        assert a == b


class TestGemmErrors:
    def test_tile_not_dividing(self):
        with pytest.raises(LTSError, match="LTS-E-GEN"):
            build_gemm(250, 640, 640, tile_m=128)

    def test_block_too_big(self):
        with pytest.raises(LTSError, match="LTS-E-MXM-BLOCK"):
            build_gemm(256, 640, 640, tile_m=128, tile_k=640)

    def test_mixed_sides(self):
        """双侧已支持（Nb ≥ 2）；Nb < 2 不够分侧时报 LTS-E-GEN。"""
        build_gemm(256, 640, 640, tile_m=128, planes=["mxm.w0", "mxm.e0"])
        with pytest.raises(LTSError, match="LTS-E-GEN"):
            build_gemm(256, 640, 320, tile_m=128, planes=["mxm.w0", "mxm.e0"])     # Nb = 1

    def test_bad_p_placement(self):
        with pytest.raises(LTSError, match="LTS-E-ARG"):
            build_gemm(256, 640, 640, tile_m=128, p_placement="sideways")


# ---- 双侧模式与 partial 交替放置（gemm.py 模块 docstring）----------------------------

FOUR = ("mxm.w0", "mxm.w1", "mxm.e0", "mxm.e1")


def _gemv(**kw):
    """8×4096×4096，tile 8×256×256：Kb = 16，合并链平铺读回需要 16×4 = 64 条流。"""
    return build_gemm(8, 4096, 4096, tile_m=8, tile_k=256, tile_n=256, **kw)


class TestDualSide:
    def test_side_split_and_names(self):
        g = _gemv(planes=FOUR)                                    # Nb = 16：n<8 西、其余东
        assert set(g.tensors) >= {"XW", "XE", "WW", "WE", "YW", "YE"}
        assert {t for t in g.tensors if t.startswith("P")} == {f"P{k}{h}" for k in range(16) for h in "WE"}
        for n in g.nodes:
            if n.kind == "mxm.load_w":
                blk = int(n.id.split(".")[1][1:].split("k")[0])   # w.n<n>k<k>.lw → n
                assert n.unit.startswith("mxm.w") == (blk < 8)   # 西侧 plane 只装 n<8 的块
        # 每个 (n,k) 只属于一个 plane：load_w 数 = Nb × Kb = 256
        assert sum(n.kind == "mxm.load_w" for n in g.nodes) == 256

    def test_odd_nb_split(self):
        g = build_gemm(128, 320, 960, tile_m=128, planes=FOUR)    # Nb = 3：n<2 西、n=2 东
        side = {0: "w", 1: "w", 2: "e"}
        for n in g.nodes:
            if n.kind == "mxm.load_w":
                blk = int(n.id.split(".")[1][1:].split("k")[0])
                assert n.unit.startswith(f"mxm.{side[blk]}")

    def test_dual_local_placement_kb2(self, m):
        """双侧 + Kb=2（V=4，不需交替）：P{k}W 在 W、P{k}E 在 E，可调可验。"""
        from ltsim.backend.chains import build_chains
        from ltsim.backend.scheduler import schedule
        g = build_gemm(256, 640, 1280, tile_m=128, planes=FOUR)
        assert (g.tensors["P0W"].hemisphere, g.tensors["P1E"].hemisphere) == ("W", "E")
        cg = build_chains(g, m)
        schedule(cg)

    def test_dual_gemv_schedules(self, m):
        """GEMV 端到端：auto → alternate，合并链两个池峰值都正好 32，不再报流宽。"""
        from ltsim.backend.bounds import compute_bounds
        from ltsim.backend.chains import build_chains
        from ltsim.backend.metrics import resource_usage
        from ltsim.backend.scheduler import schedule
        g = _gemv(planes=FOUR)
        assert (g.tensors["P0W"].hemisphere, g.tensors["P1W"].hemisphere) == ("W", "E")
        cg = build_chains(g, m)
        p = schedule(cg)
        u = resource_usage(cg, p)
        assert u["srf.W2E.W"]["peak"] == 32 and u["srf.E2W.E"]["peak"] == 32
        b = compute_bounds(cg, p)
        assert b.total >= b.lb
        # 同形状只用西侧 2 个 MXM 的对照：交替放置让它可行，但比 4 个 MXM 慢
        g2 = _gemv()
        cg2 = build_chains(g2, m)
        b2 = compute_bounds(cg2, schedule(cg2))
        assert b.total < b2.total


class TestPartialPlacement:
    def test_auto_local_when_narrow(self):
        g = build_gemm(256, 640, 640, tile_m=128, banks="single")  # Kb=2，V=4：8 条流就够
        assert {t.hemisphere for nm, t in g.tensors.items() if nm.startswith("P")} == {"W"}

    def test_auto_alternate_single_side(self):
        g = _gemv()                                                # 默认 planes 全在西侧
        hemi = {k: g.tensors[f"P{k}"].hemisphere for k in range(16)}
        assert hemi == {k: ("W" if k % 2 == 0 else "E") for k in range(16)}

    def test_default_equals_explicit_local_when_narrow(self):
        a = json.dumps(_to_dict(build_gemm(256, 640, 640, tile_m=128)))
        b = json.dumps(_to_dict(build_gemm(256, 640, 640, tile_m=128, p_placement="local")))
        assert a == b

    def test_local_still_raises_width(self, m):
        from ltsim.backend.chains import build_chains
        g = _gemv(p_placement="local")
        with pytest.raises(LTSError, match="LTS-E-SRF-WIDTH"):
            build_chains(g, m)
