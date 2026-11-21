"""B2 chains.py 的测试。

金标准来自 backend.md §8.1–8.2 的手算表（MLP），以及按 §4.4 公式手算的预留表；
负例覆盖 §4.5 的每个诊断码；row 模式按设计 §7.3 的两条约束手算。
"""
import pytest

from ltsim.backend.chains import ChainEdge, Reservation, build_chains
from ltsim.errors import LTSError
from ltsim.gen.builder import GraphBuilder, Out
from ltsim.gen.gemm import build_gemm
from ltsim.gen.mlp import build_mlp
from ltsim.graph.load import validate_graph
from ltsim.machine.model import default_machine

I8, I32, F32 = Out("i8", 1), Out("i32", 4), Out("f32", 4)


@pytest.fixture(scope="module")
def m():
    return default_machine()


@pytest.fixture(scope="module")
def cg(m):
    return build_chains(build_mlp(), m)


def R(resource, start, end, amount, node):
    return Reservation(resource, start, end, amount, node)


# ---- MLP 金标准 --------------------------------------------------------------

class TestMlpChains:
    def test_chain_ids_and_members(self, cg):
        """链 id 按"分量里在 nodes 中最靠前的节点"排序（backend.md §4.1）。"""
        assert [c.id for c in cg.chains] == [f"c{i}" for i in range(7)]
        assert cg.chain("c0").nodes == ("w1n0.rd", "w1n0.lw")
        assert cg.chain("c3").nodes == ("w2k1.rd", "w2k1.lw")
        assert cg.chain("c4").nodes == ("a0.x.rd", "a0.acc", "a0.b1.rd", "a0.vxm", "a0.h.wr")
        assert cg.chain("c6").nodes == ("b.h0.rd", "b.h1.rd", "b.acc0", "b.acc1", "b.b2.rd", "b.vxm", "b.y.wr")
        assert cg.chain_of["a1.vxm"] == "c5" and cg.chain_of["b.y.wr"] == "c6"

    def test_c0_weight_chain(self, cg):
        c = cg.chain("c0")
        assert c.rate == 16 and c.D == {"w1n0.rd": 20, "w1n0.lw": 20}
        assert c.offset == {"w1n0.rd": 0, "w1n0.lw": 9}            # 0 + L 1 + hop 8
        assert c.length == 48                                       # 9 + 0 + 20 + 19
        assert c.reservations == (
            R("mem.W.rd", 0, 20, 16, "w1n0.rd"),
            R("mxm.w0.weight", 9, 29, 1, "w1n0.lw"),
            R("srf.E2W", 1, 48, 16, "w1n0.rd"),                     # [1, 1+20+19+8)
        )

    def test_c2_east_weight_chain(self, cg):
        c = cg.chain("c2")
        assert c.offset == {"w2k0.rd": 0, "w2k0.lw": 10}           # hop MEM_E>MXM_E = 9
        assert c.length == 49
        assert c.reservations == (
            R("mem.E.rd", 0, 20, 16, "w2k0.rd"),
            R("mxm.e0.weight", 10, 30, 1, "w2k0.lw"),
            R("srf.W2E", 1, 49, 16, "w2k0.rd"),
        )

    def test_c4_first_layer(self, cg):
        c = cg.chain("c4")
        assert c.rate == 1 and set(c.D.values()) == {128}
        # 读 B1 被推迟到 52，让偏置与矩阵乘结果同时到达 VXM（backend.md §8.1）
        assert c.offset == {"a0.x.rd": 0, "a0.acc": 9, "a0.b1.rd": 52, "a0.vxm": 55, "a0.h.wr": 117}
        assert c.length == 265                                      # 117 + 1 + 128 + 19
        assert c.reservations == (
            R("mem.E.wr", 118, 246, 1, "a0.h.wr"),                  # [o+L, o+L+D)，V_in = 1
            R("mem.W.rd", 0, 128, 1, "a0.x.rd"),
            R("mem.W.rd", 52, 180, 4, "a0.b1.rd"),                  # f32 V=4
            R("mxm.w0.compute", 9, 137, 1, "a0.acc"),
            R("srf.E2W", 1, 156, 1, "a0.x.rd"),                     # [1, 1+128+19+8)
            R("srf.W2E", 34, 202, 4, "a0.acc"),                     # [9+25, 34+128+19+21)
            R("srf.W2E", 53, 202, 4, "a0.b1.rd"),
            R("srf.W2E", 115, 264, 1, "a0.vxm"),
            R("vxm.alu", 55, 183, 6, "a0.vxm"),                     # 链深 6
        )

    def test_c6_second_layer(self, cg):
        c = cg.chain("c6")
        assert c.offset == {"b.h0.rd": 0, "b.h1.rd": 0, "b.acc0": 10, "b.acc1": 10,
                            "b.b2.rd": 54, "b.vxm": 57, "b.y.wr": 91}
        assert c.length == 239                                      # 91 + 1 + 128 + 19
        res = {(r.node, r.resource): (r.start, r.end, r.amount) for r in c.reservations}
        assert res[("b.vxm", "vxm.alu")] == (57, 185, 5)
        assert res[("b.y.wr", "mem.W.wr")] == (92, 220, 1)
        assert res[("b.acc1", "mxm.e1.compute")] == (10, 138, 1)
        assert res[("b.acc0", "srf.E2W")] == (35, 204, 4)

    def test_chain_edges(self, cg):
        """backend.md §8.2。"""
        assert cg.edges == (
            ChainEdge("c0", "c4", 21, ("state", "w1n0.lw", "a0.acc"), "issue_end"),   # (9+20) + 1 − 9
            ChainEdge("c1", "c5", 21, ("state", "w1n1.lw", "a1.acc"), "issue_end"),
            ChainEdge("c2", "c6", 21, ("state", "w2k0.lw", "b.acc0"), "issue_end"),   # (10+20) + 1 − 10
            ChainEdge("c3", "c6", 21, ("state", "w2k1.lw", "b.acc1"), "issue_end"),
            ChainEdge("c4", "c6", 265, ("mem", "a0.h.wr", "b.h0.rd"), "done"),        # done(a0.h.wr) + 0 − 0
            ChainEdge("c5", "c6", 265, ("mem", "a1.h.wr", "b.h1.rd"), "done"),
        )

    def test_srf_area_matches_lb_res(self, cg):
        """旁证：W2E 的 Σ 条数×时长 = 5278，/32 ≈ 165 —— 即 backend.md §8.3 的 LB_res。"""
        area = {"srf.W2E": 0, "srf.E2W": 0}
        for c in cg.chains:
            for r in c.reservations:
                if r.resource in area:
                    area[r.resource] += r.amount * (r.end - r.start)
        assert area == {"srf.W2E": 5278, "srf.E2W": 3315}

    def test_works_carried(self, cg):
        assert cg.works["a0.vxm"].L == 60

    def test_deterministic(self, m):
        assert build_chains(build_mlp(), m) == build_chains(build_mlp(), m)


# ---- GEMM：多块换权重 --------------------------------------------------------

class TestGemm:
    def _weights(self, cg):
        return sorted({(e.edge[0], e.weight) for e in cg.edges})

    def test_alternate(self, m):
        cg = build_chains(build_gemm(256, 640, 640, tile_m=128), m)
        assert len(cg.chains) == 16 and len(cg.edges) == 20
        # state 21；acc_seq 128（等前一个 acc 占满 128 拍）；block_switch 128+5；lw_after_iw −1；
        # mem：partial 写完的 done = 42 + 1 + 128 + 19 = 190
        assert self._weights(cg) == [("mem", 190), ("order", -1), ("order", 128), ("order", 133), ("state", 21)]

    def test_single_adds_bank_war(self, m):
        cg = build_chains(build_gemm(256, 640, 640, tile_m=128, banks="single"), m)
        assert len(cg.edges) == 22
        wars = [e for e in cg.edges if e.edge == ("order", "m1n0k0.acc", "w.n0k1.lw")]
        assert len(wars) == 1 and wars[0].weight == 128               # (9+128) + 0 − 9


# ---- 偏移：等式传播 ----------------------------------------------------------

def _fork_with_sink():
    """分叉的一支是汇点：规格原来的 ASAP/ALAP 在这里会误报对齐失败。"""
    b = GraphBuilder("fork", "lpu-tile-v0")
    b.tensor("X", "i8", (8, 320), "input", "W", "row")
    b.tensor("Y", "i8", (8, 320), "input", "E", "row")
    b.tensor("P", "i32", (8, 320), "intermediate", "W", "row")
    b.tensor("Z", "i32", (8, 320), "output", "E", "row")
    b.read("x.rd", "X[0:8,0:320]", 8, I8, ["op"])
    b.acc("acc.w", "mxm.w0", 0, 8, I32, ["op"])
    b.write("p.wr", "P[0:8,0:320]", 8, ["op"])
    b.read("y.rd", "Y[0:8,0:320]", 8, I8, ["op"])
    b.acc("acc.e", "mxm.e0", 0, 8, I32, ["op"])
    b.vxm("sum", ["add.i32"], 8, I32, ["op"])
    b.write("z.wr", "Z[0:8,0:320]", 8, ["op"])
    for e in [("x.rd", "acc.w"), ("acc.w", "p.wr"), ("acc.w", "sum"),
              ("y.rd", "acc.e"), ("acc.e", "sum"), ("sum", "z.wr")]:
        b.stream(*e)
    return b.build()


class TestOffsets:
    def test_fork_with_sink_is_solvable(self, m):
        g = _fork_with_sink()
        validate_graph(g, m)
        c = build_chains(g, m).chain("c0")
        # 东侧路径 0 + (1+9) + (25+22) = 57 决定 sum；西侧整体推迟 2 拍刚好赶上
        assert c.offset == {"x.rd": 2, "acc.w": 11, "p.wr": 44, "y.rd": 0, "acc.e": 10, "sum": 57, "z.wr": 63}

    def test_true_misalignment(self, m):
        """同一个读值经 acc 到达 vxm 要 28+46 拍，直接到达只要 3 拍：无解。"""
        b = GraphBuilder("t", "lpu-tile-v0")
        b.tensor("X", "i8", (8, 320), "input", "E", "row")
        b.read("r", "X[0:8,0:320]", 8, I8, ["op"])
        b.acc("a", "mxm.w0", 0, 8, I32, ["op"])
        b.vxm("v", ["add.i32"], 8, I32, ["op"])
        b.stream("r", "a"); b.stream("a", "v"); b.stream("r", "v")
        with pytest.raises(LTSError, match="LTS-E-CHAIN-ALIGN"):
            build_chains(b.build(), m)

    def test_stream_cycle(self, m):
        b = GraphBuilder("t", "lpu-tile-v0")
        b.vxm("v", ["relu"], 8, I8, ["op"])
        b.acc("a", "mxm.w0", 0, 8, I32, ["op"])
        b.stream("v", "a"); b.stream("a", "v")
        with pytest.raises(LTSError, match="LTS-E-GRAPH-CYCLE"):
            build_chains(b.build(), m)


# ---- 链级合法性 --------------------------------------------------------------

class TestChainLegality:
    def test_srf_width(self, m):
        """9 个 f32 读汇入一个 vxm：W2E 方向 9×4 = 36 > 32。"""
        b = GraphBuilder("t", "lpu-tile-v0")
        b.vxm("v", ["add.f32"] * 8, 8, F32, ["op"])
        for i in range(9):
            b.tensor(f"A{i}", "f32", (8, 320), "input", "W", "row")
            b.read(f"r{i}", f"A{i}[0:8,0:320]", 8, F32, ["op"])
            b.stream(f"r{i}", "v")
        with pytest.raises(LTSError, match="LTS-E-SRF-WIDTH"):
            build_chains(b.build(), m)

    def test_exclusive_self_conflict(self, m):
        """同一条链里两个 acc 在同一个 plane 上同时算。"""
        b = GraphBuilder("t", "lpu-tile-v0")
        b.tensor("X", "i8", (8, 320), "input", "W", "row")
        b.tensor("Y", "i32", (8, 320), "output", "E", "row")
        for i in range(2):
            b.read(f"x{i}", "X[0:8,0:320]", 8, I8, ["op"])
            b.acc(f"acc{i}", "mxm.w0", i, 8, I32, ["op"])
            b.stream(f"x{i}", f"acc{i}")
        b.vxm("v", ["add.i32"], 8, I32, ["op"])
        b.write("y", "Y[0:8,0:320]", 8, ["op"])
        b.stream("acc0", "v"); b.stream("acc1", "v"); b.stream("v", "y")
        with pytest.raises(LTSError, match="LTS-E-CHAIN-SELF.*mxm.w0.compute"):
            build_chains(b.build(), m)

    def _two_vxm(self, d):
        b = GraphBuilder("t", "lpu-tile-v0")
        b.tensor("X", "i32", (8, 320), "input", "E", "row")
        b.read("r", "X[0:8,0:320]", 8, I32, ["op"])
        b.vxm("v1", ["add.i32"] * d, 8, I32, ["op"])
        b.vxm("v2", ["add.i32"] * d, 8, I32, ["op"])
        b.stream("r", "v1"); b.stream("r", "v2")
        return b.build()

    def test_counter_self_conflict(self, m):
        """同一条链里两个链深 16 的 vxm 同时算：vxm.alu 逐拍 32 > 16。"""
        with pytest.raises(LTSError, match="LTS-E-CHAIN-SELF.*vxm.alu"):
            build_chains(self._two_vxm(16), m)

    @pytest.mark.parametrize("r0,ok", [(128, True), (127, False)])
    def test_counter_touching_vs_overlap(self, m, r0, ok):
        """同一条链里两个链深 10 的 vxm 吃同一个 acc 的输出：v1 吃前 128 行，在 [55,183)；
        v2 从第 r0 行开始吃。r0 = 128 时 v2 在 [183,311)，首尾相接，10 + 10 不同时出现，通过；
        r0 = 127 时重叠 1 拍，20 > 16，报错。"""
        b = GraphBuilder("t", "lpu-tile-v0")
        b.tensor("X", "i8", (256, 320), "input", "W", "row")
        b.read("x.rd", "X[0:256,0:320]", 256, I8, ["op"])
        b.acc("acc", "mxm.w0", 0, 256, I32, ["op"])
        b.vxm("v1", ["add.i32"] * 10, 128, I32, ["op"])
        b.vxm("v2", ["add.i32"] * 10, 128, I32, ["op"])
        b.stream("x.rd", "acc"); b.stream("acc", "v1"); b.stream("acc", "v2", r0)
        if ok:
            c = build_chains(b.build(), m).chain("c0")
            assert (c.offset["v1"], c.offset["v2"]) == (55, 183)
        else:
            with pytest.raises(LTSError, match="LTS-E-CHAIN-SELF.*vxm.alu"):
                build_chains(b.build(), m)

    def test_out_without_consumer_takes_no_stream(self, m):
        """v1、v2 有 out 但没有流消费者：不占流。8+8 = 16 个 ALU 恰好放满。"""
        c = build_chains(self._two_vxm(8), m).chain("c0")
        assert {r.node for r in c.reservations if r.resource.startswith("srf.")} == {"r"}
        assert sum(r.amount for r in c.reservations if r.resource == "vxm.alu") == 16


# ---- 同链 mem 边（对半树第 1 层与上游融合的形状）----------------------------------

def _same_chain_mem(h, extra_order=False):
    """acc 的结果一份写回 S，一份从第 h 行起直接流进 vxm；vxm 的另一路输入从 MEM 读 S 的前 h 行。"""
    b = GraphBuilder("t", "lpu-tile-v0")
    b.tensor("X", "i8", (128, 320), "input", "W", "row")
    b.tensor("S", "i32", (128, 320), "intermediate", "W", "row")
    b.tensor("O", "i32", (h, 320), "output", "E", "row")
    b.read("x.rd", "X[0:128,0:320]", 128, I8, ["op"])
    b.acc("acc", "mxm.w0", 0, 128, I32, ["op"])
    b.write("s.wr", "S[0:128,0:320]", 128, ["op"])
    b.read("low.rd", f"S[0:{h},0:320]", h, I32, ["op"])
    b.vxm("v", ["add.i32"], h, I32, ["op"])
    b.write("o.wr", f"O[0:{h},0:320]", h, ["op"])
    b.stream("x.rd", "acc"); b.stream("acc", "s.wr"); b.stream("acc", "v", h)
    b.stream("low.rd", "v"); b.stream("v", "o.wr")
    b.mem("s.wr", "low.rd")
    if extra_order:
        b.order("acc", "low.rd", ref="issue_end", lag=0, reason="acc_seq")
    return b.build()


class TestSameChainMem:
    def test_passes_when_low_half_is_written(self, m):
        """h = 64：vxm 在 9+25+21+64 = 119 开工，读低半在 116；
        第 i 行写完可见于 42+1+i+19 = 62+i ≤ 116+i，成立。"""
        g = _same_chain_mem(64)
        validate_graph(g, m)
        cg = build_chains(g, m)
        assert len(cg.chains) == 1 and cg.edges == ()
        assert cg.chain("c0").offset["low.rd"] == 116

    def test_fails_when_h_too_small(self, m):
        """h = 8：读低半在 60，但第 0 行要到 62 才可见 —— 对应设计 §5.4 的"h 太小，需取消融合"。"""
        with pytest.raises(LTSError, match="LTS-E-CHAIN-SELF.*reads word 0 2 cycle"):
            build_chains(_same_chain_mem(8), m)

    def test_boundary(self, m):
        """读低半在 52+h，第 0 个 word 可见于 62：h = 10 刚好相等，通过；h = 9 差 1 拍，报错。"""
        build_chains(_same_chain_mem(10), m)
        with pytest.raises(LTSError, match="LTS-E-CHAIN-SELF.*reads word 0 1 cycle"):
            build_chains(_same_chain_mem(9), m)

    def test_other_same_chain_edge(self, m):
        """同链的 order 边：要求 offset(low.rd) ≥ issue_end(acc) = 9 + 128 = 137，实际 116。"""
        with pytest.raises(LTSError, match="LTS-E-CHAIN-SELF.*order edge acc->low.rd"):
            build_chains(_same_chain_mem(64, extra_order=True), m)


# ---- 链间约束图 --------------------------------------------------------------

def test_chain_cycle(m):
    b = GraphBuilder("t", "lpu-tile-v0")
    b.tensor("X", "i8", (8, 320), "input", "W", "row")
    for i, p in enumerate(("mxm.w0", "mxm.w1")):
        b.tensor(f"Y{i}", "i32", (8, 320), "output", "W", "row")
        b.read(f"x{i}", "X[0:8,0:320]", 8, I8, ["op"])
        b.acc(f"a{i}", p, 0, 8, I32, ["op"])
        b.write(f"y{i}", f"Y{i}[0:8,0:320]", 8, ["op"])
        b.stream(f"x{i}", f"a{i}"); b.stream(f"a{i}", f"y{i}")
    b.order("a0", "a1", ref="issue_end", lag=0, reason="acc_seq")
    b.order("a1", "a0", ref="issue_end", lag=0, reason="acc_seq")
    with pytest.raises(LTSError, match=r"LTS-E-CHAIN-CYCLE.*c0.*c1|LTS-E-CHAIN-CYCLE.*c1.*c0"):
        build_chains(b.build(), m)


# ---- --mem-overlap row --------------------------------------------------------

def _split_write(read_rows=128, two_writes=True):
    """两个写链各写 H 的一半（或一个写链写全部），一个读链读 H[0:read_rows]。"""
    b = GraphBuilder("t", "lpu-tile-v0")
    b.tensor("X", "i8", (128, 320), "input", "W", "row")
    b.tensor("H", "i32", (128, 320), "intermediate", "W", "row")
    b.tensor("Y", "i32", (read_rows, 320), "output", "E", "row")
    parts = [(0, 64, "mxm.w0"), (64, 128, "mxm.w1")] if two_writes else [(0, 128, "mxm.w0")]
    for i, (r0, r1, p) in enumerate(parts):
        b.read(f"x{i}", f"X[{r0}:{r1},0:320]", r1 - r0, I8, ["op"])
        b.acc(f"a{i}", p, 0, r1 - r0, I32, ["op"])
        b.write(f"w{i}", f"H[{r0}:{r1},0:320]", r1 - r0, ["op"])
        b.stream(f"x{i}", f"a{i}"); b.stream(f"a{i}", f"w{i}")
    b.read("h.rd", f"H[0:{read_rows},0:320]", read_rows, I32, ["op"])
    b.vxm("v", ["add.i32"], read_rows, I32, ["op"])
    b.write("y", f"Y[0:{read_rows},0:320]", read_rows, ["op"])
    b.stream("h.rd", "v"); b.stream("v", "y")
    for i in range(len(parts)):
        b.mem(f"w{i}", "h.rd")
    return b.build()


class TestMemOverlapRow:
    def test_mlp(self, m):
        """c4→c6：第一行 117+1+19+0−0 = 137；最后一行 265+0−128−0 = 137。state 边不变。"""
        cg = build_chains(build_mlp(), m, mem_overlap="row")
        w = {(e.src, e.dst): e.weight for e in cg.edges}
        assert w[("c4", "c6")] == 137 and w[("c5", "c6")] == 137
        assert w[("c0", "c4")] == 21

    def test_gemm_partial(self, m):
        """partial 写 → K 合并读：第一行 42+1+19 = 62；最后一行 190−128 = 62。"""
        cg = build_chains(build_gemm(256, 640, 640, tile_m=128), m, mem_overlap="row")
        assert {e.weight for e in cg.edges if e.edge[0] == "mem"} == {62}

    def test_one_to_one(self, m):
        """一个写链写全部 H，读链读全部 H。写节点 o = 9+25+8 = 42，两侧 ρ = 1：
        每个 word 都是 42+1+i+19 − (0+i) = 62。tile 语义是 done = 42+1+128+19 = 190。"""
        g = _split_write(two_writes=False)
        validate_graph(g, m)
        assert {e.weight for e in build_chains(g, m, mem_overlap="row").edges} == {62}
        assert {e.weight for e in build_chains(g, m).edges} == {190}

    def test_many_to_one(self, m):
        """读 H[0:128) 依赖两个写（各 64 行，o 都是 42）。逐 word 精确：
        w0（word 0–63）：62 + i − i = 62；w1（word 64–127）：62 + (word−64) − word = −2，即读到后半段时 w1 早已写完。
        tile 语义两条都是 done − o_r = 42+1+64+19 = 126。"""
        g = _split_write(two_writes=True)
        validate_graph(g, m)
        row = {e.edge[1]: e.weight for e in build_chains(g, m, mem_overlap="row").edges}
        assert row == {"w0": 62, "w1": -2}
        assert {e.weight for e in build_chains(g, m).edges} == {126}

    def test_different_word_range(self, m):
        """写 H[0:128)，读 H[0:64)：公共 word 0–63，每个都是 62。"""
        g = _split_write(read_rows=64, two_writes=False)
        validate_graph(g, m)
        assert {e.weight for e in build_chains(g, m, mem_overlap="row").edges} == {62}
        assert {e.weight for e in build_chains(g, m).edges} == {190}

    def test_fast_reader_runtime_weight(self, m):
        """运行时权重：写链 读 A(i32) → vxm[cvt] → 写 Wt，ρ_w = 1，o_w = 0+1+2 + (6+2)+2 = 13；
        读链 读 Wt → load_w，ρ_r = 16。第 i 个 word：13+1+i+19 − ceil(i/16)，i = 319 时最大 = 332。
        设计 §7.3 的两端点给 max(33, 353−20) = 333，保守 1 拍；tile 语义 353。"""
        b = GraphBuilder("rw", "lpu-tile-v0")
        b.tensor("A", "i32", (320, 320), "input", "W", "row")
        b.tensor("Wt", "i8", (320, 320), "intermediate", "W", "row")
        b.read("a.rd", "A[0:320,0:320]", 320, Out("i32", 4), ["op"])
        b.vxm("cvt", ["cvt"], 320, I8, ["op"])
        b.write("wt.wr", "Wt[0:320,0:320]", 320, ["op"])
        b.read("wt.rd", "Wt[0:320,0:320]", 320, I8, ["op"])
        b.load_w("lw", "mxm.w0", 0, 320, ["op"])
        b.stream("a.rd", "cvt"); b.stream("cvt", "wt.wr"); b.stream("wt.rd", "lw")
        b.mem("wt.wr", "wt.rd")
        g = b.build()
        validate_graph(g, m)
        assert [e.weight for e in build_chains(g, m, mem_overlap="row").edges] == [332]
        assert [e.weight for e in build_chains(g, m).edges] == [353]

    def test_fractional_rates_design_would_be_optimistic(self, m):
        """写侧 vxm 链深 19（ρ_w = 16/19），读侧链深 18（ρ_r = 8/9），9 行；o_w = 81，D_w = D_r = 11。
        设计两端点给 max(81+1+19, 81+1+11+19−11) = 101，此时第 6、7、8 个 word 会在可见之前被读走；
        逐 word 精确值是 102。tile 语义 112。"""
        R9 = 9
        b = GraphBuilder("dv", "lpu-tile-v0")
        b.tensor("X", "i32", (R9, 320), "input", "W", "row")
        b.tensor("S", "i32", (R9, 320), "intermediate", "W", "row")
        b.read("x.rd", f"X[0:{R9},0:320]", R9, I32, ["op"])
        b.vxm("vw", ["add.i32"] * 19, R9, I32, ["op"])
        b.write("s.wr", f"S[0:{R9},0:320]", R9, ["op"])
        b.read("s.rd", f"S[0:{R9},0:320]", R9, I32, ["op"])
        b.vxm("vr", ["add.i32"] * 18, R9, I32, ["op"])
        b.stream("x.rd", "vw"); b.stream("vw", "s.wr"); b.stream("s.rd", "vr")
        b.mem("s.wr", "s.rd")
        g = b.build()
        validate_graph(g, m)
        assert [e.weight for e in build_chains(g, m, mem_overlap="row").edges] == [102]
        assert [e.weight for e in build_chains(g, m).edges] == [112]

    def test_bad_mode(self, m):
        with pytest.raises(LTSError, match="LTS-E-ARG"):
            build_chains(build_mlp(), m, mem_overlap="word")


# ---- row 模式的紧性：w 时每个 word 都安全，w−1 时至少一个 word 被提前读走 ----------------
from fractions import Fraction as F
from math import ceil

from ltsim.graph.load import tile_region


def _assert_row_tight(g, cg, wave=19):
    by = {n.id: n for n in g.nodes}
    mem_edges = [e for e in cg.edges if e.edge[0] == "mem"]
    assert mem_edges
    for e in mem_edges:
        w_id, r_id = e.edge[1], e.edge[2]
        cw, cr = cg.chain(e.src), cg.chain(e.dst)
        _, (rw0, rw1), _ = tile_region(g, by[r_id].tile)
        _, (ww0, ww1), _ = tile_region(g, by[w_id].tile)

        def safe(delta):          # T_w = 0、T_r = delta：读到每个公共 word 时它已写完可见
            return all(delta + cr.offset[r_id] + ceil(F(x - rw0) / cr.rate)
                       >= cw.offset[w_id] + cg.works[w_id].L + ceil(F(x - ww0) / cw.rate) + wave
                       for x in range(max(rw0, ww0), min(rw1, ww1)))
        assert safe(e.weight) and not safe(e.weight - 1), e


def _alias_graph():
    """经 K（row [64,128]）写，经 Kt（K 的 lane 别名 [128,64]）读第 16–47 个 word。"""
    b = GraphBuilder("al", "lpu-tile-v0")
    b.tensor("X", "i8", (64, 128), "input", "W", "row")
    b.tensor("K", "i8", (64, 128), "intermediate", "W", "row")
    b.tensor("Kt", "i8", (128, 64), "intermediate", "W", "lane", alias_of="K")
    b.tensor("O", "i8", (32, 128), "output", "E", "row")
    b.read("x.rd", "X[0:64,0:128]", 64, I8, ["op"])
    b.vxm("v1", ["relu"], 64, I8, ["op"])
    b.write("k.wr", "K[0:64,0:128]", 64, ["op"])
    b.read("kt.rd", "Kt[0:128,16:48]", 32, I8, ["op"])
    b.vxm("v2", ["relu"], 32, I8, ["op"])
    b.write("o.wr", "O[0:32,0:128]", 32, ["op"])
    b.stream("x.rd", "v1"); b.stream("v1", "k.wr"); b.stream("kt.rd", "v2"); b.stream("v2", "o.wr")
    b.mem("k.wr", "kt.rd")
    return b.build()


def _offset_read_graph():
    """写 H[0:128)，读 H[50:128)：读的起始 word 不是 0。"""
    b = GraphBuilder("off", "lpu-tile-v0")
    b.tensor("X", "i8", (128, 320), "input", "W", "row")
    b.tensor("H", "i32", (128, 320), "intermediate", "W", "row")
    b.tensor("Y", "i32", (78, 320), "output", "E", "row")
    b.read("x.rd", "X[0:128,0:320]", 128, I8, ["op"])
    b.acc("a", "mxm.w0", 0, 128, I32, ["op"])
    b.write("h.wr", "H[0:128,0:320]", 128, ["op"])
    b.read("h.rd", "H[50:128,0:320]", 78, I32, ["op"])
    b.vxm("v", ["add.i32"], 78, I32, ["op"])
    b.write("y", "Y[0:78,0:320]", 78, ["op"])
    b.stream("x.rd", "a"); b.stream("a", "h.wr"); b.stream("h.rd", "v"); b.stream("v", "y")
    b.mem("h.wr", "h.rd")
    return b.build()


@pytest.mark.parametrize("make", [
    lambda: build_mlp(),
    lambda: build_gemm(256, 640, 640, tile_m=128),
    lambda: _split_write(two_writes=True),
    lambda: _split_write(read_rows=64, two_writes=False),
    _alias_graph,
    _offset_read_graph,
])
def test_row_mode_weight_is_tight(m, make):
    g = make()
    validate_graph(g, m)
    cg = build_chains(g, m, mem_overlap="row")
    _assert_row_tight(g, cg)
    tile = {(e.src, e.dst, e.edge): e.weight for e in build_chains(g, m).edges}
    assert all(e.weight <= tile[(e.src, e.dst, e.edge)] for e in cg.edges)      # row 不比 tile 更晚
