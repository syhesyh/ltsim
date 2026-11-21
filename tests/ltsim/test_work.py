"""B1 work.py 的测试。期望值来自 backend.md §3.2 公式 + 设计 §3.4 初值参数，
其中 a0.vxm 的 L=60、b.vxm 的 L=32、各 hop 与 backend.md §8.1 手算表一致。"""
import os
from fractions import Fraction as F

import pytest

from ltsim.backend.work import (NodeWork, direction, node_works, out_directions,
                                stream_hops, stream_preds, write_v_in)
from ltsim.errors import LTSError
from ltsim.gen.builder import GraphBuilder, Out
from ltsim.gen.gemm import build_gemm
from ltsim.graph.load import load_graph, validate_graph
from ltsim.graph.tile_graph import Node
from ltsim.machine.model import load_machine

HERE = os.path.dirname(__file__)
FIXTURE = os.path.join(HERE, "fixtures", "mlp_tile_graph.json")
MACHINE = os.path.join(HERE, "..", "..", "ltsim", "machine", "lpu-tile-v0.yaml")


@pytest.fixture(scope="module")
def m():
    return load_machine(MACHINE)


@pytest.fixture(scope="module")
def g():
    return load_graph(FIXTURE)


@pytest.fixture(scope="module")
def w(g, m):
    return node_works(g, m)


# MLP 全部 25 个节点的期望值：(cap, L, skew, category)。按 backend.md §3.2 公式手算，
# 与 §8.1 手算表一致（a*.vxm L=(6+2)×5+(18+2)=60，b.vxm L=4+4+8+8+8=32）。
MLP_WORK = {
    "w1n0.rd": (44, 1, 0, "MEM_W"), "w1n1.rd": (44, 1, 0, "MEM_W"),
    "w2k0.rd": (44, 1, 0, "MEM_E"), "w2k1.rd": (44, 1, 0, "MEM_E"),
    "w1n0.lw": (16, 0, 19, "MXM_W"), "w1n1.lw": (16, 0, 19, "MXM_W"),
    "w2k0.lw": (16, 0, 19, "MXM_E"), "w2k1.lw": (16, 0, 19, "MXM_E"),
    **{f"a{i}.x.rd": (44, 1, 0, "MEM_W") for i in (0, 1)},
    **{f"a{i}.acc": (1, 25, 0, "MXM_W") for i in (0, 1)},
    **{f"a{i}.b1.rd": (11, 1, 0, "MEM_W") for i in (0, 1)},     # f32 V=4 → 44/4
    **{f"a{i}.vxm": (1, 60, 0, "VXM") for i in (0, 1)},
    **{f"a{i}.h.wr": (44, 1, 19, "MEM_E") for i in (0, 1)},     # H 在东半球，V_in=1
    "b.h0.rd": (44, 1, 0, "MEM_E"), "b.h1.rd": (44, 1, 0, "MEM_E"),
    "b.acc0": (1, 25, 0, "MXM_E"), "b.acc1": (1, 25, 0, "MXM_E"),
    "b.b2.rd": (11, 1, 0, "MEM_W"),                              # i32 V=4
    "b.vxm": (1, 32, 0, "VXM"),
    "b.y.wr": (44, 1, 19, "MEM_W"),
}

# MLP 全部 18 条 stream 边：(hop, 源节点的流方向)
MLP_STREAMS = {
    ("w1n0.rd", "w1n0.lw"): (8, "E2W"), ("w1n1.rd", "w1n1.lw"): (8, "E2W"),      # MEM_W > MXM_W
    ("w2k0.rd", "w2k0.lw"): (9, "W2E"), ("w2k1.rd", "w2k1.lw"): (9, "W2E"),      # MEM_E > MXM_E
    **{e: v for i in (0, 1) for e, v in {
        (f"a{i}.x.rd", f"a{i}.acc"): (8, "E2W"),
        (f"a{i}.acc", f"a{i}.vxm"): (21, "W2E"),                                 # MXM_W > VXM
        (f"a{i}.b1.rd", f"a{i}.vxm"): (2, "W2E"),                                # MEM_W > VXM
        (f"a{i}.vxm", f"a{i}.h.wr"): (2, "W2E"),                                 # VXM > MEM_E
    }.items()},
    ("b.h0.rd", "b.acc0"): (9, "W2E"), ("b.h1.rd", "b.acc1"): (9, "W2E"),        # MEM_E > MXM_E
    ("b.acc0", "b.vxm"): (22, "E2W"), ("b.acc1", "b.vxm"): (22, "E2W"),          # MXM_E > VXM
    ("b.b2.rd", "b.vxm"): (2, "W2E"),
    ("b.vxm", "b.y.wr"): (2, "E2W"),                                             # VXM > MEM_W
}


class TestMlpWork:
    def test_table_is_complete(self, g):
        assert set(MLP_WORK) == {n.id for n in g.nodes}          # 25 个
        assert set(MLP_STREAMS) == {(s, d) for s, d, _ in g.edges_stream}   # 18 条

    @pytest.mark.parametrize("nid", sorted(MLP_WORK))
    def test_node(self, w, nid):
        cap, L, skew, cat = MLP_WORK[nid]
        assert w[nid] == NodeWork(F(cap), L, skew, cat)

    def test_cap_is_fraction(self, w):
        assert all(isinstance(x.cap, F) for x in w.values())


class TestMlpHops:
    def test_hops(self, g, w, m):
        h = stream_hops(g, w, m)
        assert h == {e: hop for e, (hop, _d) in MLP_STREAMS.items()}

    def test_directions(self, g, w):
        d = out_directions(g, w)
        assert d == {s: dr for (s, _t), (_h, dr) in MLP_STREAMS.items()}
        # 没有流消费者的节点（写节点、装权重）不出现
        assert "a0.h.wr" not in d and "w1n0.lw" not in d

    def test_first_layer_stream_counts(self, g, w):
        """backend.md §8.1：c4 在 W2E 方向 4+4+1=9 条流，E2W 方向 1 条。"""
        d = out_directions(g, w)
        by = {n.id: n for n in g.nodes}
        c4 = ["a0.x.rd", "a0.acc", "a0.b1.rd", "a0.vxm"]
        assert sum(by[i].out.V for i in c4 if d[i] == "W2E") == 9
        assert sum(by[i].out.V for i in c4 if d[i] == "E2W") == 1


class TestHelpers:
    def test_direction(self):
        assert direction("MEM_W", "MXM_W") == "E2W"
        assert direction("MXM_W", "VXM") == "W2E"
        assert direction("VXM", "MEM_W") == "E2W"
        with pytest.raises(LTSError, match="LTS-E-GRAPH-HOP"):
            direction("VXM", "VXM")

    def test_deep_vxm_chain_rate_halves(self, m):
        """链深 32 > 16 个 ALU：每拍推进半行（设计 §3.6）。"""
        b = GraphBuilder("t", "lpu-tile-v0")
        b.tensor("A", "i32", (8, 320), "input", "W", "row")
        b.tensor("B", "i32", (8, 320), "output", "W", "row")
        b.read("r", "A[0:8,0:320]", 8, Out("i32", 4), ["op"])
        b.vxm("v", ["add.i32"] * 32, 8, Out("i32", 4), ["op"])
        b.write("wr", "B[0:8,0:320]", 8, ["op"])
        b.stream("r", "v"); b.stream("v", "wr")
        w = node_works(b.build(), m)
        assert w["v"].cap == F(1, 2)
        assert w["v"].L == 32 * (2 + 2)

    def test_alias_hemisphere(self, m):
        """tile 指向 transpose 别名时，半球按源张量取（B0 已保证二者一致）。"""
        b = GraphBuilder("t", "lpu-tile-v0")
        b.tensor("K", "i8", (8, 64), "input", "E", "row")
        b.tensor("Kt", "i8", (64, 8), "intermediate", "E", "lane", alias_of="K")   # B0 要求半球一致
        b.tensor("O", "i8", (64, 8), "output", "W", "row")
        b.read("r", "Kt[0:64,0:8]", 64, Out("i8", 1), ["op"])
        b.vxm("v", ["relu"], 64, Out("i8", 1), ["op"])
        b.write("wr", "O[0:64,0:8]", 64, ["op"])
        b.stream("r", "v"); b.stream("v", "wr")
        assert node_works(b.build(), m)["r"].category == "MEM_E"


class TestErrors:
    def _rw(self, write_hemi):
        b = GraphBuilder("t", "lpu-tile-v0")
        b.tensor("A", "i8", (8, 320), "input", "W", "row")
        b.tensor("B", "i8", (8, 320), "output", write_hemi, "row")
        b.read("r", "A[0:8,0:320]", 8, Out("i8", 1), ["op"])
        b.write("wr", "B[0:8,0:320]", 8, ["op"])
        b.stream("r", "wr")
        return b

    def test_same_category_stream(self, m):
        g = self._rw("W").build()           # MEM_W → MEM_W
        w = node_works(g, m)
        with pytest.raises(LTSError, match="LTS-E-GRAPH-HOP"):
            stream_hops(g, w, m)

    def test_two_directions(self, m):
        b = GraphBuilder("t", "lpu-tile-v0")
        b.tensor("A", "i8", (8, 320), "input", "W", "row")
        b.tensor("BW", "i8", (8, 320), "output", "W", "row")
        b.tensor("BE", "i8", (8, 320), "output", "E", "row")
        b.read("r", "A[0:8,0:320]", 8, Out("i8", 1), ["op"])
        b.vxm("v", ["relu"], 8, Out("i8", 1), ["op"])
        b.write("ww", "BW[0:8,0:320]", 8, ["op"])
        b.write("we", "BE[0:8,0:320]", 8, ["op"])
        b.stream("r", "v"); b.stream("v", "ww"); b.stream("v", "we")
        g = b.build()
        with pytest.raises(LTSError, match="LTS-E-GRAPH-HOP"):
            out_directions(g, node_works(g, m))

    def test_write_without_stream_pred(self, m):
        b = GraphBuilder("t", "lpu-tile-v0")
        b.tensor("B", "i8", (8, 320), "output", "W", "row")
        b.write("wr", "B[0:8,0:320]", 8, ["op"])
        with pytest.raises(LTSError, match="LTS-E-GRAPH-STREAM"):
            node_works(b.build(), m)

    def test_unknown_vxm_op(self, m):
        b = GraphBuilder("t", "lpu-tile-v0")
        b.tensor("A", "i8", (8, 320), "input", "W", "row")
        b.read("r", "A[0:8,0:320]", 8, Out("i8", 1), ["op"])
        b.vxm("v", ["nosuch"], 8, Out("i8", 1), ["op"])
        b.stream("r", "v")
        with pytest.raises(LTSError, match="LTS-E-MACHINE"):
            node_works(b.build(), m)

    def test_unsupported_kind(self, m):
        b = GraphBuilder("t", "lpu-tile-v0")
        b._add(Node(id="s", kind="sxm.transpose", rows=1, origin=("op",), unit="sxm.w"))
        with pytest.raises(LTSError, match="LTS-E-GRAPH-KIND"):
            node_works(b.build(), m)


def _mxm_graph(dtype, V):
    """读权重 → load_w；读 X → acc → 写回。权重和 X 都是 dtype。"""
    b = GraphBuilder("t", "lpu-tile-v0")
    b.tensor("W", dtype, (320, 320), "param", "W", "row")
    b.tensor("X", dtype, (64, 320), "input", "W", "row")
    b.tensor("Y", "i32", (64, 320), "output", "W", "row")
    b.read("w.rd", "W[0:320,0:320]", 320, Out(dtype, V), ["mm"])
    b.load_w("w.lw", "mxm.w0", 0, 320, ["mm"])
    b.read("x.rd", "X[0:64,0:320]", 64, Out(dtype, V), ["mm"])
    b.acc("acc", "mxm.w0", 0, 64, Out("i32", 4), ["mm"])
    b.write("y.wr", "Y[0:64,0:320]", 64, ["mm"])
    b.stream("w.rd", "w.lw"); b.stream("x.rd", "acc"); b.stream("acc", "y.wr")
    b.state("w.lw", "acc")
    return b.build()


class TestPrecision:
    @pytest.mark.parametrize("dtype,V,lw_cap,acc_L", [
        ("i8", 1, 16, 25),
        ("f16", 2, 8, 50),      # 装载减半、延迟翻倍
        ("bf16", 2, 8, 50),
    ])
    def test_mxm_by_dtype(self, m, dtype, V, lw_cap, acc_L):
        w = node_works(_mxm_graph(dtype, V), m)
        assert w["w.lw"].cap == F(lw_cap)
        assert w["acc"].L == acc_L
        assert w["acc"].cap == F(1)                  # acc_ii 不随精度变化
        assert w["w.lw"].L == 0 and w["w.lw"].skew == 19

    def test_unsupported_dtype(self, m):
        # 流入 MXM 的是 i32：表里没有，报 LTS-E-UNIT-OP
        with pytest.raises(LTSError, match="LTS-E-UNIT-OP"):
            node_works(_mxm_graph("i32", 4), m)

    def test_acc_dtype_comes_from_streamed_operand(self, m):
        """acc 自己输出 i32，但延迟按流入操作数（f16）查。"""
        g = _mxm_graph("f16", 2)
        assert g.node("acc").out.dtype == "i32"
        assert node_works(g, m)["acc"].L == 50


class TestWriteVIn:
    def test_mlp(self, g):
        v = write_v_in(g)
        assert set(v) == {"a0.h.wr", "a1.h.wr", "b.y.wr"}
        assert v["a0.h.wr"] == 1                     # 前驱 vxm 输出 i8

    def test_gemm_partial(self):
        v = write_v_in(build_gemm(256, 640, 640, tile_m=128))
        assert v["m0n0k0.wr"] == 4                   # 前驱 acc 输出 i32
        assert v["m0n0.y.wr"] == 4                   # 前驱 vxm 求和输出 i32

    def test_stream_preds(self, g):
        p = stream_preds(g)
        assert p["a0.vxm"] == ["a0.acc", "a0.b1.rd"]
        assert "a0.x.rd" not in p


def test_gemm_graph_all_hops_defined(m):
    """GEMM 生成器的图（含 acc→写 MEM、K 合并 vxm）在 B1 全部能算，hop 表无缺项。"""
    g = build_gemm(256, 640, 640, tile_m=128)
    w = node_works(g, m)
    h = stream_hops(g, w, m)
    out_directions(g, w)
    assert len(h) == len(g.edges_stream)
    assert h[("m0n0k0.acc", "m0n0k0.wr")] == 8          # MXM_W > MEM_W
    assert w["m0n0.sum"].L == 2 + 2                      # 一个 add.i32


# ---- A.4：补覆盖缺口 --------------------------------------------------------

def _chain(tensors, nodes, streams):
    """小工具：tensors = [(name, dtype, shape, hemi, alias_of)]，nodes = 已构造好的 builder 调用。"""
    b = GraphBuilder("t", "lpu-tile-v0")
    for name, dt, shape, hemi, alias in tensors:
        b.tensor(name, dt, shape, "intermediate", hemi, "row", alias_of=alias)
    for fn in nodes:
        fn(b)
    for s, d in streams:
        b.stream(s, d)
    return b.build()


class TestMoreErrors:
    def test_write_with_two_stream_preds_builder(self, m):
        g = _chain([("A", "i8", (8, 320), "W", None), ("B", "i8", (8, 320), "E", None)],
                   [lambda b: b.read("r1", "A[0:8,0:320]", 8, Out("i8", 1), ["op"]),
                    lambda b: b.read("r2", "A[0:8,0:320]", 8, Out("i8", 1), ["op"]),
                    lambda b: b.vxm("v", ["relu"], 8, Out("i8", 1), ["op"]),
                    lambda b: b.write("wr", "B[0:8,0:320]", 8, ["op"])],
                   [("r1", "v"), ("v", "wr"), ("r2", "wr")])
        with pytest.raises(LTSError, match="LTS-E-GRAPH-STREAM"):
            node_works(g, m)

    def test_write_with_two_stream_preds_json(self, tmp_path):
        """经 JSON 加载时，B0 就拦下。"""
        import json
        d = json.load(open(FIXTURE))
        d["edges"]["stream"].append(["a0.b1.rd", "a0.h.wr"])
        p = tmp_path / "g.json"
        p.write_text(json.dumps(d))
        with pytest.raises(LTSError, match="LTS-E-GRAPH-STREAM"):
            load_graph(str(p))

    def test_alias_dangling_json(self, tmp_path):
        import json
        d = json.load(open(FIXTURE))
        d["tensors"]["H"]["alias_of"] = "NOPE"
        p = tmp_path / "g.json"
        p.write_text(json.dumps(d))
        with pytest.raises(LTSError, match="LTS-E-GRAPH-REF"):
            load_graph(str(p))

    def test_alias_cycle(self, m):
        g = _chain([("A", "i8", (8, 320), "W", "B"), ("B", "i8", (8, 320), "W", "A"),
                    ("C", "i8", (8, 320), "E", None)],
                   [lambda b: b.read("r", "A[0:8,0:320]", 8, Out("i8", 1), ["op"]),
                    lambda b: b.vxm("v", ["relu"], 8, Out("i8", 1), ["op"])],
                   [("r", "v")])
        with pytest.raises(LTSError, match="LTS-E-GRAPH-REF"):
            node_works(g, m)

    def test_tile_unknown_tensor(self, m):
        g = _chain([], [lambda b: b.read("r", "GHOST[0:8,0:320]", 8, Out("i8", 1), ["op"])], [])
        with pytest.raises(LTSError, match="LTS-E-GRAPH-REF"):
            node_works(g, m)

    def test_missing_tile(self, m):
        g = _chain([], [lambda b: b._add(Node(id="r", kind="mem.read", rows=8, origin=("op",),
                                               out=Out("i8", 1)))], [])
        with pytest.raises(LTSError, match="LTS-E-GRAPH-TILE"):
            node_works(g, m)

    def test_read_without_out(self, m):
        g = _chain([("A", "i8", (8, 320), "W", None)],
                   [lambda b: b.read("r", "A[0:8,0:320]", 8, None, ["op"])], [])
        with pytest.raises(LTSError, match="LTS-E-GRAPH-STREAM"):
            node_works(g, m)

    def test_vxm_empty_ops(self, m):
        g = _chain([], [lambda b: b.vxm("v", [], 8, Out("i8", 1), ["op"])], [])
        with pytest.raises(LTSError, match="LTS-E-GRAPH-KIND"):
            node_works(g, m)

    def test_unknown_kind(self, m):
        g = _chain([], [lambda b: b._add(Node(id="x", kind="foo.bar", rows=1, origin=("op",)))], [])
        with pytest.raises(LTSError, match="LTS-E-GRAPH-KIND"):
            node_works(g, m)


class TestMoreFormulas:
    def test_hop_cycles_scales(self, g, m):
        """hop = hops 表 × srf.hop_cycles；把 hop_cycles 改成 2，hop 全部翻倍。"""
        import copy
        from ltsim.machine.model import Machine
        raw = copy.deepcopy(dict(m.raw))
        raw["srf"]["hop_cycles"] = 2
        m2 = Machine(raw)
        h = stream_hops(g, node_works(g, m2), m2)
        assert h == {e: 2 * hop for e, (hop, _d) in MLP_STREAMS.items()}

    def test_non_integer_caps(self, m):
        """链深 17 > 16 个 ALU → 16/17 行/拍。tile 为合法的 320 lane（i8，V=1）。"""
        g = _chain([("A", "i8", (8, 320), "W", None), ("B", "i8", (8, 320), "E", None)],
                   [lambda b: b.read("r", "A[0:8,0:320]", 8, Out("i8", 1), ["op"]),
                    lambda b: b.vxm("v", ["relu"] * 17, 8, Out("i8", 1), ["op"]),
                    lambda b: b.write("wr", "B[0:8,0:320]", 8, ["op"])],
                   [("r", "v"), ("v", "wr")])
        validate_graph(g, m)                     # 先过 B0 的 lane 宽检查
        w = node_works(g, m)
        assert w["v"].cap == F(16, 17)
        assert w["r"].cap == F(44) and w["wr"].cap == F(44)

    def test_directions_match_mxm_in_out_dir(self, m):
        """按 x 序推出的方向，与 yaml 中每个 MXM 实例的 in_dir / out_dir 一致：
        数据从本侧 MEM 流入 MXM 走 in_dir，从 MXM 流向 VXM 走 out_dir。"""
        for inst in m.raw["mxm"]["instances"]:
            side = inst["side"]
            assert direction(f"MEM_{side}", f"MXM_{side}") == inst["in_dir"], inst["id"]
            assert direction(f"MXM_{side}", "VXM") == inst["out_dir"], inst["id"]
