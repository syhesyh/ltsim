"""用 GraphBuilder 复刻设计文档 §6.3 的两层 MLP tile graph（金标准）。

128×320×640，fc1 绑西侧 {w0,w1} n_split，fc2 绑东侧 {e0,e1} k_split。
25 节点 / 7 链 / 24 边（18 stream + 4 state + 2 mem）。
"""
from __future__ import annotations

from ..graph.tile_graph import Out, TileGraph
from .builder import GraphBuilder, tile_str

I8 = Out("i8", 1)
I32 = Out("i32", 4)
F32 = Out("f32", 4)


def build_mlp() -> TileGraph:
    b = GraphBuilder("mlp_128x320x640", "lpu-tile-v0", source="mlp.yaml")   # 与设计 §6.3 一致

    # ---- 张量 ----
    b.tensor("X",  "i8",  (128, 320), "input",        "W", "row")
    b.tensor("W1", "i8",  (320, 640), "param",        "W", "row")
    b.tensor("B1", "f32", (640,),     "param",        "W", "row")
    b.tensor("H",  "i8",  (128, 640), "intermediate", "E", "row")
    b.tensor("W2", "i8",  (640, 320), "param",        "E", "row")
    b.tensor("B2", "i32", (320,),     "param",        "W", "row")
    b.tensor("Y",  "i8",  (128, 320), "output",       "W", "row")

    # ---- 权重链（西：W1 两个 N 块；东：W2 两个 K 块）----
    b.read("w1n0.rd", tile_str("W1", 0, 320, 0, 320),   320, I8, ["fc1"])
    b.load_w("w1n0.lw", "mxm.w0", 0, 320, ["fc1"])
    b.stream("w1n0.rd", "w1n0.lw")

    b.read("w1n1.rd", tile_str("W1", 0, 320, 320, 640), 320, I8, ["fc1"])
    b.load_w("w1n1.lw", "mxm.w1", 0, 320, ["fc1"])
    b.stream("w1n1.rd", "w1n1.lw")

    b.read("w2k0.rd", tile_str("W2", 0, 320, 0, 320),   320, I8, ["fc2"])
    b.load_w("w2k0.lw", "mxm.e0", 0, 320, ["fc2"])
    b.stream("w2k0.rd", "w2k0.lw")

    b.read("w2k1.rd", tile_str("W2", 320, 640, 0, 320), 320, I8, ["fc2"])
    b.load_w("w2k1.lw", "mxm.e1", 0, 320, ["fc2"])
    b.stream("w2k1.rd", "w2k1.lw")

    # ---- 第一层左半（a0：fc1 的 N 块 0 → H[:, 0:320]）----
    b.read("a0.x.rd",  tile_str("X", 0, 128, 0, 320), 128, I8, ["fc1"])
    b.acc("a0.acc",   "mxm.w0", 0, 128, I32, ["fc1"])
    b.read("a0.b1.rd", tile_str("B1", 0, 320), 128, F32, ["b1"], repeat=128)
    b.vxm("a0.vxm", ["cvt", "mul.f32", "add.f32", "gelu.f32", "mul.f32", "cvt"], 128, I8,
          ["dq1", "b1", "act", "q1"])
    b.write("a0.h.wr", tile_str("H", 0, 128, 0, 320), 128, ["q1"])
    b.stream("a0.x.rd", "a0.acc")
    b.stream("a0.acc", "a0.vxm")
    b.stream("a0.b1.rd", "a0.vxm")
    b.stream("a0.vxm", "a0.h.wr")

    # ---- 第一层右半（a1：fc1 的 N 块 1 → H[:, 320:640]）----
    b.read("a1.x.rd",  tile_str("X", 0, 128, 0, 320), 128, I8, ["fc1"])
    b.acc("a1.acc",   "mxm.w1", 0, 128, I32, ["fc1"])
    b.read("a1.b1.rd", tile_str("B1", 320, 640), 128, F32, ["b1"], repeat=128)
    b.vxm("a1.vxm", ["cvt", "mul.f32", "add.f32", "gelu.f32", "mul.f32", "cvt"], 128, I8,
          ["dq1", "b1", "act", "q1"])
    b.write("a1.h.wr", tile_str("H", 0, 128, 320, 640), 128, ["q1"])
    b.stream("a1.x.rd", "a1.acc")
    b.stream("a1.acc", "a1.vxm")
    b.stream("a1.b1.rd", "a1.vxm")
    b.stream("a1.vxm", "a1.h.wr")

    # ---- 第二层（b：fc2 两个 K 块 K 合并 + b2 + requant → Y）----
    b.read("b.h0.rd", tile_str("H", 0, 128, 0, 320),   128, I8, ["fc2"])
    b.read("b.h1.rd", tile_str("H", 0, 128, 320, 640), 128, I8, ["fc2"])
    b.acc("b.acc0", "mxm.e0", 0, 128, I32, ["fc2"])
    b.acc("b.acc1", "mxm.e1", 0, 128, I32, ["fc2"])
    b.read("b.b2.rd", tile_str("B2", 0, 320), 128, I32, ["b2"], repeat=128)
    b.vxm("b.vxm", ["add.i32", "add.i32", "cvt", "mul.f32", "cvt"], 128, I8, ["fc2", "b2", "y"])
    b.write("b.y.wr", tile_str("Y", 0, 128, 0, 320), 128, ["y"])
    b.stream("b.h0.rd", "b.acc0")
    b.stream("b.h1.rd", "b.acc1")
    b.stream("b.acc0", "b.vxm")
    b.stream("b.acc1", "b.vxm")
    b.stream("b.b2.rd", "b.vxm")
    b.stream("b.vxm", "b.y.wr")

    # ---- MXM 顺序边：每个 plane 只有一个权重块，所以只产生 state 边 ----
    b.link_plane([("w1n0.lw", ["a0.acc"])], banks="alternate")
    b.link_plane([("w1n1.lw", ["a1.acc"])], banks="alternate")
    b.link_plane([("w2k0.lw", ["b.acc0"])], banks="alternate")
    b.link_plane([("w2k1.lw", ["b.acc1"])], banks="alternate")

    # ---- mem 边：第一层写 H → 第二层读 H ----
    b.mem("a0.h.wr", "b.h0.rd")
    b.mem("a1.h.wr", "b.h1.rd")

    return b.build()
