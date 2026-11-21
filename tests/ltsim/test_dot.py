"""Graphviz 导出（ltsim/graph/dot.py）。"""
import json
import os

from ltsim.backend.chains import build_chains
from ltsim.cli import main
from ltsim.gen.gemm import build_gemm
from ltsim.gen.mlp import build_mlp
from ltsim.graph.dot import to_dot
from ltsim.machine.model import default_machine

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "mlp_tile_graph.json")


def test_mlp_structure():
    g = build_mlp()
    cg = build_chains(g, default_machine())
    d = to_dot(g, cg.chain_of)
    assert d.count("subgraph \"cluster_") == 7                       # 每条链一个框
    assert d.count("[label=") == 25                                  # 25 个节点
    assert d.count(" -> ") == 24                                     # 18 stream + 4 state + 2 mem
    assert d.count('label="state"') == 4 and d.count('label="mem"') == 2
    assert '"a0.vxm" [label="a0.vxm\\nvxm.chain\\nrows=128\\n[cvt, mul.f32, add.f32, gelu.f32, mul.f32, cvt]"' in d
    assert d == to_dot(g, cg.chain_of)                               # 确定性


def test_without_chains_and_order_labels():
    g = build_gemm(256, 640, 640, tile_m=128, banks="single")
    d = to_dot(g)
    assert "cluster_" not in d
    assert '"m0n0k0.acc" -> "w.n0k1.lw"' in d and 'label="lw_after_iw (start-1)"' in d
    assert 'label="block_switch (issue_end mxm.block_switch_gap)"' in d
    assert d.count("bank_war") == 2


def test_quotes_are_escaped():
    from ltsim.gen.builder import GraphBuilder, Out
    b = GraphBuilder("t", "lpu-tile-v0")
    b.tensor("A", "i8", (8, 320), "input", "W", "row")
    b.read('r"1', "A[0:8,0:320]", 8, Out("i8", 1), ["op"])
    assert '"r\\"1" [label="r\\"1\\n' in to_dot(b.build())


def test_cli_run_dot(tmp_path, capsys):
    assert main(["run", FIXTURE, "--out", str(tmp_path / "r"), "--dot"]) == 0
    assert "graph.dot" in capsys.readouterr().out
    assert (tmp_path / "r" / "graph.dot").read_text().startswith('digraph "mlp_tile_graph.json" {')
    assert main(["run", FIXTURE, "--out", str(tmp_path / "r2")]) == 0
    assert not (tmp_path / "r2" / "graph.dot").exists()              # 默认不写
