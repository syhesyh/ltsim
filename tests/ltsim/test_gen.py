"""gen/ 的测试：builder 校验 + MLP 金标准与 fixture 逐字段一致。"""
import json
import os

import pytest

from ltsim.errors import LTSError
from ltsim.gen.builder import GraphBuilder, Out, _to_dict, bank_for, plane_for, tile_str
from ltsim.gen.mlp import build_mlp
from ltsim.graph.load import load_graph

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "mlp_tile_graph.json")


class TestHelpers:
    def test_tile_str_rank2(self):
        assert tile_str("W1", 0, 320, 0, 320) == "W1[0:320,0:320]"

    def test_tile_str_rank1(self):
        assert tile_str("B1", 0, 320) == "B1[0:320]"

    def test_bank_for(self):
        assert bank_for(0, "alternate") == 0
        assert bank_for(3, "alternate") == 1
        assert bank_for(5, "single") == 0

    def test_plane_for(self):
        planes = ["mxm.w0", "mxm.w1"]
        # 块 (n=1, k=0)，在 (n,k) 字典序中的序号 j=2
        assert plane_for(planes, "n_split", n=1, k=0, j=2) == "mxm.w1"
        assert plane_for(planes, "k_split", n=1, k=0, j=2) == "mxm.w0"
        assert plane_for(planes, "round_robin", n=1, k=0, j=2) == "mxm.w0"
        with pytest.raises(LTSError, match="LTS-E-GEN"):
            plane_for(planes, "bogus", 0, 0, 0)


class TestBuilderValidation:
    def test_dup_node_id(self):
        b = GraphBuilder("t", "m")
        b.read("a", "X[0:1,0:1]", 1, Out("i8", 1), ["op"])
        with pytest.raises(LTSError, match="LTS-E-GRAPH-DUP"):
            b.read("a", "X[0:1,0:1]", 1, Out("i8", 1), ["op"])

    def test_stream_src_no_out(self):
        b2 = GraphBuilder("t2", "m")
        b2.write("w", "Y[0:1,0:1]", 1, ["op"])
        b2.read("r", "Y[0:1,0:1]", 1, Out("i8", 1), ["op"])
        b2.stream("w", "r")                        # w 无 out，当 stream src → 报错
        with pytest.raises(LTSError, match="LTS-E-GRAPH-STREAM"):
            b2.build()

    def test_edge_endpoint_missing(self):
        b = GraphBuilder("t", "m")
        b.read("r", "X[0:1,0:1]", 1, Out("i8", 1), ["op"])
        b.stream("r", "ghost")
        with pytest.raises(LTSError, match="LTS-E-GRAPH-REF"):
            b.build()


class TestMlpGolden:
    def test_matches_fixture_exactly(self):
        """严格比较：所有顶层字段（含 source）、节点与边数组的顺序都必须一致。
        dict 的键顺序不影响相等，列表顺序影响。"""
        assert _to_dict(build_mlp()) == json.load(open(FIXTURE))

    def test_seven_chains(self):
        """设计 §6.3：按 stream 边求连通分量应得到 7 条链。"""
        g = build_mlp()
        parent = {n.id: n.id for n in g.nodes}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for s, d, _ in g.edges_stream:
            parent[find(s)] = find(d)
        sizes = sorted(sum(1 for n in g.nodes if find(n.id) == r) for r in {find(n.id) for n in g.nodes})
        assert sizes == [2, 2, 2, 2, 5, 5, 7]

    def test_counts(self):
        g = build_mlp()
        assert len(g.nodes) == 25
        assert len(g.edges_stream) == 18
        assert len(g.edges_state) == 4
        assert len(g.edges_mem) == 2

    def test_deterministic(self):
        a = json.dumps(_to_dict(build_mlp()), sort_keys=True)
        b = json.dumps(_to_dict(build_mlp()), sort_keys=True)
        assert a == b

    def test_roundtrip_loads(self, tmp_path):
        p = tmp_path / "mlp.json"
        p.write_text(json.dumps(_to_dict(build_mlp()), ensure_ascii=False, indent=2))
        g2 = load_graph(str(p))
        assert len(g2.nodes) == 25
        assert g2.node("a0.acc").unit == "mxm.w0"
