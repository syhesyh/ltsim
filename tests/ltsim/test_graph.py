"""graph/load.py 的单元测试：结构校验 + lag 解析。"""
import json
import os

import pytest

from ltsim.errors import LTSError
from ltsim.graph.load import load_graph, resolve_lags, validate_units
from ltsim.machine.model import load_machine

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "mlp_tile_graph.json")
MACHINE = os.path.join(os.path.dirname(__file__), "..", "..", "ltsim", "machine", "lpu-tile-v0.yaml")


@pytest.fixture(scope="module")
def g():
    return load_graph(FIXTURE)


@pytest.fixture(scope="module")
def m():
    return load_machine(MACHINE)


class TestLoadGolden:
    def test_counts(self, g):
        assert len(g.nodes) == 25
        assert len(g.edges_stream) == 18
        assert len(g.edges_state) == 4
        assert len(g.edges_mem) == 2
        assert len(g.edges_order) == 0

    def test_tensor_fields(self, g):
        assert g.tensors["X"].hemisphere == "W"
        assert g.tensors["H"].hemisphere == "E"
        assert g.tensors["B1"].layout == "row"
        assert g.tensors["H"].role == "intermediate"

    def test_node_fields(self, g):
        n = g.node("a0.acc")
        assert n.kind == "mxm.acc"
        assert n.unit == "mxm.w0"
        assert n.bank == 0
        assert n.rows == 128
        assert n.out.V == 4
        assert n.out.dtype == "i32"

        v = g.node("a0.vxm")
        assert v.ops == ("cvt", "mul.f32", "add.f32", "gelu.f32", "mul.f32", "cvt")
        assert v.origin == ("dq1", "b1", "act", "q1")

        b = g.node("a0.b1.rd")
        assert b.repeat == 128

    def test_stream_r0_default(self, g):
        # 所有 stream 边 r0 默认 0
        assert all(r0 == 0 for (_s, _d, r0) in g.edges_stream)


class TestValidateUnits:
    def test_golden_units_ok(self, g, m):
        validate_units(g, m)  # 不抛

    def test_bad_unit(self, g, m, tmp_path):
        d = json.load(open(FIXTURE))
        # load_w 与对应的 acc 一起改，state 边保持一致，才能测到 unit 名单检查
        for n in d["nodes"]:
            if n["id"] in ("w1n0.lw", "a0.acc"):
                n["unit"] = "mxm.nope"
        p = tmp_path / "bad.json"
        p.write_text(json.dumps(d))
        g2 = load_graph(str(p))
        with pytest.raises(LTSError, match="LTS-E-GRAPH-UNIT"):
            validate_units(g2, m)


class TestStructureErrors:
    def _write(self, tmp_path, mutate):
        d = json.load(open(FIXTURE))
        mutate(d)
        p = tmp_path / "g.json"
        p.write_text(json.dumps(d))
        return str(p)

    def test_dup_node_id(self, tmp_path):
        def mut(d):
            d["nodes"].append(dict(d["nodes"][0]))
        with pytest.raises(LTSError, match="LTS-E-GRAPH-DUP"):
            load_graph(self._write(tmp_path, mut))

    def test_edge_unknown_node(self, tmp_path):
        def mut(d):
            d["edges"]["state"] = [["w1n0.lw", "ghost"]]
        with pytest.raises(LTSError, match="LTS-E-GRAPH-REF"):
            load_graph(self._write(tmp_path, mut))

    def test_stream_src_no_out(self, tmp_path):
        def mut(d):
            # a0.h.wr 是 mem.write 无 out，让它当 stream 源
            d["edges"]["stream"] = [["a0.h.wr", "b.h0.rd"]]
        with pytest.raises(LTSError, match="LTS-E-GRAPH-STREAM"):
            load_graph(self._write(tmp_path, mut))

    def test_bad_schema(self, tmp_path):
        def mut(d):
            d["schema"] = "wrong"
        with pytest.raises(LTSError, match="LTS-E-GRAPH-PARSE"):
            load_graph(self._write(tmp_path, mut))


class TestResolveLags:
    def test_defaults(self, g, m):
        rg = resolve_lags(g, m)
        kinds = {(e[0], e[3]) for e in rg.non_stream_edges}
        # state → issue_end, mem → done
        assert ("state", "issue_end") in kinds
        assert ("mem", "done") in kinds
        # lag 数值：state 用 lw_to_acc_gap=1，mem 用 raw_gap=0
        lags = {e[0]: e[4] for e in rg.non_stream_edges}
        assert lags["state"] == 1
        assert lags["mem"] == 0
        # 4 state + 2 mem = 6 条非 stream 边
        assert len(rg.non_stream_edges) == 6

    def test_order_lag_string_resolved(self, tmp_path, m):
        d = json.load(open(FIXTURE))
        d["edges"]["order"] = [
            {"src": "a0.acc", "dst": "a1.acc", "ref": "issue_end", "lag": "mxm.block_switch_gap", "reason": "block_switch"}
        ]
        p = tmp_path / "g.json"
        p.write_text(json.dumps(d))
        g2 = load_graph(str(p))
        rg = resolve_lags(g2, m)
        order = [e for e in rg.non_stream_edges if e[0] == "order"]
        assert order == [("order", "a0.acc", "a1.acc", "issue_end", 5, "block_switch")]


# ---- B0 lane 宽检查 -----------------------------------------------------------
from ltsim.gen.builder import GraphBuilder, Out
from ltsim.gen.gemm import build_gemm
from ltsim.graph.load import parse_tile, tile_lanes, validate_graph, validate_lane_width


def _one_read(tensor, tile, out=Out("i8", 1), alias_src=None):
    """只有一个 mem.read 的图。tensor = (name, dtype, shape, layout)。"""
    b = GraphBuilder("t", "lpu-tile-v0")
    if alias_src is not None:
        b.tensor(*alias_src)
    name, dt, shape, layout = tensor
    b.tensor(name, dt, shape, "input", "W", layout,
             alias_of=alias_src[0] if alias_src else None)
    b.read("r", tile, 1, out, ["op"])
    return b.build()


class TestParseTile:
    def test_rank2(self):
        assert parse_tile("H[0:128,0:320]") == ("H", ((0, 128), (0, 320)))

    def test_rank1(self):
        assert parse_tile("B1[320:640]") == ("B1", ((320, 640),))

    def test_spaces_ok(self):
        assert parse_tile("W1[0:320, 0:320]") == ("W1", ((0, 320), (0, 320)))

    @pytest.mark.parametrize("bad", ["H", "[0:1]", "H[0:128,0:320", "H[a:b]", "H[0-1]"])
    def test_malformed(self, bad):
        with pytest.raises(LTSError, match="LTS-E-GRAPH-TILE"):
            parse_tile(bad)


class TestTileLanes:
    def test_row_counts_second_dim(self):
        g = _one_read(("X", "i8", (128, 640), "row"), "X[0:128,0:320]")
        assert tile_lanes(g, "X[0:128,0:320]") == 320

    def test_lane_counts_first_dim(self):
        """lane 布局 X[R, C]：C 个 word、每个 word R 个 lane。"""
        g = _one_read(("S", "f32", (197, 197), "lane"), "S[0:197,0:197]")
        assert tile_lanes(g, "S[0:197,0:197]") == 197

    def test_alias_uses_own_layout(self):
        """Kt 是 K（row，[197,64]）的 lane 版本 [64,197]：按 Kt 自己的 lane 布局算，lane = 64。
        与设计 §4.5 的"mem.read Kt（197 行 × 64 lane）"一致。"""
        g = _one_read(("Kt", "i8", (64, 197), "lane"), "Kt[0:64,0:197]",
                      alias_src=("K", "i8", (197, 64), "input", "W", "row"))
        assert tile_lanes(g, "Kt[0:64,0:197]") == 64

    def test_rank1_row(self):
        g = _one_read(("B", "f32", (640,), "row"), "B[0:320]", Out("f32", 4))
        assert tile_lanes(g, "B[0:320]") == 320

    def test_rank1_lane_not_checkable(self):
        g = _one_read(("B", "f32", (640,), "lane"), "B[0:640]", Out("f32", 4))
        assert tile_lanes(g, "B[0:640]") is None

    def test_out_of_bounds(self):
        g = _one_read(("X", "i8", (128, 320), "row"), "X[0:128,0:400]")
        with pytest.raises(LTSError, match="LTS-E-GRAPH-TILE"):
            tile_lanes(g, "X[0:128,0:400]")

    def test_rank_mismatch(self):
        g = _one_read(("X", "i8", (128, 320), "row"), "X[0:128]")
        with pytest.raises(LTSError, match="LTS-E-GRAPH-TILE"):
            tile_lanes(g, "X[0:128]")


class TestValidateLaneWidth:
    def test_mlp_ok(self, g, m):
        validate_graph(g, m)

    def test_gemm_ok(self, m):
        validate_graph(build_gemm(256, 640, 640, tile_m=128), m)

    def test_row_too_wide(self, m):
        g = _one_read(("X", "i8", (128, 640), "row"), "X[0:128,0:640]")
        with pytest.raises(LTSError, match="LTS-E-GRAPH-LANES"):
            validate_lane_width(g, m)

    def test_lane_too_wide(self, m):
        """lane 布局下逻辑第一维超过 320 也不行。"""
        g = _one_read(("S", "f32", (400, 64), "lane"), "S[0:400,0:64]", Out("f32", 4))
        with pytest.raises(LTSError, match="LTS-E-GRAPH-LANES"):
            validate_lane_width(g, m)

    def test_rank1_too_wide(self, m):
        g = _one_read(("B", "f32", (640,), "row"), "B[0:640]", Out("f32", 4))
        with pytest.raises(LTSError, match="LTS-E-GRAPH-LANES"):
            validate_lane_width(g, m)

    def test_exactly_320_ok(self, m):
        g = _one_read(("X", "i8", (128, 640), "row"), "X[0:128,320:640]")
        validate_lane_width(g, m)


# ---- B0：字段取值与一致性检查 --------------------------------------------------
from ltsim.graph.load import tile_region, validate_structure


def _load_mut(tmp_path, mutate):
    d = json.load(open(FIXTURE))
    mutate(d)
    p = tmp_path / "g.json"
    p.write_text(json.dumps(d))
    return load_graph(str(p))


def _node(d, nid):
    return next(n for n in d["nodes"] if n["id"] == nid)


class TestFieldValues:
    @pytest.mark.parametrize("mutate", [
        lambda d: d["tensors"]["X"].update(dtype="int8"),
        lambda d: d["tensors"]["X"].update(layout="col"),
        lambda d: d["tensors"]["X"].update(hemisphere="N"),
        lambda d: d["tensors"]["X"].update(role="weight"),
        lambda d: d["tensors"]["X"].update(shape=[128, 0]),
        lambda d: _node(d, "a0.acc").update(rows="128"),          # 字符串不是整数
        lambda d: _node(d, "a0.acc").update(bank=-1),
        lambda d: _node(d, "a0.acc")["out"].update(V=0),
        lambda d: _node(d, "a0.acc")["out"].update(dtype="int32"),
        lambda d: _node(d, "a0.vxm").update(ops=[]),
        lambda d: _node(d, "a0.h.wr").update(out={"dtype": "i8", "V": 1}),   # 写节点不得有 out
        lambda d: _node(d, "a0.h.wr").pop("tile"),                         # 写节点必须有 tile
        lambda d: _node(d, "w1n0.lw").pop("bank"),
        lambda d: _node(d, "a0.acc").update(kind="mxm.matmul"),
        lambda d: d["edges"]["stream"].__setitem__(0, ["w1n0.rd", "w1n0.lw", -1]),
        lambda d: d["edges"]["order"].append(
            {"src": "a0.acc", "dst": "a1.acc", "ref": "end", "lag": 0, "reason": "acc_seq"}),
        lambda d: d["edges"]["order"].append(
            {"src": "a0.acc", "dst": "a1.acc", "ref": "start", "lag": 0, "reason": "because"}),
        lambda d: d["edges"]["order"].append(
            {"src": "a0.acc", "dst": "a1.acc", "ref": "start", "lag": 1.5, "reason": "acc_seq"}),
    ])
    def test_bad_value(self, tmp_path, mutate):
        with pytest.raises(LTSError, match="LTS-E-GRAPH-FIELD"):
            _load_mut(tmp_path, mutate)

    @pytest.mark.parametrize("mutate", [
        lambda d: _node(d, "a0.acc").pop("kind"),
        lambda d: d["tensors"]["X"].pop("dtype"),
        lambda d: d["edges"]["order"].append({"src": "a0.acc", "dst": "a1.acc"}),
    ])
    def test_missing_field_is_diagnosed_not_keyerror(self, tmp_path, mutate):
        with pytest.raises(LTSError, match="LTS-E-GRAPH-FIELD"):
            _load_mut(tmp_path, mutate)


class TestRowsConsistency:
    def test_rows_mismatch(self, tmp_path):
        with pytest.raises(LTSError, match="LTS-E-GRAPH-ROWS"):
            _load_mut(tmp_path, lambda d: _node(d, "a0.x.rd").update(rows=64))

    def test_broadcast_rows_must_equal_repeat(self, tmp_path):
        with pytest.raises(LTSError, match="LTS-E-GRAPH-ROWS"):
            _load_mut(tmp_path, lambda d: _node(d, "a0.b1.rd").update(repeat=64))

    def test_broadcast_needs_one_word_tile(self, tmp_path):
        """repeat 读的 tile 必须只有 1 个 word；X[0:128,...] 是 128 个 word。"""
        def mut(d):
            _node(d, "a0.x.rd").update(repeat=128)
        with pytest.raises(LTSError, match="LTS-E-GRAPH-ROWS"):
            _load_mut(tmp_path, mut)


class TestMemEdgeConsistency:
    def test_endpoint_kinds(self, tmp_path):
        with pytest.raises(LTSError, match="LTS-E-GRAPH-MEM"):
            _load_mut(tmp_path, lambda d: d["edges"]["mem"].append(["a0.acc", "b.h0.rd"]))

    def test_different_image(self, tmp_path):
        """写 H、读 Y：不是同一份 MEM 映像。"""
        with pytest.raises(LTSError, match="LTS-E-GRAPH-MEM"):
            _load_mut(tmp_path, lambda d: d["edges"]["mem"].append(["a0.h.wr", "a0.x.rd"]))

    def test_no_overlap(self, tmp_path):
        """写 H[:, 0:320]、读 H[:, 320:640]：lane 不相交。"""
        with pytest.raises(LTSError, match="LTS-E-GRAPH-MEM"):
            _load_mut(tmp_path, lambda d: d["edges"]["mem"].append(["a0.h.wr", "b.h1.rd"]))

    def test_alias_same_image_ok(self):
        """写 K（row），读 Kt（K 的 lane 版本）：同一份映像，物理区域相同。"""
        b = GraphBuilder("t", "lpu-tile-v0")
        b.tensor("K", "i8", (8, 64), "intermediate", "W", "row")
        b.tensor("Kt", "i8", (64, 8), "intermediate", "W", "lane", alias_of="K")
        b.tensor("O", "i8", (8, 64), "output", "E", "row")
        b.read("src", "O[0:8,0:64]", 8, Out("i8", 1), ["op"])
        b.vxm("v", ["relu"], 8, Out("i8", 1), ["op"])
        b.write("k.wr", "K[0:8,0:64]", 8, ["op"])
        b.read("kt.rd", "Kt[0:64,0:8]", 8, Out("i8", 1), ["op"])
        b.stream("src", "v"); b.stream("v", "k.wr")
        b.mem("k.wr", "kt.rd")
        g = b.build()
        validate_structure(g)
        assert tile_region(g, "K[0:8,0:64]") == ("K", (0, 8), (0, 64))
        assert tile_region(g, "Kt[0:64,0:8]") == ("K", (0, 8), (0, 64))


class TestStateEdgeConsistency:
    def test_endpoint_kinds(self, tmp_path):
        with pytest.raises(LTSError, match="LTS-E-GRAPH-STATE"):
            _load_mut(tmp_path, lambda d: d["edges"]["state"].append(["w1n0.rd", "a0.acc"]))

    def test_bank_mismatch(self, tmp_path):
        with pytest.raises(LTSError, match="LTS-E-GRAPH-STATE"):
            _load_mut(tmp_path, lambda d: _node(d, "a0.acc").update(bank=1))

    def test_unit_mismatch(self, tmp_path):
        with pytest.raises(LTSError, match="LTS-E-GRAPH-STATE"):
            _load_mut(tmp_path, lambda d: _node(d, "a0.acc").update(unit="mxm.w1"))


def test_generated_graphs_pass_full_b0(m):
    from ltsim.gen.mlp import build_mlp
    for g in (build_mlp(), build_gemm(256, 640, 640, tile_m=128),
              build_gemm(256, 640, 640, tile_m=128, banks="single", map="k_split")):
        validate_graph(g, m)


# ---- B0 加固：类型、白名单、元素个数、r0 上界、alias 一致性 ----------------------

class TestTypeErrorsAreDiagnosed:
    @pytest.mark.parametrize("mutate", [
        lambda d: _node(d, "a0.vxm").update(ops="cvt"),                # 字符串不能当列表
        lambda d: _node(d, "a0.vxm").update(origin="dq1"),
        lambda d: _node(d, "a0.vxm").pop("origin"),
        lambda d: _node(d, "a0.acc").update(kind=["mxm.acc"]),
        lambda d: _node(d, "a0.x.rd").update(tile=3),
        lambda d: _node(d, "a0.acc").update(unit=0),
        lambda d: _node(d, "a0.acc").update(out=[1]),
        lambda d: d["edges"]["stream"].__setitem__(0, [["w1n0.rd"], "w1n0.lw"]),
        lambda d: d["edges"]["stream"].__setitem__(0, ["w1n0.rd", "w1n0.lw", 0, 9]),   # 多于 3 个元素
        lambda d: d["edges"]["mem"].__setitem__(0, ["a0.h.wr", "b.h0.rd", 0]),         # 多于 2 个元素
        lambda d: d["edges"].update(misc=[]),
        lambda d: d.pop("machine"),
        lambda d: _node(d, "a0.acc").update(color="red"),               # 未知键
        lambda d: d["tensors"]["X"].update(color="red"),
        lambda d: _node(d, "a0.acc")["out"].update(extra=1),
    ])
    def test_field(self, tmp_path, mutate):
        with pytest.raises(LTSError, match="LTS-E-GRAPH-FIELD"):
            _load_mut(tmp_path, mutate)


class TestKindAllowlist:
    @pytest.mark.parametrize("nid,field,value", [
        ("a0.acc", "tile", "X[0:128,0:320]"),
        ("a0.acc", "repeat", 128),
        ("a0.x.rd", "bank", 0),
        ("a0.x.rd", "unit", "mxm.w0"),
        ("a0.vxm", "bank", 0),
        ("a0.h.wr", "repeat", 128),
        ("w1n0.lw", "ops", ["relu"]),
    ])
    def test_field_not_allowed(self, tmp_path, nid, field, value):
        with pytest.raises(LTSError, match="LTS-E-GRAPH-FIELD.*not allowed"):
            _load_mut(tmp_path, lambda d: _node(d, nid).update({field: value}))


def test_r0_must_be_less_than_src_rows(tmp_path):
    with pytest.raises(LTSError, match="LTS-E-GRAPH-STREAM.*r0=128"):
        _load_mut(tmp_path, lambda d: d["edges"]["stream"].__setitem__(5, ["a0.acc", "a0.vxm", 128]))
    _load_mut(tmp_path, lambda d: d["edges"]["stream"].__setitem__(5, ["a0.acc", "a0.vxm", 127]))   # 合法


class TestAliasConsistency:
    def _g(self, k=("i8", (8, 64), "W", "row"), kt=("i8", (64, 8), "W", "lane")):
        b = GraphBuilder("t", "lpu-tile-v0")
        b.tensor("K", k[0], k[1], "intermediate", k[2], k[3])
        b.tensor("Kt", kt[0], kt[1], "intermediate", kt[2], kt[3], alias_of="K")
        return b.build()

    def test_valid(self):
        validate_structure(self._g())

    @pytest.mark.parametrize("kt,why", [
        (("i8", (8, 64), "W", "lane"), "transpose"),
        (("i8", (64, 8), "W", "row"), "opposite"),
        (("i32", (64, 8), "W", "lane"), "dtype"),
        (("i8", (64, 8), "E", "lane"), "hemisphere"),
    ])
    def test_inconsistent(self, kt, why):
        with pytest.raises(LTSError, match=f"LTS-E-GRAPH-REF.*{why}"):
            validate_structure(self._g(kt=kt))

    def test_rank1_alias_rejected(self):
        with pytest.raises(LTSError, match="LTS-E-GRAPH-REF.*rank 2"):
            validate_structure(self._g(k=("f32", (64,), "W", "row"), kt=("f32", (64,), "W", "lane")))


def test_fuzz_never_crashes(tmp_path):
    """对 MLP fixture 的每个键：删除，或换成各种类型的错值。结果只能是正常加载或 LTSError，
    不能是 KeyError / TypeError / AttributeError 等未诊断的异常。"""
    base = json.load(open(FIXTURE))
    bad_values = [None, "zz", -3, 1.5, [], {}, True]

    def paths(x, prefix=()):
        if isinstance(x, dict):
            for k, v in x.items():
                yield prefix + (k,)
                yield from paths(v, prefix + (k,))
        elif isinstance(x, list):
            for i, v in enumerate(x):
                yield prefix + (i,)
                yield from paths(v, prefix + (i,))

    p = tmp_path / "g.json"
    n = 0
    for path in list(paths(base)):
        for mode in ["delete"] + bad_values:
            d = json.loads(json.dumps(base))
            parent = d
            for k in path[:-1]:
                parent = parent[k]
            if mode == "delete":
                if isinstance(parent, list):
                    continue
                del parent[path[-1]]
            else:
                parent[path[-1]] = mode
            p.write_text(json.dumps(d))
            try:
                load_graph(str(p))
            except LTSError:
                pass
            n += 1
    assert n > 2000


# ---- schema.json 与 load.py 的规则同步 -------------------------------------------
import os as _os


@pytest.fixture(scope="module")
def schema():
    path = _os.path.join(_os.path.dirname(__file__), "..", "..", "ltsim", "graph", "schema.json")
    return json.load(open(path))


class TestSchemaJsonInSync:
    """schema.json 只作文档、运行时不读取；这里确保它与 load.py 里手写的规则一致，改了一边忘了另一边就失败。"""

    def test_enums(self, schema):
        from ltsim.graph import load as L
        t = schema["properties"]["tensors"]["additionalProperties"]["properties"]
        assert (t["dtype"]["enum"], t["role"]["enum"], t["hemisphere"]["enum"], t["layout"]["enum"]) == \
            (list(L.DTYPES), list(L.ROLES), list(L.HEMISPHERES), list(L.LAYOUTS))
        o = schema["properties"]["edges"]["properties"]["order"]["items"]["properties"]
        assert (o["ref"]["enum"], o["reason"]["enum"]) == (list(L.REFS), list(L.REASONS))
        assert list(schema["properties"]["edges"]["properties"]) == list(L.EDGE_GROUPS)

    def test_node_fields_per_kind(self, schema):
        from ltsim.graph import load as L
        node = schema["properties"]["nodes"]["items"]
        assert node["additionalProperties"] is False
        assert set(node["properties"]) == set(L.COMMON_NODE_FIELDS) | set(L.OPTIONAL_NODE_FIELDS)
        rules = {}
        for r in node["allOf"]:
            cond = r["if"]["properties"]["kind"]
            if "const" in cond:
                rules[cond["const"]] = r["then"]
            elif cond.get("pattern") == "^sxm\\.":
                rules["sxm.*"] = r["then"]
        specs = {**L.KIND_FIELDS, "sxm.*": L.SXM_FIELDS}
        assert set(rules) == set(specs)
        for kind, spec in specs.items():
            allowed = set(spec["required"]) | set(spec["optional"])
            assert set(rules[kind]["required"]) == set(L.COMMON_NODE_FIELDS) | set(spec["required"]), kind
            assert set(rules[kind]["properties"]) == set(L.OPTIONAL_NODE_FIELDS) - allowed, kind
