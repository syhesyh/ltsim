"""Perfetto trace（backend.md §7，设计 §8.2）：结构、MLP 的关键时间段、流向事件规则、计数器、自检本身。"""
import copy
import json

import pytest

from ltsim.backend.chains import build_chains
from ltsim.backend.metrics import resource_usage
from ltsim.backend.scheduler import schedule
from ltsim.errors import LTSError
from ltsim.gen.gemm import build_gemm
from ltsim.gen.mlp import build_mlp
from ltsim.machine.model import Machine, default_machine
from ltsim.trace.perfetto import build_trace, check_trace, counter_names, dumps_trace, write_trace

# 手算核对过的期望值（513、E2W 顶满、c4 在 28）都是旧选项下的（backend.md §8.4）；新默认选项的用例见 TestDefaultOptions
LEGACY = dict(srf_pool="direction", lw_streams="shared")


@pytest.fixture(scope="module")
def m():
    return default_machine()


def _trace(g, m, opts=LEGACY, **kw):
    cg = build_chains(g, m, **opts)
    p = schedule(cg)
    return cg, p, build_trace(g, cg, p, m, **kw)


@pytest.fixture(scope="module")
def mlp(m):
    return _trace(build_mlp(), m)


def events(t, ph):
    return [e for e in t["traceEvents"] if e["ph"] == ph]


def us(cycle):                      # cycle_ns = 1
    return cycle / 1000


class TestStructure:
    def test_processes(self, mlp):
        _cg, _p, t = mlp
        names = {e["pid"]: e["args"]["name"] for e in events(t, "M") if e["name"] == "process_name"}
        assert names == {1: "Ops (LTS-IR)", 2: "Chains", 3: "MXM", 4: "VXM", 5: "SXM",
                         6: "MEM-W", 7: "MEM-E", 8: "Counters", 9: "Critical path"}
        assert t["displayTimeUnit"] == "ns" and t["otherData"]["cycle_ns"] == 1.0

    def test_slice_counts(self, mlp):
        """8 个算子 + 7 条链 + 25 个节点 + 关键路径 3 段 = 43。节点分布：MXM 8（4 个 load_w + 4 个 acc）、VXM 3；
        MEM-W 8（读 W1×2、读 X×2、读 B1×2、读 B2、写 Y）；MEM-E 6（读 W2×2、写 H×2、读 H×2）。"""
        _cg, _p, t = mlp
        per_pid = {}
        for e in events(t, "X"):
            per_pid[e["pid"]] = per_pid.get(e["pid"], 0) + 1
        assert per_pid == {1: 8, 2: 7, 3: 8, 4: 3, 6: 8, 7: 6, 9: 3}

    def test_mxm_threads_one_per_resource(self, mlp):
        _cg, _p, t = mlp
        mxm = {e["args"]["name"] for e in events(t, "M") if e["name"] == "thread_name" and e["pid"] == 3}
        assert mxm == {f"mxm.{p}.{k}" for p in ("w0", "w1", "e0", "e1") for k in ("weight", "compute")}

    def test_self_check_passes(self, mlp):
        assert check_trace(mlp[2]) == []

    def test_deterministic(self, m):
        assert dumps_trace(_trace(build_mlp(), m)[2]) == dumps_trace(_trace(build_mlp(), m)[2])


class TestMlpSlices:
    def _slice(self, t, node):
        return next(e for e in events(t, "X") if e["args"].get("node") == node)

    def test_acc_on_compute_thread(self, mlp):
        """a0.acc：链 c4 在 28 开始，偏移 9 → [37, 165)，在 mxm.w0.compute 线程上。"""
        _cg, _p, t = mlp
        e = self._slice(t, "a0.acc")
        thread = {(x["pid"], x["tid"]): x["args"]["name"] for x in events(t, "M") if x["name"] == "thread_name"}
        assert (e["ts"], e["dur"]) == (us(37), us(128))
        assert thread[(e["pid"], e["tid"])] == "mxm.w0.compute"
        assert e["name"] == "mxm.acc:fc1"
        assert e["args"]["bound_by"] == {"kind": "resource", "resource": "srf.E2W", "blocker": "c0"}

    def test_write_slice_starts_after_latency(self, mlp):
        """写节点画 [start+L, start+L+D)：a0.h.wr start = 28+117 = 145，L = 1 → [146, 274)。"""
        _cg, _p, t = mlp
        e = self._slice(t, "a0.h.wr")
        assert (e["pid"], e["ts"], e["dur"]) == (7, us(146), us(128))     # H 在东半球

    def test_first_layer_delayed_visible(self, mlp):
        """§8.3 的现象：第一层两条链在 28 才开始（被 E2W 流卡住，依赖边只要求 21）。"""
        _cg, _p, t = mlp
        chains = {e["name"]: e for e in events(t, "X") if e["pid"] == 2}
        assert chains["c4"]["ts"] == chains["c5"]["ts"] == us(28)
        assert chains["c4"]["args"]["bound_by"]["resource"] == "srf.E2W"

    def test_critical_path_rows(self, mlp):
        """c0 [0,48) 与 c4 [28,293) 重叠，分到两行；c6 [274,513) 与 c4 重叠、与 c0 不重叠，回到第 1 行。"""
        _cg, _p, t = mlp
        cp = sorted((e["args"]["step"], e["args"]["chain"], e["tid"]) for e in events(t, "X") if e["pid"] == 9)
        assert cp == [(1, "c0", 1), (2, "c4", 2), (3, "c6", 1)]


class TestCounters:
    def test_e2w_tops_out_at_32(self, mlp):
        """§8.3：[1,29) 内 E2W 被两块西侧权重装载占满 32 条。"""
        _cg, _p, t = mlp
        e2w = [(e["ts"], e["args"]["value"]) for e in events(t, "C") if e["name"] == "srf.E2W"]
        assert e2w[:2] == [(0.0, 0), (us(1), 32)]
        assert max(v for _ts, v in e2w) == 32 and e2w[-1][1] == 0

    def test_counters_match_resource_peaks(self, mlp, m):
        cg, p, t = mlp
        usage = resource_usage(cg, p)
        assert counter_names(cg.resources) == ["srf.E2W", "srf.W2E", "vxm.alu",
                                               "mem.W.rd", "mem.W.wr", "mem.E.rd", "mem.E.wr"]
        for name in counter_names(cg.resources):
            vals = [e["args"]["value"] for e in events(t, "C") if e["name"] == name]
            assert max(vals) == usage[name]["peak"] and vals[-1] == 0, name


class TestFlows:
    def test_mlp_inter_chain_flows(self, mlp):
        """6 条链间边：4 条 state（s 在 load_w 最后一拍 28，f 在 acc start）、2 条 mem。"""
        _cg, _p, t = mlp
        s = {e["id"]: e for e in events(t, "s")}
        f = {e["id"]: e for e in events(t, "f")}
        assert len(s) == len(f) == 6
        st = next(i for i, e in s.items() if e["args"] == {"src": "w1n0.lw", "dst": "a0.acc"})
        assert (s[st]["ts"], f[st]["ts"], f[st]["bp"], s[st]["cat"]) == (us(28), us(37), "e", "state")

    def test_ref_start_edge_points_forward(self, m):
        """lw_after_iw（ref = start）：s 放在上一块首个 acc 的 start，不是它的最后一拍；f 不早于 s。
        规格原来的放法会画出 164 → 37 的反向箭头。"""
        _cg, _p, t = _trace(build_gemm(256, 640, 1280, tile_m=128), m)
        s = {e["id"]: e for e in events(t, "s")}
        f = {e["id"]: e for e in events(t, "f")}
        iw = [i for i, e in s.items() if e["name"] == "lw_after_iw"]
        assert len(iw) == 6
        for i in iw:
            assert f[i]["ts"] >= s[i]["ts"]
        one = next(i for i in iw if s[i]["args"] == {"src": "m0n0k0.acc", "dst": "w.n0k1.lw"})
        assert s[one]["ts"] == us(37)                      # m0n0k0.acc 的 start：28 + 9
        assert t["otherData"]["skipped_flows"] == []

    def test_stream_flows_optional(self, m):
        _cg, _p, off = _trace(build_mlp(), m)
        _cg, _p, on = _trace(build_mlp(), m, stream_flows=True)
        assert len(events(on, "s")) - len(events(off, "s")) == 18      # 18 条 stream 边
        assert {e["cat"] for e in events(on, "s")} == {"state", "mem", "stream"}
        assert check_trace(on) == []

    def test_gemm_variants_pass_self_check(self, m):
        for kw in (dict(banks="single"), dict(map="k_split"), dict(planes=["mxm.e0", "mxm.e1"])):
            g = build_gemm(256, 640, 640, tile_m=64, **kw)
            _cg, _p, t = _trace(g, m, stream_flows=True)
            assert check_trace(t) == [] and t["otherData"]["skipped_flows"] == []


def test_cycle_ns_scales_time(m):
    raw = copy.deepcopy(dict(m.raw))
    raw["display"] = {"cycle_ns": 2.0}
    _cg, _p, t = _trace(build_mlp(), Machine(raw))
    acc = next(e for e in events(t, "X") if e["args"].get("node") == "a0.acc")
    assert (acc["ts"], acc["dur"]) == (2 * us(37), 2 * us(128))


class TestCheckerCatchesProblems:
    def _bad(self, mlp, mutate):
        t = json.loads(dumps_trace(mlp[2]))
        mutate(t["traceEvents"])
        return check_trace(t)

    def test_partial_overlap(self, mlp):
        def mut(ev):
            x = [e for e in ev if e["ph"] == "X" and e["pid"] == 3 and e["args"].get("node") == "a0.acc"][0]
            ev.append({**x, "ts": x["ts"] + 0.05, "name": "dup"})
        assert any("partial overlap" in s for s in self._bad(mlp, mut))

    def test_missing_f(self, mlp):
        def mut(ev):
            ev.remove(next(e for e in ev if e["ph"] == "f"))
        assert any("missing its f" in s for s in self._bad(mlp, mut))

    def test_flow_outside_slice_and_backwards(self, mlp):
        def mut(ev):
            f = next(e for e in ev if e["ph"] == "f")
            f["ts"] = 0.0
        msgs = self._bad(mlp, mut)
        assert any("not inside a slice" in s for s in msgs) and any("earlier than s" in s for s in msgs)

    def test_negative_ts(self, mlp):
        def mut(ev):
            next(e for e in ev if e["ph"] == "C")["ts"] = -1.0
        assert any("negative ts" in s for s in self._bad(mlp, mut))


def test_write_trace(tmp_path, mlp):
    path = tmp_path / "trace.json"
    write_trace(str(path), mlp[2])
    assert json.loads(path.read_text())["otherData"]["total_cycles"] == 513


# ---- 覆盖缺口 -----------------------------------------------------------------
from ltsim.gen.builder import graph_to_dict
from ltsim.graph.load import validate_graph
from ltsim.graph.tile_graph import OrderEdge


def _with_orders(*orders):
    """MLP 上追加几条 order 边（经 JSON 往返得到新的 TileGraph）。"""
    g = build_mlp()
    d = graph_to_dict(g)
    d["edges"]["order"] += [{"src": s, "dst": t, "ref": r, "lag": lag, "reason": why} for s, t, r, lag, why in orders]
    import os
    import tempfile
    from ltsim.graph.load import load_graph
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "g.json")
        with open(p, "w") as f:
            json.dump(d, f)
        return load_graph(p)


def test_parallel_order_edges_keep_their_own_ref_and_reason(m):
    """同一对节点上两条 order 边：bank_war（ref = start）的 s 在 w1n0.lw 的 start 9；acc_seq（ref = done）在最后一拍 28。"""
    g = _with_orders(("w1n0.lw", "w2k0.lw", "start", 0, "bank_war"), ("w1n0.lw", "w2k0.lw", "done", 0, "acc_seq"))
    validate_graph(g, m)
    _cg, _p, t = _trace(g, m)
    s = {e["name"]: e["ts"] for e in events(t, "s") if e["args"] == {"src": "w1n0.lw", "dst": "w2k0.lw"}}
    assert s == {"bank_war": us(9), "acc_seq": us(28)}
    assert check_trace(t) == []


def test_skipped_flow_is_recorded(m):
    """lag 很负的 order 边 a0.acc → w2k0.lw（c4 → c2；c2 不是 c4 的祖先，不成环）：c2 仍在 0 开始，
    w2k0.lw 的时间段是 [10,30)；s 在 a0.acc 的最后一拍 164，已晚于目的段 → 不画，记在 skipped_flows。"""
    g = _with_orders(("a0.acc", "w2k0.lw", "issue_end", -1000, "acc_seq"))
    _cg, p, t = _trace(g, m)
    assert p.start["c2"] == 0
    assert t["otherData"]["skipped_flows"] == ["order:a0.acc->w2k0.lw"]
    assert check_trace(t) == []


def test_mem_flow_from_write_node(mlp):
    """写节点 a0.h.wr 的时间段是 [146,274)（start + L）：mem 边的 s 在最后一拍 273，f 在读 H 的 start 274。
    依赖时刻 issue_end + write_latency + raw_gap = 145+128+1+0 = 274 恰好是时间段的终点。"""
    _cg, _p, t = mlp
    s = next(e for e in events(t, "s") if e["args"] == {"src": "a0.h.wr", "dst": "b.h0.rd"})
    f = next(e for e in events(t, "f") if e["id"] == s["id"])
    assert (s["ts"], f["ts"], s["cat"]) == (us(273), us(274), "mem")


def test_rows_and_op_threads(mlp):
    _cg, _p, t = mlp
    row = {e["args"]["node"]: (e["pid"], e["tid"]) for e in events(t, "X") if "node" in e["args"]}
    assert (row["a0.vxm"], row["a1.vxm"], row["b.vxm"]) == ((4, 1), (4, 2), (4, 1))
    assert (row["a0.x.rd"], row["a1.x.rd"], row["a0.b1.rd"], row["a1.b1.rd"]) == ((6, 1), (6, 2), (6, 3), (6, 4))
    ops = [e["args"]["name"] for e in events(t, "M") if e["name"] == "thread_name" and e["pid"] == 1]
    assert ops == ["fc1", "fc2", "b1", "dq1", "act", "q1", "b2", "y"]


def test_self_check_failure_raises(m, monkeypatch):
    import ltsim.trace.perfetto as pf
    monkeypatch.setattr(pf, "check_trace", lambda t: ["boom"])
    with pytest.raises(LTSError, match="LTS-E-TRACE.*boom"):
        _trace(build_mlp(), m)


def test_deterministic_across_processes(tmp_path):
    """不同 PYTHONHASHSEED 的两个进程生成的 trace.json 逐字节相同。"""
    import os, subprocess, sys
    code = ("import sys; from ltsim.backend.chains import build_chains; from ltsim.backend.scheduler import schedule; "
            "from ltsim.trace.perfetto import build_trace, dumps_trace; from ltsim.gen.gemm import build_gemm; "
            "from ltsim.machine.model import default_machine; m = default_machine(); "
            "g = build_gemm(256, 640, 640, tile_m=64); cg = build_chains(g, m); p = schedule(cg); "
            "sys.stdout.write(dumps_trace(build_trace(g, cg, p, m, stream_flows=True)))")
    outs = []
    for seed in ("0", "12345"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        outs.append(subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env,
                                   check=True, cwd=os.path.dirname(os.path.dirname(os.path.dirname(__file__)))).stdout)
    assert outs[0] == outs[1] and len(outs[0]) > 1000


class TestCheckerMore:
    def _x(self, ts, dur, tid=1, name="a"):
        return {"ph": "X", "name": name, "pid": 1, "tid": tid, "ts": ts, "dur": dur, "args": {}}

    def test_zero_and_negative_dur(self):
        assert any("non-positive dur" in s for s in check_trace({"traceEvents": [self._x(0.0, 0.0)]}))
        assert any("non-positive dur" in s for s in check_trace({"traceEvents": [self._x(0.0, -0.001)]}))

    def test_touching_is_not_overlap(self):
        """0.1 + 0.2 与 0.3 在浮点下不完全相等，首尾相接不应误报。"""
        assert check_trace({"traceEvents": [self._x(0.1, 0.2, name="a"), self._x(0.3, 0.1, name="b")]}) == []

    def test_only_f(self):
        f = {"ph": "f", "bp": "e", "id": 1, "cat": "mem", "name": "mem", "pid": 1, "tid": 1, "ts": 0.0}
        assert any("missing its s" in s for s in check_trace({"traceEvents": [self._x(0.0, 1.0), f]}))

    def test_missing_field_reported_not_raised(self):
        assert any("missing field" in s for s in check_trace({"traceEvents": [{"ph": "X", "name": "a"}]}))

    def test_f_needs_bp_and_matching_cat_name(self):
        base = {"id": 1, "pid": 1, "tid": 1, "ts": 0.5}
        s = {**base, "ph": "s", "cat": "mem", "name": "mem"}
        f = {**base, "ph": "f", "cat": "state", "name": "mem", "bp": "x"}
        msgs = check_trace({"traceEvents": [self._x(0.0, 1.0), s, f]})
        assert any('bp "e"' in m for m in msgs) and any("differ in cat/name" in m for m in msgs)


class TestDefaultOptions:
    """新默认选项（srf_pool = half、lw_streams = separate）：计数器列出资源表里的全部 8 个 SRF 池；
    MLP 的 c4 在 21 开始（只受 state 边约束），读 X 与读权重分在 srf.E2W.W 与 srf.E2W.W.lw。"""

    def test_counters_are_all_pools(self, m):
        cg, p, t = _trace(build_mlp(), m, opts={})
        names = counter_names(cg.resources)
        assert names[:8] == ["srf.E2W.E", "srf.E2W.E.lw", "srf.E2W.W", "srf.E2W.W.lw",
                             "srf.W2E.E", "srf.W2E.E.lw", "srf.W2E.W", "srf.W2E.W.lw"]
        assert names[8:] == ["vxm.alu", "mem.W.rd", "mem.W.wr", "mem.E.rd", "mem.E.wr"]
        usage = resource_usage(cg, p)
        for name in names:
            vals = [e["args"]["value"] for e in events(t, "C") if e["name"] == name]
            assert vals[0] == 0 or name.startswith("mem."), name
            assert max(vals) == usage[name]["peak"] and vals[-1] == 0, name
        lw = [(e["ts"], e["args"]["value"]) for e in events(t, "C") if e["name"] == "srf.E2W.W.lw"]
        assert lw == [(0.0, 0), (us(1), 32), (us(29), 0)]
        assert check_trace(t) == []

    def test_first_layer_not_delayed(self, m):
        _cg, p, t = _trace(build_mlp(), m, opts={})
        chains = {e["name"]: e for e in events(t, "X") if e["pid"] == 2}
        assert chains["c4"]["ts"] == chains["c5"]["ts"] == us(21)
        assert chains["c4"]["args"]["bound_by"]["kind"] == "edge"
        assert t["otherData"]["total_cycles"] == p.total == 506

    def test_no_hop_tail_total_in_other_data(self, m):
        g = build_mlp()
        cg = build_chains(g, m, **LEGACY)
        t = build_trace(g, cg, schedule(cg), m, no_hop_tail_total=506)
        assert (t["otherData"]["total_cycles"], t["otherData"]["no_hop_tail_total_cycles"]) == (513, 506)
        assert "no_hop_tail_total_cycles" not in _trace(g, m)[2]["otherData"]
