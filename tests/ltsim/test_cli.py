"""CLI（interfaces.md §6）：gen / run 端到端、输出目录规则、错误退出码、机器 id 不一致的警告。"""
import json
import os
import subprocess
import sys

import pytest
import yaml

from ltsim.cli import main
from ltsim.machine.model import DEFAULT_MACHINE_PATH

HERE = os.path.dirname(__file__)
FIXTURE = os.path.join(HERE, "fixtures", "mlp_tile_graph.json")
ROOT = os.path.dirname(os.path.dirname(HERE))


def run(argv, capsys):
    code = main(argv)
    out = capsys.readouterr()
    return code, out.out, out.err


class TestGen:
    def test_mlp_matches_golden(self, tmp_path, capsys):
        code, out, _ = run(["gen", "mlp", "--out", str(tmp_path / "g")], capsys)
        assert code == 0 and "25 nodes, 24 edges" in out
        assert json.load(open(tmp_path / "g" / "tile_graph.json")) == json.load(open(FIXTURE))

    def test_gemm(self, tmp_path, capsys):
        code, out, _ = run(["gen", "gemm", "--M", "256", "--K", "640", "--N", "640", "--tile-m", "128",
                            "--planes", "w0,w1", "--banks", "single", "--out", str(tmp_path / "g")], capsys)
        assert code == 0 and "48 nodes" in out
        g = json.load(open(tmp_path / "g" / "tile_graph.json"))
        assert sum(1 for o in g["edges"]["order"] if o["reason"] == "bank_war") == 2


class TestRun:
    def test_mlp_end_to_end(self, tmp_path, capsys):
        """默认选项（srf_pool = half、lw_streams = separate）：506 = LB，c4 只受 state 边约束（backend.md §8）。"""
        code, out, err = run(["run", FIXTURE, "--out", str(tmp_path / "r")], capsys)
        assert code == 0 and err == ""
        lines = out.splitlines()
        assert lines[0] == "total_cycles 506   lower bound 506 (path 506, resource 136 on vxm.alu)   gap 0.00%"
        assert lines[1] == ("critical path: c0 [0,48) (source) → c4 [21,286) (edge state w1n0.lw→a0.acc) → "
                            "c6 [267,506) (edge mem a0.h.wr→b.h0.rd)")
        assert lines[2] == "without SRF hop tail: total_cycles 506   lower bound 506   gap 0.00%"
        assert lines[3].startswith("wrote ") and len(lines) == 4
        r = json.load(open(tmp_path / "r" / "result.json"))
        t = json.load(open(tmp_path / "r" / "trace.json"))
        assert r["total_cycles"] == 506 and t["otherData"]["total_cycles"] == 506
        assert t["otherData"]["no_hop_tail_total_cycles"] == 506
        assert r["schema"] == "lts-result/0.2"
        assert r["no_hop_tail"]["total_cycles"] == 506 and r["no_hop_tail"]["critical_path"] == ["c0", "c4", "c6"]
        assert sorted(r["srf"]) == sorted(f"srf.{d}.{h}{x}" for d in ("W2E", "E2W") for h in "WE" for x in ("", ".lw"))
        prov = r["provenance"]
        assert prov["command"].startswith("python -m ltsim run ") and prov["policy"] == "bottom_level"
        assert (prov["srf_pool"], prov["lw_streams"]) == ("half", "separate")
        import hashlib
        assert prov["tile_graph_sha256"] == hashlib.sha256(open(FIXTURE, "rb").read()).hexdigest()
        assert prov["machine_sha256"] == hashlib.sha256(open(DEFAULT_MACHINE_PATH, "rb").read()).hexdigest()

    def test_options_pass_through(self, tmp_path, capsys):
        code, _, _ = run(["run", FIXTURE, "--out", str(tmp_path / "r"), "--policy", "topo",
                          "--mem-overlap", "row", "--trace-stream-flows"], capsys)
        assert code == 0
        r = json.load(open(tmp_path / "r" / "result.json"))
        t = json.load(open(tmp_path / "r" / "trace.json"))
        assert (r["provenance"]["policy"], r["provenance"]["mem_overlap"]) == ("topo", "row")
        assert any(e.get("cat") == "stream" for e in t["traceEvents"])

    def test_legacy_srf_options(self, tmp_path, capsys):
        """旧选项（每个方向一个池、权重与其他数据共用）：513，c4 被 srf.E2W 卡在 28；不带 hop 尾巴时 506。"""
        code, out, _ = run(["run", FIXTURE, "--out", str(tmp_path / "r"), "--srf-pool", "direction",
                            "--lw-streams", "shared"], capsys)
        assert code == 0
        assert "total_cycles 513" in out and "gap 1.38%" in out
        assert "c0 [0,48) (source) → c4 [28,293) (resource srf.E2W, held by c0)" in out
        assert "without SRF hop tail: total_cycles 506   lower bound 506   gap 0.00%" in out
        r = json.load(open(tmp_path / "r" / "result.json"))
        assert (r["provenance"]["srf_pool"], r["provenance"]["lw_streams"]) == ("direction", "shared")
        assert r["srf"] == {"srf.E2W": {"peak": 32, "avg": 4.869396}, "srf.W2E": {"peak": 32, "avg": 8.214425}}
        assert r["no_hop_tail"]["total_cycles"] == 506
        assert r["no_hop_tail"]["bounds"] == {"path": 506, "resource": 136, "resource_exact": "136",
                                              "resource_name": "vxm.alu", "lb": 506, "gap": 0.0}

    @pytest.mark.parametrize("pool,lw,total", [("half", "shared", 513), ("direction", "separate", 506)])
    def test_mixed_srf_options(self, tmp_path, capsys, pool, lw, total):
        code, out, _ = run(["run", FIXTURE, "--out", str(tmp_path / "r"), "--srf-pool", pool, "--lw-streams", lw], capsys)
        assert code == 0 and f"total_cycles {total} " in out
        r = json.load(open(tmp_path / "r" / "result.json"))
        assert (r["provenance"]["srf_pool"], r["provenance"]["lw_streams"]) == (pool, lw)

    def test_row_mode_has_no_hop_tail_too(self, tmp_path, capsys):
        code, out, _ = run(["run", FIXTURE, "--out", str(tmp_path / "r"), "--mem-overlap", "row",
                            "--srf-pool", "direction", "--lw-streams", "shared"], capsys)
        assert code == 0 and "total_cycles 393 " in out and "without SRF hop tail: total_cycles 386 " in out

    def test_same_input_same_output(self, tmp_path, capsys):
        for d in ("a", "b"):
            assert run(["run", FIXTURE, "--out", str(tmp_path / d)], capsys)[0] == 0

        def strip(path):
            x = json.load(open(path))
            prov = x["provenance"] if "provenance" in x else x["otherData"]["provenance"]
            prov.pop("command")                  # 命令行里的输出目录不同
            return x
        for f in ("result.json", "trace.json"):
            assert strip(tmp_path / "a" / f) == strip(tmp_path / "b" / f)

    def test_machine_by_id_and_by_path(self, tmp_path, capsys):
        assert run(["run", FIXTURE, "--machine", "lpu-tile-v0", "--out", str(tmp_path / "a")], capsys)[0] == 0
        assert run(["run", FIXTURE, "--machine", DEFAULT_MACHINE_PATH, "--out", str(tmp_path / "b")], capsys)[0] == 0


class TestErrors:
    def test_existing_out_dir(self, tmp_path, capsys):
        (tmp_path / "r").mkdir()
        (tmp_path / "r" / "keep.txt").write_text("x")
        code, _, err = run(["run", FIXTURE, "--out", str(tmp_path / "r")], capsys)
        assert code == 2 and "LTS-E-ARG" in err and "already exists" in err
        assert os.listdir(tmp_path / "r") == ["keep.txt"]           # 没有动已有目录

    def test_bad_graph_creates_nothing(self, tmp_path, capsys):
        d = json.load(open(FIXTURE))
        d["nodes"][0]["rows"] = 64                                   # rows 与 tile 不符
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps(d))
        code, _, err = run(["run", str(bad), "--out", str(tmp_path / "r")], capsys)
        assert code == 2 and "LTS-E-GRAPH-ROWS" in err
        assert not (tmp_path / "r").exists()                         # 失败不建目录，可以用同名重跑

    def test_missing_input(self, tmp_path, capsys):
        code, _, err = run(["run", str(tmp_path / "nope.json"), "--out", str(tmp_path / "r")], capsys)
        assert code == 2 and "LTS-E-ARG" in err

    @pytest.mark.parametrize("flag", ["--srf-pool", "--lw-streams"])
    def test_bad_srf_option(self, tmp_path, capsys, flag):
        """命令行上由 argparse 拒绝（退出码 2，与 --mem-overlap 一样）；API 层报 LTS-E-ARG（test_chains.py）。"""
        with pytest.raises(SystemExit) as e:
            main(["run", FIXTURE, "--out", str(tmp_path / "r"), flag, "bogus"])
        assert e.value.code == 2 and "invalid choice" in capsys.readouterr().err
        assert not (tmp_path / "r").exists()

    def test_unknown_machine(self, tmp_path, capsys):
        code, _, err = run(["run", FIXTURE, "--machine", "lpu-tile-v9", "--out", str(tmp_path / "r")], capsys)
        assert code == 2 and "LTS-E-MACHINE" in err


def test_machine_id_mismatch_warns_and_continues(tmp_path, capsys):
    """换一份改过 id 和参数的机器文件跑同一张图：照常出结果，diagnostics 里记一条警告。"""
    raw = yaml.safe_load(open(DEFAULT_MACHINE_PATH))
    raw["id"] = "lpu-tile-dse1"
    raw["mxm"]["block_switch_gap"] = 9
    mpath = tmp_path / "dse1.yaml"
    mpath.write_text(yaml.safe_dump(raw))
    code, _, err = run(["run", FIXTURE, "--machine", str(mpath), "--out", str(tmp_path / "r")], capsys)
    assert code == 0 and "LTS-W-MACHINE-ID" in err
    r = json.load(open(tmp_path / "r" / "result.json"))
    assert r["diagnostics"] == [{"level": "warning", "code": "LTS-W-MACHINE-ID",
                                 "msg": f"graph targets machine 'lpu-tile-v0' but running on 'lpu-tile-dse1' ({mpath})"}]


def test_python_dash_m(tmp_path):
    """python -m ltsim 能直接用。"""
    p = subprocess.run([sys.executable, "-m", "ltsim", "run", FIXTURE, "--out", str(tmp_path / "r")],
                       capture_output=True, text=True, cwd=ROOT)
    assert p.returncode == 0 and "total_cycles 506" in p.stdout
    p = subprocess.run([sys.executable, "-m", "ltsim", "run", FIXTURE, "--out", str(tmp_path / "r")],
                       capture_output=True, text=True, cwd=ROOT)
    assert p.returncode == 2 and "already exists" in p.stderr
