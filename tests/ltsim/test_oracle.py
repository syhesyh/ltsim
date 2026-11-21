"""CP-SAT oracle（backend.md §9）：LB ≤ OPT ≤ SSGS 恒成立；MLP 的 513 是最优；已知的 SSGS 损失用例。
没装 ortools 时整个文件跳过（ltsim 包本身不依赖它）。"""
import random

import pytest

pytest.importorskip("ortools")

from ltsim.backend.bounds import compute_bounds
from ltsim.backend.chains import build_chains
from ltsim.backend.oracle import solve_optimal
from ltsim.backend.scheduler import schedule
from ltsim.gen.gemm import build_gemm
from ltsim.gen.mlp import build_mlp
from ltsim.machine.model import Resource, default_machine

# 手算核对过的期望值都是旧选项下的（backend.md §8.4）；新默认选项的用例单独写
LEGACY = dict(srf_pool="direction", lw_streams="shared")


@pytest.fixture(scope="module")
def m():
    return default_machine()


def _check(cg):
    """返回 (LB, OPT, SSGS, topo)，并断言 LB ≤ OPT ≤ min(SSGS, topo)。"""
    r = solve_optimal(cg, time_limit_s=20)
    assert r.status == "OPTIMAL"
    ssgs = schedule(cg)
    topo = schedule(cg, "topo")
    lb = compute_bounds(cg, ssgs).lb
    assert lb <= r.makespan <= ssgs.total and r.makespan <= topo.total
    assert r.ssgs == ssgs.total and r.bound == r.makespan
    return lb, r.makespan, ssgs.total, topo.total


def test_mlp_513_is_optimal(m):
    """MLP：SSGS 的 513 就是最优；与下界 506 之间的 7 拍全部来自下界偏松（路径界不看第一层与权重装载
    争 E2W 流，c4 只能在 28 而不是 21 开始），不是调度器的损失。"""
    assert _check(build_chains(build_mlp(), m, **LEGACY))[:3] == (506, 513, 513)


def test_gemm_alternate_ssgs_is_optimal(m):
    assert _check(build_chains(build_gemm(256, 640, 1280, tile_m=128), m, **LEGACY))[1:3] == (1267, 1267)


def test_gemm_single_known_ssgs_loss(m):
    """banks = single：SSGS 把两个 plane 上对称的链放在同一时刻，最优解把它们错开（例如 w1 的第一块权重装载
    推迟到 28），换块时少被 E2W 流卡住。SSGS 只多 3 拍（0.22%）；旧规则（流预留加 wave 尾巴）下这里差 60 拍
    （4.37%），大部分来自那 19 拍的尾巴拉长了流冲突的窗口。仍是贪心列表调度的已知弱点。"""
    lb, opt, ssgs, _topo = _check(build_chains(build_gemm(256, 640, 1280, tile_m=128, banks="single"), m, **LEGACY))
    assert (lb, opt, ssgs) == (1308, 1336, 1339)


def test_random_gemm_variants(m):
    rng = random.Random(5)
    for _ in range(12):
        Mb, Kb, Nb = rng.randint(1, 2), rng.randint(1, 2), rng.randint(1, 3)
        tile_m = rng.choice([64, 128])
        g = build_gemm(Mb * tile_m, Kb * 320, Nb * 320, tile_m=tile_m,
                       planes=rng.choice([["mxm.w0"], ["mxm.w0", "mxm.w1"]]),
                       map=rng.choice(["k_split", "n_split"]), banks=rng.choice(["alternate", "single"]))
        cg = build_chains(g, m, **LEGACY)
        assert len(cg.chains) <= 30
        _check(cg)


OPTIONS = [dict(srf_pool=p, lw_streams=lw, hop_tail=ht)
           for p in ("half", "direction") for lw in ("separate", "shared") for ht in (True, False)]


@pytest.mark.parametrize("opts", OPTIONS, ids=lambda o: f"{o['srf_pool']}-{o['lw_streams']}-{'tail' if o['hop_tail'] else 'notail'}")
def test_all_srf_options(m, opts):
    """四种流池划分 × 带 / 不带 hop 尾巴：oracle 用同一张资源表（cg.resources），LB ≤ OPT ≤ SSGS 恒成立。"""
    for g in (build_mlp(), build_gemm(256, 640, 640, tile_m=128, banks="single"),
              build_gemm(128, 640, 640, tile_m=64, planes=["mxm.w0"], map="k_split"),
              build_gemm(128, 320, 960, tile_m=128, planes=["mxm.e0", "mxm.e1"], map="round_robin")):
        _check(build_chains(g, m, **opts))


def test_new_defaults_known_numbers(m):
    """默认选项（half + separate）：MLP、alternate / single 的 GEMM 都等于 LB_path，SSGS 即最优。"""
    assert _check(build_chains(build_mlp(), m))[:3] == (506, 506, 506)
    assert _check(build_chains(build_gemm(256, 640, 1280, tile_m=128), m))[:3] == (1260, 1260, 1260)
    assert _check(build_chains(build_gemm(256, 640, 1280, tile_m=128, banks="single"), m))[:3] == (1308, 1308, 1308)


def test_random_synthetic_graphs():
    from tests.ltsim.test_scheduler import _synthetic
    resources = {"x.a": Resource("x.a", "exclusive", 1), "x.b": Resource("x.b", "exclusive", 1),
                 "k.c": Resource("k.c", "counter", 5), "k.d": Resource("k.d", "counter", 3)}
    rng = random.Random(6)
    for _ in range(60):
        _check(_synthetic(rng, resources))
