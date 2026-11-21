"""CP-SAT oracle（backend.md §9）：LB ≤ OPT ≤ SSGS 恒成立；MLP 的 551 是最优；已知的 SSGS 损失用例。
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


@pytest.fixture(scope="module")
def m():
    return default_machine()


def _check(cg, resources):
    """返回 (LB, OPT, SSGS, topo)，并断言 LB ≤ OPT ≤ min(SSGS, topo)。"""
    r = solve_optimal(cg, resources, time_limit_s=20)
    assert r.status == "OPTIMAL"
    ssgs = schedule(cg, resources)
    topo = schedule(cg, resources, "topo")
    lb = compute_bounds(cg, ssgs, resources).lb
    assert lb <= r.makespan <= ssgs.total and r.makespan <= topo.total
    assert r.ssgs == ssgs.total and r.bound == r.makespan
    return lb, r.makespan, ssgs.total, topo.total


def test_mlp_551_is_optimal(m):
    """MLP：SSGS 的 551 就是最优；与下界 525 之间的 26 拍全部来自下界偏松，不是调度器的损失。"""
    assert _check(build_chains(build_mlp(), m), m.resources())[:3] == (525, 551, 551)


def test_gemm_alternate_ssgs_is_optimal(m):
    assert _check(build_chains(build_gemm(256, 640, 1280, tile_m=128), m), m.resources())[1:3] == (1305, 1305)


def test_gemm_single_known_ssgs_loss(m):
    """banks = single：最优解把两个 plane 的权重装载错开（w1 第一块推迟到 56），避免两块同时占满 E2W 的 32 条流；
    SSGS 把对称的链放在同一时刻，每次换块都被流卡住，多 60 拍（4.37%）。这是贪心列表调度的已知弱点。"""
    lb, opt, ssgs, _topo = _check(build_chains(build_gemm(256, 640, 1280, tile_m=128, banks="single"), m),
                                  m.resources())
    assert (lb, opt, ssgs) == (1327, 1374, 1434)


def test_random_gemm_variants(m):
    rng = random.Random(5)
    for _ in range(12):
        Mb, Kb, Nb = rng.randint(1, 2), rng.randint(1, 2), rng.randint(1, 3)
        tile_m = rng.choice([64, 128])
        g = build_gemm(Mb * tile_m, Kb * 320, Nb * 320, tile_m=tile_m,
                       planes=rng.choice([["mxm.w0"], ["mxm.w0", "mxm.w1"]]),
                       map=rng.choice(["k_split", "n_split"]), banks=rng.choice(["alternate", "single"]))
        cg = build_chains(g, m)
        assert len(cg.chains) <= 30
        _check(cg, m.resources())


def test_random_synthetic_graphs():
    from tests.ltsim.test_scheduler import _synthetic
    resources = {"x.a": Resource("x.a", "exclusive", 1), "x.b": Resource("x.b", "exclusive", 1),
                 "k.c": Resource("k.c", "counter", 5), "k.d": Resource("k.d", "counter", 3)}
    rng = random.Random(6)
    for _ in range(60):
        _check(_synthetic(rng, resources), resources)
