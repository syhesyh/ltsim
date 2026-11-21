"""金标准快照（设计 §10.3：MLP 的 result.json 逐字节固定）。

快照用默认选项（srf_pool = half、lw_streams = separate），带 no_hop_tail（第二次调度，流预留不带 hop 尾巴）。
快照不含 provenance（其中的机器文件 sha256 会因 yaml 注释改动而变），其余部分逐字节比较。
结果有意改变时，用 `LTSIM_UPDATE_GOLDEN=1 python -m pytest tests/ltsim/test_golden.py` 重新生成，并在提交说明里写清原因。
"""
import os

import pytest

from ltsim.backend.chains import build_chains
from ltsim.backend.result import build_result, dumps_result, make_provenance
from ltsim.backend.scheduler import schedule
from ltsim.gen.gemm import build_gemm
from ltsim.graph.load import load_graph
from ltsim.machine.model import default_machine

HERE = os.path.dirname(__file__)
FIX = os.path.join(HERE, "fixtures")

CASES = {
    "mlp_result.json": lambda: load_graph(os.path.join(FIX, "mlp_tile_graph.json")),
    "gemm_single_result.json": lambda: build_gemm(256, 640, 640, tile_m=128, banks="single"),
}


def _current(make):
    m = default_machine()
    g = make()
    cg = build_chains(g, m)
    cg2 = build_chains(g, m, hop_tail=False)
    r = build_result(g, cg, schedule(cg), m,
                     make_provenance(g, m, policy="bottom_level", mem_overlap="tile", srf_pool="half",
                                     lw_streams="separate"),
                     no_hop_tail=(cg2, schedule(cg2)))
    r.pop("provenance")
    return dumps_result(r)


@pytest.mark.parametrize("name", sorted(CASES))
def test_matches_golden(name):
    path = os.path.join(FIX, name)
    text = _current(CASES[name])
    if os.environ.get("LTSIM_UPDATE_GOLDEN"):
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
    with open(path, encoding="utf-8") as f:
        assert text == f.read(), f"{name} changed; if intended, regenerate with LTSIM_UPDATE_GOLDEN=1"
