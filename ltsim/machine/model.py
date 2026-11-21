"""LTSim 机器模型：加载 lpu-tile-v0.yaml 并提供参数查询与资源注册表。

对应设计文档 §3.4。只做查询与资源推导，不做任何周期计算（那是 B1/B3 的事）。
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Literal, Mapping

import yaml

from ..errors import LTSError


SRF_POOL_MODES = ("half", "direction")
LW_STREAM_MODES = ("separate", "shared")


def srf_pool_names(srf_pool: str, lw_streams: str) -> tuple[str, ...]:
    """全部 SRF 流池的 id，按名字排序。例如 half + separate：srf.E2W.E、srf.E2W.E.lw、srf.E2W.W、…"""
    base = [f"srf.{d}" for d in ("W2E", "E2W")]
    if srf_pool == "half":
        base = [f"{b}.{h}" for b in base for h in ("W", "E")]
    if lw_streams == "separate":
        base += [f"{b}.lw" for b in base]
    return tuple(sorted(base))


@dataclass(frozen=True)
class Resource:
    """一个可被调度器竞争的资源。"""
    id: str                                  # "mxm.w0.compute" / "vxm.alu" / "srf.E2W.W" / "mem.W.rd" ...
    kind: Literal["exclusive", "counter"]
    cap: int                                 # exclusive 恒为 1；counter 为容量


@dataclass(frozen=True)
class Machine:
    """机器参数表。

    raw 是 yaml 原文（Mapping）。所有查询方法在参数缺失时抛 LTSError("LTS-E-MACHINE")。
    """
    raw: Mapping[str, Any]

    # ---- 参数查询 -----------------------------------------------------

    def param(self, path: str) -> int:
        """按点分路径取整数参数，如 "mxm.lw_to_acc_gap" → 1。"""
        cur: Any = self.raw
        for part in path.split("."):
            if not isinstance(cur, Mapping) or part not in cur:
                raise LTSError("LTS-E-MACHINE", f"machine param not found: {path!r}")
            cur = cur[part]
        if isinstance(cur, bool) or not isinstance(cur, int):
            raise LTSError("LTS-E-MACHINE", f"machine param {path!r} is not an int: {cur!r}")
        return cur

    def hop(self, src_cat: str, dst_cat: str) -> int:
        """查 hops 表，如 hop("MEM_W", "MXM_W") → 8。"""
        key = f"{src_cat}>{dst_cat}"
        hops = self.raw.get("hops", {})
        if key not in hops:
            raise LTSError("LTS-E-MACHINE", f"hop not found: {key!r}")
        return int(hops[key])

    def op_latency(self, alu_op: str) -> int:
        """vxm 操作延迟，如 op_latency("gelu.f32") → 18。"""
        table = self.raw.get("vxm", {}).get("op_latency", {})
        if alu_op not in table:
            raise LTSError("LTS-E-MACHINE", f"vxm op_latency not found: {alu_op!r}")
        return int(table[alu_op])

    # ---- 资源注册表 ---------------------------------------------------

    def resources(self, *, srf_pool: str = "half", lw_streams: str = "separate") -> Mapping[str, Resource]:
        """推导全部资源 id → Resource（backend.md §4.4.1「流资源池」）。

        只供 B2 的 build_chains 调用：它把这张表存进 ChainGraph.resources，B3/B4/trace/oracle 都从那里取，
        所以链图和资源表不会各算各的。srf_pool / lw_streams 只决定 SRF 流池怎么分，其余资源与选项无关。"""
        if srf_pool not in SRF_POOL_MODES:
            raise LTSError("LTS-E-ARG", f"srf_pool must be one of {SRF_POOL_MODES}, got {srf_pool!r}")
        if lw_streams not in LW_STREAM_MODES:
            raise LTSError("LTS-E-ARG", f"lw_streams must be one of {LW_STREAM_MODES}, got {lw_streams!r}")
        res: dict[str, Resource] = {}

        # MXM：每 plane 有独立的 weight 与 compute 两个独占资源（lw_abc_acc=true）
        for inst in self.raw["mxm"]["instances"]:
            pid = inst["id"]                                # mxm.w0 / mxm.w1 / mxm.e0 / mxm.e1
            res[f"{pid}.weight"] = Resource(f"{pid}.weight", "exclusive", 1)
            res[f"{pid}.compute"] = Resource(f"{pid}.compute", "exclusive", 1)

        # VXM：计数资源，alu 数
        res["vxm.alu"] = Resource("vxm.alu", "counter", self.param("vxm.alus"))

        # SRF：流池，每个池 streams_per_direction 条流。direction：每个方向一个池；half：每个方向再分东西两半；
        # lw_streams = separate 时每个池另有一个权重装载专用的 ".lw" 池
        n = self.param("srf.streams_per_direction")
        for name in srf_pool_names(srf_pool, lw_streams):
            res[name] = Resource(name, "counter", n)

        # MEM：每半球读/写各一个计数资源
        rd = self.param("mem.read_vectors_per_cycle")
        wr = self.param("mem.write_vectors_per_cycle")
        for h in self.raw["mem"]["hemispheres"]:
            res[f"mem.{h}.rd"] = Resource(f"mem.{h}.rd", "counter", rd)
            res[f"mem.{h}.wr"] = Resource(f"mem.{h}.wr", "counter", wr)

        # SXM：独占（v0 仅预留）
        for inst in self.raw.get("sxm", {}).get("instances", []):
            res[inst["id"]] = Resource(inst["id"], "exclusive", 1)

        return res


DEFAULT_MACHINE_PATH = os.path.join(os.path.dirname(__file__), "lpu-tile-v0.yaml")


def default_machine() -> Machine:
    """加载仓库自带的 lpu-tile-v0.yaml。"""
    return load_machine(DEFAULT_MACHINE_PATH)


def load_machine(path: str) -> Machine:
    """加载机器参数 yaml，校验 schema 与必需字段。"""
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, Mapping):
        raise LTSError("LTS-E-MACHINE", f"machine file {path} is not a mapping")
    if raw.get("schema") != "lts-machine/0.1":
        raise LTSError("LTS-E-MACHINE", f"machine schema must be 'lts-machine/0.1', got {raw.get('schema')!r}")
    for field in ("geometry", "srf", "hops", "mem", "mxm", "vxm"):
        if field not in raw:
            raise LTSError("LTS-E-MACHINE", f"machine file missing required section: {field!r}")
    m = Machine(raw)
    _check_geometry(m)
    return m


def _check_geometry(m: Machine) -> None:
    """推导量必须与源参数一致：
    vector_bytes == superlanes × lanes_per_superlane；
    srf.wave_skew_cycles == superlanes − 1（命令波每个 superlane 晚 1 拍）。
    """
    sl = m.param("geometry.superlanes")
    lps = m.param("geometry.lanes_per_superlane")
    vb = m.param("geometry.vector_bytes")
    if vb != sl * lps:
        raise LTSError("LTS-E-MACHINE",
                       f"geometry.vector_bytes={vb} != superlanes {sl} × lanes_per_superlane {lps} = {sl * lps}")
    ws = m.param("srf.wave_skew_cycles")
    if ws != sl - 1:
        raise LTSError("LTS-E-MACHINE",
                       f"srf.wave_skew_cycles={ws} != superlanes {sl} − 1 = {sl - 1}")
