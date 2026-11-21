"""用 GraphBuilder 按设计 §5.4 的 matmul 规则展开一个参数化 GEMM：Y = X · W。

覆盖多块换权重的场景（每个 plane 上多个权重块），用来检验 MXM 顺序边
（state / acc_seq / block_switch / lw_after_iw / bank_war）的生成。

范围（有意简化）：
- row 形式（X 为 row 布局流入 MXM，W 装进 MXM），INT8 输入、INT32 输出；
- 单侧模式（planes 全在同一半球）：所有张量放在该半球，输出与早期版本逐字节一致；
- 双侧模式（planes 跨东西两侧，可用满 4 个 MXM）：n 块前一半归西侧、后一半归东侧，
  X/W/Y 按侧复制为 XW/XE、WW/WE、YW/YE（全形；X 每侧全量读，W/Y 每侧只用自己的 n 列范围）；
- K 合并（Kb > 1）不融合：每个 partial 写回 MEM，再由一个 vxm.chain 读回求和后写 Y；
  Kb == 1 时 acc 直接写 Y。partial 的放置由 p_placement 决定（见下）；
- 不加 bias / 激活，不做 lane 形式与运行时权重。

p_placement（Kb > 1 时的 partial 张量半球）：
- "local"：全放本侧；
- "alternate"：P{k} 放本侧（k 偶）/ 对侧（k 奇），合并链的读流对半分到东西两个流池，
  可容纳的 Kb × V 上限从 32 翻倍到 64；
- "auto"（默认）：Kb × V > srf.streams_per_direction 时按 alternate，否则按 local。
  ceil(Kb/2) × V > 32 的形状仍然会报 LTS-E-SRF-WIDTH：那需要树形 K 合并，目前未实现。
"""
from __future__ import annotations

from math import ceil

from ..errors import LTSError
from ..graph.tile_graph import Out, TileGraph
from ..machine.model import Machine, default_machine
from .builder import GraphBuilder, bank_for, plane_for, tile_str

P_PLACEMENT_MODES = ("auto", "local", "alternate")
_OTHER = {"W": "E", "E": "W"}


def build_gemm(M: int, K: int, N: int, *, tile_m: int, tile_k: int | None = None, tile_n: int | None = None,
               planes: list[str] = ("mxm.w0", "mxm.w1"), map: str = "n_split", banks: str = "alternate",
               p_placement: str = "auto", name: str | None = None,
               machine: Machine | None = None) -> TileGraph:
    """machine 省略时用仓库自带的 lpu-tile-v0.yaml。块大小上限与每个向量的 lane 数都从它读取；
    tile_k / tile_n 省略时取 MXM 块的大小。planes / p_placement 的含义见模块 docstring。"""
    m = machine if machine is not None else default_machine()
    block_k, block_n = m.param("mxm.block_k_rows"), m.param("mxm.block_n_cols")
    tile_k = block_k if tile_k is None else tile_k
    tile_n = block_n if tile_n is None else tile_n
    lanes = m.param("geometry.superlanes") * m.param("geometry.lanes_per_superlane")   # 每个向量的 lane 数
    if tile_k > block_k or tile_n > block_n:
        raise LTSError("LTS-E-MXM-BLOCK",
                       f"tile_k={tile_k}, tile_n={tile_n} must be ≤ block {block_k}×{block_n}")
    for dim, t, nm in ((M, tile_m, "tile_m"), (K, tile_k, "tile_k"), (N, tile_n, "tile_n")):
        if t <= 0 or dim % t != 0:
            raise LTSError("LTS-E-GEN", f"{nm}={t} must divide {dim}")
    if p_placement not in P_PLACEMENT_MODES:
        raise LTSError("LTS-E-ARG", f"p_placement must be one of {P_PLACEMENT_MODES}, got {p_placement!r}")
    planes = list(planes)
    if not planes:
        raise LTSError("LTS-E-GEN", "planes must not be empty")
    side_of_plane = {p: p.split(".")[1][0].upper() for p in planes}     # mxm.w0 → "W"
    sides = [h for h in ("W", "E") if h in set(side_of_plane.values())]
    dual = len(sides) == 2

    Mb, Kb, Nb = M // tile_m, K // tile_k, N // tile_n
    if dual and Nb < 2:
        raise LTSError("LTS-E-GEN", f"planes span both sides but Nb={Nb} < 2: N 块不够分侧")
    # 每行向量数 V = ceil(lane 数 / 每个向量的 lane 数) × dtype 字节数（设计 §3.2）
    acc_out = Out("i32", ceil(tile_n / lanes) * 4)
    x_out = Out("i8", ceil(tile_k / lanes) * 1)
    w_out = Out("i8", ceil(tile_n / lanes) * 1)

    # ---- n 块分侧（双侧模式）：前一半 W、后一半 E；单侧模式全部在唯一的一侧 ----
    split_n = (Nb + 1) // 2
    side_of_n = {n: ("W" if n < split_n else "E") if dual else sides[0] for n in range(Nb)}
    side_planes = {h: [p for p in planes if side_of_plane[p] == h] for h in sides}

    # ---- partial 放置（见模块 docstring）----
    eff = p_placement
    if eff == "auto":
        eff = ("alternate" if Kb > 1 and Kb * acc_out.V > m.param("srf.streams_per_direction") else "local")

    def p_hemi(h: str, k: int) -> str:
        return _OTHER[h] if (eff == "alternate" and k % 2 == 1) else h

    def tname(base: str, h: str) -> str:
        return f"{base}{h}" if dual else base

    def pname(k: int, h: str) -> str:
        return f"P{k}{h}" if dual else f"P{k}"

    b = GraphBuilder(name or f"gemm_{M}x{K}x{N}", m.raw["id"])
    for h in sides:
        b.tensor(tname("X", h), "i8", (M, K), "input", h, "row")
        b.tensor(tname("W", h), "i8", (K, N), "param", h, "row")
        b.tensor(tname("Y", h), "i32", (M, N), "output", h, "row")
    if Kb > 1:
        for h in sides:
            for k in range(Kb):
                b.tensor(pname(k, h), "i32", (M, N), "intermediate", p_hemi(h, k), "row")

    # ---- 权重块分配：双侧时先分侧，再在侧内按 map 分到 plane ----
    per_plane: dict[str, list[tuple[int, int]]] = {p: [] for p in planes}
    j_of: dict[str, int] = {h: 0 for h in sides}                        # 侧内块序号（round_robin 用）
    for n in range(Nb):
        h = side_of_n[n]
        local_n = n if h == "W" else n - split_n
        for k in range(Kb):
            p = plane_for(side_planes[h], map, local_n, k, j_of[h])
            per_plane[p].append((n, k))
            j_of[h] += 1

    plane_of: dict[tuple[int, int], str] = {}
    bank_of: dict[tuple[int, int], int] = {}
    for p in planes:
        for j, nk in enumerate(per_plane[p]):
            plane_of[nk] = p
            bank_of[nk] = bank_for(j, banks)

    # ---- ① 权重装载链：每块 mem.read W[k, n] ─▶ load_w ----
    for p in planes:
        for (n, k) in per_plane[p]:
            rd, lw = f"w.n{n}k{k}.rd", f"w.n{n}k{k}.lw"
            b.read(rd, tile_str(tname("W", side_of_n[n]), k * tile_k, (k + 1) * tile_k, n * tile_n, (n + 1) * tile_n),
                   tile_k, w_out, ["mm"])
            b.load_w(lw, p, bank_of[(n, k)], tile_k, ["mm"])
            b.stream(rd, lw)

    # ---- ② 计算链：每个 (m, n, k) 一条 mem.read X ─▶ acc ─▶ mem.write ----
    def acc_id(m, n, k):
        return f"m{m}n{n}k{k}.acc"

    for n in range(Nb):
        h = side_of_n[n]
        for m in range(Mb):
            for k in range(Kb):
                base = f"m{m}n{n}k{k}"
                rows = tile_m
                b.read(f"{base}.x.rd", tile_str(tname("X", h), m * tile_m, (m + 1) * tile_m, k * tile_k, (k + 1) * tile_k),
                       rows, x_out, ["mm"])
                b.acc(acc_id(m, n, k), plane_of[(n, k)], bank_of[(n, k)], rows, acc_out, ["mm"])
                dst = pname(k, h) if Kb > 1 else tname("Y", h)
                b.write(f"{base}.wr", tile_str(dst, m * tile_m, (m + 1) * tile_m, n * tile_n, (n + 1) * tile_n),
                        rows, ["mm"])
                b.stream(f"{base}.x.rd", acc_id(m, n, k))
                b.stream(acc_id(m, n, k), f"{base}.wr")

    # ---- ③ K 合并（不融合）：读回 Kb 个 partial ─▶ vxm 求和 ─▶ 写 Y ----
    if Kb > 1:
        for n in range(Nb):
            h = side_of_n[n]
            for m in range(Mb):
                base = f"m{m}n{n}"
                for k in range(Kb):
                    rd = f"{base}.p{k}.rd"
                    b.read(rd, tile_str(pname(k, h), m * tile_m, (m + 1) * tile_m, n * tile_n, (n + 1) * tile_n),
                           tile_m, acc_out, ["mm"])
                b.vxm(f"{base}.sum", ["add.i32"] * (Kb - 1), tile_m, acc_out, ["mm"])
                b.write(f"{base}.y.wr", tile_str(tname("Y", h), m * tile_m, (m + 1) * tile_m, n * tile_n, (n + 1) * tile_n),
                        tile_m, ["mm"])
                for k in range(Kb):
                    b.stream(f"{base}.p{k}.rd", f"{base}.sum")
                    b.mem(f"m{m}n{n}k{k}.wr", f"{base}.p{k}.rd")
                b.stream(f"{base}.sum", f"{base}.y.wr")

    # ---- ④ MXM 顺序边：每个 plane 按块顺序，块内 m 升序 ----
    for p in planes:
        blocks = [(f"w.n{n}k{k}.lw", [acc_id(m, n, k) for m in range(Mb)]) for (n, k) in per_plane[p]]
        b.link_plane(blocks, banks)

    return b.build()
