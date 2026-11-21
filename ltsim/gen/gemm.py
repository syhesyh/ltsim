"""用 GraphBuilder 按设计 §5.4 的 matmul 规则展开一个参数化 GEMM：Y = X · W。

覆盖多块换权重的场景（每个 plane 上多个权重块），用来检验 MXM 顺序边
（state / acc_seq / block_switch / lw_after_iw / bank_war）的生成。

范围（有意简化）：
- row 形式（X 为 row 布局流入 MXM，W 装进 MXM），INT8 输入、INT32 输出；
- 所有张量放在绑定 plane 所在的半球；
- K 合并（Kb > 1）不融合：每个 partial 写回 MEM，再由一个 vxm.chain 读回求和后写 Y；
  Kb == 1 时 acc 直接写 Y；
- 不加 bias / 激活，不做 lane 形式与运行时权重。
"""
from __future__ import annotations

from math import ceil

from ..errors import LTSError
from ..graph.tile_graph import Out, TileGraph
from ..machine.model import Machine, default_machine
from .builder import GraphBuilder, bank_for, plane_for, tile_str

def build_gemm(M: int, K: int, N: int, *, tile_m: int, tile_k: int | None = None, tile_n: int | None = None,
               planes: list[str] = ("mxm.w0", "mxm.w1"), map: str = "n_split",
               banks: str = "alternate", name: str | None = None,
               machine: Machine | None = None) -> TileGraph:
    """machine 省略时用仓库自带的 lpu-tile-v0.yaml。块大小上限与每个向量的 lane 数都从它读取；
    tile_k / tile_n 省略时取 MXM 块的大小。"""
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
    planes = list(planes)
    sides = {p.split(".")[1][0].upper() for p in planes}        # mxm.w0 → "W"
    if len(sides) != 1:
        raise LTSError("LTS-E-GEN", f"planes must be on one side, got {planes}")
    hemi = sides.pop()

    Mb, Kb, Nb = M // tile_m, K // tile_k, N // tile_n
    # 每行向量数 V = ceil(lane 数 / 每个向量的 lane 数) × dtype 字节数（设计 §3.2）
    acc_out = Out("i32", ceil(tile_n / lanes) * 4)
    x_out = Out("i8", ceil(tile_k / lanes) * 1)
    w_out = Out("i8", ceil(tile_n / lanes) * 1)

    b = GraphBuilder(name or f"gemm_{M}x{K}x{N}", m.raw["id"])
    b.tensor("X", "i8", (M, K), "input", hemi, "row")
    b.tensor("W", "i8", (K, N), "param", hemi, "row")
    b.tensor("Y", "i32", (M, N), "output", hemi, "row")
    if Kb > 1:
        for k in range(Kb):
            b.tensor(f"P{k}", "i32", (M, N), "intermediate", hemi, "row")

    # ---- 权重块分配：(n, k) 字典序，按 map 分到 plane ----
    per_plane: dict[str, list[tuple[int, int]]] = {p: [] for p in planes}
    for j, (n, k) in enumerate((n, k) for n in range(Nb) for k in range(Kb)):
        per_plane[plane_for(planes, map, n, k, j)].append((n, k))

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
            b.read(rd, tile_str("W", k * tile_k, (k + 1) * tile_k, n * tile_n, (n + 1) * tile_n),
                   tile_k, w_out, ["mm"])
            b.load_w(lw, p, bank_of[(n, k)], tile_k, ["mm"])
            b.stream(rd, lw)

    # ---- ② 计算链：每个 (m, n, k) 一条 mem.read X ─▶ acc ─▶ mem.write ----
    def acc_id(m, n, k):
        return f"m{m}n{n}k{k}.acc"

    for n in range(Nb):
        for m in range(Mb):
            for k in range(Kb):
                base = f"m{m}n{n}k{k}"
                rows = tile_m
                b.read(f"{base}.x.rd", tile_str("X", m * tile_m, (m + 1) * tile_m, k * tile_k, (k + 1) * tile_k),
                       rows, x_out, ["mm"])
                b.acc(acc_id(m, n, k), plane_of[(n, k)], bank_of[(n, k)], rows, acc_out, ["mm"])
                dst = f"P{k}" if Kb > 1 else "Y"
                b.write(f"{base}.wr", tile_str(dst, m * tile_m, (m + 1) * tile_m, n * tile_n, (n + 1) * tile_n),
                        rows, ["mm"])
                b.stream(f"{base}.x.rd", acc_id(m, n, k))
                b.stream(acc_id(m, n, k), f"{base}.wr")

    # ---- ③ K 合并（不融合）：读回 Kb 个 partial ─▶ vxm 求和 ─▶ 写 Y ----
    if Kb > 1:
        for n in range(Nb):
            for m in range(Mb):
                base = f"m{m}n{n}"
                for k in range(Kb):
                    rd = f"{base}.p{k}.rd"
                    b.read(rd, tile_str(f"P{k}", m * tile_m, (m + 1) * tile_m, n * tile_n, (n + 1) * tile_n),
                           tile_m, acc_out, ["mm"])
                b.vxm(f"{base}.sum", ["add.i32"] * (Kb - 1), tile_m, acc_out, ["mm"])
                b.write(f"{base}.y.wr", tile_str("Y", m * tile_m, (m + 1) * tile_m, n * tile_n, (n + 1) * tile_n),
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
