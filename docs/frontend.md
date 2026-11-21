# LTSim 前端开发文档：LTS-IR → tile graph

> 状态：开发文档，代码尚未开始。**本文是前端的权威规格。** 设计文档（下称"设计"；lpu-compiler 仓库的 `Docs/lpu_tile_sim_design.md`，私有，出处见 [README](../README.md#参考资料)）
> 是起点和参考，但不保证正确；本文对它的修改与偏离都在正文里注明。
> 本文回答"具体怎么写代码"：数据结构、每个 pass 的算法、命名规则、测试和开发顺序。

## 0. 前端做什么

```text
lts-ir.yaml ──P1──▶ Program ──P2──▶ ResolvedSchedule ──P3──▶ TileTable ──P4/P5──▶ TileGraph ──P6──▶ tile_graph.json
                                 └─▶ resolved_schedule.json
```

| pass | 输入 | 输出 | 模块 |
|---|---|---|---|
| P1 解析校验 | YAML 文本 | `Program`（算法部分）+ 原始 schedule | `ltsim/ir/parse.py`、`validate.py` |
| P2 补全决策 | `Program` + 原始 schedule | `ResolvedSchedule` | `ltsim/frontend/resolve.py` |
| P3 分块 | `Program` + `ResolvedSchedule` | `TileTable` | `ltsim/frontend/tiling.py` |
| P4 展开 | 以上全部 | 节点与组内边 | `ltsim/frontend/expand/*.py` |
| P5 建边 | P4 的结果 | 组间边（mem / state / order） | `ltsim/frontend/edges.py` |
| P6 合法性与输出 | `TileGraph` | `tile_graph.json` | `ltsim/frontend/legality.py`、`emit.py` |

前端**不知道任何周期数**，只用机器参数里的几何量：320 lane、MXM 块 320×320、plane 名单、半球容量。

## 1. 数据结构

全部用 `@dataclass(frozen=True)`；集合用 `tuple` 或 `Mapping`，每个 pass 返回新对象，不修改输入。

### 1.1 算法部分（P1 产出）

```python
DType = Literal["i8", "i32", "f16", "bf16", "f32"]
BYTES = {"i8": 1, "i32": 4, "f16": 2, "bf16": 2, "f32": 4}

@dataclass(frozen=True)
class TensorSpec:
    name: str
    shape: tuple[int, ...]          # (R, C) 或 (N,)
    dtype: DType
    role: Literal["input", "param", "output", "intermediate"]

@dataclass(frozen=True)
class OpSpec:
    id: str
    op: str                         # 设计 4.2 表中的算子名
    inputs: tuple[str, ...]
    output: str
    attrs: Mapping[str, Any]

@dataclass(frozen=True)
class Program:
    name: str
    machine: str
    tensors: Mapping[str, TensorSpec]   # 含 P1 推导出的中间张量
    ops: tuple[OpSpec, ...]             # 拓扑序
    producer: Mapping[str, str]         # 张量名 → 产生它的 op id
    consumers: Mapping[str, tuple[str, ...]]
```

### 1.2 补全后的决策（P2 产出）

```python
@dataclass(frozen=True)
class TensorPlan:
    layout: Literal["row", "lane"]
    hemisphere: Literal["W", "E"]
    alias_of: str | None = None         # transpose 输出共享映像时指向源张量

@dataclass(frozen=True)
class MatmulPlan:
    form: Literal["row", "lane"]        # 由输出布局决定（设计 4.3 ⓪ 规则 4）
    tile_m: int; tile_k: int; tile_n: int
    planes: tuple[str, ...]             # 例如 ("mxm.w0", "mxm.w1")
    map: Literal["k_split", "n_split", "round_robin"]
    banks: Literal["alternate", "single"]
    order: tuple[tuple[int, int], ...]  # 每个 plane 上 (n, k) 块的装载顺序，已按 plane 展开

@dataclass(frozen=True)
class GroupPlan:
    id: str                             # "g0", "g1", ...（按组内第一个算子的拓扑序编号）
    ops: tuple[str, ...]                # 组内算子，按链顺序
    tile_m: int                         # 组内统一的物理行块大小

@dataclass(frozen=True)
class ResolvedSchedule:
    tensors: Mapping[str, TensorPlan]
    matmuls: Mapping[str, MatmulPlan]
    groups: tuple[GroupPlan, ...]       # 覆盖全部算子，每个算子恰好属于一个组
    group_of: Mapping[str, str]         # op id → group id
```

### 1.3 tile 与物理坐标

```python
@dataclass(frozen=True)
class Tile:
    tensor: str
    r0: int; r1: int; c0: int; c1: int  # 逻辑坐标，左闭右开；rank 1 张量 r0=0, r1=1

    def tag(self) -> str: ...           # "X[0:128,0:320]"，写进 tile graph
```

逻辑坐标到物理坐标由布局决定（设计 3.2）：

```python
def phys(tile: Tile, plan: TensorPlan) -> tuple[int, int]:
    """返回 (rows, lanes)：row 布局取 (r1-r0, c1-c0)；lane 布局取 (c1-c0, r1-r0)。"""

def vectors_per_row(lanes: int, dtype: DType) -> int:
    return ceil(lanes / 320) * BYTES[dtype]     # v0 中 lanes ≤ 320，所以就是 BYTES[dtype]
```

### 1.4 tile graph（P4–P6 产出，对应设计 6.2 的 0.2 格式）

```python
@dataclass(frozen=True)
class Out:
    dtype: DType
    V: int

@dataclass(frozen=True)
class Node:
    id: str
    kind: Literal["mem.read", "mem.write", "mxm.load_w", "mxm.acc", "vxm.chain"]
    rows: int
    origin: tuple[str, ...]
    tile: Tile | None = None            # mem.read / mem.write
    unit: str | None = None             # mxm.*
    bank: int | None = None             # mxm.*
    repeat: int | None = None           # 广播读
    ops: tuple[str, ...] | None = None  # vxm.chain
    out: Out | None = None              # 往流上送数据的节点

@dataclass(frozen=True)
class OrderEdge:
    src: str; dst: str
    ref: Literal["start", "issue_end", "done"]
    lag: int | str
    reason: Literal["acc_seq", "block_switch", "lw_after_iw", "bank_war"]

@dataclass(frozen=True)
class TileGraph:
    tensors: Mapping[str, TensorSpec | TensorPlan]   # 输出时合并成一个对象
    nodes: tuple[Node, ...]                          # 创建顺序 = 输出顺序
    stream: tuple[tuple[str, str, int], ...]         # (src, dst, r0)，r0 默认 0
    state: tuple[tuple[str, str], ...]
    mem: tuple[tuple[str, str], ...]
    order: tuple[OrderEdge, ...]
```

### 1.5 诊断

```python
class LtsError(Exception):
    def __init__(self, code: str, message: str, where: str = ""): ...
    # code 取设计附录 A，例如 "LTS-E-LAYOUT"；where 例如 "ops[3].in[1]" 或 "schedule.fusion[0]"
```

规则：遇到第一个错误就抛出，不尝试继续；CLI 把它打印成一行 JSON，退出码 2。警告（如果以后有）收集到 `TileGraph` 之外的列表里，不影响输出。

## 2. P1 解析与校验

### 2.1 读取

`yaml.safe_load`，再按 `ltsim/ir/schema.json`（JSON Schema）校验结构。schema 只管形状：必填字段、类型、枚举值。
语义检查在下面单独做，这样错误信息能落到具体的 op。

### 2.2 语义检查（按顺序）

1. `schema == "lts-ir/0.1"`，`machine` 能在 `ltsim/machine/` 找到；
2. `ops` 的 `id` 唯一；每个输入张量在此之前已存在（声明的 input/param，或前面某个 op 的输出）——这同时保证了拓扑序；
3. 每个张量最多一个生产者；声明为 `output` 的张量必须由某个 op 产生；
4. 对每个 op 调用推导函数，得到输出的 `TensorSpec`，并检查输入约束（下表）；
5. 没有被任何 op 使用、也不是 output 的中间张量：报 `LTS-E-PARSE-DEAD`（避免写错名字悄悄被忽略）。

### 2.3 推导函数

每个算子一个函数 `infer_<op>(inputs: list[TensorSpec], attrs) -> (shape, dtype)`，放在 `ltsim/ir/ops.py` 的表里：

| 算子 | 检查 | 输出 |
|---|---|---|
| `matmul` | 两个都是 rank 2、i8；`x.shape[1] == w.shape[0]` | `(x.shape[0], w.shape[1])`, i32 |
| `add` / `mul` | dtype 相同；第二个与第一个同形，或是 rank 1 且长度 = 第一个的列数 | 同第一个 |
| `mul_scalar` | 有 `attrs.value` | 同输入 |
| `relu` / `gelu` | gelu 要求 f32 | 同输入 |
| `dequantize` | i8 或 i32，有 `scale > 0` | 同形，f32 |
| `quantize` | f32，有 `scale > 0` | 同形，i8 |
| `requant` | i32，有 `scale > 0` | 同形，i8 |
| `layer_norm` | x 为 f32 rank 2；gamma、beta 为 param、rank 1、长度 = 列数；有 `eps` | 同 x |
| `softmax` | x 为 f32 rank 2 | 同 x |
| `transpose` | rank 2，`perm == [1, 0]` | `(C, R)`，同 dtype |

不满足时报 `LTS-E-OP-<算子大写>`，例如 `LTS-E-OP-MATMUL`，message 写清楚哪个维度对不上。

## 3. P2 补全决策

严格按设计 4.3 的顺序 ⓪ 布局 → ① 分块 → ② 绑定 → ③ 半球 → ④ 权重 → ⑤ 融合。
**例外：** 融合组要最先校验并建好，因为布局分量和分块传播都要用到它。实际代码顺序：

```text
resolve_fusion(⑤ 的校验与建组) → resolve_layout(⓪) → resolve_tiling(①)
  → resolve_binding(②) → resolve_hemisphere(③) → resolve_weights(④) → check_capacity
```

### 3.1 融合组（⑤）

1. 读 `schedule.fusion`，每个组按给出的顺序；检查：
   - 组内第 i+1 个算子的某个输入是第 i 个算子的输出（链状）；
   - 每个算子最多出现在一个组；
   - matmul 至多一个且在组首（否则 `LTS-E-FUSE-*`）；归约至多一个（`LTS-E-FUSE-REDUCE`）。
2. 没出现在任何组里的算子各自成单元素组；
3. 组按"组内第一个算子在 `ops` 里的位置"排序，依次编号 `g0, g1, …`。

### 3.2 布局（⓪）

用并查集把"必须同布局"的张量连成分量：

```python
uf = UnionFind(all tensors)
for op in ops:
    if op.op in ELEMENTWISE | {"layer_norm", "softmax"}:
        uf.union(op.inputs[0], op.output)
        for t in op.inputs[1:]:
            if rank(t) == 2: uf.union(t, op.output)    # rank 1 广播参数不连
```

然后对每个分量决定布局，按优先级：

1. 分量里有归约的输入或输出 → `lane`；分量里有显式给出 `row` 的张量则报 `LTS-E-LAYOUT`；
2. 分量里有显式给出的布局 → 用它（多个不一致时报 `LTS-E-LAYOUT`）；
3. 否则暂不定，等第 4 步。

接着按拓扑序处理 `transpose` 和 `matmul`，处理到哪个就定哪个：

```python
for op in ops:                                    # 拓扑序
    if op.op == "transpose":
        src = layout(op.inputs[0]) or "row"       # 输入还没定就取 row
        set_layout(op.output, flip(src)); alias(op.output, op.inputs[0])
    elif op.op == "matmul":
        x, w, y = op.inputs[0], op.inputs[1], op.output
        y_layout = layout(y) or ("row" if can_be(x, "row") else "lane")
        stream_operand = x if y_layout == "row" else w
        require(stream_operand, y_layout)          # 已定且不同 → LTS-E-LAYOUT；未定且是 input/param → 设上
        prefer(weight_operand, y_layout)           # 未定就设成 y_layout，已定不同也接受（权重通路调整，乐观）
        set_layout(y, y_layout)
```

`set_layout` 作用于整个分量；最后仍未定的分量一律 `row`。
graph input / param 被两个消费者要求不同布局时，生成第二份 `TensorPlan`（名字加 `@lane` / `@row` 后缀），并在容量检查里算两份（设计 4.3 ⓪ 规则 5）。
lane 布局的张量检查逻辑第一维 ≤ 320，否则 `LTS-E-LAYOUT-WIDTH`。

### 3.3 分块（①）

每个组只有一个 `tile_m`，指组的**物理行**块大小：

1. 收集组内所有显式的 `tile_m`（matmul 的或逐元素算子的），必须相等（`LTS-E-FUSE-TILE`）；
2. 没有显式值时取组内输出张量的全部物理行数；
3. 组内有归约时，`tile_m` 必须等于全部物理行数（`LTS-E-REDUCE-TILE`）；
4. matmul 的 `tile_k`、`tile_n` 缺省为 320，超过 320 报 `LTS-E-MXM-BLOCK`。

### 3.4 绑定、半球、权重（②③④）

- **绑定：** 缺省时，权重张量显式放在 E 用 `("mxm.e0", "mxm.e1")`，否则 `("mxm.w0", "mxm.w1")`；`map` 缺省 `k_split`。
  按 `map` 把每个 `(n, k)` 块分到一个 plane，得到每个 plane 的块列表。
- **半球：** 显式值优先；matmul 的 x、w 取绑定 plane 的一侧；其余张量取 W。transpose 的输出跟随源张量。
- **权重：** `banks` 缺省 `alternate`；`order` 缺省按 `(n, k)` 字典序；按 plane 拆开，第 j 个块的 bank = `j % 2`（alternate）或 0（single）。

### 3.5 容量检查

```text
每个半球：Σ ceil(该张量的向量总数) ≤ 44 × 8192
张量的向量总数 = 物理行数 × vectors_per_row(物理 lane 数, dtype)
```

transpose 的别名张量不重复计算；按需复制的 input/param 算两份。超出报 `LTS-E-MEM-FULL`，message 列出该半球最大的几个张量。

### 3.6 输出 `resolved_schedule.json`

把 `ResolvedSchedule` 原样写出，键排序固定。以 MLP（设计 4.4）为例：

```json
{
  "tensors": {"X": {"layout": "row", "hemisphere": "W"}, "H": {"layout": "row", "hemisphere": "E"}, "...": {}},
  "matmuls": {
    "fc1": {"form": "row", "tile_m": 128, "tile_k": 320, "tile_n": 320,
            "planes": ["mxm.w0", "mxm.w1"], "map": "n_split", "banks": "alternate",
            "order": {"mxm.w0": [[0, 0]], "mxm.w1": [[1, 0]]}},
    "fc2": {"form": "row", "...": "...", "order": {"mxm.e0": [[0, 0]], "mxm.e1": [[0, 1]]}}
  },
  "groups": [{"id": "g0", "ops": ["fc1", "dq1", "b1", "act", "q1"], "tile_m": 128},
             {"id": "g1", "ops": ["fc2", "b2", "y"], "tile_m": 128}]
}
```

## 4. P3 分块：tile 表

对每个组算出它的**位置集合**，位置就是组输出 tile 的坐标：

```python
def positions(group) -> list[Position]:
    # Position = (mi, ni)：物理行块序号、lane 块序号
    rows, lanes = phys(output tensor 的整体 tile, plan)
    return [(mi, ni) for mi in range(ceil(rows / tile_m)) for ni in range(ceil(lanes / 320))]
```

- matmul 组：lane 块就是 N 块（`tile_n`）；另外还有 K 块序号 `ki in range(ceil(K / tile_k))`；
- 逐元素组：lane 块按 320 切，与生产者一致（P2 已保证组内一致）；
- 张量上的任意物理区域可以反算回逻辑 `Tile`，写进节点的 `tile` 字段。

`TileTable` 就是 `group id → [Position]` 加上一个 `region(tensor, mi, ni[, ki]) -> Tile` 的函数，不必预先存所有 tile。

## 5. P4 展开

### 5.1 节点命名规则

命名必须确定，同一输入两次运行得到相同 id：

| 节点 | id |
|---|---|
| 组内节点 | `<组 id>.<位置>.<角色>`，例如 `g0.m0n0.acc`、`g0.m0n0.x.rd`、`g0.m0n0.vxm` |
| 权重装载链 | `<matmul id>.w.<plane>.<块>`，例如 `fc1.w.w0.n0k0.rd`、`fc1.w.w0.n0k0.lw` |
| 对半树第 j 层 | `<组 id>.<位置>.t<树号>.L<j>.<角色>`，例如 `g1.m0n0.t0.L2.vxm` |

角色名：`<张量>.rd`（读某个张量）、`acc`、`acc.k<ki>`、`vxm`、`<张量>.wr`。
设计 6.3 的例子为了可读性手工起了短名（`a0.acc`、`b.vxm`），生成器的输出与它**同构**，id 不同；M1 的金标准测试按同构比较（见第 8 节）。

### 5.2 组的展开：一个"待发 VXM"累加器

每个组在每个位置上从头到尾走一遍组内算子，用一个累加器收集还没落成节点的 VXM 操作：

```python
@dataclass
class PendingVxm:
    ops: list[str]                          # 已收集的 ALU 操作
    inputs: list[tuple[str, int]]           # 流入的 (节点 id, 起始行 r0)
    origin: list[str]

def expand_group(group, pos, b: GraphBuilder):
    cur = None                              # 当前"流上的数据"由哪个节点产生
    pend = PendingVxm([], [], [])
    for op in group.ops:
        if op.op == "matmul":               # 只可能在组首
            cur, pend = expand_matmul(op, pos, b)        # 见 5.3；K 合并会把加法放进 pend
        elif op.op in ELEMENTWISE:
            if cur is None and not pend.inputs:          # 组首就是逐元素算子：从 MEM 读第一个操作数
                cur = b.read(op.inputs[0], pos)
            if cur is not None:
                pend.inputs.append((cur, 0)); cur = None
            for t in op.inputs[1:]:                      # 第二操作数：广播参数或另一个张量，都从 MEM 读
                pend.inputs.append((b.read(t, pos, broadcast=(rank(t) == 1)), 0))
            pend.ops += ALU_OPS[op.op](op)               # 例如 requant → ["cvt", "mul.f32", "cvt"]
            pend.origin.append(op.id)
        elif op.op in ("layer_norm", "softmax"):
            cur = flush(pend, b)                         # 先把前面的 VXM 操作落成节点
            cur, pend = expand_reduction(op, pos, b, upstream=cur)   # 见 5.5
        elif op.op == "transpose":
            pass                                         # 不生成节点（5.6）
    cur = flush(pend, b) or cur
    if needs_mem(group.ops[-1].output):                  # graph output，或有组外消费者
        b.write(group.ops[-1].output, pos, src=cur)
```

`flush` 把累加器落成一个 `vxm.chain` 节点：`ops = pend.ops`，`rows` 取位置的物理行数，`out` 取最后一个 ALU 操作的输出 dtype；
再给 `pend.inputs` 里每一项加一条 `stream` 边 `(输入节点, 新节点, r0)`。累加器为空时返回 `None`。

`b.read(t, pos, broadcast)` 生成 `mem.read`：

- `tile = region(t, pos)`；
- `rows` 取物理行数；广播时 `rows` 取消费者的行数，`repeat = rows`；
- `out = Out(dtype(t), vectors_per_row(物理 lane, dtype))`。

`b.write(t, pos, src)` 生成 `mem.write`，并加 `stream` 边 `(src, 写节点, 0)`。

### 5.3 matmul

```python
def expand_matmul(op, pos, b):
    plan = matmuls[op.id]
    S, Wt = (x, w) if plan.form == "row" else (w_transposed_view, x)     # 设计 5.4
    for (plane, j, (n, k), bank) in plan.blocks():          # 权重链：每个块只建一次，与位置无关
        b.once(f"{op.id}.w.{plane}.n{n}k{k}", lambda:
               b.chain(read(Wt, block(n, k)) -> load_w(plane, bank)))
    mi, ni = pos
    accs = []
    for k in range(Kb):
        rd = b.read(S, region_S(mi, k))
        acc = b.node("mxm.acc", unit=plane_of(n=ni, k=k), bank=bank_of(ni, k), rows=rows(mi),
                     out=Out("i32", 4), origin=[op.id])
        b.stream(rd, acc); accs.append(acc)
    if Kb == 1:
        return accs[0], PendingVxm([], [], [])
    if group_is_fused_beyond(op):                           # K 合并并入后继 VXM
        return None, PendingVxm(["add.i32"] * (Kb - 1), [(a, 0) for a in accs], [op.id])
    # 不融合：每个 partial 写回 MEM，再建一个单独的合并组（id 为 "<op>.kmerge"）
    ...
```

- `plane_of`、`bank_of` 来自 P2 的块分配；融合的 K 合并要求同一 `(mi, ni)` 的 Kb 个块在不同 plane 上（`LTS-E-FUSE-KMERGE`），在这里检查；
- 权重链的建立与位置无关，用 `b.once` 去重；
- 运行时权重（Wt 是前面算出的张量）不在这里处理依赖，P5 按区域重叠给权重链的读节点补 `mem` 边。

### 5.4 逐元素算子的 ALU 操作表

| 算子 | ALU 操作（设计 3.4 `op_latency` 的键） |
|---|---|
| `add` | `add.i32` / `add.f32`（按 dtype） |
| `mul` / `mul_scalar` | `mul.i32` / `mul.f32` |
| `relu` | `relu` |
| `gelu` | `gelu.f32` |
| `dequantize` | `cvt`, `mul.f32` |
| `quantize` | `mul.f32`, `cvt` |
| `requant` | `cvt`, `mul.f32`, `cvt` |

### 5.5 归约（layer_norm / softmax）

按设计 5.4 的规则展开，`n` 是物理行数（word 数）：

```python
def tree_sum(src_tensor, pos, pre, b, upstream=None):
    """返回最后一层写出的 1-word 张量（临时张量，名字 '<组>.<位置>.t<树号>.L<j>'）。"""
    levels = halving_levels(n)                  # 与 compiler/reference/lane_numeric.py 同一算法
    prev = src_tensor
    for j, (n_j, h_j) in enumerate(levels, start=1):
        low  = b.read(prev, rows=[0, h_j])
        if j == 1 and upstream is not None:     # 第 1 层与上游融合（设计 5.4）
            high_inputs = [(upstream, h_j)]     # stream 边带起始行 h_j
            b.same_chain_mem(writer_of(src_tensor), low)   # 同链 mem 边，B2 按逐行检查
        else:
            high_inputs = [(b.read(prev, rows=[h_j, n_j]), 0)]
        v = b.vxm(ops=(pre if j == 1 else []) * 2 + ["add.f32"], inputs=[(low, 0), *high_inputs])
        prev = b.write_temp(v, rows=h_j)        # 每层写回 MEM；层与层之间的 mem 边由 P5 按区域补
    return prev
```

- softmax：`S = tree_sum(x, ["exp.f32"])`；OUT 段：`read x`、`read S`（广播，`repeat = n`）→ 累加器
  `["exp.f32", "rcp.f32", "mul.f32"]`，返回给 `expand_group` 继续接后面的逐元素算子；
- layer_norm：`S1 = tree_sum(x, [])`；`S2 = tree_sum(x, ["mul.f32", "sub.f32", "mul.f32"])`，第 1 层另读 S1（广播）；
  OUT 段累加器为 8 个操作（设计 5.4）；
- 只有第一棵树可以带 `upstream`；S2 的第 1 层永远从 MEM 读；
- 组内第一个算子就是归约时，`upstream=None`，第 1 层从 MEM 读两半；
- 第 1 层的 `pre` 对两半各做一次，所以 ALU 操作是 `pre * 2 + [add]`。

### 5.6 transpose

不生成节点。P2 已把输出标成源张量的别名；之后 `b.read(Kt, ...)` 实际读的是源张量的映像，
`tile` 字段写成别名名字下的逻辑坐标，`tensors` 表里记 `alias_of`。后端按别名找到半球，不需要别的处理。

## 6. P5 建边

P4 已经生成了全部 `stream` 边。P5 补三类组间约束。

### 6.1 `mem` 边：按区域重叠

```python
writes = defaultdict(list)                 # 物理张量名 → [(写节点, 物理行区间, 物理 lane 区间)]
for n in nodes:
    if n.kind == "mem.write": writes[physical(n.tile.tensor)].append((n.id, prow(n), plane_range(n)))
for n in nodes:
    if n.kind == "mem.read":
        for (w, rr, ll) in writes[physical(n.tile.tensor)]:
            if overlaps(rr, prow(n)) and overlaps(ll, plane_range(n)) and w != same_chain_writer(n):
                mem.append((w, n.id))
```

- 比较用**物理**坐标，别名先解析到源张量；
- graph input / param 没有写节点，读它们不产生边；
- 第 1 层融合时的"同链 mem 边"在 P4 已加，这里跳过同一对；
- 一个读依赖多个写是正常的（分块不同的两组）；结果按 `(src, dst)` 排序后输出。

### 6.2 `state` 与 `order` 边：按每个 plane 的块序列

```python
for plane, blocks in plan.per_plane():                    # blocks 按装载顺序
    for j, blk in enumerate(blocks):
        accs = acc_nodes(plane, blk)                       # 该块的所有 acc，按 m 升序
        state.append((lw_node(blk), accs[0]))
        for a, b2 in pairwise(accs):   order.append(OrderEdge(a, b2, "issue_end", 0, "acc_seq"))
        if j > 0:
            prev_accs = acc_nodes(plane, blocks[j - 1])
            order.append(OrderEdge(prev_accs[-1], accs[0], "issue_end", "mxm.block_switch_gap", "block_switch"))
            order.append(OrderEdge(prev_accs[0], lw_node(blk), "start", -1, "lw_after_iw"))
            if banks == "single":
                order.append(OrderEdge(prev_accs[-1], lw_node(blk), "issue_end", 0, "bank_war"))
```

`state` 边只连到块的**第一个** acc；后面的 acc 由 `acc_seq` 保证在其后（设计 5.5）。

## 7. P6 合法性与输出

### 7.1 检查

P1、P2 已经查过大部分；P6 只查只有在展开后才看得到的：

| 检查 | 诊断码 |
|---|---|
| 每个 `stream` 边的起点都有 `out` | 内部错误（`LTS-E-INTERNAL`），说明生成器有 bug |
| 节点 id 唯一；边的两端都存在 | 内部错误 |
| matmul 块不超过 320×320 | `LTS-E-MXM-BLOCK` |
| MXM 只接受 i8 输入 | `LTS-E-UNIT-OP` |
| 融合 K 合并的块在不同 plane | `LTS-E-FUSE-KMERGE` |

### 7.2 输出

- `tensors` 按名字排序，每项合并 `TensorSpec` 和 `TensorPlan`；
- `nodes` 按创建顺序；
- `edges.stream` 写成 `[src, dst]`，r0 非 0 时写 `[src, dst, r0]`；`state`、`mem` 写 `[src, dst]`；`order` 写对象；
- 各组边按 `(src, dst)` 排序；
- `json.dumps(..., ensure_ascii=False, indent=2)`；同一输入两次运行字节相同（测试要求）。

## 8. 用 MLP 走一遍（设计 4.4 → 6.3）

| 步骤 | 结果 |
|---|---|
| P1 | 推导出 A1 `[128,640] i32`、F1/F2/F3 `f32`、H `[128,640] i8`、A2 `[128,320] i32`、A3 `i32` |
| P2 融合 | `g0 = [fc1, dq1, b1, act, q1]`，`g1 = [fc2, b2, y]` |
| P2 布局 | 所有分量没有归约、没有显式值，fc1、fc2 输出先试 row 成功 → 全部 row |
| P2 分块 | 两组 `tile_m = 128`（全部行） |
| P2 绑定 | fc1：N=640 两块，`n_split` → n0→w0、n1→w1；fc2：K=640 两块，`k_split` → k0→e0、k1→e1 |
| P2 半球 | X、W1 → W（fc1 在西）；H、W2 → E（fc2 在东）；B1、B2、Y → W |
| P3 | g0 位置 `(m0, n0)`、`(m0, n1)`；g1 位置 `(m0, n0)`，Kb = 2 |
| P4 g0 | 每个位置：读 X → acc → 累加器收 dq1、b1（另读 B1 广播）、act、q1 → flush 成 6 个操作的 vxm → H 有组外消费者，写 H |
| P4 g1 | 读 H 两个 K 块 → 两个 acc → 累加器先放 1 个 `add.i32`（K 合并），再收 b2（读 B2 广播）、y → flush → 写 Y |
| P5 mem | g1 读 `H[0:128,0:320]` 与 g0 写的同一区域重叠 → 1 条边；另一块同理 |
| P5 state | 4 个块各一条 |
| 结果 | 25 个节点，stream 18、state 4、mem 2、order 0，与设计 6.3 同构 |

**同构比较的做法（测试用）：** 把每个节点映射成 `(kind, unit, bank, tile, rows, ops, out, repeat)`，
再把边映射成两端的签名，比较两边的多重集合。这样 id 的命名差异不影响结果。

## 9. 测试

| 文件 | 内容 |
|---|---|
| `tests/ltsim/test_parse.py` | 每个算子的推导；每种 `LTS-E-PARSE-*` / `LTS-E-OP-*` 至少一个反例 |
| `tests/ltsim/test_resolve_layout.py` | MLP 全 row；attention（设计 4.5）的布局表逐项一致；transpose 两端同布局报错；归约分量强制 row 报错 |
| `tests/ltsim/test_resolve_misc.py` | 融合组校验；`tile_m` 传播与冲突；绑定与块分配；容量超限 |
| `tests/ltsim/test_expand.py` | 单个 matmul（Kb=1、Kb=2 融合 / 不融合）；逐元素链合并成一个 vxm；广播读的 `repeat` |
| `tests/ltsim/test_reduction.py` | `halving_levels` 与 `compiler/reference/lane_numeric.py` 结果一致；softmax 节点数；第 1 层融合产生带 r0 的 stream 边和同链 mem 边 |
| `tests/ltsim/test_edges.py` | 区域重叠：分块相同一对一、分块不同一对多；`acc_seq` / `block_switch` / `lw_after_iw` / `bank_war` 的生成条件 |
| `tests/ltsim/test_frontend_golden.py` | MLP 与 6.3 同构；attention 展开为 12 条链、47 个节点；MLP `tile_m: 32` 变体 16 条链、76 个节点；两次运行字节相同 |

链数用一个小工具函数按 `stream` 边求连通分量来数（后端 B2 也用同一个函数）。

## 10. 开发顺序

每一步都有可运行的测试再进行下一步：

1. **数据结构与 P1：** `ir/types.py`、`ir/ops.py`、`ir/parse.py`。完成标准：MLP 和 attention 的 YAML 都能解析，推导结果正确。
2. **P2 融合与布局：** 先做最难的布局求解。完成标准：attention 的布局表与设计 4.5 一致。
3. **P2 其余与 P3：** 分块、绑定、半球、权重、容量；位置集合。
4. **P4 matmul 与逐元素：** 先不管归约。完成标准：MLP 展开后与 6.3 同构。
5. **P5：** 区域重叠和 MXM 顺序边。完成标准：MLP `tile_m: 32` 变体的 16 条链、76 个节点，mem 边一对一。
6. **P4 归约：** 对半树、layer_norm、softmax、第 1 层融合。完成标准：attention 12 条链、47 个节点。
7. **P6 与 CLI：** `python -m ltsim lower`，输出 `resolved_schedule.json` 与 `tile_graph.json`。

## 11. 开工前要确认的点

- **`tile_n` 对 lane 形式 matmul 的含义：** 设计约定 `tile_m`、`tile_n` 分别切物理行和输出 lane。
  lane 形式下它们对应逻辑的 N 和 M，P2 要按这个换算；写测试时以 attention 的 `s`、`o` 为例逐一核对。
- **广播参数在 lane 布局下：** 设计规定按 word 复制到每个 lane，由主机离线准备。
  实现上把它当作一个物理形状为 `(C, R)` 的 param 读，`repeat` 为 1；不是 row 布局下的 `repeat = rows`。
- **非融合 K 合并的合并组：** 设计只写了"再由一个单独的 vxm.chain 读回求和"。
  实现时它作为一个合成组（id `<op>.kmerge`），参与 P5 的区域重叠建边，和普通组一样处理。
