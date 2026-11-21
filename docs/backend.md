# LTSim 后端开发文档：tile graph → 估计结果

> 状态：开发文档，B0–B4、Perfetto trace 与 CLI 已实现（B3 的 random 策略推迟）。**本文是后端的权威规格。** 设计文档（下称"设计"；lpu-compiler 仓库的 `Docs/lpu_tile_sim_design.md`，私有，出处见 [README](../README.md#参考资料)）
> 第 3、6、7、8 节是起点和参考，但不保证正确；本文对它的修改与偏离都在正文里注明。
> 本文回答"具体怎么写代码"：数据结构、每一步的算法、一个可以手算核对的例子、测试和开发顺序。

## 0. 后端做什么

```text
tile_graph.json ┐
                ├─▶ B0 加载校验 ─▶ B1 节点工作量 ─▶ B2 组链 ─▶ B3 调度 ─▶ B4 指标 ─▶ result.json
lpu-tile-v0.yaml┘                                                                  └─▶ trace.json
```

| 步骤 | 做什么 | 模块 |
|---|---|---|
| B0 | 读机器参数和 tile graph，校验字段、引用、一致性与 lane 宽，解析 `lag` 里的参数名 | `ltsim/machine/model.py`、`ltsim/graph/load.py` |
| B1 | 每个节点的速率上限 `cap`、首出延迟 `L`、尾巴 `skew`；每条 stream 边的 hop | `ltsim/backend/work.py` |
| B2 | 按 stream 边组链；链速率、占用时长、链内偏移、预留表、链级合法性；链间约束图 | `ltsim/backend/chains.py` |
| B3 | 资源日历；串行生成法（SSGS）给每条链定起始时刻 | `ltsim/backend/calendar.py`、`scheduler.py` |
| B4 | 下界、利用率、关键路径；写 `result.json` 和 Perfetto `trace.json` | `ltsim/backend/bounds.py`、`metrics.py`、`ltsim/trace/perfetto.py` |

后端**不知道算子语义**，只认节点类型、行数、`out` 和边。改硬件参数只需改 `lpu-tile-v0.yaml`，不用重跑前端。

## 1. 数据结构

全部 `@dataclass(frozen=True)`；时间一律用整数拍。

```python
@dataclass(frozen=True)
class Machine:
    raw: Mapping[str, Any]                      # yaml 原文
    def param(self, path: str) -> int: ...      # "mxm.lw_to_acc_gap" → 1；找不到抛 LTS-E-MACHINE
    def hop(self, src_cat: str, dst_cat: str) -> int: ...   # hops["MEM_W>MXM_W"]
    def op_latency(self, alu_op: str) -> int: ...           # vxm.op_latency["gelu.f32"]

@dataclass(frozen=True)
class NodeWork:                                 # B1
    cap: Fraction                               # 每拍最多几行；用 Fraction 避免浮点误差
    L: int
    skew: int
    category: str                               # "MEM_W" / "MEM_E" / "MXM_W" / "MXM_E" / "VXM"

@dataclass(frozen=True)
class Reservation:                              # 相对链起点的占用
    resource: str                               # "mxm.w0.compute"、"vxm.alu"、"srf.E2W.W"、"mem.W.rd" ...
    start: int
    end: int                                    # 左闭右开
    amount: int                                 # 独占资源恒为 1
    node: str                                   # 由哪个节点产生（trace 和 bound_by 用）

@dataclass(frozen=True)
class Chain:                                    # B2
    id: str                                     # "c0", "c1", ...（见 4.1）
    nodes: tuple[str, ...]
    rate: Fraction                              # ρ
    D: Mapping[str, int]
    offset: Mapping[str, int]                   # o_n
    length: int                                 # len(c)
    reservations: tuple[Reservation, ...]

@dataclass(frozen=True)
class ChainEdge:                                # 链间约束 T_dst ≥ T_src + weight
    src: str; dst: str; weight: int
    edge: tuple[str, str, str]                  # 来源边 (kind, src 节点, dst 节点)，用于 bound_by
    ref: str = "done"                           # 来源边的 ref；trace 按它放流向事件的起点
    reason: str | None = None                   # order 边的 reason；state / mem 为 None

@dataclass(frozen=True)
class ChainGraph:                               # B2 的输出
    chains: tuple[Chain, ...]
    edges: tuple[ChainEdge, ...]                # 全部链间边，不去重（见 4.6）
    chain_of: Mapping[str, str]                 # 节点 id → 链 id
    works: Mapping[str, NodeWork]               # B1 结果，供 B3/B4 使用
    resources: Mapping[str, Resource]           # 组链所用的资源表（含按选项划分的 SRF 流池，见 4.4.1）；
                                                # B3、B4、trace、oracle 只用它，不再各自调 m.resources()

@dataclass(frozen=True)
class Placed:                                   # B3 的结果
    policy: str
    start: Mapping[str, int]                    # 链 id → T_c
    bound_by: Mapping[str, BoundBy]
    order: tuple[str, ...]                      # 放置顺序
    total: int                                  # max(T_c + len(c))

@dataclass(frozen=True)
class BoundBy:
    kind: Literal["source", "edge", "resource", "policy"]
    edge: tuple[str, str, str] | None = None    # edge：取到 est 的那条链间边
    resource: str | None = None                 # resource：最后一次冲突的资源
    blocker: str | None = None                  # 卡住它的链：edge 为边的源链；resource 为占着资源的链；policy 为前一条已放的链
```

## 2. B0 加载与校验

**机器参数**（`load_machine`）：

1. `yaml.safe_load`；检查 `schema == "lts-machine/0.1"`，必需的节齐全；推导量与源参数一致：
   `geometry.vector_bytes == superlanes × lanes_per_superlane`，`srf.wave_skew_cycles == superlanes − 1`。

**tile graph**（`load_graph` 做结构部分，`validate_graph(g, m)` 做完整的 B0）：

2. **字段与类型**（`LTS-E-GRAPH-FIELD`）。在 `load.py` 里手写校验，不在运行时读取 `ltsim/graph/schema.json`：它只作文档，
   而约定只依赖标准库与 PyYAML，没有 jsonschema。
   - `schema == "lts-tile-graph/0.2"`；`machine` 是非空字符串，`source` 是字符串；
   - tensors 是对象、nodes 是列表、edges 是对象；edges 只有 stream / state / mem / order 四组；
   - stream 边 2 或 3 个元素，state / mem 边恰好 2 个；
   - 张量、节点、`out`、order 边上出现未知键报错；
   - 取值：dtype / role / hemisphere / layout / ref / reason 是规定的枚举；shape 是 1–2 个正整数；
   - 类型：id / kind / tile / unit / alias_of / 边端点是字符串；rows、repeat 是正整数，bank 是非负整数，r0 是非负整数；
     out 是 `{dtype, V}`；ops 是非空字符串列表；origin 是字符串列表（必需）。字符串不能当列表用，例如 `ops: "cvt"` 报错；
   - **每种 kind 允许的字段（白名单）**。公共字段 id / kind / rows / origin 每种都必需；除此之外：

     | kind | 必需 | 可选 |
     |---|---|---|
     | `mem.read` | tile、out | repeat |
     | `mem.write` | tile | — |
     | `mxm.load_w` | unit、bank | — |
     | `mxm.acc` | unit、bank、out | — |
     | `vxm.chain` | ops、out | — |
     | `sxm.*` | unit | out（v0 仅预留，B1 报 `LTS-E-GRAPH-KIND`） |

     表外的字段一律报错，例如 acc 带 tile、读节点带 bank。

   缺字段、类型不对都报这个码，不会抛 `KeyError` / `TypeError`（`test_fuzz_never_crashes` 守着这一点）。
3. **引用**：节点 id 唯一（`LTS-E-GRAPH-DUP`）；所有边的两端都存在（`LTS-E-GRAPH-REF`）；
   每条 stream 边的起点有 `out`，`r0` 小于起点的 rows；每个 mem.write 恰好有一个 stream 前驱（`LTS-E-GRAPH-STREAM`）；
4. **alias**（transpose 的输出，设计 §4.3 ⓪ 规则 3）：`alias_of` 指向存在的张量、不成环；
   alias 必须是 rank 2，形状是源张量的转置，布局相反，dtype 与半球相同（`LTS-E-GRAPH-REF`）。
   所以按 alias 取半球和按源张量取半球结果相同；
5. **tile**：格式合法、维数与张量一致、坐标落在张量形状内（`LTS-E-GRAPH-TILE`）；
6. **一致性**（防止生成器写错）：
   - 节点的 `rows` 等于 tile 的物理 word 数；广播读要求 `rows == repeat` 且 tile 只有 1 个 word（`LTS-E-GRAPH-ROWS`）。
     广播读的含义见设计 §5.4 通用规则 2：每拍重读同一行，`rows` 等于消费者的行数。所以 `repeat` 不影响 B1 的 cap；
   - `mem` 边必须从 `mem.write` 指向 `mem.read`，两端在同一份 MEM 映像上（展开 alias 后是同一个张量），物理区域相交（`LTS-E-GRAPH-MEM`）；
   - `state` 边必须从 `mxm.load_w` 指向 `mxm.acc`，且两者的 unit、bank 相同（`LTS-E-GRAPH-STATE`）；

**需要机器参数的检查**（只在 `validate_graph` 里做）：

7. `mxm.*` 节点的 `unit` 在机器参数的 plane 名单里（`LTS-E-GRAPH-UNIT`）；
8. **lane 宽限制**：每个 tile 的物理 lane 数不超过 `superlanes × lanes_per_superlane`（当前 320），否则报 `LTS-E-GRAPH-LANES`。
   这条限制保证 B1 的 VXM `groups = 1`、每行向量数 V 只由 dtype 字节数决定。
   与设计的偏离：设计把它放在前端（§3.2 的规则、§5.6 只检查 lane 布局的 `LTS-E-LAYOUT-WIDTH`）；
   这里放在 B0，对手写图和 builder 生成的图同样生效。

`lag` 是字符串时，由 `resolve_lags` 用 `machine.param` 解析成整数（B2 调用）。

**物理区域**由 `tile_region` 按 tile 所指张量**自己的**布局换算（alias 有自己的布局，例如 Kt 是 K 的 lane 版本）：

- row 布局：word 是逻辑第一维，lane 是第二维；lane 布局相反；
- rank 1 的 row 张量是 1 个 word，lane 区间是它的跨度；
- rank 1 的 lane 张量每个元素一个 word，lane 区间不确定。这种张量不能做 alias（第 4 条），所以不会出现前后不一致的映像。

`load_graph` 自动做第 2–6 项；`builder.build()` 只查引用。内存里搭的图（builder、DSL）进后端前调 `validate_graph(g, m)`，
它做第 2–8 项（重复调用无副作用）。

任何一项不满足：报 `LTS-E-GRAPH-*`（结构）或 `LTS-E-MACHINE`（参数缺失），退出码 2。

## 3. B1 节点工作量

### 3.1 节点所在位置（category）

| 节点 | category |
|---|---|
| `mem.read` / `mem.write` | 由 `tile` 的张量查 `tensors[...].hemisphere`：`MEM_W` 或 `MEM_E` |
| `mxm.*` | 由 `unit` 查机器参数的 `side`：`mxm.w*` → `MXM_W`，`mxm.e*` → `MXM_E` |
| `vxm.chain` | `VXM` |

**方向**由两个 category 的 x 顺序决定：`MXM_W < MEM_W < VXM < MEM_E < MXM_E`。
dst 在 src 东边就是 `W2E`，否则 `E2W`；同一 category 之间不会出现 stream 边（出现即报 `LTS-E-GRAPH-HOP`）。

### 3.2 公式（设计 3.6）

```python
def work(node, m: Machine) -> NodeWork:
    k = node.kind
    if k == "mem.read":   cap, L, skew = F(m.param("mem.read_vectors_per_cycle"), node.out.V), m.param("mem.read_latency"), 0
    if k == "mem.write":  cap, L, skew = F(m.param("mem.write_vectors_per_cycle"), V_in(node)), m.param("mem.write_latency"), m.param("srf.wave_skew_cycles")
    if k == "mxm.load_w": cap, L, skew = F(m.param(f"mxm.lw_rows_per_cycle.{dt_in(node)}")), 0, m.param("srf.wave_skew_cycles")
    if k == "mxm.acc":    cap, L, skew = F(1, m.param("mxm.acc_ii")), m.param(f"mxm.acc_latency.{dt_in(node)}"), 0
    if k == "vxm.chain":
        d = len(node.ops)
        cap = min(F(1), F(m.param("vxm.alus"), d))          # v0 中 groups = 1
        L = sum(m.op_latency(o) + m.param("vxm.op_overhead") for o in node.ops)
        skew = 0
```

- `V_in(node)` 是写节点收到的流的 V，等于它的 stream 前驱的 `out.V`；由公开函数 `write_v_in(g)` 给出，B2 算写带宽时复用；
- `dt_in(node)` 是 MXM 节点流入数据的精度，等于它唯一 stream 前驱的 `out.dtype`。acc 取的是流入操作数的精度，
  不是它自己的 i32 输出。`lw_rows_per_cycle` 和 `acc_latency` 按精度分表，键名与 tile graph 的 dtype 一致（i8 / f16 / bf16）；
  表里没有的精度报 `LTS-E-UNIT-OP`。v0 初值：i8 装载每拍 16 行、延迟 25；f16 / bf16 装载每拍 8 行、延迟 50；
  `acc_ii` 不随精度变化。f16 / bf16 的值是 assumed，待 M2 校准。
  **与设计的偏离：** ① 设计 §5.6 规定 MXM v0 只支持 INT8，这里扩展为 i8 / f16 / bf16，前端 P6 的
  `LTS-E-UNIT-OP` 检查届时要相应放宽；② 设计 §3.4 写的是 `acc_latency: {int8: 25}`，这里键名改成了 `i8`，以便与 tile graph 一致；
- `F` 是 `fractions.Fraction`；
- `hop(u, v) = m.hop(category(u), category(v)) × srf.hop_cycles`；
- **`skew` 的用途：** 每个单元的命令都以 superlane 波的方式执行，superlane r 在 `launch + r` 执行；LTSim 的时刻都记在
  波前（superlane 0）。`skew` 只用于节点的 `done` 和链长（§4.4），即程序结束时最后一个 superlane 还要再收尾 19 拍。
  **比较两个都跟着命令波走的操作时不加它**：两边在 superlane r 上各晚 r 拍，相互抵消。所以流预留（§4.4）、
  mem 边（§4.6）、state 边、ref 为 start / issue_end 的 order 边和所有资源预留里都没有 `wave_skew`。只有用户手写的
  `ref = done` 的 order 边会经 `refoff` 带上源节点的 skew（builder 从不生成这种边）。

## 4. B2 组链

### 4.1 求链

对 `stream` 边做并查集，每个连通分量是一条链。**链 id 的确定规则：**
按分量里"在 `nodes` 数组中最靠前的节点"的位置排序，依次编号 `c0, c1, …`。
这样同一输入的链 id 总是相同，调度的平局也按它来决（4.6、5.3）。

### 4.2 速率与占用

```python
rate = min(work[n].cap for n in chain)
D[n] = ceil(rows[n] / rate)
```

### 4.3 链内偏移

流不能等待，所以链内**每条** stream 边都是等式（汇点也一样）：

```python
def arr(u, v, r0):  return work[u].L + hop(u, v) + ceil(r0 / rate)
# 对每条 stream 边：o[v] == o[u] + arr(u, v, r0)
```

链是连通的，这组等式的解如果存在，就唯一（整体平移一个常数除外）。求法：

1. 链内 stream 边有向环 → `LTS-E-GRAPH-CYCLE`；
2. 从链里在图中最靠前的节点出发（o = 0），沿无向边按广度优先传播：顺着边 `o[v] = o[u] + arr`，逆着边 `o[u] = o[v] − arr`；
3. 核对每条 stream 边的等式，不成立报 `LTS-E-CHAIN-ALIGN`，诊断信息给出这个流值、两个要求的时刻，建议拆开融合组让它经过 MEM；
4. 整体减去 `min(o)`。

**与设计的偏离：** 设计 §7.2（以及本文早先的版本）用"正向 ASAP → 反向 ALAP（汇点保持正向值）→ 核对"。
汇点保持正向值这一步没有依据，会误报：一个流值分叉、一支直接进汇点、另一支汇入由更慢输入决定时刻的节点时，
ALAP 不推汇点，于是这个流值被要求在两个时刻送出，报 `LTS-E-CHAIN-ALIGN`，而实际存在合法解（整体推迟这一支）。
反例见 `tests/ltsim/test_chains.py::TestOffsets::test_fork_with_sink_is_solvable`。在 ASAP/ALAP 能通过的图上，
两种求法结果相同（MLP 已核对）；ASAP/ALAP 误报的图，等式传播能解；真正矛盾的图，两者都报错。

### 4.4 链长与预留表

```python
length = max(o[n] + work[n].L + D[n] + work[n].skew for n in chain)
```

每个节点产生的预留（设计 7.3）：

| 节点 | 资源 | 区间（相对链起点） | 数量 |
|---|---|---|---|
| `mem.read` | `mem.<H>.rd` | `[o, o+D)` | `ceil(V × rate)` |
| `mem.write` | `mem.<H>.wr` | `[o+L, o+L+D)` | `ceil(V_in × rate)` |
| `mxm.load_w` | `<unit>.weight` | `[o, o+D)` | 1 |
| `mxm.acc` | `<unit>.compute` | `[o, o+D)` | 1 |
| `vxm.chain` | `vxm.alu` | `[o, o+D)` | `ceil(d × rate)` |
| 有 `out` 的节点 u | SRF 流池（见 4.4.1），每个池一条 | `[o+L, o+L+D+hop_last(u))`；`hop_tail=False` 时 `[o+L, o+L+D)` | `ceil(out.V × rate)` |

- `hop_last(u)` 是 u 到它所有 stream 后继的 hop 的最大值；
- **有 `out` 但没有 stream 消费者的节点不占流**（没有方向，也没有 `hop_last`）；
- 一个节点的后继分在两个方向：v0 在 B1 报 `LTS-E-GRAPH-HOP`（见 §11）。将来放开时，每个方向各一条流预留；
- 预留表按 (资源, start, end, 节点) 排序。B3 不直接逐条检查它，而是先把计数资源上的重叠预留合并成分段（§5.3）；
- 每条预留至少 1 拍：B0 保证 rows ≥ 1，所以 D ≥ 1。B3 依赖这条不变量（零长度区间一律视为不冲突、提交时忽略）。

**与设计的偏离（2026-11-21）：流预留与 mem 边不加 `wave_skew`。** 设计（`Docs/lpu_tile_sim_design.md` 约第 1012、1023、
1095 行）在流预留的尾巴、mem 边的 tile 语义（等写节点的 `done`，含 19 拍 skew）和 row 语义的逐行约束里都加了
`wave_skew`，没有说明理由。ltsim 去掉这三处：流预留是 `[o+L, o+L+D+hop_last)`，mem 边默认
`("issue_end", mem.write_latency + mem.raw_gap)`（§4.6），`mem_row_lag` 也不含 `wave_skew`（§4.5）。

- **理由：** 时刻都记在 superlane 0。前后两个占同一条流（或先写后读同一块 MEM）的操作都跟着同一个命令波走，
  在 superlane r 上两边各晚 r 拍，比较时抵消；每个 superlane 上的流与 MEM 是各自独立的。lpu-compiler 的静态检查
  （`compiler/backend/schedule/static.py`）也是按 (拍, 站点, 方向, 流, superlane) 查流冲突，按基准发射时刻比较 MEM 的先写后读
  （读在写的 launch + 1 之后即可），不加 19。这也与 ltsim 已有的处理一致：state 边用 `issue_end + lw_to_acc_gap`，
  MXM 的 order 边用 start / issue_end，MEM / MXM / VXM 的预留都是 `[o, o+D)`，都不加 wave；
- **保留的地方：** 节点 `done`、链长、程序总周期（最后一个 superlane 真的晚 19 拍结束，§3.2）；
- **假设：** 所有单元都跟着同一个命令波走。将来引入跨 superlane 搬数据的 SXM 操作（M6）时，这条规则要重新审视；
- **验证：** M2 校准集里的 GEMV 用例对这条规则最敏感，用它确认（见 [roadmap.md](roadmap.md) M2）。

#### 4.4.1 流资源池

`build_chains(g, m, *, mem_overlap="tile", srf_pool="half", lw_streams="separate", hop_tail=True)` 的后三个选项决定
SRF 流预留进哪个计数资源、区间多长。取值不对报 `LTS-E-ARG`。每个池都是计数资源，容量 `srf.streams_per_direction`（32）。
资源表由 `Machine.resources(srf_pool=…, lw_streams=…)` 按选项生成，存进 `ChainGraph.resources`（§1、§5）。

**`srf_pool`：怎么分池。**

| 取值 | 池 | 说明 |
|---|---|---|
| `half`（默认） | `srf.W2E.W`、`srf.W2E.E`、`srf.E2W.W`、`srf.E2W.E` | 每个方向再按芯片的东西两半各一个池；不同方向、不同半边互不干扰 |
| `direction` | `srf.W2E`、`srf.E2W` | 每个方向一个池（M0 早先的行为，也是设计的做法） |

`half` 时一条流预留占哪几半，由节点 u **和它全部 stream 后继**的 category 决定，取并集：

- `MEM_W`、`MXM_W` → W；`MEM_E`、`MXM_E` → E；`VXM` 在中间，不属于任何一半；
- 所以 `MEM_W → MXM_W`、`MXM_W → VXM`、`MEM_W → VXM` 只占 W；`VXM → MEM_E`、`MEM_E → MXM_E` 只占 E；
- 跨中线的传输（例如 `MEM_W → MXM_E`、`MXM_W → MEM_E`、`MEM_E → MXM_W`）两半都占；
- 集合为空（只可能是 VXM → VXM，v0 在 B1 就拒绝了）时两半都占。

占几半就产生几条预留，区间与数量相同。

**`lw_streams`：权重装载用不用专门的流。**

| 取值 | 规则 |
|---|---|
| `separate`（默认） | 节点 u 的 stream 后继**全部**是 `mxm.load_w` 时，它的流预留进权重专用池：普通池名加 `.lw`（`srf.E2W.W.lw`；`direction` 时 `srf.E2W.lw`）。其他流量留在普通池。含义：权重装载与其他数据（例如送进 MXM 的激活）不争同一批流 |
| `shared` | 不分，权重装载与其他数据共用普通池（M0 早先的行为） |

后继里既有 load_w 又有别的（v0 的 builder 不会生成）：保守处理，`.lw` 池和普通池都占。`separate` 时资源表里总有全部
`.lw` 池，没用到也列出。两个选项组合起来：`half + separate` 有 8 个池，`half + shared` 4 个，`direction + separate` 4 个，
`direction + shared` 2 个。

**`hop_tail`：流预留带不带 hop 尾巴。** `True`（默认）时区间是 `[o+L, o+L+D+hop_last)`，`False` 时是 `[o+L, o+L+D)`。
其他一切（偏移、链长、链间边、其他预留）不变。模型不知道 MEM slice 的位置，hop 尾巴到底多长说不准，所以 CLI 的 `run`
**总是调度两次**：带尾巴的是主结果（`total_cycles`、trace、关键路径），不带的写进 `result.json` 的 `no_hop_tail`（§6），
两个数夹住这一不确定性。

**与设计的偏离（2026-11-21）：** 设计每个方向只有一个流池，流预留总带 hop 尾巴。ltsim 默认按东西两半分池、并给权重装载
单独的流。这两条都是**假设**：

- 按半边分池的依据是流沿 x 方向传，只穿过源到目的之间的那一段；但真实的流占用取决于 MEM slice 放在哪，模型目前没有 slice 放置；
- `lw_streams = separate` 假设编译器总能给权重装载分到与其他数据不冲突的流，这是**乐观**的；
- 两条都待 M2 校准（GEMV 用例对流最敏感）或 M6 引入 slice 放置后决定（[roadmap.md](roadmap.md) 待定项 T11–T13）。
  在那之前，hop 尾巴由主结果与 `no_hop_tail` 两个值夹住。旧行为用 `--srf-pool direction --lw-streams shared` 得到，
  手算的旧数字见 §8.4。

### 4.5 链级合法性

| 检查 | 做法 | 诊断码 |
|---|---|---|
| 流宽 | 对本链的 `srf.*` 预留**逐池**做扫描线，任一时刻每个池的数量和 ≤ 32 | `LTS-E-SRF-WIDTH` |
| 资源自冲突 | 本链在同一独占资源上的预留两两不相交 | `LTS-E-CHAIN-SELF` |
| 计数资源自超限 | 本链在 `vxm.alu`、`mem.*` 上**逐拍**的占用之和 ≤ 容量（扫描线，与流宽同法；不是总量之和） | `LTS-E-CHAIN-SELF` |
| 同链 `mem` 边 | 逐行检查（设计 7.2 表），见表后 | `LTS-E-CHAIN-SELF` |
| 同链其他非 stream 边 | `o_v ≥ refoff(u) + lag` | `LTS-E-CHAIN-SELF` |

先查流宽，再查其他资源。**同链 `mem` 边的逐行检查：** 用 `tile_region` 把读、写两个 tile 换算成物理 word 区间
（各自按自己的布局），对每个公共 word 检查"读到它时它已写完且可见"。同链与链间（§4.6 的 row 模式）共用一个函数：

```python
def mem_row_lag(w, r):          # T_r − T_w 至少要多大
    return max(o_w + L_w + ceil(i_w / ρ_w) + raw_gap − o_r − ceil(i_r / ρ_r)
               for word in common_words(r.tile, w.tile))      # i_r、i_w：该 word 在读、写节点里的行号
# 同链：T_r = T_w，要求 mem_row_lag ≤ 0
```

两端起始 word 相同、同速率时化简为 `o_r ≥ o_w + L_w + raw_gap`。读写都跟着命令波走，不加 `wave_skew`（§4.4 的偏离）。
`raw_gap` 是 `mem.raw_gap` 本身：`L_w`（= `mem.write_latency`）已在式中，不能再用 §4.6 展开后的 lag，否则写延迟会算两次。广播读的 tile 只有 1 个 word，
i_r 恒为 0，即只检查第一次读。没有公共 word 报 `LTS-E-GRAPH-MEM`（B0 已拦，这里兜底）。

### 4.6 链间约束图

```python
def refoff(u, ref):
    return {"start": o[u], "issue_end": o[u] + D[u], "done": o[u] + work[u].L + D[u] + work[u].skew}[ref]

for (kind, u, v, ref, lag, reason) in non_stream_edges:   # state / mem / order，已展开默认 ref 与 lag
    if chain_of(u) != chain_of(v):
        w = refoff(u, ref) + lag - o[v]
        add ChainEdge(chain_of(u), chain_of(v), w, (kind, u, v), ref, reason)
```

**链间边不去重。** 设计 §7.2 说同一对链只保留 `w` 最大的一条，这里全部保留：B3 求 `est` 时本来就取最大值，结果不变；
B4 的 trace 要为每条链间边画一对流向事件（设计 §8.2），去重会丢信息。

默认值：`state` → `("issue_end", mxm.lw_to_acc_gap)`；`mem` → `("issue_end", mem.write_latency + mem.raw_gap)`；
`order` 用边上写的。`mem` 边的源一定是 `mem.write`（B0 保证），它的 L 就是 `mem.write_latency`：最后一行在 superlane 0
发出（issue_end）后再过 L + raw_gap，读者在 superlane 0 就能读整个 tile，其余 superlane 两边同样晚 r 拍，所以不等 `done`
（§4.4 的偏离）。tile graph 里 `order` 边的 `ref` 仍只接受 start / issue_end / done。
`--mem-overlap row` 时（`build_chains(..., mem_overlap="row")`），**所有**跨链的 `mem` 边改用逐 word 语义：
读第 i 个 word 只等第 i 个 word 写完可见。边的权重就是 4.5 的 `mem_row_lag(w, r)`，即逐 word 条件给出的
`T_r − T_w` 的精确下界。它不超过 tile 语义的权重（每个 word 都不晚于写节点的 issue_end + L_w + raw_gap 可见，
而读的每个 word 不早于读节点的 start）。

**与设计的偏离：** 设计 §7.3 只对"一对一、行范围相同"的读写对启用 row，并用首行、末行两条约束近似：

```text
第一行：start(r) ≥ start(w) + L_w + wave_skew + raw_gap
最后一行：start(r) ≥ done(w) + raw_gap − D_r
```

两条端点约束不等于逐行条件。下面的数字都按同一个 wave 约定比较（端点约束也去掉 `wave_skew`，`done` 换成
`issue_end + L_w`）：

- 两侧速率都是小于 1 的分数、且不相等时，会**乐观**。例如写侧链深 19、读侧链深 18、9 行：
  端点约束给 82，第 6–8 个 word 会在可见之前被读走；精确值是 83；
- 另一些情况下会**保守**，最多约 3 拍。例如运行时权重读链（ρ = 16）：端点约束给 314，精确值是 313。

逐 word 公式没有这两个问题，也能直接处理多个写 → 一个读、word 区间不同的情况，所以不再需要"一对一"的限制。
用例见 `tests/ltsim/test_chains.py::TestMemOverlapRow`。
用 Kahn 算法检查无环，有环报 `LTS-E-CHAIN-CYCLE`，诊断信息列出环上的链。

## 5. B3 调度

**资源表只有一个来源：** `schedule(cg, policy)`、`verify(cg, placed)`、B4 的 `compute_bounds(cg, placed)` /
`resource_usage(cg, placed)`、trace 与 oracle 都取 `cg.resources`，即 B2 组链时按选项生成的那张表（§4.4.1），
不再接受单独传入的资源表。这样换了 `srf_pool` / `lw_streams` 之后，各处看到的流池不会不一致。

### 5.1 日历

```python
class ExclusiveCalendar:          # 每个独占资源一个
    def conflict(self, s, e) -> tuple[int, str] | None:
        """与 [s, e) 重叠的已占区间里，结束最晚的那个的 (end, chain)；没有重叠返回 None。"""

class CounterCalendar:            # 每个计数资源一个，记录分段常数的占用量
    cap: int
    def conflict(self, s, e, k) -> tuple[int, str] | None:
        """[s, e) 内占用量 > cap − k 的最后一拍 t，返回 (t + 1, 那一拍贡献最大的链)；放得下返回 None。"""
```

- 两个类都只追加、不删除。只用标准库：`bisect` 维护有序列表（约定不依赖 `sortedcontainers`）。
- 独占日历里提交的区间互不重叠，按起点排序后终点也有序，`conflict` 是一次二分查找。
- 计数日历存断点和每段的占用量；`conflict` 从右往左扫与 `[s, e)` 相交的段。
- **blocker 的平局规则：** 那一拍贡献相同的链里，取最先提交进日历的那条（§8.3 第 3 步因此是 c0）。
- 零长度区间 `[s, s)` 不与任何区间冲突，提交时忽略（4.4 的不变量保证合法输入不会出现）。
- 一段用量就超过容量（`k > cap`）报 `LTS-E-CHAIN-SELF`；B2 的链级检查已拦下，这里兜底。
  预留用了资源表里没有的资源，报 `LTS-E-MACHINE`。
- `CounterCalendar.steps()` 给出 (时刻, 占用量) 的变化点，供 trace 的计数器事件使用。

### 5.2 优先级

```python
bl[c] = max(length[c], max((e.weight + bl[e.dst] for e in out_edges(c)), default=0))   # 逆拓扑序计算
```

### 5.3 串行生成法

```python
T, bound_by = {}, {}
while len(T) < n_chains:
    ready = [c for c in chains if c not in T and all(e.src in T for e in in_edges(c))]
    c = max(ready, key=lambda c: (prio(c), -index(c)))          # 优先级高者先，平局取 id 小的
    est, why = 0, BoundBy("source")
    for e in in_edges(c):
        if T[e.src] + e.weight > est:                           # 同值时保留第一条达到的边
            est, why = T[e.src] + e.weight, BoundBy("edge", edge=e.edge, blocker=e.src)
    if policy == "topo" and 已放过链 and T[上一条] > est:        # 不插空
        est, why = T[上一条], BoundBy("policy", blocker=上一条)
    t, blocked = earliest_fit(segments(c), est)
    T[c] = t
    bound_by[c] = why if blocked is None else blocked
    commit(c, t)                                                # 原始预留表写入日历

def earliest_fit(segs, t):
    blocked = None
    while True:
        for s in segs:                                          # 固定顺序：按资源名、再按 start
            hit = calendar(s.resource).conflict(t + s.start, t + s.end, s.amount)
            if hit:
                end, owner = hit
                t = end - s.start
                blocked = BoundBy("resource", resource=s.resource, blocker=owner)
                break
        else:
            return t, blocked
```

**按合并后的分段检查（与设计的偏离）：** 设计 §7.3 和本文早先的版本对预留表**逐条**检查。
但同一条链在同一个计数资源上常有多条重叠的预留，例如 c4 在 W2E 上有 acc 4 条、读 B1 4 条、vxm 1 条。
逐条检查时每条都放得下，加起来却会超容量。反例：链 A 占 24 条 W2E，链 B 的 3 条读各占 4 条且彼此重叠。
逐条检查把 B 放在 T=0，峰值 36 > 32。所以 `segments(c)` 先把每个计数资源上的预留按断点切开、
重叠部分的用量相加，得到互不重叠的分段（c4 的 W2E 变成 4 / 8 / 9 / 1 四段），再逐段检查。
独占资源仍然每条预留一段（B2 保证同一链在独占资源上不重叠）。这一改动当时没有改变 MLP 的结果；
反例见 `tests/ltsim/test_scheduler.py::TestAggregatedFit`。

**`earliest_fit` 的正确性：**

- 每次冲突后 `t` 严格增大，日历有限，所以必然终止；
- 不会跳过可行位置：
  - 独占资源上，任何落在 `(t, end − a)` 内的起点，都还会和那个结束最晚的区间重叠；
  - 计数资源上，任何落在 `(t, 最后超载拍 − a]` 内的起点，同一分段仍然覆盖那一拍。

  所以返回的是 `est` 之后最早的可行起点。

**策略：**

- `bottom_level`：`prio(c) = bl[c]`，默认；
- `topo`：模拟按顺序发射的朴素编译器，作对照。链按链间约束图的 Kahn 拓扑序依次放置：
  每一步在当前入度为 0 的链里取 id 序号最小的（最小堆，不是"初始按 id 排好的 FIFO 队列"）；
  起始时刻不早于前一条已放链的 T，即不插空。因不插空而推迟时，`bound_by` 记为 `policy`，指向前一条链。
  设计 §7.3 写的是按"所属算子在 LTS-IR 中的序号"排序，但 tile graph 里没有算子顺序，所以改用拓扑序；
- `random:K:seed`：**推迟实现**（`schedule` 遇到它报 `LTS-E-ARG`）。等粗粒度编译器（M3）需要时再加。
  届时的规格：第 k 次运行用 `bl[c] × (1 + u)`，`u ~ U(-0.2, 0.2)` 由 `random.Random(f"{seed}:{k}")` 按链 id 顺序生成；
  K 次结果全部记录，取总周期最小的一次，平局取 k 最小。

**已知弱点（oracle 实测，§9）：** SSGS 是贪心的，每条链放一次就不再动，会把对称的工作放在同一时刻、互相阻塞。
例如旧选项（`srf_pool = direction`、`lw_streams = shared`）下 banks = single 的 GEMM 256×640×1280：SSGS 让两个 plane 同时装权重，两块合计占满 E2W 的 32 条流，换块时被卡住，总周期 1339；
最优解把两个 plane 的链错开，总周期 1336（SSGS 多 0.22%）。§4.4 的 wave 规则修订（2026-11-21）之前，同一个例子是 1434 对 1374
（多 4.37%）：流预留多出的 19 拍尾巴拉长了每次阻塞。alternate 配置下装载藏在计算里，SSGS 正好最优。默认选项下权重装载有自己的流，这个例子的 SSGS 是 1308，等于 LB_path。
将来的 `random:K:seed` 策略或调度后的改进步骤（如前后向修正）可以缓解。

**自检：** `verify(cg, placed)` 不复用日历的代码，直接按预留表逐拍累加。它检查：

- 每条链都有 T 且 T ≥ 0；
- order 覆盖全部链；
- `total = max(T + len)`；
- 全部链间约束成立；
- 任何一拍、任何资源都不超容量。

返回违反项列表，不抛异常。测试和 B4 都调用它。

**最早可行性的测试：** `tests/ltsim/test_scheduler.py` 里有一个逐拍的参考调度器：选链规则相同，但从 est 起逐个试 T、
逐拍检查容量，不用日历、不做跳跃。它在 GEMM 变体和随机合成的链图（独占资源插空、计数资源冲突都会成为瓶颈）上，
与 `schedule()` 的 T 和放置顺序逐项比较。

## 6. B4 下界、指标与输出

模块：`bounds.py`（下界、关键路径、roofline）、`metrics.py`（节点时刻、资源、算子）、`result.py`（组装与写出）。

```python
total = max(T[c] + length[c] for c in chains)
ES   = longest_path_from_source(chain_edges)            # 不考虑资源；虚拟源保证 T ≥ 0
LB_path = max(ES[c] + length[c] for c in chains)
LB_res  = max(sum(r.amount * (r.end - r.start) for r in all_res(R)) / cap(R) for R in resources)   # 平局取资源名在前的
LB = max(LB_path, ceil(LB_res)); gap = total / LB - 1
```

- `LB_path` 与 B3 用同一组边权；`LB_res` 成立是因为 B3 保证每一拍占用 ≤ 容量。所以 `total ≥ LB` 必然成立，
  `build_result` 会检查它，并调用 `verify`；不成立报 `LTS-E-INTERNAL`。
- **MXM roofline（与设计的偏离）：** `总 MAC / 每拍峰值 MAC`。每个 acc 的 MAC = 行数 × K × N，K、N 取它**实际**所用权重块的大小。
  它所用的块 = 同一 unit、同一 bank 上、在 acc 开始之前（`issue_end ≤ acc.start`）最后一次装完的 load_w：
  K = 该 load_w 的行数，N = 流进它的权重 tile 的 lane 数（没有 tile 时取 `block_n_cols`）；找不到时按满块计。
  不能只看 state 边：builder 只给每个权重块的第一个 acc 连 state 边，同一块后面的 acc 靠 acc_seq 串起来。
  峰值 = plane 数 × `block_k_rows` × `block_n_cols`，都从机器参数读。对 GEMM，roofline 恰好等于 M·K·N / 峰值
  （`test_roofline_is_true_mac` 守着这一点）。
  设计 §7.4 按"行数 × 320 × 320"累加，那其实是"各 plane 的忙碌下界"，已经包含在 `LB_res` 的 `mxm.*.compute` 里；
  按实际块大小算，权重块不满时能看出算力浪费。例如 `tile_n = 160` 的 GEMM：roofline 是 32，按满块算是 64。
- **关键路径：** 从最后结束的链出发（平局取在链列表里序号小的，不按字符串比较），沿 `bound_by.blocker` 回溯：
  - `edge` 的 blocker 是边的源链；
  - `resource` 的 blocker 是占着资源的链；
  - `policy`（topo 策略因不插空而推迟）的 blocker 是前一条已放的链；
  - `source` 停止。

  回溯的每一步都回到更早放置的链，所以必然终止；万一重复访问或 blocker 缺失，报 `LTS-E-INTERNAL`。
- **节点时刻：** `start = T_c + o_n`，`issue_end = start + D_n`，`done = start + L_n + D_n + skew_n`（设计 §3.5）。
- **资源：** 每个资源（含没用到的）给出忙碌拍数（占用 > 0 的拍数）、峰值和利用率。
  独占资源的利用率 = 忙碌拍数 / `total`；计数资源 = Σ(数量 × 时长) / (容量 × `total`)。
- **流：** 每个 SRF 池各给峰值与时间加权平均（Σ 数量 × 时长 / `total`）。
- **每个 LTS-IR 算子的起止：** 取 `origin` 含该算子的所有节点里，最早的 `start` 和最晚的 `done`；
  按算子在节点里第一次出现的顺序列出。不靠节点名匹配，避免 GTSim 那种耦合。

**`result.json`**（schema `lts-result/0.2`，在设计 §8.1 的基础上补了几个字段；下面是 MLP 在默认选项下的结果，列表只留一项）：

```json
{
  "schema": "lts-result/0.2",
  "provenance": {"ltsim_commit": null, "dirty": null, "command": "...", "ir_sha256": null,
                 "tile_graph_sha256": "...", "machine_sha256": "...", "contract_sha256": null,
                 "policy": "bottom_level", "seed": null, "mem_overlap": "tile",
                 "srf_pool": "half", "lw_streams": "separate"},
  "total_cycles": 506,
  "bounds": {"path": 506, "resource": 136, "resource_exact": "136", "resource_name": "vxm.alu",
             "lb": 506, "gap": 0.0, "mxm_roofline": 128.0},
  "chains": [{"id": "c4", "start": 21, "len": 265, "end": 286, "rate": 1.0, "rate_exact": "1",
              "bound_by": {"kind": "edge", "edge": ["state", "w1n0.lw", "a0.acc"], "from_chain": "c0"}}],
  "nodes": [{"id": "a0.acc", "chain": "c4", "start": 30, "issue_end": 158, "done": 183}],
  "resources": [{"id": "mxm.w0.compute", "kind": "exclusive", "cap": 1, "busy": 128, "peak": 1, "utilization": 0.252964}],
  "srf": {"srf.E2W.W": {"peak": 2, "avg": 0.794466}, "srf.E2W.W.lw": {"peak": 32, "avg": 1.770751},
          ...},                          // 默认选项下共 8 个池，按名字排序全部列出
  "critical_path": ["c0", "c4", "c6"],
  "critical_path_detail": [{"chain": "c6", "start": 267, "end": 506,
                            "bound_by": {"kind": "edge", "edge": ["mem", "a0.h.wr", "b.h0.rd"], "from_chain": "c4"}}],
  "no_hop_tail": {"total_cycles": 506,
                  "bounds": {"path": 506, "resource": 136, "resource_exact": "136", "resource_name": "vxm.alu",
                             "lb": 506, "gap": 0.0},
                  "critical_path": ["c0", "c4", "c6"],
                  "critical_path_detail": [{"chain": "c6", "start": 267, "end": 506, "bound_by": {"kind": "edge", "...": "..."}}]},
  "ops": [{"id": "fc1", "start": 0, "end": 183}],
  "diagnostics": []
}
```

- `bound_by` 按种类写：edge 写来源边和它的源链；resource 写资源和 blocker；policy 写 blocker；source 只写 kind；
- 链速率写两遍：`rate` 是浮点，`rate_exact` 是精确分数字符串，例如 `"16/17"`；
- 列表顺序：`chains` 按链 id 的序号，`nodes` 按图中节点的顺序，`resources` 按资源名，`ops` 按算子第一次出现的顺序；
- `resources` 列出 `cg.resources` 里的全部资源，自然包含选项决定的全部 SRF 池；
- `srf`：以池 id 为键（如 `srf.E2W.W`），列出资源表里的**全部** SRF 池，按名字排序；每个池给峰值与时间加权平均；
- `no_hop_tail`：同一张图、同样选项，但 `build_chains(..., hop_tail=False)`（§4.4.1）的第二次调度：总周期、下界
  （与主结果的 `bounds` 同格式，不含 `mxm_roofline`）、关键路径。`build_result(..., no_hop_tail=(cg2, placed2))` 传入，
  也做 `verify` 与 `total ≥ LB` 自检；两次的链或资源表对不上报 `LTS-E-ARG`；不传时为 `null`。CLI 的 `run` 总是传；
- `provenance` 记录 `srf_pool`、`lw_streams`（`make_provenance` 的必填关键字参数）；
- 浮点统一 `round(x, 6)`（极小的数会按 JSON 的写法写成 `1e-06`）；不写时间戳；键的顺序固定。
  同一输入两次运行，`result.json` 除 `provenance.command` 记录的命令行外逐字节相同；输出路径（`--out`）也相同时完全相同；
- 来源信息：给出文件路径时取文件内容的 sha256，内存里的图和机器参数取规范化内容的 sha256。
  M0 不读取 git 信息，`ltsim_commit` / `dirty` 写 null（待定项 T10）；M0 没有 LTS-IR 和合同文件，`ir_sha256` / `contract_sha256` 写 null。
- 与 0.1 的差别：`srf` 从 `{"W2E": …, "E2W": …}` 改成按池 id；新增 `no_hop_tail`；`provenance` 新增两个选项。

下界与 roofline 只作内部诊断，对外只报 tile 估计（设计 14.2）。

## 7. Perfetto trace

模块：`ltsim/trace/perfetto.py`。`build_trace(g, cg, placed, m, provenance=None, stream_flows=False, no_hop_tail_total=None)` 返回
Chrome Trace Event JSON：`{"traceEvents": [...], "displayTimeUnit": "ns", "otherData": {...}}`，可以直接拖进 <https://ui.perfetto.dev>。

### 7.1 生成步骤

1. **进程（固定 pid，设计 §8.2）：** 1 Ops (LTS-IR)、2 Chains、3 MXM、4 VXM、5 SXM、6 MEM-W、7 MEM-E、8 Counters、9 Critical path。
   用 `M` 事件写 `process_name`、`process_sort_index`、`thread_name`、`thread_sort_index`。
2. **线程：**
   - Ops：每个 LTS-IR 算子一个线程，一段 `[最早 start, 最晚 done)`（按 origin 聚合，见 §6）；
   - MXM：`cg.resources` 里每个 MXM 独占资源一个线程（`mxm.w0.weight`、`mxm.w0.compute` …，按资源名排序，含没用到的）；SXM 同理（v0 留空）；
   - Chains、VXM、MEM-W、MEM-E、Critical path：时间段会重叠，用贪心区间着色分行。按 (start, end, 名字) 排序，
     放进第一个"上一段已结束"的行，每行一个 tid，线程名 `row k`。排序键里的"名字"是节点 id / 链 id（字符串比较）；
   - **关键路径（与设计的偏离）：** 设计只给 1 个线程，但关键路径上相邻的链在时间上本来就会重叠：
     一条链在前一条的流释放后就能开始，而前一条的链长还包含命令波尾巴。例如 MLP 的 c0 `[0,48)` 与 c4 `[21,286)`。
     所以和 Chains 一样分行，每段的名字是 `k. 链 id`，从路径起点开始编号（1 是回溯到的最早那条链）。
3. **`X` 事件：** 节点、链、算子、关键路径各一段。
   - 节点的区间取占用区间 `[start, issue_end)`；`mem.write` 取 `[start+L, start+L+D)`；
   - 名字是 `kind:origin`，origin 用 `+` 连接；
   - 节点的 `args` 带：node、chain、kind、origin、tile、unit、rows、rate / rate_exact、V（写节点为 V_in）、
     usage（该节点在各资源上的每拍用量）、L、start、issue_end、done、bound_by（所在链的）。
4. **`C` 事件：** 计数器取自 `cg.resources`（`counter_names`）：全部 SRF 池（按名字排序；默认选项下 8 个，`srf.E2W.E` …
   `srf.W2E.W.lw`，旧选项下 `srf.E2W`、`srf.W2E`），然后 `vxm.alu`，然后 `mem.W.rd`、`mem.W.wr`、`mem.E.rd`、`mem.E.wr`。
   从预留表和 T 直接算占用量的阶梯函数，每次变化一个点，`args = {"value": 占用量}`。第 0 拍总有一个点：
   第 0 拍已有占用就直接写该值（例如 MLP 的 `mem.W.rd` 在第 0 拍是 32），否则写 0。
5. **流向事件（与设计的偏离）：** 每条链间边（mem、state、order，不去重）一对 `s` / `f`，`id` 从 1 连续编号，
   `cat` 是边的类型（`state` / `mem` / `order`），order 边的 `name` 是 reason：
   - `s` 放在源节点的线程上，`ts` 取这条边**依赖的时刻**：`ref = start` 取源节点时间段的起点；`issue_end` / `done`
     取源节点时间段的最后一拍（写节点的时间段从 start + L 起，所以这里说"时间段的起点"而不是节点的 start）。
     每条链间边的 ref / reason 由 `ChainEdge` 自己携带，同一对节点之间的平行边互不干扰。设计一律取最后一拍，那样 `lw_after_iw`（ref = start、lag = −1）会画成反向箭头。
     例如 GEMM 的 `m0n0k0.acc → w.n0k1.lw` 会从第 164 拍指向第 37 拍；Perfetto 按时间顺序配对 s/f，这样的箭头很可能被丢掉；
   - `f` 放在目的节点的线程上，`ts = max(目的节点时间段的起点, s 的 ts)`，带 `bp: "e"`，`cat` / `name` 与 s 相同。这保证 f 不早于 s，
     且落在目的节点的时间段内；万一落不进去（只在 lag 很负时可能），这条箭头不画，记在 `otherData.skipped_flows`；
   - `stream_flows=True` 时，链内的 stream 边也画（`cat = "stream"`）。s 在流给目的节点的第一行离开源节点的时刻
     `start + L + ceil(r0/ρ)`（截到源节点时间段内），f 同上规则。默认关闭。
6. **时间换算：** `ts = cycle × cycle_ns / 1000`（单位 µs），`cycle_ns` 取机器参数 `display.cycle_ns`（默认 1，
   即界面上 1 ns = 1 拍）；`dur` 由整数拍差直接换算，不由两个浮点相减。
7. **`otherData`：** `schema: "lts-trace/0.1"`、`cycle_ns`、`total_cycles`、`skipped_flows`、`provenance`；
   给了 `no_hop_tail_total` 时另有 `no_hop_tail_total_cycles`（CLI 总是给）。trace 只画主结果（带 hop 尾巴的那次调度）。
   事件顺序固定（元数据 → X → C → 流向），同一输入两次运行除 `provenance.command` 外逐字节相同，命令行（含 `--out`）相同时完全相同。

### 7.2 自检（`check_trace`，每次生成后都跑，不通过报 `LTS-E-TRACE`）

- 同一 `(pid, tid)` 上的 `X` 事件两两不部分重叠（嵌套允许，本实现不会出现）；
- 每个 `s` 都有同 `id` 的 `f`，二者都落在所在线程的某个 `X` 事件内部，且 f 不早于 s；
- 所有 `ts`、`dur` 非负，`X` 的 `dur` > 0。

- `f` 带 `bp: "e"`；同一 id 的 s 与 f 的 `cat`、`name` 相同（Perfetto 按 id + cat + name 配对流向事件）；
- 事件缺字段时报出来，不抛异常。

浮点比较留 1e-9 µs 的余量（1 拍是 `cycle_ns / 1000` µs，远大于它）。

**Perfetto 的时间换算（已核对源码）：** Perfetto 主分支 `src/trace_processor/importers/json/json_trace_tokenizer.cc`
的 `CoerceToTs` 对整数 µs 直接乘 1000，对带小数的 µs 用 `std::llround(value * 1000.0)` 四舍五入成 ns，而不是截断。
所以 `1.001` µs 得到 1001 ns，首尾相接的段不会因为浮点误差变成部分重叠。很老的 Perfetto 版本可能不同。

### 7.3 怎么看

用 README「快速上手」里的命令生成 MLP 和两种 GEMM（banks = alternate / single）的 `trace.json`，拖进 Perfetto。
在 MLP 里（默认选项）应能看到：

- Counters 的 `srf.E2W.W.lw` 在 `[1,29)` 顶到 32（西侧两块权重同时装载），而读 X 所在的 `srf.E2W.W` 最多只有 2；
- Chains 的 c4、c5 在 21 开始，点开看 `bound_by` 是 state 边；
- c6 在 267 开始，早于 c4、c5 结束（286）：mem 边不等写节点最后 19 拍的 superlane 收尾（§4.4 的偏离）；
- Critical path 是 `1. c0 → 2. c4 → 3. c6`。

加 `--srf-pool direction --lw-streams shared`（旧选项）再跑一次，能看到流争用：`srf.E2W` 在 `[1,29)` 顶到 32，
c4、c5 被它卡到 28（`bound_by` 是 `srf.E2W`），c6 在 274，总周期 513（§8.4）。

在 GEMM 里对比 alternate 与 single：MXM 的 `mxm.w0.compute` 线程上，alternate 时块与块之间只隔 5 拍；single 时空 21 拍
（等上一块算完才能装下一块：装载 20 拍，加 `lw_to_acc_gap` 1 拍）。旧选项下 single 空 28 拍：读 X 的流还要等权重装载的流
连同 8 拍 hop 尾巴一起释放。

## 8. 手算一遍：设计 6.3 的 MLP

用设计 3.4 的初值参数、默认选项（`srf_pool = half`、`lw_streams = separate`、`hop_tail = True`，§4.4.1）手算。
**这些数字依赖目前标为 assumed / derived 的参数和 §4.4.1 的两条假设，只是开发用的金标准，不是性能结论。**
M2 校准改动参数后，金标准要重新生成。旧选项（`direction` + `shared`）下的数字见 8.4。

### 8.1 B1 与 B2：每条链

链 id 按 4.1 的规则：`c0` = 装 W1 左半，`c1` = 装 W1 右半，`c2`、`c3` = 装 W2 两块，`c4` = 第一层左半（a0），`c5` = 第一层右半（a1），`c6` = 第二层（b）。

**c0（装 W1 左半）：**

| 节点 | cap | L | hop 到后继 | D | o |
|---|---|---|---|---|---|
| `w1n0.rd` | 44 | 1 | MEM_W→MXM_W = 8 | 20 | 0 |
| `w1n0.lw` | 16 | 0 | — | 20 | 9 |

ρ = 16，D = 320 / 16 = 20。`len = max(0+1+20, 9+0+20+19) = 48`。
流：16 条，区间 `[1, 1+20+8) = [1, 29)`（不加 wave_skew，§4.4）。方向 E2W；MEM_W → MXM_W 只在西半；后继是 load_w，
所以进权重专用池 **`srf.E2W.W.lw`**。c1 相同；c2、c3 在东侧，hop 为 9，所以 `o_lw = 10`、`len = 49`，
流为 16 条 `srf.W2E.E.lw`，区间 `[1, 1+20+9) = [1, 30)`。链长仍含 skew 19。

**c4（第一层左半）：** ρ = 1（acc 和 vxm 都是每拍 1 行），D = 128。

| 节点 | L | 正向 o | 反向后 o |
|---|---|---|---|
| `a0.x.rd` | 1 | 0 | 0 |
| `a0.acc` | 25 | 0+1+8 = 9 | 9 |
| `a0.b1.rd` | 1 | 0 | 55−1−2 = **52** |
| `a0.vxm` | (6+2)×5 + (18+2) = 60 | max(9+25+21, 0+1+2) = 55 | 55 |
| `a0.h.wr` | 1 | 55+60+2 = 117 | 117 |

- 反向这一步把读 B1 从 0 推迟到 52，让偏置和矩阵乘的结果同时到达 VXM；
- `len = 117 + 1 + 128 + 19 = 265`；
- 流：读 X 1 条，`[1, 1+128+8) = [1, 137)`，在 **`srf.E2W.W`**（MEM_W → MXM_W，后继是 acc，不是 load_w）；
  acc 输出 4 条 `[34, 183)` 与 B1 4 条 `[53, 183)` 都在 `srf.W2E.W`（MXM_W / MEM_W → VXM，VXM 不属于任何一半）；
  vxm 送去写 H 的 1 条 `[115, 245)` 在 `srf.W2E.E`（VXM → MEM_E）。c5 相同。

**c6（第二层）：** ρ = 1，D = 128。两个 acc 的 `o = 0+1+9 = 10`；
vxm 的 `L = 4+4+8+8+8 = 32`，`o = max(10+25+22, 0+1+2) = 57`；写 Y 的 `o = 57+32+2 = 91`；读 B2 反向后 `o = 54`。
`len = 91 + 1 + 128 + 19 = 239`。

### 8.2 链间约束

| 边 | 计算 | w |
|---|---|---|
| state c0 → c4 | `issue_end(w1n0.lw)` = 9 + 20，加 lag 1，减 `o(a0.acc)` = 9 | 21 |
| state c2 → c6 | `(10 + 20) + 1 − o(b.acc0) = 10` | 21 |
| mem c4 → c6 | `issue_end(a0.h.wr)` = 117 + 128 = 245，加 write_latency 1 + raw_gap 0，减 `o(b.h0.rd)` = 0 | 246 |

c1 → c5、c3 → c6、c5 → c6 同理。mem 边不等 `done`（265），少了写节点的 19 拍 skew（§4.6）。

### 8.3 B3：调度过程

优先级：

- `bl(c6) = 239`；
- `bl(c4) = bl(c5) = 246 + 239 = 485`；
- `bl(c0) = bl(c1) = 21 + 485 = 506`；
- `bl(c2) = bl(c3) = 21 + 239 = 260`。

| 步 | 放哪条链 | est | 结果 | 原因 |
|---|---|---|---|---|
| 1 | c0 | 0 | T = 0 | 无前驱 |
| 2 | c1 | 0 | T = 0 | `srf.E2W.W.lw` 16 + 16 = 32，刚好放下 |
| 3 | c4 | 0 + 21 = 21 | **T = 21** | 读 X 要 1 条 `srf.E2W.W`（链内 `[1, 137)`），这个池里还没有占用；权重读占的是 `.lw` 池，不冲突 |
| 4 | c5 | 21 | T = 21 | 同上：`srf.E2W.W` 两条链合计 2 条，`srf.W2E.W` 合计最多 16 条，都放得下 |
| 5 | c2 | 0 | T = 0 | 东侧，`srf.W2E.E.lw` 在 `[1, 30)` 内没有占用 |
| 6 | c3 | 0 | T = 0 | `srf.W2E.E.lw` 16 + 16 = 32 |
| 7 | c6 | max(21 + 246, 0 + 21) = 267 | T = 267 | 等第一层写完 H；早于 c4 结束的 286（= 267 + 19，链长含 skew） |

**结果：**

- **总周期：** 267 + 239 = **506**；
- **下界：** `LB_path` = 0 + 21 + 246 + 239 = 506；`LB_res` 最大的是 `vxm.alu`：(6 × 128 × 2 + 5 × 128) / 16 = 136
  （流池里最大的是 `srf.W2E.W`：2752 / 32 = 86）。所以 LB = 506，gap = 0；
- **关键路径：** c6 ←(mem 边 a0.h.wr → b.h0.rd) c4 ←(state 边 w1n0.lw → a0.acc) c0，全部是依赖边，没有资源等待；
- **不带 hop 尾巴（`no_hop_tail`）：** 流预留都短了 hop_last，这里本来就没有资源等待，同样是 506。

### 8.4 旧选项下的数字（`--srf-pool direction --lw-streams shared`）

M0 早先只有每个方向一个流池、权重与其他数据共用（§4.4.1）。8.1 的偏移、链长与 8.2 的链间边都不变，只是流预留进
`srf.E2W` / `srf.W2E`：c0、c1 的权重读各 16 条 E2W `[1, 29)`，读 X 1 条 E2W `[1, 137)`，两者在同一个池里竞争。调度过程：

| 步 | 放哪条链 | est | 结果 | 原因 |
|---|---|---|---|---|
| 1–2 | c0、c1 | 0 | T = 0 | E2W 16 + 16 = 32，刚好放下 |
| 3 | c4 | 21 | **T = 28** | 读 X 要 1 条 E2W，但 `[1, 29)` 已被 c0、c1 占满 32 条，只能推到 29 − 1 = 28 |
| 4 | c5 | 21 | T = 28 | 同上 |
| 5–6 | c2、c3 | 0 | T = 0 | 东侧，W2E 16 + 16 = 32 |
| 7 | c6 | max(28 + 246, 0 + 21) = 274 | T = 274 | 早于 c4 结束的 293 |

- **总周期** 274 + 239 = **513**；LB 仍是 506（`srf.W2E` 是 4214 / 32 ≈ 131.7），gap = 7 / 506 ≈ 1.38%；
  **关键路径** c6 ←(mem) c4 ←(资源 srf.E2W，被 c0 占用) c0。oracle 证明 513 在这个模型下最优；
- 晚开始的 7 拍（28 − 21）就是权重读流预留里的 hop 尾巴：不带尾巴时 E2W 只在 `[1, 21)` 占满，c4 在 21 就能开始，
  所以这组选项的 `no_hop_tail` 是 506。`--mem-overlap row` 时是 393 / 386（LB 378）；
- 两种新选项单独打开时：`half + shared` 与旧选项相同（513 / 506），读 X 和权重读都在 `srf.E2W.W`；
  `direction + separate` 与默认相同（506 / 506）。`srf.streams_per_direction` 改成 64 时四种组合都是 506；
- 这些数字由 `test_chains.py` / `test_scheduler.py` / `test_bounds.py` / `test_trace.py` 里显式传 `LEGACY` 选项的用例、
  `test_cli.py::TestRun::test_legacy_srf_options` 与 `test_oracle.py::test_mlp_513_is_optimal` 守着。

## 9. 测试

在仓库根目录运行：`python -m pytest tests -q`。oracle 测试需要 OR-Tools（`pip install -e ".[oracle]"`），没装时自动跳过。

| 文件 | 内容 |
|---|---|
| `test_machine.py` | 参数路径解析；按精度分表的 MXM 参数；资源表（含四种选项下的 SRF 池与 `LTS-E-ARG`）；推导量校验（vector_bytes、wave_skew）；缺字段报 `LTS-E-MACHINE` |
| `test_graph.py` | B0：加载；字段取值、类型与每种 kind 的白名单（`LTS-E-GRAPH-FIELD`）；rows / mem / state 一致性；alias；tile 解析与物理区域；lane 宽；"永不崩溃"的模糊测试；`schema.json` 与 `load.py` 的同步 |
| `test_gen.py`、`test_gen_gemm.py` | builder；MLP 与设计 §6.3 逐字段一致、7 条链；MXM 顺序边辅助函数；GEMM 生成器 |
| `test_work.py` | B1：MLP 全部 25 个节点的 cap / L / skew、18 条 stream 边的 hop 与方向；多精度；各诊断码 |
| `test_chains.py` | B2：第 8.1–8.2 节逐项（含完整预留表，旧选项）；等式传播（分叉的一支是汇点）；链级合法性的边界（首尾相接 / 重叠一拍、刚好相等 / 差一拍）；row 模式的精确值与紧性；流资源池（§4.4.1）：各类传输进哪个半边、跨中线占两半、权重读进 `.lw` 池、后继混合的情形、`hop_tail=False` 的区间、`ChainGraph.resources`、逐池的流宽检查、选项取值 |
| `test_calendar.py` | 两种日历的 `conflict`：无重叠、首尾相接、部分重叠、刚好放满、超限一拍、blocker 平局 |
| `test_scheduler.py` | B3：第 8.3 节（默认选项）与 8.4 节（旧选项）逐项；流加倍时四种选项都是 506；合并分段；topo；double weight bank 的装载与计算重叠；与逐拍参考调度器逐项一致（GEMM 变体在旧选项与默认选项下、随机合成链图）；`verify` 的负例 |
| `test_bounds.py` | B4：下界、关键路径（含 policy）、roofline 等于 M·K·N / 峰值、各项指标、`result.json` 字段与确定性、`total ≥ LB`；默认选项的下界与关键路径；`no_hop_tail`（旧选项 513 / 506，row 393 / 386）、按池的 `srf`、对不上的第二次调度报 `LTS-E-ARG` |
| `test_trace.py` | trace：结构、MLP 的关键时间段（旧选项）、计数器（取自资源表；默认选项下 8 个 SRF 池）、`no_hop_tail_total_cycles`、流向箭头规则（含平行边、被跳过的箭头）、自检本身、跨进程确定性（默认选项） |
| `test_dot.py` | Graphviz 导出 |
| `test_cli.py` | CLI 端到端：`gen` / `run`（默认选项的三行输出、旧选项与混合选项、row 模式的 no_hop_tail）、`--srf-pool` / `--lw-streams` 与非法取值、输出目录规则、错误退出码、机器 id 不一致的警告、`python -m ltsim` |
| `test_golden.py` | **金标准快照**：MLP 与一个 single-bank GEMM 的 `result.json`（默认选项，含 `no_hop_tail`，不含 provenance）逐字节固定 |
| `test_oracle.py` | CP-SAT oracle（需要 ortools，没装时跳过）：`LB ≤ OPT ≤ SSGS`（含四种流池选项 × 带 / 不带 hop 尾巴）等，见下 |

**性质测试**用 `random.Random` 固定种子生成随机 GEMM 变体与合成链图（设计原本计划用 Hypothesis，环境里没有，改用此法），检查：

- 所有链间约束 `T_dst ≥ T_src + w` 成立；
- 任意时刻每个资源的占用不超过容量；
- `total ≥ LB`；
- 同一输入两次运行结果相同；
- `schedule()` 返回的是 est 之后最早的可行时刻（与逐拍参考调度器比较）。

**金标准快照**（`tests/ltsim/fixtures/mlp_result.json`、`gemm_single_result.json`）：设计 §10.3 要求 MLP 的 `result.json` 逐字节固定。
快照用默认选项（`half` + `separate`），带 `no_hop_tail`；去掉 `provenance`（机器文件的 sha256 会因 yaml 注释改动而变），其余逐字节比较。
MLP 的数字（506 / LB 506 / c0 → c4 → c6，no_hop_tail 506）与第 8.3 节的手算和 oracle 一致；GEMM 那份（754，no_hop_tail 754）只作回归，锁住当前行为。
2026-11-21 引入流资源池时按默认选项重新生成过一次（此前是旧选项下的 513 与 769）。结果**有意**改变时，用
`LTSIM_UPDATE_GOLDEN=1 python -m pytest tests/ltsim/test_golden.py` 重新生成，并写明原因。

**调度器 oracle（`ltsim/backend/oracle.py`，测试 `tests/ltsim/test_oracle.py`）：** 用 OR-Tools CP-SAT 求同一张链图、同一套资源下的最优链调度。
模型与 B3 同构：每条链的起始时刻是整数变量，链间边 `T_dst ≥ T_src + w`；独占资源上的预留区间 `NoOverlap`；
计数资源用合并后的分段（`fit_segments`）做 `Cumulative`；目标是最小化总周期。以 SSGS 的解作上界和初始提示；
求出的解再交给 `verify` 独立检查。ortools 只在函数里导入，ltsim 包不依赖它；没装时测试跳过。

- 性质：`LB ≤ OPT ≤ SSGS`，且 `OPT ≤ topo`。这同时校验了 B4 的下界确实是下界；
- oracle 只取 `cg.resources`，与 B3 用同一张资源表；`LB ≤ OPT ≤ SSGS` 在四种流池选项 × 带 / 不带 hop 尾巴下都测；
- 已知数（旧选项 `direction` + `shared`）：MLP 的 513 是最优（与 LB 506 的差全部来自下界偏松）；GEMM 256×640×1280（tile_m = 128）alternate 时 SSGS = OPT = 1267；
  single 时 OPT 是 1336，SSGS 是 1339，LB 1308（旧规则下是 1374 / 1434，SSGS 的损失大部分来自流预留的 wave 尾巴）；
- 已知数（默认选项）：MLP 506、GEMM alternate 1260、single 1308，三者都是 LB = OPT = SSGS；
- **观察（2026-11-08，一次性实验，报告未入库；是 §4.4 的 wave 规则修订之前测的，没有重测）：** 74 个随机 GEMM 变体（≤ 30 条链）上，SSGS 有 93% 恰好最优，平均多 0.30%，最多 7.47%
  （single + round_robin）；topo 平均多 6.00%。400 张随机合成链图（2–7 条链，资源更紧）上，SSGS 有 68% 恰好最优，平均多 4.69%。
  OPT 比 LB 平均高 2.46%（GEMM）；
- 与设计的偏离：设计说"≤ 30 条链用 CP-SAT"，这里一致；性质测试没有用 Hypothesis（环境里没有），改用 `random.Random` 固定种子生成随机图。

## 10. 开发顺序（即设计里程碑 M0，已全部完成；进度见 [roadmap.md](roadmap.md)）

1. ✅ **机器参数：** 先手写 `ltsim/machine/lpu-tile-v0.yaml`，取设计 3.4 的值；`derive_machine.py` 等 `LPU_SIM_ROOT` 可用时再补。
   完成标准：`test_machine.py` 通过。
2. ✅ **B0 + B1：** 能读入设计 6.3 的 JSON。完成标准：`test_work.py` 通过。
3. ✅ **B2：** 完成标准：第 8.1 节的数字全部对上。
4. ✅ **日历与 B3：** 完成标准：第 8.3 节的调度过程和总周期对上（当时是 551；2026-11-21 的 wave 规则修订后是 513，见 §4.4；同日引入流资源池后，默认选项下是 506，旧选项仍是 513，见 §8.3、§8.4）。
5. ✅ **B4 与 `result.json`：** 完成标准：下界与关键路径对上。
6. ✅ **Perfetto：** 完成标准：自检通过，在 <https://ui.perfetto.dev> 打开后能看到第 8 节描述的 E2W 流顶满和第一层推迟（用户已于 2026-11-08 目视确认）。
7. ✅ **CLI：** `python -m ltsim gen mlp|gemm`、`python -m ltsim run <tile_graph.json> --out <新目录>`（见 interfaces.md §6）。
8. ✅ **oracle 与性质测试。**
9. ✅ **收尾：** `schema.json` 与 B0 规则同步、金标准快照、Graphviz 导出（`run --dot`）。

## 11. 开工前要确认的点

- **一个节点的流同时向两个方向送出：** 4.4 约定每个方向各一条流预留。v0 的例子里没有这种情况，先实现成报错（`LTS-E-GRAPH-HOP`），
  遇到真实需要时再放开。
- **计数资源冲突时 `blocker` 的选法：** 5.1 取"那一拍贡献最大的链"。这只影响关键路径的显示，不影响调度结果。
- **`mem.write` 的 `V_in`：** 写节点没有 `out`，它的 V 从唯一的 stream 前驱取。写节点有多个 stream 前驱属于前端 bug，报 `LTS-E-GRAPH-*`。
- **节点内第 i 行的时刻（待定，M2 校准时定）：** 现在照设计 §7.2 用 `ceil(i / ρ)`，§4.3 的 r0 项、4.5 的同链逐 word 检查、
  4.6 的 row 模式都用它。ρ > 1 时它与占用区间 `[o, o+D)` 不自洽：ρ = 16、320 行时第 1–16 行都在第 1 拍，
  第 319 行在第 20 拍，已在区间之外。改用 `floor(i / ρ)` 则自洽。两者在 ρ ≤ 1 时多数相同或差 1 拍，MLP 与现有 GEMM 不受影响。
  依赖这一选择的数字：4.6 里运行时权重例子的 313（floor 下是 314）、`TestMemOverlapRow` 里的分数速率用例。
  M2 用 lpu-sim 校准时按实测选定，并重算这些数字。
- **VXM → VXM 的 stream 边（待定，M1 前要定）：** 设计 §5.4 中，对半树第 1 层与上游融合时有一条
  `[上游 vxm, 第 1 层 vxm, h]` 的 stream 边（例如 §4.5 attention 的 Cs 链）。B1 目前把同类别之间的 stream 边当作错误
  （`LTS-E-GRAPH-HOP`），hop 表里也没有 `VXM>VXM`。M0 的 MLP、GEMM 不涉及；M1 做 softmax / layer_norm 前要定：
  VXM 内部的传递计不计 hop、计多少，占不占流、占哪个方向。
