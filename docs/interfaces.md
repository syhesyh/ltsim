# LTSim 接口文档（M0）

> 状态：接口契约。本文只定义**模块边界与接口**，算法细节以 [backend.md](backend.md) 为准，两者冲突时以 backend.md 为准并回头修正本文。
> 设计文档（下称"设计"；lpu-compiler 仓库的 `Docs/lpu_tile_sim_design.md`，私有，出处见 [README](../README.md#参考资料)）只作参考，不保证正确。

## 0. 模块总览

```text
gen/builder.py, mlp.py, gemm.py ─▶ tile_graph.json ─▶ graph/load.py（B0：load_graph / validate_graph）
                                                            │ TileGraph           machine/model.py ◀── lpu-tile-v0.yaml
                                                            ▼                          │ Machine
                                    backend/work.py     (B1)  NodeWork  ◀─────────────┤
                                    backend/chains.py   (B2)  ChainGraph（Chain, ChainEdge, Reservation）
                                    backend/calendar.py + scheduler.py (B3)  Placed
                                    backend/bounds.py + metrics.py + result.py (B4) ─▶ result.json
                                    trace/perfetto.py ─▶ trace.json        graph/dot.py ─▶ graph.dot（可选）
                                    backend/oracle.py：CP-SAT 最优调度（只用于测试与诊断）

cli.py：python -m ltsim gen mlp|gemm … ／ python -m ltsim run <graph> [--machine …] [--dot] …
```

约定（设计 §10.1）：全部 `@dataclass(frozen=True)`；时间一律整数拍；速率用 `fractions.Fraction`；
只依赖标准库 + PyYAML（oracle 测试可用 OR-Tools）；错误用带编号的诊断码，退出码 2。

## 1. `ltsim/machine/` — 机器模型（B0 的硬件侧）

### 文件

- `lpu-tile-v0.yaml`：手写初值，取设计 §3.4 全表。每个参数注明来源（contract/derived/assumed/calibrated）。
- `model.py`：`Machine` 类与资源注册表。
- `derive_machine.py`：**本次不实现**（等 `LPU_SIM_ROOT` 可用）。

### 接口

```python
@dataclass(frozen=True)
class Machine:
    raw: Mapping[str, Any]
    def param(self, path: str) -> int: ...        # "mxm.lw_to_acc_gap" → int；缺则 LTS-E-MACHINE
    def hop(self, src_cat: str, dst_cat: str) -> int: ...   # hops["MEM_W>MXM_W"]；缺则 LTS-E-MACHINE
    def op_latency(self, alu_op: str) -> int: ...           # vxm.op_latency["gelu.f32"]；缺则 LTS-E-MACHINE
    def resources(self, *, srf_pool="half", lw_streams="separate") -> Mapping[str, Resource]: ...
        # 全部资源 id → Resource（见下）；只供 build_chains 调用，其余模块从 ChainGraph.resources 取

@dataclass(frozen=True)
class Resource:
    id: str                   # "mxm.w0.compute" / "vxm.alu" / "srf.E2W.W" / "mem.W.rd" / "sxm.w" ...
    kind: Literal["exclusive", "counter"]
    cap: int                  # exclusive 恒为 1；counter 为容量（16 / 32 / 44）
```

`resources()` 从 yaml 推导：4 个 plane 各 `weight`/`compute`（独占）、`vxm.alu`(16)、
SRF 流池（每个 32，按选项划分，见下）、`mem.W.rd`/`mem.W.wr`/`mem.E.rd`/`mem.E.wr`(44)、`sxm.w`/`sxm.e`（独占）。

SRF 流池（backend.md §4.4.1；`srf_pool_names(srf_pool, lw_streams)` 给出全部池 id，按名字排序）：

| `srf_pool` | `lw_streams` | 池 |
|---|---|---|
| `half`（默认） | `separate`（默认） | `srf.{W2E,E2W}.{W,E}` 与各自的 `.lw`，共 8 个 |
| `half` | `shared` | `srf.{W2E,E2W}.{W,E}`，4 个 |
| `direction` | `separate` | `srf.W2E`、`srf.E2W`、`srf.W2E.lw`、`srf.E2W.lw` |
| `direction` | `shared` | `srf.W2E`、`srf.E2W`（M0 早先的资源表） |

常量 `SRF_POOL_MODES = ("half", "direction")`、`LW_STREAM_MODES = ("separate", "shared")`；取值不对报 `LTS-E-ARG`。

加载入口在 `model.py`：`load_machine(path) -> Machine`、`default_machine() -> Machine`（内置 `lpu-tile-v0.yaml`）。

### 诊断

- `LTS-E-MACHINE`：参数路径不存在、yaml 结构/schema 不符、必需字段缺失。
- `LTS-E-ARG`：`resources()` 的 `srf_pool` / `lw_streams` 取值不对。

## 2. `ltsim/graph/` — tile graph 表示与加载（B0 的图侧）

### 文件

- `schema.json`：JSON Schema，对应设计 §6.2（schema 字符串 `lts-tile-graph/0.2`）。
- `tile_graph.py`：数据结构。
- `load.py`：加载 + 校验。
- `dot.py`：`to_dot(g, chain_of=None, name=...) -> str`，Graphviz 导出；给出 `chain_of` 时每条链一个框；CLI `run --dot` 写 `graph.dot`。

### 数据结构

```python
@dataclass(frozen=True)
class Tensor:
    dtype: str                # i8/i32/f16/bf16/f32
    shape: tuple[int, ...]    # rank 2 或 rank 1
    role: str                 # input/param/output/intermediate
    hemisphere: str           # "W" | "E"
    layout: str               # "row" | "lane"
    alias_of: str | None      # transpose 输出共享映像；None 表示非别名

@dataclass(frozen=True)
class Out:                    # 流值：节点送出的数据
    dtype: str
    V: int                    # 每行占几条流（向量数）

@dataclass(frozen=True)
class Node:
    id: str
    kind: str                 # mem.read/mem.write/mxm.load_w/mxm.acc/vxm.chain/sxm.*
    rows: int                 # 物理行数
    origin: tuple[str, ...]   # 来源 LTS-IR 算子 id
    tile: str | None          # mem.* 用，"<T>[r0:r1, c0:c1]" 或 "<T>[n0:n1]"
    unit: str | None          # mxm.* / sxm.* 用
    bank: int | None          # mxm.load_w / mxm.acc 用
    ops: tuple[str, ...] | None   # vxm.chain 用
    repeat: int | None        # mem.read 广播时用
    out: Out | None           # 有流输出的节点

@dataclass(frozen=True)
class TileGraph:
    schema: str               # "lts-tile-graph/0.2"
    machine: str              # 机器参数文件 id
    source: str
    tensors: Mapping[str, Tensor]
    nodes: tuple[Node, ...]
    edges_stream: tuple[tuple[str, str, int], ...]      # (src, dst, r0)；r0 默认 0
    edges_state:  tuple[tuple[str, str], ...]
    edges_mem:    tuple[tuple[str, str], ...]
    edges_order:  tuple[OrderEdge, ...]

@dataclass(frozen=True)
class OrderEdge:
    src: str; dst: str
    ref: str                  # "start" | "issue_end" | "done"
    lag: int | str            # 整数，或机器参数路径（加载时解析成 int）
    reason: str               # acc_seq / block_switch / lw_after_iw / bank_war
```

### 接口与校验（`load.py`）

```python
def load_graph(path: str) -> TileGraph: ...
def validate_structure(g: TileGraph) -> None: ...                 # B0 中不需要机器参数的全部检查（load_graph 自动调用）
def validate_graph(g: TileGraph, m: Machine) -> None: ...         # 完整 B0 = validate_structure + unit 名单 + lane 宽
def tile_region(g, tile) -> (root, word 区间, lane 区间 | None): ... # 物理区域；B0 一致性检查与 B2 逐行检查共用
def alias_root(g, name) -> str: ...                               # 沿 alias_of 展开到源张量
def parse_tile(tile: str) -> tuple[str, tuple[tuple[int, int], ...]]: ...   # "H[0:128,0:320]" → ("H", ((0,128),(0,320)))
def tile_lanes(g, tile) -> int | None: ...                        # 物理 lane 数；rank 1 的 lane 布局返回 None
def resolve_lags(g: TileGraph, m: Machine) -> ResolvedGraph: ...   # lag 字符串 → int；state/mem 展开默认 ref/lag
```

校验规则（backend.md §2）：节点 id 唯一；边两端存在；stream 起点有 `out`；
`mxm.*.unit` 在 plane 名单；`alias_of` 展开（节点 tile 指向别名时按源张量取半球）；
`mem.write` 有唯一 stream 前驱（取 `V_in`）；tile 合法且在界内（`LTS-E-GRAPH-TILE`）；
每个 tile 的 lane 数不超过 `superlanes × lanes_per_superlane`（`LTS-E-GRAPH-LANES`，见 backend.md §2 第 7 条）。
字段取值、类型、每种 kind 的字段白名单、边的元素个数、未知键（`LTS-E-GRAPH-FIELD`，手写校验，不读 schema.json）；
r0 小于起点 rows（`LTS-E-GRAPH-STREAM`）；alias 是 rank 2 的转置、布局相反、dtype 与半球相同（`LTS-E-GRAPH-REF`）；rows 与 tile 的 word 数一致（`LTS-E-GRAPH-ROWS`）；
mem 边 write→read、同一映像、区域相交（`LTS-E-GRAPH-MEM`）；state 边 load_w→acc、同 unit 同 bank（`LTS-E-GRAPH-STATE`）。
`builder.build()` 只查引用，完整 B0 校验调 `validate_graph`。
失败报 `LTS-E-GRAPH-*` / `LTS-E-MACHINE`。

## 3. `ltsim/gen/` — 通用 DAG builder

不做算子级模式；用户逐节点构造，builder 负责校验、字段补全与确定性输出。

### 接口

```python
class GraphBuilder:
    def __init__(self, name: str, machine: str): ...
    def tensor(self, name, dtype, shape, role, hemisphere, layout, alias_of=None) -> None: ...
    def read(self, id, tile, rows, out: Out, origin, repeat=None) -> None: ...
    def write(self, id, tile, rows, origin) -> None: ...
    def load_w(self, id, unit, bank, rows, origin) -> None: ...
    def acc(self, id, unit, bank, rows, out: Out, origin) -> None: ...
    def vxm(self, id, ops: list[str], rows, out: Out, origin) -> None: ...
    def stream(self, src, dst, r0=0) -> None: ...
    def state(self, src, dst) -> None: ...
    def mem(self, src, dst) -> None: ...
    def order(self, src, dst, ref, lag, reason) -> None: ...
    def build(self) -> TileGraph: ...        # 走 graph/ 的同一套校验
    def save(self, path) -> None: ...        # 确定性输出：同输入两次运行字节相同
```

### 展开辅助（循环友好的小工具）

```python
def tile_str(tensor, r0, r1, c0=None, c1=None) -> str: ...   # "W1[0:320,0:320]"；rank 1 时 "B1[0:320]"
def plane_for(planes, map, n, k, j) -> str: ...              # k_split 取 k、n_split 取 n、round_robin 取块序号 j
def bank_for(j, banks) -> int: ...                           # alternate: j%2；single: 0

# GraphBuilder 的方法。blocks 是一个 plane 上按装载顺序排列的权重块：
#   [(load_w 节点 id, [该块的 acc id，按执行顺序即 m 升序]), ...]
b.link_weight_order(blocks, banks)   # state + lw_after_iw（+ bank_war，仅 single）
b.link_acc_order(blocks)             # acc_seq + block_switch
b.link_plane(blocks, banks)          # 以上两者
```

生成的边与设计 §5.5 的表一一对应；`block_switch` 的 lag 写参数名 `"mxm.block_switch_gap"`，
由后端 `resolve_lags` 解析，tile graph 中不出现周期数。

### `gemm.py` 参数化 GEMM

`build_gemm(M, K, N, *, tile_m, tile_k=None, tile_n=None, planes=("mxm.w0","mxm.w1"), map="n_split", banks="alternate", p_placement="auto", machine=None)`
（`machine` 省略时用 `default_machine()`；`tile_k`/`tile_n` 省略时取 `mxm.block_k_rows`/`block_n_cols`；每行 V 按 `superlanes × lanes_per_superlane` 计算，不写死 320）
按设计 §5.4 的 matmul 规则展开 `Y = X · W`，用来覆盖"每个 plane 多个权重块"的场景。有意简化：
row 形式、INT8→INT32、无 bias/激活；Kb > 1 时 K 合并不融合（partial 写回 MEM，再由 vxm.chain 读回求和）。

planes 可以跨东西两侧（用满 4 个 MXM）：n 块前一半归西侧 plane、后一半归东侧（要求 Nb ≥ 2，否则 `LTS-E-GEN`），
`map` 在侧内适用；双侧模式下 X/W/Y 按侧复制为 XW/XE、WW/WE、YW/YE，partial 张量按侧拆分为 `P{k}W`/`P{k}E`。
单侧模式（默认）输出与此前的版本逐字节一致。

`p_placement` 决定 Kb > 1 时 partial 张量的半球（常量 `P_PLACEMENT_MODES`，非法值报 `LTS-E-ARG`）：
`local` 全放本侧；`alternate` 按 k 奇偶在本侧/对侧交替，合并链的读流对半分到东西两个流池（Kb × V 上限 32 → 64）；
`auto`（默认）只在 Kb × V > `srf.streams_per_direction` 时按 alternate。ceil(Kb/2) × V > 32 仍报
`LTS-E-SRF-WIDTH`：那需要树形 K 合并，目前未实现。

### `mlp.py` 金标准

`python -m ltsim gen mlp --out <dir>` 用 builder 复刻设计 §6.3 的 MLP：
25 节点 / 7 链 / 24 边，输出 JSON 与设计 §6.3 **逐字段一致**（节点 id、字段、顺序）。

## 4. `ltsim/backend/` — B1–B4

数据结构（`NodeWork`/`Reservation`/`Chain`/`ChainEdge`/`Placed`/`BoundBy`）
**按 backend.md §1 原文定义，不改**。以下为函数签名。

### B1 `work.py`

```python
def node_works(g: TileGraph, m: Machine) -> dict[str, NodeWork]: ...                  # backend.md §3.2 公式
def stream_hops(g, works, m) -> dict[tuple[str, str], int]: ...                       # 每条 stream 边的 hop × hop_cycles
def out_directions(g, works) -> dict[str, str]: ...                                   # 有流消费者的节点 → "W2E"/"E2W"
def stream_preds(g) -> dict[str, list[str]]: ...                                      # 每个节点的 stream 前驱
def write_v_in(g) -> dict[str, int]: ...                                              # mem.write → V_in（B2 算写带宽用）
def node_category(n, g, m) -> str: ...      # MEM_W/MEM_E（按 tile 张量的半球，展开 alias_of）/MXM_W/MXM_E/VXM
def direction(src_cat, dst_cat) -> str: ... # x 序 MXM_W < MEM_W < VXM < MEM_E < MXM_E
```

诊断：同 category 的 stream 边、同一流值送往两个方向 → `LTS-E-GRAPH-HOP`；
mem.write 没有唯一 stream 前驱 → `LTS-E-GRAPH-STREAM`；v0 不支持的 kind（含 `sxm.*`）→ `LTS-E-GRAPH-KIND`；
参数或 vxm 操作缺失 → `LTS-E-MACHINE`；
MXM 节点流入数据的精度不在 `lw_rows_per_cycle` / `acc_latency` 表里 → `LTS-E-UNIT-OP`。

### B2 `chains.py`

```python
def build_chains(g: TileGraph, m: Machine, *, mem_overlap: str = "tile", srf_pool: str = "half",
                 lw_streams: str = "separate", hop_tail: bool = True) -> ChainGraph: ...
# ChainGraph(chains, edges, chain_of, works, resources)；Chain / ChainEdge / Reservation 见 backend.md §1
def srf_pools(n, succ, by, works, direction, srf_pool, lw_streams) -> tuple[str, ...]: ...   # 一个节点的流预留进哪些池（§4.4.1）
```

前提：`g` 已通过 B0（`validate_graph`）。内部调用 B1 的 `node_works` / `stream_hops` / `out_directions` / `write_v_in`
和 B0 的 `resolve_lags` / `tile_region`。

内含：并查集组链（链 id 规则 §4.1）；rate / D；链内偏移用等式传播（§4.3）；预留表（§4.4）；
流预留按 `srf_pool` / `lw_streams` 分池、`hop_tail=False` 时不带 hop 尾巴（§4.4.1）；
链级合法性（§4.5，含同链 mem 边逐 word 检查，流宽逐池检查）；链间约束（§4.6，全部保留不去重；`mem_overlap="row"` 时所有跨链 mem 边用逐 word 精确公式）；
链间约束图用 Kahn 检查无环。

诊断：`LTS-E-CHAIN-ALIGN` / `LTS-E-CHAIN-SELF` / `LTS-E-CHAIN-CYCLE` / `LTS-E-SRF-WIDTH` / `LTS-E-GRAPH-CYCLE`；
`mem_overlap` / `srf_pool` / `lw_streams` / `hop_tail` 取值不对报 `LTS-E-ARG`。

**资源表只有一个来源：`ChainGraph.resources`。** 它是 `build_chains` 按选项调 `m.resources(srf_pool=…, lw_streams=…)`
得到的那张表，组链的检查就是对着它做的。B3、B4、trace、oracle 的函数都不再接受资源表参数，一律取 `cg.resources`；
手工构造的 `ChainGraph`（测试里的合成链图）要自己带上资源表。

### B3 `calendar.py` + `scheduler.py`

```python
class ExclusiveCalendar:
    def conflict(self, s: int, e: int) -> tuple[int, str] | None: ...   # backend.md §5.1
    def commit(self, s: int, e: int, owner: str) -> None: ...

class CounterCalendar:
    def __init__(self, cap: int): ...
    def conflict(self, s: int, e: int, k: int) -> tuple[int, str] | None: ...   # §5.1
    def commit(self, s: int, e: int, k: int, owner: str) -> None: ...
    def level_at(self, t: int) -> int: ...
    def steps(self) -> list[tuple[int, int]]: ...                     # (时刻, 占用量) 变化点

def schedule(cg: ChainGraph, policy: str = "bottom_level") -> Placed: ...          # 资源表取 cg.resources
# policy: "bottom_level" | "topo"；"random:K:seed" 推迟实现，报 LTS-E-ARG
def bottom_levels(cg) -> dict[str, int]: ...
def fit_segments(chain, resources) -> tuple[Segment, ...]: ...      # 计数资源上同一链的重叠预留合并成分段（§5.3）；调用方传 cg.resources
def verify(cg, placed) -> list[str]: ...                            # 独立检查约束与容量（对 cg.resources），空表示合法
```

`Placed(policy, start, bound_by, order, total)`；`BoundBy.kind ∈ {source, edge, resource, policy}`；
除 source 外都有 `blocker`（卡住它的链，edge 为边的源链），关键路径沿它回溯（backend.md §1、§6）。

### B4 `bounds.py` + `metrics.py` + `result.py`

```python
def compute_bounds(cg, placed) -> Bounds: ...                       # 资源表取 cg.resources
#   Bounds: total, path, resource(ceil), resource_exact(Fraction), resource_name, lb, gap(Fraction), earliest(ES)
def critical_path(cg, placed) -> tuple[CriticalStep, ...]: ...     # 起点 → 终点；CriticalStep(chain, bound_by)
def mxm_roofline(g, cg, placed, m) -> Fraction: ...                # 按实际所用权重块算 MAC（backend.md §6）

def node_times(cg, placed) -> dict[str, (start, issue_end, done)]: ...
def resource_usage(cg, placed) -> dict[str, {kind, cap, busy, area, peak, utilization, avg}]: ...   # cg.resources 的每个资源
def op_spans(g, cg, placed) -> dict[str, (start, end)]: ...         # 按 origin 聚合

def make_provenance(g, m, *, policy, mem_overlap, srf_pool, lw_streams,
                    tile_graph_path=None, machine_path=None, command=None) -> dict: ...
def build_result(g, cg, placed, m, provenance, diagnostics=None, *,
                 no_hop_tail: tuple[ChainGraph, Placed] | None = None) -> dict: ...
#   schema lts-result/0.2；内含 verify 与 total ≥ LB 自检（两次调度都做）。cg / placed 是主结果（带 hop 尾巴）；
#   no_hop_tail 是同一张图、同样选项、build_chains(..., hop_tail=False) 的 (链图, 调度)，写进 "no_hop_tail" 字段，
#   不给时为 null；两次的链或资源表对不上报 LTS-E-ARG
def dumps_result(result) -> str: ...
def write_result(path, result) -> None: ...                         # 字段见 backend.md §6
```

### 调度器 oracle `oracle.py`（只用于测试和诊断）

```python
def solve_optimal(cg, *, time_limit_s=10.0, workers=8) -> OracleResult: ...      # 资源表取 cg.resources，与 B3 相同
#   OracleResult(status "OPTIMAL"|"FEASIBLE", makespan, bound, start, ssgs, wall_s)；需要 ortools，没装报 LTS-E-ARG
```

## 5. `ltsim/trace/perfetto.py`

```python
def build_trace(g, cg, placed, m, *, provenance=None, stream_flows=False,
                no_hop_tail_total=None) -> dict: ...         # 内含 check_trace 自检；资源取 cg.resources
def counter_names(resources) -> list[str]: ...    # 计数器：全部 SRF 池（排序）、vxm.alu、mem.*
def check_trace(trace) -> list[str]: ...          # backend.md §7.2，空表示通过
def dumps_trace(trace) -> str: ...
def write_trace(path, trace) -> None: ...
```

- 9 个固定 pid；节点、链、算子、关键路径的 `X` 事件；计数器的 `C` 事件（资源表里全部 SRF 池 + `vxm.alu` + 4 个 `mem.*`）；
  `no_hop_tail_total` 给出时写进 `otherData.no_hop_tail_total_cycles`，trace 只画主结果；链间边（可选加链内 stream 边）的 `s` / `f` 流向事件。
- 流向事件按边依赖的时刻放 s，f 不早于 s；关键路径分行。这两处与设计不同，见 backend.md §7.1。
- 自检不通过报 `LTS-E-TRACE`。

## 6. `ltsim/cli.py`

```bash
python -m ltsim gen mlp  --out <新目录>                                   # 金标准 tile_graph.json（设计 §6.3）
python -m ltsim gen gemm --M 256 --K 640 --N 1280 --tile-m 128 \
       [--tile-k K --tile-n N --planes w0,w1 --map n_split --banks alternate \
        [--p-placement auto|local|alternate] --machine <id|yaml>] --out <新目录>
python -m ltsim run <tile_graph.json> [--machine <id|yaml>] --out <新目录> \
       [--policy bottom_level|topo] [--mem-overlap tile|row] \
       [--srf-pool half|direction] [--lw-streams separate|shared] [--trace-stream-flows] [--dot]
```

- `run`：B0 → B1 → B2 → B3 → B4 → trace，在输出目录写 `result.json`、`trace.json`；终端打印总周期、下界、gap 与关键路径；
- `run` **调度两次**，只差 `hop_tail`：带 hop 尾巴的是主结果（`total_cycles`、trace、关键路径），不带的写进 `result.json` 的
  `no_hop_tail`，终端多打一行 `without SRF hop tail: total_cycles N   lower bound L   gap x%`（`--mem-overlap tile|row` 都这样）；
- `--srf-pool half|direction`（默认 half）、`--lw-streams separate|shared`（默认 separate）：SRF 流池的划分（backend.md §4.4.1）。
  旧行为是 `--srf-pool direction --lw-streams shared`。非法取值由 argparse 拒绝（退出码 2，与 `--mem-overlap` 相同）；
  API 层（`build_chains` / `Machine.resources`）报 `LTS-E-ARG`；
- `--machine` 写机器 id（在 `ltsim/machine/<id>.yaml` 找）或 yaml 路径；省略时用图里的 `machine` 字段。
  与图里的 id 不一致时**照常运行**，在 `result.json` 的 `diagnostics` 里记 `LTS-W-MACHINE-ID` 警告，并打到 stderr。
  这样同一张图可以直接换一份改过参数的机器文件做设计空间探索；
- 输出目录必须是新目录（已存在报 `LTS-E-ARG`）。先全部算完、成功后才建目录，失败时可以用同名重跑；
- `result.json` 的 `provenance` 记录完整命令行、输入文件和机器文件的 sha256、策略、mem 语义、`srf_pool`、`lw_streams`；
- 出错时把 `error: LTS-E-…: 说明` 打到 stderr，退出码 2；
- `random:K:seed` 策略推迟实现（backend.md §5.3）；`sim`（LTS-IR → 前端 → 后端）等 M1 前端完成后再加。

## 7. 测试（`tests/ltsim/`）

文件清单与每个文件测什么，见 [backend.md](backend.md) §9。要点：

- 金标准：设计 §6.3 的 MLP 在 B1–B4 的每一步都与 backend.md §8 的手算表逐项一致；`result.json` 有快照（`test_golden.py`）；
- 性质测试用 `random.Random` 固定种子（环境里没有 Hypothesis）：约束成立、资源不超容量、`total ≥ LB`、确定性、
  `schedule()` 与逐拍参考调度器一致；
- oracle（需要 OR-Tools，即可选依赖 `oracle`；没装时跳过）：`LB ≤ OPT ≤ SSGS`，含四种流池选项 × 带 / 不带 hop 尾巴。

## 8. 开发顺序

M0 的 10 步已全部完成（backend.md §10）。后续阶段的计划与进度见 [roadmap.md](roadmap.md)。

## 9. 已定的实现约定

- 节点流分两个方向送出：v0 报 `LTS-E-GRAPH-HOP`（backend.md §11）。
- `CounterCalendar.conflict` 的 blocker 取"该拍贡献最大的链"，平局取最先提交的：只影响关键路径显示。
- `mem.write` 的 `V_in` 取唯一 stream 前驱的 `out.V`；没有或有多个前驱报 `LTS-E-GRAPH-STREAM`。
- `--mem-overlap row`：所有跨链 mem 边用逐 word 的精确公式（backend.md §4.6），默认 `tile`。
- SRF 流池默认 `half` + `separate`，CLI 总是同时给出带 / 不带 hop 尾巴的两个结果（backend.md §4.4.1；两条划分规则是假设，见 roadmap 待定项 T11–T13）。

跨阶段的待定项见 [roadmap.md](roadmap.md) 第 3 节。

## 10. 诊断码一览

错误以 `LTSError(code, msg)` 抛出（CLI 打印到 stderr，退出码 2）；警告写进 `result.json` 的 `diagnostics`，不中断运行。
新增诊断码时更新本表。

| 码 | 阶段 | 含义 |
|---|---|---|
| `LTS-E-ARG` | CLI / API | 参数不对：输出目录已存在、`policy` / `mem_overlap` / `srf_pool` / `lw_streams` / `hop_tail` 取值不对、`build_result` 的 `no_hop_tail` 与主结果对不上、用 oracle 但没装 OR-Tools 等 |
| `LTS-E-MACHINE` | 机器参数 | yaml 结构或推导量不符、参数路径不存在、节点用到机器里没有的资源 |
| `LTS-E-GEN` | 生成器 | builder / GEMM 参数不合法：tile 不整除维度、plane 不在同一侧、未知的 `map` / `banks`、权重块没有 acc |
| `LTS-E-MXM-BLOCK` | 生成器 | GEMM 的 `tile_k` / `tile_n` 超过 MXM 块大小 |
| `LTS-E-GRAPH-PARSE` | B0 | 顶层不是 JSON 对象，或 `schema` 不是 `lts-tile-graph/0.2`（JSON 文本本身的语法错误目前不会转成诊断码，会直接抛出 Python 异常） |
| `LTS-E-GRAPH-FIELD` | B0 | 字段缺失、类型不对、未知键、边的元素个数不对 |
| `LTS-E-GRAPH-DUP` | B0 / builder | 节点 id 重复 |
| `LTS-E-GRAPH-REF` | B0 | 引用了不存在的节点或张量；`alias_of` 不合法或成环 |
| `LTS-E-GRAPH-TILE` | B0 | tile 写法错误、维数不符或越界 |
| `LTS-E-GRAPH-ROWS` | B0 | `rows` 与 tile 的 word 数不一致；广播读的形式不对 |
| `LTS-E-GRAPH-LANES` | B0 | tile 的 lane 数超过 `superlanes × lanes_per_superlane` |
| `LTS-E-GRAPH-STREAM` | B0 / B1 | stream 边的源没有 `out`、`r0` 越界；`mem.write` / MXM 节点没有唯一的 stream 前驱 |
| `LTS-E-GRAPH-MEM` | B0 / B2 | mem 边不是 write → read、不在同一个 MEM 映像、读写区域不相交 |
| `LTS-E-GRAPH-STATE` | B0 | state 边不是 load_w → acc，或两端的 unit / bank 不一致 |
| `LTS-E-GRAPH-UNIT` | B0 / B1 | 节点的 `unit` 不在机器的 MXM plane 列表里 |
| `LTS-E-GRAPH-KIND` | B1 | v0 不支持的节点 kind（含 `sxm.*`），或 `vxm.chain` 没有 op |
| `LTS-E-GRAPH-HOP` | B1 | 同一位置类别之间的 stream 边；一个节点的输出流向东西两个方向 |
| `LTS-E-UNIT-OP` | B1 | MXM 输入精度不在 `lw_rows_per_cycle` / `acc_latency` 表里 |
| `LTS-E-GRAPH-CYCLE` | B2 | 一条链内的 stream 边成环 |
| `LTS-E-CHAIN-ALIGN` | B2 | 链内两条路径要求同一节点有不同的偏移。流不能等待，需要拆开融合组，让这个值经 MEM 中转 |
| `LTS-E-CHAIN-SELF` | B2 | 链自身不合法：独占资源重叠、计数资源超容量、同链 mem 边读早于写可见、同链 order 边不满足 |
| `LTS-E-SRF-WIDTH` | B2 | 一条链在某一时刻、某个 SRF 池里需要的流超过池的容量（`srf.streams_per_direction`） |
| `LTS-E-CHAIN-CYCLE` | B2 | 链间约束图成环 |
| `LTS-E-TRACE` | trace | 生成的 trace 自检不通过 |
| `LTS-E-INTERNAL` | B3 / B4 / oracle | 内部自检失败（调度违反约束、总周期小于下界、关键路径成环等），属于 bug |
| `LTS-W-MACHINE-ID` | CLI（警告） | `--machine` 的 id 与图里写的不同；照常运行 |
