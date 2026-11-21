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
    def resources(self) -> Mapping[str, Resource]: ...      # 全部资源 id → Resource（见下）

@dataclass(frozen=True)
class Resource:
    id: str                   # "mxm.w0.compute" / "vxm.alu" / "srf.W2E" / "mem.W.rd" / "sxm.w" ...
    kind: Literal["exclusive", "counter"]
    cap: int                  # exclusive 恒为 1；counter 为容量（16 / 32 / 44）
```

`resources()` 从 yaml 推导：4 个 plane 各 `weight`/`compute`（独占）、`vxm.alu`(16)、
`srf.W2E`/`srf.E2W`(32)、`mem.W.rd`/`mem.W.wr`/`mem.E.rd`/`mem.E.wr`(44)、`sxm.w`/`sxm.e`（独占）。

### 诊断

- `LTS-E-MACHINE`：参数路径不存在、yaml 结构/schema 不符、必需字段缺失。

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
def load_machine(path: str) -> Machine: ...
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

`build_gemm(M, K, N, *, tile_m, tile_k=None, tile_n=None, planes=("mxm.w0","mxm.w1"), map="n_split", banks="alternate", machine=None)`
（`machine` 省略时用 `default_machine()`；`tile_k`/`tile_n` 省略时取 `mxm.block_k_rows`/`block_n_cols`；每行 V 按 `superlanes × lanes_per_superlane` 计算，不写死 320）
按设计 §5.4 的 matmul 规则展开 `Y = X · W`，用来覆盖"每个 plane 多个权重块"的场景。有意简化：
row 形式、INT8→INT32、无 bias/激活；Kb > 1 时 K 合并不融合（partial 写回 MEM，再由 vxm.chain 读回求和）。

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
def build_chains(g: TileGraph, m: Machine, *, mem_overlap: str = "tile") -> ChainGraph: ...
# ChainGraph(chains, edges, chain_of, works)；Chain / ChainEdge / Reservation 见 backend.md §1
```

前提：`g` 已通过 B0（`validate_graph`）。内部调用 B1 的 `node_works` / `stream_hops` / `out_directions` / `write_v_in`
和 B0 的 `resolve_lags` / `tile_region`。

内含：并查集组链（链 id 规则 §4.1）；rate / D；链内偏移用等式传播（§4.3）；预留表（§4.4）；
链级合法性（§4.5，含同链 mem 边逐 word 检查）；链间约束（§4.6，全部保留不去重；`mem_overlap="row"` 时所有跨链 mem 边用逐 word 精确公式）；
链间约束图用 Kahn 检查无环。

诊断：`LTS-E-CHAIN-ALIGN` / `LTS-E-CHAIN-SELF` / `LTS-E-CHAIN-CYCLE` / `LTS-E-SRF-WIDTH` / `LTS-E-GRAPH-CYCLE`；
`mem_overlap` 取值不对报 `LTS-E-ARG`。

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

def schedule(cg: ChainGraph, resources: Mapping[str, Resource], policy: str = "bottom_level") -> Placed: ...
# policy: "bottom_level" | "topo"；"random:K:seed" 推迟实现，报 LTS-E-ARG
def bottom_levels(cg) -> dict[str, int]: ...
def fit_segments(chain, resources) -> tuple[Segment, ...]: ...      # 计数资源上同一链的重叠预留合并成分段（§5.3）
def verify(cg, placed, resources) -> list[str]: ...                 # 独立检查约束与容量，空表示合法
```

`Placed(policy, start, bound_by, order, total)`；`BoundBy.kind ∈ {source, edge, resource, policy}`；
除 source 外都有 `blocker`（卡住它的链，edge 为边的源链），关键路径沿它回溯（backend.md §1、§6）。

### B4 `bounds.py` + `metrics.py` + `result.py`

```python
def compute_bounds(cg, placed, resources) -> Bounds: ...
#   Bounds: total, path, resource(ceil), resource_exact(Fraction), resource_name, lb, gap(Fraction), earliest(ES)
def critical_path(cg, placed) -> tuple[CriticalStep, ...]: ...     # 起点 → 终点；CriticalStep(chain, bound_by)
def mxm_roofline(g, cg, placed, m) -> Fraction: ...                # 按实际所用权重块算 MAC（backend.md §6）

def node_times(cg, placed) -> dict[str, (start, issue_end, done)]: ...
def resource_usage(cg, placed, resources) -> dict[str, {kind, cap, busy, area, peak, utilization, avg}]: ...
def op_spans(g, cg, placed) -> dict[str, (start, end)]: ...         # 按 origin 聚合

def make_provenance(g, m, *, policy, mem_overlap, tile_graph_path=None, machine_path=None, command=None) -> dict: ...
def build_result(g, cg, placed, m, provenance) -> dict: ...         # 内含 verify 与 total ≥ LB 自检
def dumps_result(result) -> str: ...
def write_result(path, result) -> None: ...                         # 字段见 backend.md §6
```

### 调度器 oracle `oracle.py`（只用于测试和诊断）

```python
def solve_optimal(cg, resources, *, time_limit_s=10.0, workers=8) -> OracleResult: ...
#   OracleResult(status "OPTIMAL"|"FEASIBLE", makespan, bound, start, ssgs, wall_s)；需要 ortools，没装报 LTS-E-ARG
```

## 5. `ltsim/trace/perfetto.py`

```python
def build_trace(g, cg, placed, m, *, provenance=None, stream_flows=False) -> dict: ...   # 内含 check_trace 自检
def check_trace(trace) -> list[str]: ...          # backend.md §7.2，空表示通过
def dumps_trace(trace) -> str: ...
def write_trace(path, trace) -> None: ...
```

- 9 个固定 pid；节点、链、算子、关键路径的 `X` 事件；7 个计数器的 `C` 事件；链间边（可选加链内 stream 边）的 `s` / `f` 流向事件。
- 流向事件按边依赖的时刻放 s，f 不早于 s；关键路径分行。这两处与设计不同，见 backend.md §7.1。
- 自检不通过报 `LTS-E-TRACE`。

## 6. `ltsim/cli.py`

```bash
python -m ltsim gen mlp  --out <新目录>                                   # 金标准 tile_graph.json（设计 §6.3）
python -m ltsim gen gemm --M 256 --K 640 --N 1280 --tile-m 128 \
       [--tile-k K --tile-n N --planes w0,w1 --map n_split --banks alternate --machine <id|yaml>] --out <新目录>
python -m ltsim run <tile_graph.json> [--machine <id|yaml>] --out <新目录> \
       [--policy bottom_level|topo] [--mem-overlap tile|row] [--trace-stream-flows]
```

- `run`：B0 → B1 → B2 → B3 → B4 → trace，在输出目录写 `result.json`、`trace.json`；终端打印总周期、下界、gap 与关键路径；
- `--machine` 写机器 id（在 `ltsim/machine/<id>.yaml` 找）或 yaml 路径；省略时用图里的 `machine` 字段。
  与图里的 id 不一致时**照常运行**，在 `result.json` 的 `diagnostics` 里记 `LTS-W-MACHINE-ID` 警告，并打到 stderr。
  这样同一张图可以直接换一份改过参数的机器文件做设计空间探索；
- 输出目录必须是新目录（已存在报 `LTS-E-ARG`）。先全部算完、成功后才建目录，失败时可以用同名重跑；
- `result.json` 的 `provenance` 记录完整命令行、输入文件和机器文件的 sha256、策略、mem 语义；
- 出错时把 `error: LTS-E-…: 说明` 打到 stderr，退出码 2；
- `random:K:seed` 策略推迟实现（backend.md §5.3）；`sim`（LTS-IR → 前端 → 后端）等 M1 前端完成后再加。

## 7. 测试（`tests/ltsim/`）

文件清单与每个文件测什么，见 [backend.md](backend.md) §9。要点：

- 金标准：设计 §6.3 的 MLP 在 B1–B4 的每一步都与 backend.md §8 的手算表逐项一致；`result.json` 有快照（`test_golden.py`）；
- 性质测试用 `random.Random` 固定种子（环境里没有 Hypothesis）：约束成立、资源不超容量、`total ≥ LB`、确定性、
  `schedule()` 与逐拍参考调度器一致；
- oracle（需要 OR-Tools，即可选依赖 `oracle`；没装时跳过）：`LB ≤ OPT ≤ SSGS`。

## 8. 开发顺序

M0 的 10 步已全部完成（backend.md §10）。后续阶段的计划与进度见 [roadmap.md](roadmap.md)。

## 9. 已定的实现约定

- 节点流分两个方向送出：v0 报 `LTS-E-GRAPH-HOP`（backend.md §11）。
- `CounterCalendar.conflict` 的 blocker 取"该拍贡献最大的链"，平局取最先提交的：只影响关键路径显示。
- `mem.write` 的 `V_in` 取唯一 stream 前驱的 `out.V`；没有或有多个前驱报 `LTS-E-GRAPH-STREAM`。
- `--mem-overlap row`：所有跨链 mem 边用逐 word 的精确公式（backend.md §4.6），默认 `tile`。

跨阶段的待定项见 [roadmap.md](roadmap.md) 第 3 节。
