# LTSim（LPU-Tile-Sim）

LTSim 是面向 LPU 类静态调度加速器的 **tile 级性能仿真器**。输入一张 tile graph（"算什么 + 怎么编译"展开到 tile 粒度的数据流图），
按机器参数算出每个 tile 操作的速率与延迟，在 MXM / VXM / MEM / 流等资源上做依赖驱动的静态调度，输出总周期、下界、
关键路径、资源利用率，以及可以在 Perfetto 里查看的时间线。

它的定位是**估计"好编译下"的周期**，用来比较编译方案、定位瓶颈，不是周期精确的仿真器：模型在 tile 粒度上，偏乐观，
机器参数大多还没有对实测校准（见[开发状态](#开发状态)）。

## 目录

- [开发状态](#开发状态)
- [安装](#安装)
- [快速上手](#快速上手)
- [用 Python 搭自己的 tile graph](#用-python-搭自己的-tile-graph)
- [核心概念](#核心概念)
- [修改与扩展](#修改与扩展)
- [仓库结构](#仓库结构)
- [文档](#文档)
- [参考资料](#参考资料)

## 开发状态

当前版本 **0.1.0，即里程碑 M0（后端）已完成**；下一步是 M1（前端：LTS-IR → tile graph）。完整计划见 [路线图](docs/roadmap.md)。

| 阶段 | 内容 | 状态 |
|---|---|---|
| M0 | 机器参数、tile graph 格式、后端 B0–B4、Perfetto trace、通用 builder、CLI、CP-SAT oracle | **完成** |
| M1 | LTS-IR 格式与前端（6 个 pass） | 下一步 |
| M2 | 用 lpu-sim 实测校准机器参数 | 未开始 |
| M3 | 粗粒度编译器：自动补全并搜索编译决策 | 未开始 |
| M4 | 算子级 Python DSL | 未开始 |
| M5 | 与真实编译器 + lpu-sim 对比验证 | 未开始 |
| M6 / M7 | 扩展算子与模式 / 多芯片 C2C | 未开始 |

M0 的验收：设计文档中的两层 MLP 例子与手算表逐拍一致。默认选项下是 506 拍，等于下界；用旧的流池选项
（`--srf-pool direction --lw-streams shared`，每个方向一个流池、权重与其他数据共用）是 513 拍，CP-SAT oracle
证明它在同一模型下最优（2026-11-21 修订 wave 规则之前是 551）。74 个随机 GEMM 变体上，默认调度器 93% 恰好最优、
平均只比最优多 0.30%（这组统计是在旧的 wave 规则和旧的流池下测的，没有重测）。测试 435 个。

### 已支持

**输入与图构造**

- tile graph JSON（格式 `lts-tile-graph/0.2`）。加载时严格校验字段白名单与类型、引用、tile 区域、张量别名、lane 宽、
  mem 边读写区域是否相交等。只要 JSON 能解析，任何字段或结构错误都会给出 `LTS-E-*` 诊断码，不会崩溃（有模糊测试）；
- 5 种节点：`mem.read`（含广播读）、`mem.write`、`mxm.load_w`、`mxm.acc`、`vxm.chain`；4 种边：`stream`、`state`、`mem`、`order`；
- 数据类型 `i8` / `i32` / `f16` / `bf16` / `f32`；MXM 的输入精度支持 `i8` / `f16` / `bf16`；布局 `row` / `lane`；
- 三种建图方式：通用的 `GraphBuilder`（逐节点搭）、内置的 MLP 例子、参数化 GEMM（可调 tile 大小、绑定的 MXM plane——
  可以跨东西两侧用满 4 个、`k_split` / `n_split` / `round_robin` 分配、权重 bank 策略、K 合并 partial 的放置）。

**机器模型**（`ltsim/machine/lpu-tile-v0.yaml`，可整份替换）

- 4 个 MXM plane（东西各 2 个），每个 plane 有双权重 bank，下一块权重的装载可以藏在当前块的计算里；
- SRF 流：按单元位置查 hop 延迟，建模命令波的逐 superlane 偏移。流资源分池，每个池 32 条：默认每个方向再分东西两半、
  权重装载另用一批流（`--srf-pool half|direction`、`--lw-streams separate|shared`）；
- MEM：东西两个半球，各有每拍读写带宽；
- VXM：16 个 ALU，按 op 查延迟（`add` / `sub` / `mul` / `relu` / `cvt` / `exp` / `rsqrt` / `gelu` / `rcp`，按 i32 / f32 区分）。

**仿真**

- 按 stream 边把节点组成"链"（流不能停顿，一条链内时间刚性），计算链内偏移、流占用并检查合法性；
- 串行生成法调度，资源分独占型（MXM 计算 / 权重口）和计数型（流、MEM 带宽、VXM ALU）；策略 `bottom_level`（默认）、`topo`；
- 层间经 MEM 传数据的两种语义：`tile`（整块写完再读，默认）、`row`（逐 word，写完一个就能读一个）；
- 下界（依赖路径界、资源界）与 gap、MXM roofline、关键路径（每条链为什么在那个时刻开始）；
- 可选的 CP-SAT oracle：求同一模型下的最优调度，用来衡量调度器本身的损失。

**输出**

- `result.json`：总周期、下界、每条链 / 每个节点的时刻、资源利用率、关键路径、来源信息（输入文件的 sha256 与命令行）；
- `trace.json`：Perfetto 时间线，含各单元的占用、链、资源计数器、依赖箭头、关键路径；
- `graph.dot`（可选）：Graphviz 图，每条链一个框；
- 输出是确定的：同一输入运行两次，文件除 `provenance.command` 记录的命令行外逐字节相同；输出路径也相同时完全相同。

### 尚不支持 / 已知限制

- **还没有前端**：不能从算子描述自动生成 tile graph，只能用 `GraphBuilder` 手搭或用内置生成器（M1 解决）；
- 不支持 SXM 节点（transpose / permute 等）；不支持 VXM → VXM 的直连流，所以 softmax / layer_norm 的归约还不能与上游融合；
- 一个节点的输出不能同时流向东西两个方向；
- 不建模 MEM 容量与地址分配（所有张量全程驻留），不建模 slice 级放置；
- **流资源池的划分是假设**：因为没有 slice 放置，默认的"按东西两半分池"与"权重装载有自己的流"（后者偏乐观）都还没有依据，
  流预留末尾的 hop 尾巴有多长也说不准。所以 `run` 总是给出带 / 不带 hop 尾巴两个结果，用来夹住这一不确定性；
  这几条待 M2 校准或 M6 的 slice 放置来定（见[路线图](docs/roadmap.md)待定项 T11–T13）；
- 只有单芯片；
- **机器参数大多是推导值或假设值，尚未用 lpu-sim 校准**（每个参数在 yaml 里标注了来源）。绝对周期数只作参考，适合做方案间的相对比较；
- `random` 调度策略未实现；流资源紧张时（例如旧的流池选项下权重 bank 用 `single` 策略），默认调度器会把对称的链放在同一时刻，
  比最优偏悲观（已知的例子上约 0.2%：1339 对 1336）。

## 安装

需要 Python 3.10+。

```bash
git clone https://github.com/syhesyh/ltsim.git
cd ltsim
pip install -e .                    # 只装仿真器（依赖 PyYAML）
pip install -e ".[test,oracle]"     # 再加 pytest 与 OR-Tools（跑测试、用 oracle）
```

装好后，`ltsim …` 与 `python -m ltsim …` 等价，在任何目录都能用。

## 快速上手

### 1. 生成 tile graph，运行仿真

```bash
ltsim gen mlp --out build/mlp                              # 内置的两层 MLP：128×320×640，int8
ltsim run build/mlp/tile_graph.json --out build/mlp_run    # 仿真，写 result.json 与 trace.json
```

终端输出：

```text
total_cycles 506   lower bound 506 (path 506, resource 136 on vxm.alu)   gap 0.00%
critical path: c0 [0,48) (source) → c4 [21,286) (edge state w1n0.lw→a0.acc) → c6 [267,506) (edge mem a0.h.wr→b.h0.rd)
without SRF hop tail: total_cycles 506   lower bound 506   gap 0.00%
```

- `total_cycles`：调度结果的总周期；
- `lower bound`：取两个下界中较大的一个。`path` 是只看依赖、不考虑资源冲突时的最短完成时间；`resource` 是最忙的那个资源
  （这里是 VXM 的 ALU `vxm.alu`）总占用量除以容量；
- `gap`：总周期比下界多出的比例。它是调度损失的上限，下界本身偏松的部分也算在里面；
- `critical path`：从头到尾决定总周期的那串链。括号里是这条链为什么在这个时刻开始：`source`（无约束）、
  `edge`（等某条依赖边）、`resource`（等某个资源，`held by` 指出占着资源的链）、`policy`（调度策略的排序）；
- `without SRF hop tail`：同一张图再调度一次，流预留不带末尾的 hop 尾巴（不知道 MEM slice 放在哪，尾巴长度说不准），
  写进 `result.json` 的 `no_hop_tail`。上面两行与 trace 都来自带尾巴的那次。

把 `trace.json` 拖进 <https://ui.perfetto.dev> 就能看时间线。

### 2. 常用选项

```bash
ltsim run G --out D --mem-overlap row          # 层间逐 word 重叠：MLP 506 → 386
ltsim run G --out D --srf-pool direction --lw-streams shared
                                               # 旧的流池：每个方向一个池，权重与其他数据共用。MLP 506 → 513
                                               #   （第一层读激活要等两块权重装载释放流；不带 hop 尾巴时仍是 506）
ltsim run G --out D --machine my-machine.yaml  # 换一份机器参数（见"修改与扩展"）
ltsim run G --out D --policy topo              # 换调度策略
ltsim run G --out D --dot                      # 另写 graph.dot
ltsim run G --out D --trace-stream-flows       # trace 里也画链内的 stream 箭头

# 参数化 GEMM：Y[M,N] = X[M,K] · W[K,N]
ltsim gen gemm --M 256 --K 640 --N 1280 --tile-m 128 --banks single --out build/gemm
#   --tile-k / --tile-n   K、N 方向的块大小（默认取 MXM 块大小 320）
#   --planes w0,w1        绑定哪些 MXM plane（w0 w1 在西侧，e0 e1 在东侧）；
#                         跨两侧写 w0,w1,e0,e1 时用满 4 个 MXM：N 块前一半归西侧、后一半归东侧，
#                         X/W/Y 按侧各复制一份（要求 N 方向至少 2 块）
#   --map n_split         权重块分到各 plane 的方式：k_split / n_split / round_robin（双侧时在侧内适用）
#   --banks alternate     权重 bank 策略：alternate（双缓冲）/ single
#   --p-placement auto    K 合并的 partial 放哪：auto（默认）只在 Kb×每行向量数 > 32 时按 K 块奇偶
#                         分放东西两侧（合并的读流对半进两个流池，上限 32 → 64）；local 全放本侧；
#                         alternate 总是交替。超过 64 仍会报 LTS-E-SRF-WIDTH（树形合并未实现）

# 例子：8×4096×4096 的 GEMV（decode 阶段的典型负载），4 个 MXM 全用上
ltsim gen gemm --M 8 --K 4096 --N 4096 --tile-m 8 --tile-k 256 --tile-n 256 \
               --planes w0,w1,e0,e1 --out build/gemv
ltsim run build/gemv/tile_graph.json --out build/gemv_run
#   total_cycles 1905（不带 hop 尾巴 1280），下界 1600：瓶颈是权重装载的流
```

### 3. 三个模型开关（为什么不写死）

`run` 的三个选项控制模型里几条"还没有实测依据的规则"怎么算——它们不改图、不改调度器，只改资源表与依赖边。
每次运行的取值记录在 `result.json` 的 `provenance` 里；等 M2 用 lpu-sim 校准后才会定死（roadmap 待定项 T11–T13）。

| 开关 | 取值 | 含义 |
|---|---|---|
| `--mem-overlap` | `tile`（默认） | 经 MEM 传数据：整块 tile 写完可见后，读方才能开始读 |
| | `row` | 逐 word 精确：写完一个 word 就能读一个，读写两边流水重叠（MLP 506 → 386） |
| `--srf-pool` | `half`（默认） | SRF 流按 方向 × 东西半区 分 4 个池（各 32 条）：流只占用源到目的之间的线路，不同半区、不同方向互不干扰（VXM 在中线，跨中线的传输两边都占） |
| | `direction` | 每个方向一个池，线路全程独占、不能空间复用（旧行为） |
| `--lw-streams` | `separate`（默认） | 权重装载（`load_w`）用专用流池（池名加 `.lw`）：认为好编译器总能让权重与其他数据各用各的流（乐观假设） |
| | `shared` | 权重装载与激活等数据共用同一组流，会互相抢（旧行为；GEMV 带 hop 尾巴时 2291 vs separate 的 1716） |

第四个不确定项没有做成开关：流预留末尾的 **hop 尾巴**（流在最后一个消费者之后还要占多久，取决于 MEM slice 的放置，模型没有这个位置信息）。
所以每次 `run` 都同时给出带尾巴（主结果）与不带尾巴（`no_hop_tail`）两个数，把不确定性夹住。

约定：

- **输出目录必须是新目录**；所有结果算完、成功之后才会建目录写文件，失败可以用同一个名字重跑；
- 出错时向 stderr 打印 `error: LTS-E-…: 说明`，退出码 2。诊断码的含义见 [接口文档](docs/interfaces.md) 第 10 节；
- `--machine` 的 id 与图里写的不一致时照常运行，只给出 `LTS-W-MACHINE-ID` 警告，方便用同一张图对比不同机器。

全部选项见 [接口文档](docs/interfaces.md) 第 6 节。

## 用 Python 搭自己的 tile graph

下面的例子从零搭一个单块矩阵乘 `Y[64,320] = X[64,320] · W[320,320]`，然后跑完整的仿真流程：

```python
from ltsim.gen.builder import GraphBuilder, tile_str
from ltsim.graph.tile_graph import Out
from ltsim.graph.load import validate_graph
from ltsim.machine.model import default_machine
from ltsim.backend.chains import build_chains
from ltsim.backend.scheduler import schedule
from ltsim.backend.result import build_result, make_provenance, write_result
from ltsim.trace.perfetto import build_trace, write_trace

b = GraphBuilder("matmul_64x320x320", "lpu-tile-v0")
#        名字  dtype  shape        role      半球  布局
b.tensor("X", "i8",  (64, 320),  "input",  "W", "row")
b.tensor("W", "i8",  (320, 320), "param",  "W", "row")
b.tensor("Y", "i32", (64, 320),  "output", "W", "row")

#      节点 id   tile                               行数  输出(dtype, 每行向量数)  所属算子
b.read("w.rd", tile_str("W", 0, 320, 0, 320), 320, Out("i8", 1), ["mm"])    # 读权重
b.load_w("w.lw", "mxm.w0", 0, 320, ["mm"])                                    # 装进 mxm.w0 的 bank 0
b.read("x.rd", tile_str("X", 0, 64, 0, 320), 64, Out("i8", 1), ["mm"])       # 读激活
b.acc("acc", "mxm.w0", 0, 64, Out("i32", 4), ["mm"])                          # 矩阵乘
b.write("y.wr", tile_str("Y", 0, 64, 0, 320), 64, ["mm"])                     # 写回 MEM

b.stream("w.rd", "w.lw")                                # 流直连
b.stream("x.rd", "acc")
b.stream("acc", "y.wr")
b.link_plane([("w.lw", ["acc"])], banks="alternate")    # 自动生成 MXM 的顺序边（这里是 state：装完权重才能算）

g, m = b.build(), default_machine()                     # build() 做结构校验
validate_graph(g, m)                                    # 对照机器参数校验
cg = build_chains(g, m)                                 # B1 节点工作量 + B2 组链
placed = schedule(cg)                                   # B3 调度（资源表取 cg.resources）
prov = make_provenance(g, m, policy="bottom_level", mem_overlap="tile", srf_pool="half", lw_streams="separate")
result = build_result(g, cg, placed, m, prov)           # B4 下界、指标
print(result["total_cycles"], result["bounds"]["lb"])   # 147 147
write_result("result.json", result)
write_trace("trace.json", build_trace(g, cg, placed, m, provenance=prov))
```

也可以 `b.save("my/tile_graph.json")` 存成文件，再用 `ltsim run` 仿真。更完整的写法可以参考 `ltsim/gen/mlp.py`（含 VXM 融合、
广播读、两层之间的 mem 边）和 `ltsim/gen/gemm.py`（每个 plane 多个权重块、K 方向合并；planes 可跨东西两侧用满 4 个 MXM）。

## 核心概念

**tile graph 里没有周期数。** 它只描述"哪些 tile 操作、在哪个单元上、按什么依赖"；所有周期都由后端按机器参数算出来，
所以同一张图换一份机器参数就能做设计空间探索。

| 节点 kind | 做什么 | 主要字段 |
|---|---|---|
| `mem.read` | 从 MEM 读一个 tile，送上流 | `tile`、`rows`、`out`（dtype 与每行向量数 V）、`repeat`（广播读） |
| `mem.write` | 把流上的数据写回 MEM | `tile`、`rows` |
| `mxm.load_w` | 把一块权重装进 MXM 的某个 bank | `unit`、`bank`、`rows` |
| `mxm.acc` | 用已装好的权重做矩阵乘 | `unit`、`bank`、`rows`、`out` |
| `vxm.chain` | VXM 上融合执行的一串逐元素 op | `ops`、`rows`、`out` |

| 边 | 含义 |
|---|---|
| `stream` | 数据经流直接从一个单元送到下一个，不落 MEM。`[src, dst, r0]`，`r0` 是 dst 从 src 的第几行开始用 |
| `state` | `load_w → acc`：权重装完才能开始计算 |
| `mem` | `mem.write → mem.read`：经 MEM 传数据，读要等写可见 |
| `order` | MXM 上的顺序约束（`acc_seq`、`block_switch`、`lw_after_iw`、`bank_war`），一般用 `link_plane` 自动生成 |

**后端流程：**

```text
B0 加载校验 → B1 节点工作量（速率、首出延迟）→ B2 组链（stream 连通分量）→ B3 调度 → B4 下界、指标、关键路径 → result.json / trace.json
```

流不能停顿，所以一条 stream 连起来的链里所有节点按同一速率运行，相对时刻固定；调度的基本单位是链。
完整格式与算法见 [后端开发文档](docs/backend.md)，机器可读的格式说明见 `ltsim/graph/schema.json`。

## 修改与扩展

### 改机器参数：不用改代码

复制一份机器文件，改 `id` 和参数，用 `--machine` 指定即可：

```bash
cp ltsim/machine/lpu-tile-v0.yaml my-machine.yaml
# 编辑 my-machine.yaml：id: my-machine，srf.streams_per_direction: 16
ltsim run build/mlp/tile_graph.json --machine my-machine.yaml --out build/mlp_srf16
# 流减半后西侧两块权重不能同时装：MLP 506 → 534（不带 hop 尾巴 526）
```

- 加载时会校验结构与推导量：`geometry.vector_bytes` 必须等于 `superlanes × lanes_per_superlane`，
  `srf.wave_skew_cycles` 必须等于 `superlanes − 1`；
- 新增 VXM op：在 `vxm.op_latency` 里加一项；MXM 新精度：在 `mxm.lw_rows_per_cycle` 与 `mxm.acc_latency` 里各加一个键；
- 每个参数的注释标注了来源（contract / derived / assumed / calibrated），修改时请一并更新。

### 改代码：去哪里改

| 想做的事 | 改哪里 |
|---|---|
| 新的负载生成器 | `ltsim/gen/` 新建模块，用 `GraphBuilder`；需要命令行入口时在 `ltsim/cli.py` 的 `build_parser` / `cmd_gen` 里注册 |
| tile graph 新字段或新规则 | `ltsim/graph/load.py`（字段白名单与校验）、`ltsim/graph/schema.json`、`docs/backend.md` §2 |
| 节点的速率、延迟公式 | `ltsim/backend/work.py`（B1） |
| 组链、流占用（流资源池）、跨链约束 | `ltsim/backend/chains.py`（B2）；资源表在 `ltsim/machine/model.py` 的 `Machine.resources` |
| 调度策略 | `ltsim/backend/scheduler.py`（B3，`POLICIES`） |
| 下界、指标、`result.json` | `ltsim/backend/bounds.py`、`metrics.py`、`result.py`（B4） |
| trace 的显示 | `ltsim/trace/perfetto.py` |

### 开发约定

- **规格先行**：`docs/backend.md` 是后端的权威规格。改算法时先改文档（包括 §8 的手算例子），再改代码；
  与设计文档不同的地方在"与设计的偏离"里逐条写明；
- **代码风格**：数据结构用 frozen dataclass；时间一律是整数拍，速率用 `fractions.Fraction`；
  错误统一抛 `LTSError(code, msg)`，错误码为 `LTS-E-*`，警告为 `LTS-W-*`，新增的码要写进 [接口文档](docs/interfaces.md) 第 10 节的一览表；
- **确定性**：不写时间戳，键和列表顺序固定，同一输入两次运行输出逐字节相同（`provenance.command` 记录命令行，`--out` 不同时只有它不同）；
- **依赖**：运行时只用标准库和 PyYAML；OR-Tools 只给 oracle 用，必须保持可选；
- **进度**：完成一个开发步骤，或做出影响计划的决定，更新 [路线图](docs/roadmap.md) 的状态和待定项。

### 测试

```bash
python -m pytest tests -q          # 在仓库根目录运行；没装 OR-Tools 时 oracle 测试自动跳过
```

- `tests/ltsim/fixtures/` 里有 MLP 与一个 GEMM 的 `result.json` 金标准快照。结果**有意**改变时，用
  `LTSIM_UPDATE_GOLDEN=1 python -m pytest tests/ltsim/test_golden.py` 重新生成，并在提交说明里写清原因；
- 改了 tile graph 的校验规则，要同步 `ltsim/graph/schema.json`，`test_graph.py` 会检查两者是否一致；
- 测试文件与覆盖内容的对应表见 [后端开发文档](docs/backend.md) §9。

## 仓库结构

```text
ltsim/              Python 包
  machine/            机器参数（lpu-tile-v0.yaml）与加载
  graph/              tile graph 数据结构、加载与 B0 校验、schema.json、Graphviz 输出
  gen/                GraphBuilder、MLP 例子、参数化 GEMM
  backend/            B1–B4：节点工作量、组链、调度、下界与指标、result.json、CP-SAT oracle
  trace/              Perfetto trace
  cli.py              命令行
tests/ltsim/        测试与金标准快照（fixtures/）
docs/               开发文档
```

## 文档

| 文档 | 内容 |
|---|---|
| [路线图](docs/roadmap.md) | **长期计划与进度的权威版本**：M0–M7 的目标、完成判据、依赖、状态，跨阶段的待定项 |
| [后端开发文档](docs/backend.md) | tile graph → 结果：B0–B4 的算法、日历与调度、Perfetto、手算例子、测试；与设计的偏离逐条注明 |
| [接口文档](docs/interfaces.md) | 各模块的函数签名、数据结构、诊断码与 CLI |
| [前端开发文档](docs/frontend.md) | LTS-IR → tile graph（M1，尚未实现）：数据结构、P1–P6 的算法、命名规则、测试、开发顺序 |

## 参考资料

各文档中的"设计 §x.y"指下面的设计文档。它在 lpu-compiler 仓库里（私有），不随本仓库发布。

| 文档 | 位置 | 内容 |
|---|---|---|
| LPU-Tile-Sim 设计文档 | lpu-compiler 仓库 `Docs/lpu_tile_sim_design.md`（`ltsim` 分支，commit `df719cf`） | 原始设计：机器模型、LTS-IR、tile graph、后端算法、输出格式、里程碑 |
| LPU 仿真与编译总览 | lpu-compiler 仓库 `Docs/lpu_sim_compile_overview.md`（同上） | 为什么要做 LTSim，它和 lpu-sim、lpu-compiler 的关系 |

设计文档**只作参考，不保证正确**；与本仓库 `docs/` 下的文档冲突时，以本仓库为准。偏离之处在各开发文档里逐条注明。
