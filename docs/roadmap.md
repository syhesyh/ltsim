# LTSim 路线图（M0–M7）

> **本文是 ltsim 长期计划与进度的权威版本。** 里程碑的划分以
> 设计文档 §11 为起点（lpu-compiler 仓库的 `Docs/lpu_tile_sim_design.md`，私有，出处见 [README](../README.md#参考资料)）；设计文档只作参考，
> 这里的调整、状态与待定项以本文为准。各阶段内部的开发步骤不在这里重复，见各自的开发文档。
>
> **维护规则：** 每完成一个阶段或其中一步，或做出影响计划的决定，就更新本文的"状态"和"待定项"。
>
> 更新：2026-11-21。

## 1. 分层与里程碑

```text
算子级 Python DSL ─▶ 粗粒度编译器 ─▶ LTS-IR ─▶ 前端 P1–P6 ─▶ tile graph ─▶ 后端 B0–B4 ─▶ result.json / trace.json
     M4                  M3            (文件)       M1           (文件)         M0
                                                                               ▲
                              M2 校准：用 lpu-sim 实测拟合机器参数 ──────────────┘（只改 yaml，不改代码）
                              M5 验证：与真实编译器 + lpu-sim 对比，给出 Go/No-Go
                              M6 扩展算子与模式；M7 多芯片 C2C
```

与 GTSim 的对应：TileLang DSL → TileLang 编译器 → IR → 前端 → tile graph → 后端。区别是 GTSim 的决策来自成熟编译器，
LTSim 的决策由 M3 自己搜索。所以"好编译下的性能"取决于搜得好不好，也取决于 M0 的调度器离最优多远（见 M0 的 oracle）。

| 阶段 | 做什么 | 完成判据 | 依赖 | 状态 |
|---|---|---|---|---|
| **M0** | 机器参数、tile graph 格式、后端 B0–B4、Perfetto trace（另加通用 builder、CLI、oracle） | 设计 §6.3 的 MLP 能跑出结果，trace 能在 Perfetto 打开 | 无 | **完成**（2026-11-08） |
| **M1** | LTS-IR 格式与前端 P1–P6 | 设计 §4.4 的 MLP 生成的 tile graph 与 §6.3 一致；§4.5 的 attention 展开出 12 条链、47 个节点；§4.4 的 `tile_m: 32` 变体展开出 16 条链、76 个节点；两个例子的几个调度变体都能运行，周期的相对大小能用 trace 解释 | M0 | 未开始，下一步 |
| **M2** | 校准 | 校准集结果记录在案，参数来源完成标记（assumed → calibrated） | M0；外部 lpu-sim（`LPU_SIM_ROOT`）与 lpu-compiler | 未开始 |
| **M3** | 粗粒度编译器（自动补全并搜索 schedule） | 只给 §4.4、§4.5 的算法部分，能补全决策并搜索；结果不差于手写 schedule；搜索日志可复现 | M1（最好有 M2 的参数） | 未开始 |
| **M4** | 算子级 Python DSL | 用 DSL 写出 §4.4 的 MLP 和 §4.5 的 attention，生成的 LTS-IR 算法部分与手写版逐字段一致 | M1；`simulate()` 需要 M3 | 未开始 |
| **M5** | 决策镜像适配器与验证 | 报告乐观性、差距分布与 Kendall τ，按阈值给出 Go/No-Go | M1、M2；lpu-compiler 的通用 DAG fixture | 未开始 |
| **M6** | 扩展 | 见第 2 节 | M1 | 未开始 |
| **M7** | 多芯片 C2C | 多芯片负载能仿真 | M1 | 未开始 |

**顺序：** 设计的顺序是 M0 → M1 → M2 → M3 → M4 → M5 → M6 → M7。M1 不依赖外部，应当紧接着做。
M2 依赖外部的 lpu-sim 和开发不完善的 lpu-compiler，是最大的外部风险。如果到时候 lpu-sim 不可用，可以先做 M3、M4：
校准只改 yaml 参数，不改代码，事后补做不会返工。

## 2. 各阶段内容

### M0 后端（完成）

开发步骤与验收见 [backend.md](backend.md) §10，接口见 [interfaces.md](interfaces.md)。

| 部分 | 模块 | 要点 |
|---|---|---|
| 机器参数 | `ltsim/machine/` | `lpu-tile-v0.yaml`（设计 §3.4 的初值）；MXM 按精度分表（i8 / f16 / bf16）；推导量校验 |
| tile graph | `ltsim/graph/` | 加载与 B0 校验：字段白名单、类型、一致性、alias、lane 宽；有"永不崩溃"的模糊测试 |
| 通用 builder | `ltsim/gen/` | `GraphBuilder`、MXM 顺序边辅助函数、MLP 金标准、参数化 GEMM |
| B1–B4 | `ltsim/backend/` | 节点工作量 → 组链（等式传播求偏移）→ SSGS 调度（计数资源合并分段）→ 下界、关键路径、指标、`result.json` |
| trace | `ltsim/trace/perfetto.py` | 9 个进程、计数器、流向箭头（按边依赖的时刻放）、自检 |
| CLI | `ltsim/cli.py` | `python -m ltsim gen mlp|gemm`、`python -m ltsim run`（`--dot` 另写 Graphviz 图） |
| oracle | `ltsim/backend/oracle.py` | CP-SAT 求最优链调度；需要 OR-Tools（可选依赖 `oracle`） |

**关键结果：**

- 设计 §6.3 的 MLP：总周期 551，下界 525，关键路径 c0 → c4 → c6，与 backend.md §8 的手算表一致；
- **551 是最优。** 与下界之间的 26 拍全部来自下界偏松，不是调度器的损失；
- 74 个随机 GEMM 变体上，SSGS 有 93% 恰好最优，平均比最优多 0.30%。已知弱点：banks = single 时会把两个 plane 的权重装载放在同一时刻，
  偏悲观约 4%（见待定项 T7）。统计见 backend.md §9；
- 测试：`python -m pytest tests -q` 共 368 个；没装 OR-Tools 时 oracle 测试跳过。
  MLP 与一个 single-bank GEMM 的 `result.json` 有金标准快照（`test_golden.py`）；`schema.json` 与 B0 的规则由测试保持同步；
- 验收：trace 已由用户在 Perfetto UI 中目视确认（2026-11-08）；
- 复核：B0–B4、trace 都经过独立 verifier 复核；CLI 与 oracle 没有单独复核（用户决定先不做，2026-11-08）。

**与设计的偏离**（逐条见 backend.md 中"与设计的偏离"）：链内偏移改为等式传播；计数资源先合并分段再检查；
row 模式用逐 word 的精确公式；MXM roofline 按实际权重块算；流向箭头按边依赖的时刻放；关键路径分行；
lane 宽检查放到 B0；MXM 支持 f16 / bf16。

**M0 内推迟的事项：** `random:K:seed` 调度策略（待定项 T6）。

### M1 前端：LTS-IR → tile graph

开发文档：[frontend.md](frontend.md)。它写于后端实现之前，开工前要按 M0 的偏离更新一遍（待定项 T2、T3）。

- LTS-IR 是一个 YAML 文件：`tensors` / `ops` 描述算什么，`schedule` 描述怎么算（分块、绑定、融合、布局、权重双缓冲）；
- 6 个 pass：

  | pass | 做什么 |
  |---|---|
  | P1 解析校验 | SSA 形式、拓扑序，推导中间张量的 shape / dtype |
  | P2 补全决策 | 按固定顺序补默认值；最难的是布局求解（row / lane） |
  | P3 分块 | 生成 tile 表 |
  | P4 展开 | matmul、逐元素算子；softmax / layer_norm 的对半树归约 |
  | P5 建边 | mem 边按区域重叠；MXM 顺序边按每个 plane 的块序列 |
  | P6 合法性与输出 | 写出 `tile_graph.json`、`resolved_schedule.json` |

- 开发顺序（frontend.md §10）：P1 → P2 的融合与布局 → P2 其余与 P3 → P4 的 matmul 与逐元素（MLP）→ P5（`tile_m: 32` 变体）
  → P4 的归约（attention）→ P6 与 CLI；
- CLI 新增 `lower`（只跑前端）和 `sim`（前端 + 后端）。

### M2 校准：把 assumed 参数变成 calibrated

- 用 lpu-compiler 生成单算子程序，在 lpu-sim 上测周期：
  - matmul 单块，M ∈ {1, 8, 64, 197, 320}，以及连续两块（测 `block_switch_gap`）；
  - 逐元素链，链深 1–4，dtype 为 INT32 / FP32；
  - MEM 到 MEM 的拷贝；
  - 对半树的单层，n ∈ {2, 25, 197}；
  - 运行时权重的装载；
- 拟合 `acc_latency`、`op_overhead`、`write_latency`、`raw_gap` 等参数，把来源标记改为 calibrated。校准集一经固定不再改；
- 写 `derive_machine.py`：从 `$LPU_SIM_ROOT/arch/full_lpu_v2.json` 生成 contract / derived 类参数，并记录合同的 sha256；
- 顺带定下待定项 T4、T5。

### M3 粗粒度编译器：自动做编译决策

设计 §14.2。

- **输入**：只有算法部分的 LTS-IR；也可以带部分 schedule，给出的字段固定不动；
- **决策空间**：融合在哪里切、布局、`tile_m`、绑定的 plane 与 `map`、bank 策略与装载顺序；
- **搜索**：从默认规则加启发出发，坐标下降（每次改一项、变好就留），再加随机重启；预算按评价次数和墙钟时间限制；固定 seed 时可复现；
- **输出**：完整的 LTS-IR、`search_log.jsonl`、最好方案的 `result.json` / `trace.json`；
- **对外的数字**：每个负载只报一个"tile 估计"，即搜到的最好方案的总周期。

### M4 算子级 Python DSL

设计 §14.1。只描述算法，不含任何决策，与 LTS-IR 的 `tensors` + `ops` 一一对应。`g.save_ir()` 写出算法部分；
`lt.simulate(g, budget, seed)` 依次做粗粒度编译、前端、后端。以后可以从 PyTorch / ONNX / lpu-compiler 的图 JSON 导入。

### M5 验证：估计可不可信

设计 §9。

- 写决策镜像适配器：把 lpu-compiler 实际做出的决策翻译成 LTS-IR 的 schedule；表达不了的 fixture 记为不适用；
- 在镜像决策下，用通用 DAG 的固定 fixture 报告三件事：
  - **乐观性**：tile 估计不高于 lpu-sim 实测的比例；
  - **差距分布**：`实测 / 估计 − 1`，这部分是编译器还能改进的空间；
  - **排序一致性**：同一负载多个调度变体之间的 Kendall τ；
- **Go/No-Go**：至少 95% 的 fixture 上估计 ≤ 实测 × 1.05，且 τ ≥ 0.8。阈值是假设，运行前要与导师确认（待定项 T9）；
- CP-SAT oracle 已在 M0 完成。

### M6 扩展

row 布局下经 SXM 折叠的 LayerNorm / Softmax（作对照）；需要真实搬数据的同布局转置；conv2d、池化等 CNN 算子；
MEM 存活期复用（v0 所有张量全程占用空间）；可选的显式 slice 级放置模式。

### M7 多芯片 C2C

设计 §13：机器参数增加 `chips` 和 `c2c.links`；单元名加芯片前缀；新增 `c2c.send` / `c2c.recv` 节点；
LTS-IR 增加 `partition` 决策。后端不需要新机制，只是多了资源和一种边；Perfetto 中每个芯片一组进程。

## 3. 待定项（跨阶段）

| # | 事项 | 影响 | 详情 |
|---|---|---|---|
| ~~T1~~ | ~~M0 的 trace 在 Perfetto UI 中目视确认~~ —— **已解决**：用户于 2026-11-08 确认 | M0 验收 | backend.md §7.3 |
| T2 | **VXM → VXM 的 stream 边**：对半树第 1 层与上游融合时（设计 §5.4；例如 §4.5 attention 的 Cs 链里，`vxm[cvt, mul]` 从第 99 行起直接流进 `vxm[exp, exp, add]`），B1 现在把同类别之间的流当作错误，hop 表里也没有 `VXM>VXM`。要定：延迟、占不占流、方向。候选：**A** 乐观：新参数 `hops.VXM>VXM` 初值 0（assumed），不占流，不参与方向判断，与设计 §3.6"ALU 之间的传递已含在 op_overhead 里"一致；**B** 保守：3 拍（穿过 VXM 的 3 个站点），占上游另一消费者方向的流；**C** v0 不支持这种融合，归约只能在融合组开头，attention 的判据要改。A、B 都属于"好编译下"的乐观假设（设计 §12：现有编译器把中间结果写回 MEM）。**决定于 M1 开工、过规格时做** | **M1，会卡住 attention** | backend.md §11 |
| T3 | 前端要跟上 M0 的偏离：MXM 已支持 f16 / bf16，P6 的 `LTS-E-UNIT-OP` 检查要放宽；lane 宽检查已在 B0 做，前端是否重复做 | M1 | backend.md §3.2、§2 |
| T4 | 节点内第 i 行的时刻用 `ceil(i/ρ)` 还是 `floor(i/ρ)`（ρ > 1 时 `ceil` 与占用区间不自洽） | M2 | backend.md §11 |
| T5 | assumed 参数待实测：f16 / bf16 的 acc 延迟与装载速率（用户给定）、MEM 每拍 44 个向量、设计 §12 列出的各项乐观假设 | M2 | `lpu-tile-v0.yaml` 的来源注释 |
| T6 | `random:K:seed` 调度策略（推迟实现，规格已写在 backend.md §5.3） | M3 | backend.md §5.3 |
| T7 | SSGS 在 banks = single 时偏悲观约 4%（对称的链被放在同一时刻），搜索打分会偏向 alternate | M3、M5 | backend.md §5.3、§9 |
| T8 | 性质测试没有用 Hypothesis（环境里没有），改用 `random.Random` 固定种子；需要时可以补 | — | backend.md §9 |
| T9 | Go/No-Go 阈值（95% / ×1.05 / τ ≥ 0.8）需要与导师确认 | M5 | 设计 §9 |
| T10 | `result.json` 的 `provenance` 里 `ltsim_commit` / `dirty` 仍写 null。ltsim 已是独立 git 仓库，可以改为取 `git rev-parse HEAD` 与工作区是否有未提交改动（不在 git 仓库里、或没有 git 时仍写 null）；金标准快照本来就去掉 `provenance`，不受影响 | — | backend.md §6 |

## 4. 仓库与环境

- ltsim 是独立的 git 仓库（2026-11-21 从工作区拆出，计划公开）。仓库根目录下：包 `ltsim/`、测试 `tests/ltsim/`、文档 `docs/`、
  `pyproject.toml`；
- 依赖：Python 3.10+、PyYAML；跑测试要 pytest；oracle 要 OR-Tools（可选）。安装：`pip install -e ".[test,oracle]"`，
  之后在任意目录都能用 `ltsim …` 或 `python -m ltsim …`；
- 在仓库根目录运行测试：`python -m pytest tests -q`；
- 生成的示例输出放在仓库根目录的 `build/`（已在 `.gitignore` 里，不入库）；
- **公开仓库的约束：** 不放 lpu-compiler 的文件（它是私有仓库），引用设计文档只写出处（见 README「参考资料」）；不写个人绝对路径。
