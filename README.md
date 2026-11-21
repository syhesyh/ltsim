# LTSim（LPU-Tile-Sim）

tile 粒度的 LPU 仿真器：把"算法 + 编译决策"展开成 tile graph，用单元、流、MEM 的吞吐模型和依赖驱动的调度，
估计好编译下的周期，并输出 Perfetto trace。

**状态：** M0（后端 + 通用 tile graph builder）已实现：B0 加载校验 → B1 节点工作量 → B2 组链 → B3 调度 → B4 下界与指标 →
Perfetto trace，加 CLI。M1 前端（LTS-IR → tile graph）尚未开始。

## 安装

依赖：Python 3.10+、PyYAML；跑测试要 pytest；CP-SAT oracle 要 OR-Tools（可选）。

```bash
pip install -e .                    # 只装仿真器
pip install -e ".[test,oracle]"     # 再加测试与 oracle 的依赖
```

## 用法

```bash
ltsim gen mlp --out build/mlp                              # 生成设计 §6.3 的 MLP tile graph
ltsim run build/mlp/tile_graph.json --out build/mlp_run    # 仿真：result.json + trace.json
ltsim gen gemm --M 256 --K 640 --N 1280 --tile-m 128 --banks single --out build/gemm
```

`ltsim` 与 `python -m ltsim` 等价。输出目录必须是新目录。`trace.json` 可以直接拖进 <https://ui.perfetto.dev>。
命令的全部选项见 [接口文档](docs/interfaces.md) 第 6 节。

## 测试

在仓库根目录运行：

```bash
python -m pytest tests -q
```

没装 OR-Tools 时，oracle 测试自动跳过。

## 目录

```text
ltsim/          Python 包
  machine/        机器参数（lpu-tile-v0.yaml）与加载
  graph/          tile graph 数据结构、加载与 B0 校验、schema.json、Graphviz 输出
  gen/            通用 GraphBuilder、MLP 金标准、参数化 GEMM
  backend/        B1–B4：节点工作量、组链、调度、下界与指标、result.json、CP-SAT oracle
  trace/          Perfetto trace
  cli.py
tests/ltsim/    测试与金标准快照（fixtures/）
docs/           开发文档
```

## 文档

| 文档 | 内容 |
|---|---|
| [路线图](docs/roadmap.md) | **长期计划与进度的权威版本**：M0–M7 的目标、完成判据、依赖、状态，跨阶段的待定项 |
| [前端开发文档](docs/frontend.md) | LTS-IR → tile graph：数据结构、P1–P6 的算法、命名规则、测试、开发顺序 |
| [后端开发文档](docs/backend.md) | tile graph → 结果：B0–B4 的算法、日历与调度、Perfetto、手算例子、测试、开发顺序；与设计的偏离逐条注明 |
| [接口文档](docs/interfaces.md) | 各模块的函数签名、数据结构、诊断码与 CLI |

## 分层

```text
算子级 Python DSL ─▶ 粗粒度编译器 ─▶ LTS-IR ─▶ 前端 ─▶ tile graph ─▶ 后端 ─▶ result.json / trace.json
      (M4)              (M3)                     (M1)                  (M0)
```

里程碑 M0–M7、当前状态和跨阶段的待定项见 [路线图](docs/roadmap.md)：M0 后端（完成）→ M1 前端 → M2 校准 → M3 粗粒度编译器 → M4 DSL → M5 验证 → M6 扩展 → M7 C2C。

## 参考资料

各文档中的"设计 §x.y"指下面的设计文档。它在 lpu-compiler 仓库里（私有），不随本仓库发布。

| 文档 | 位置 | 内容 |
|---|---|---|
| LPU-Tile-Sim 设计文档 | lpu-compiler 仓库 `Docs/lpu_tile_sim_design.md`（`ltsim` 分支，commit `df719cf`） | 原始设计：机器模型、LTS-IR、tile graph、后端算法、输出格式、里程碑 |
| LPU 仿真与编译总览 | lpu-compiler 仓库 `Docs/lpu_sim_compile_overview.md`（同上） | 为什么要做 LTSim，它和 lpu-sim、lpu-compiler 的关系 |

设计文档**只作参考，不保证正确**；与本仓库 `docs/` 下的文档冲突时，以本仓库为准。偏离之处在各开发文档里逐条注明。
