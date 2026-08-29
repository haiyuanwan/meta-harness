# Continual Harness：方法依据、实验设计与实现进度

> 本文合并并取代原 `META_HARNESS_METHOD_AND_PIPELINE_CN.md` 与
> `AGENTSTREAM_SEQUENTIAL_META_HARNESS_PLAN.md`。它是本项目关于 Meta-Harness
> 方法、continual/sequential 实验协议、迁移与遗忘评测、HDA 以及当前工程状态的
> 唯一主文档。
>
> Meta-Harness 方法说明以仓库原 `main@44b9942` 的发布代码和论文
> [Meta-Harness: End-to-End Optimization of Model Harnesses](https://arxiv.org/abs/2603.28052)
> 为依据；具体执行以本仓库当前代码与本文状态为准。

## 0. 当前状态

- 更新时间：2026-08-26。
- 状态：方案已 review；benchmark-native runtime、固定划分、六格评测、显著性门、
  HDA 审批报告，以及 Terminal-Bench/Harbor 风格的 late-verifier 隔离均已完成。
- 尚未完成：controller 进程级精确 resume、BrowseCompPlus 真实 task smoke，
  以及获得人工批准后的 Bcc/Eneutral 执行。
- 尚未启动 BFCL/BrowseCompPlus 正式计分与五轮付费实验；仅执行了下述 BFCL
  三任务基础设施 smoke。
- 目标顺序：`BFCL -> BrowseCompPlus`。
- 进化预算：每个 benchmark 5 轮，每轮 1 个 candidate。
- Base model：`anthropic/Claude-Opus-4.6-hq`。
- Proposer：Claude Code，model `Claude-Opus-4.6-hq`。

本计划解决三个问题：

1. harness 在当前 benchmark 上有没有学到能力；
2. 在 BFCL 上进化出的能力能否迁移到 BrowseCompPlus，以及在 BrowseCompPlus 上继续进化后是否遗忘 BFCL；
3. 对显著变化，用 Harness Delta Attribution（HDA）区分 test-time scaling、
   overfitting 和 generalizable improvement 的贡献。

核心原则是：**使用两个完整任务池做一次固定划分，但不让每一轮 candidate 跑完整任务池。** 大部分任务保持隐藏，只在迁移评测或最终审计时使用。

### 0.1 已经实现的工程改动

本轮实现把“任务执行”和“评分”从同一个 Python environment 中彻底拆开：

```text
solver snapshot
  → 运行 candidate 与 benchmark tools
  → candidate.close() 产生 state_after
  → 导出最小 JSON grading artifact
  → controller 原子提交 harness_store.json
  → 销毁 solver sandbox

grader snapshot
  → 只接收 grading artifact
  → 加载私有答案、checker/judge
  → 返回 score 与 grader usage
  → 不把 verifier 输出或评测后状态传回 harness
```

具体完成项：

1. `TaskEnvironment` 删除 `grade()`，改为只允许导出
   `grading_artifact()`；solver backend 不再拥有评分职责。
2. 新建 `benchmark_graders/`，BFCL 与 BrowseCompPlus 的私有 checker、答案读取和
   judge 调用全部移动到这个包。
3. BFCL solver 不再 import `load_ground_truth_entry`、`ast_checker` 或
   `multi_turn_checker`；构建 solver snapshot 后删除
   `bfcl_eval/data/possible_answer`。
4. BrowseCompPlus 数据被拆成两份：solver 清单只有 `query_id/query`；grader 清单
   才包含 answer 与 evidence doc IDs。solver snapshot 删除完整解密文件和 grader
   清单。
5. OpenSandbox runtime identity 增加 `role=solver|grader`，每个 benchmark 构建两个
   独立 snapshot，并使用 role-specific source archive；solver archive 不包含
   `benchmark_graders`、`grading.py` 或 grader worker。
6. solver request 会剥离 `grader_kwargs`。candidate 只接触任务、工具、observations
   和自身状态，不能接触 grader 配置、答案或 verifier 源码。
7. solver retry 与 grader retry 分账：grader timeout/5xx 只重启 grader，绝不重跑
   model actions，也不回滚已经提交的 `harness_store.json`；合法 score=0 不重试。
8. 本地模式保留相同的时间顺序用于低成本 smoke，但只有 OpenSandbox 双 snapshot
   提供进程和文件系统层面的正式隔离。
9. 当前专项测试为 `54 passed`；其中包括 grader 连续失败三次而 solver 只执行
   一次、state 不回滚、solver source archive 无 grader 包、公开 Browse 清单无
   私有字段等回归测试。
10. 已增加可选 Harbor 0.20 executor 原型：外层 continual controller 保持不变，
    一个 Harbor trial 对应一个 task chunk，Harbor 负责 solver lifecycle、artifact
    collection 和 separate verifier；自定义 environment 让 Harbor 可以恢复已有
    OpenSandbox snapshot ID。四个 role snapshot 已真实构建；BFCL 三任务 smoke 已通过
    完整 solver、状态提交、separate verifier 和 hidden-test 状态隔离流程。
11. solver snapshot identity 额外包含构建配方版本。模型客户端依赖变化时，`require`
    不会静默恢复旧 solver；grader snapshot 不会因 solver-only 依赖变化而失效。
12. BrowseCompPlus 官方包使用 `--no-deps` 安装，再精确安装 FAISS 路径依赖，避免
    未使用的 `pyserini>=1.2.0` 导致大规模 pip 版本回溯；Tevatron 固定到 commit
    `dd063104c81a76d6a77c845f667b46b9e5abd625`。

### 0.2 代码落点

| 文件/目录 | 当前职责 |
|---|---|
| `benchmark_backends/base.py` | solver-only `TaskEnvironment` 与 backend contract |
| `benchmark_backends/bfcl.py` | BFCL 公开 prompt、tools、状态执行与 artifact |
| `benchmark_backends/browsecompplus.py` | Browse query、搜索工具与 response artifact |
| `benchmark_backends/prepare_browsecompplus.py` | 从解密源生成 public solver/private grader 两份清单 |
| `benchmark_graders/bfcl.py` | BFCL ground truth 加载与官方 checker |
| `benchmark_graders/browsecompplus.py` | Browse 私有答案、evidence、judge 和 citation metrics |
| `grading.py` | verifier-only retry 与 score merge |
| `sandbox_worker.py` | OpenSandbox solver-only worker |
| `sandbox_grader_worker.py` | OpenSandbox grader-only worker |
| `sandbox_evaluation.py` | candidate task loop、state commit、artifact 导出；本地 smoke 的 late grading |
| `opensandbox_backend.py` | role-specific snapshot、两阶段调度、artifact transport 与独立重试 |
| `harbor_executor.py` | snapshot-backed Harbor environment、chunk agent、separate verifier 与 trial hook |
| `harbor_backend.py` | 将 Harbor trial 接入 continual controller，并在 verification 前原子提交 solver state |
| `experiment_protocol.py` | 完整任务池 fingerprint、固定 split 与 hidden streams |
| `transfer_evaluation.py` | H0/H1/H2 六格评测与 paired bootstrap |
| `hda_reporting.py` | HDA gate、exact diff、审批点与自包含 HTML |

专项验证命令为：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
pytest -q integrations/agentstream_sequential_meta/tests
```

当前结果是 `54 passed`。2026-08-26 已在真实 OpenSandbox 上构建 BFCL/Browse 各
solver+grader snapshot，并以 Harbor 跑完 BFCL 三任务 smoke。该 smoke 不是正式实验：
train/validation/test 分别为 `1.0/0.0/0.0`，对应模型调用 `13/10/11` 次；两次 verifier
分别收到 `2/1` 条 grader 结果。状态从 `session_count=0` 经 train+validation 变为 `2`，
hidden test 的临时状态为 `3`，但 outgoing 仍为 `2`，确认 test 状态未泄漏进后续流。

## Part I：Meta-Harness 方法依据

### A. 一句话理解

Meta-Harness 不训练 base model 权重，而是让一个 coding agent 自动搜索“模型外面
的程序”。这个可执行程序就是 harness，它决定 prompt、memory、检索、工具、
控制流、验证、重试、上下文压缩和任务结束条件。

每一轮中，proposer 读取历代候选的代码、分数和原始轨迹，提出一个可验证的修改；
固定 evaluator 再用同一个 base model 独立跑分。新代码与新轨迹继续进入经验库，
形成可长期检索的工程闭环。

### B. 核心角色与优化目标

| 角色 | 含义 | 本项目中的对应物 |
|---|---|---|
| 固定模型 `M` | 真正解决任务的 solver，搜索期间权重不变 | Claude-Opus-4.6-hq |
| 候选 harness `H` | 包装模型的可执行、可带状态程序 | `candidate.py + harness_store.json` |
| proposer `P` | 阅读历史并编写新候选的 coding agent | Claude Code |
| evaluator | 固定协议、工具环境和评分边界 | benchmark backend + private grader |
| 经验库 `D` | 历代代码、状态、分数、轨迹和日志 | `evolution/`、`public/`、私有评测目录 |

方法目标可写为：

$$
H^{*} = \underset{H}{\operatorname{arg\,max}}\;
\mathbb{E}_{x \sim \mathcal{X},\; \tau \sim p_M(H,x)}[r(\tau,x)]
$$

也就是在固定模型和 evaluator 下，寻找期望任务奖励最高的 harness。若同时考虑
准确率、上下文长度、调用次数或成本，可以维护 Pareto frontier，而不是过早压成
单一分数。代码里的 `learn` 或 state update 指外部 memory 更新，不是反向传播。

### C. 为什么保存完整历史

Harness 的早期存储或控制流决策，可能在几十步后才造成成功或失败。只把最近一次
分数或摘要交给 proposer，会丢失这种因果链。因此 Meta-Harness 把以下信息外置到
文件系统：

- 每个候选的完整源码与状态迁移；
- 每个任务的 reward、status 和资源消耗；
- prompt、模型输出、工具调用、observations 和状态更新；
- proposer 的诊断、改动和日志。

proposer 可以像工程师一样选择性搜索历史、比较同一种失败、回退有害改动，或组合
不同候选中的有效机制。它并不要求一个固定 mutation operator，也不要求每轮只能
从上一轮赢家做局部修改。

### D. 通用外循环与层级

```text
评测初始 baselines
  → 写入代码、分数和 trajectories
  → proposer 读取历史并生成 candidate
  → import/interface/smoke 校验
  → 在固定 search-phase tasks 上独立评测
  → 保存完整产物并更新 frontier
  → 重复 N 轮
  → 冻结赢家
  → 一次性运行 held-out evaluation
```

需要区分：

| 名称 | 含义 |
|---|---|
| evolution iteration | proposer 生成候选、评测并更新一次历史/frontier |
| candidate | 一份新的完整 harness；一轮可以有一个或多个 |
| candidate evaluation | 候选跑完整 search/validation task flow |
| task/trial/attempt | 某候选在某一道任务上的执行或基础设施重试 |
| agent turn | 一个 task 内的一次模型回复或工具调用阶段 |
| LLM call | 最底层的一次 provider 请求 |

所以“五轮进化”不是只调用模型五次。baseline 是 Phase 0，最终 hidden/test 也不属于
普通 evolution iteration。

### E. Train、validation、test 与实时 sequential stream

“搜索集”是 outer loop 的角色名称，不是第四种数据。凡是会反复影响候选状态、
proposer 决策或 frontier 选择的数据，都属于 search phase：

- train 可以更新候选的外部 memory/state；
- validation 给 proposer 反馈并选择候选；
- hidden/test 必须在搜索冻结后使用，不能影响提案与选择。

真实日常使用中确实没有一个预先声明的 train split，每道题到来时都是当下的 test。
但做研究时如果要判断“是否泛化、是否迁移、是否遗忘”，仍必须人为保留不可见任务。
本项目因此模拟在线状态语义：stream 内 task 连续到来并更新 harness state；同时在
outer-loop 层面固定 search/validation/hidden/audit 边界，避免用未来答案选择程序。

### F. 原仓库三套实现及其含义

| 实现 | 被搜索对象 | evaluator | 主要意义 |
|---|---|---|---|
| `reference_examples/text_classification` | `MemorySystem` | 本地 train/val/test inner loop | 搜索有状态 memory 管理 |
| `reference_examples/terminal_bench_2` | `Terminus2` 子类 | Harbor + TB2 verifier | 搜索完整 terminal agent scaffold |
| `experimental/harbor_meta_harness` | 可编辑 terminal harness | 外层 Harbor + child Harbor jobs | 验证嵌套优化、权限与预算闭环 |

它们是三套独立实验，不是先后执行的三个步骤。论文还包含数学检索实验，但当前仓库
没有发布相应 reference implementation。

文本分类发布版默认从 `no_memory` 和 `fewshot_all` 开始，在 train 上构造 memory，
用 validation accuracy/context cost 更新 frontier，冻结后显式运行 test。论文原始
文本分类设置则是 20 轮 × 每轮 2 candidates，并把 zero-shot、few-shot、ACE、MCE
作为更丰富的初始 population；cleaned release 默认是 20 轮 × 每轮 3 candidates，
两者不能混为一谈。

TB2 reference 会先评测 Terminus-KIRA 和 vanilla Terminus2，但 proposer 主要从
成熟的 KIRA scaffold 出发。默认是 5 轮 × 每轮 1 candidate；论文附录分析过一个
10 轮 run。它在相同公开 89 tasks 上搜索并增加 trials 复评，因此更接近 benchmark
discovery，不是严格的未见任务泛化实验。

Harbor pilot 则故意使用很弱的 1-turn seed，并给 outer agent 有限次数的
`evaluate_harness` 工具。它证明闭环和 trust boundary 能工作，但不能单独支持超过
KIRA 等强 baseline 的研究结论。

### G. Base harness 应如何选择

proposer 强不代表 seed 必须弱。proposer 是“写程序的研究员”，base harness 是被
部署评测的程序；评测期间 proposer 不会临场帮助候选解题。

教学 demo 可以用透明弱 seed，闭环 smoke 可以用故意留有改进空间的 seed，但正式
研究应从简单、高效、可审计且已经能稳定完成工具循环的 harness 出发。否则大量预算
会被浪费在重新实现 timeout、工具解析、上下文交接和完成判定，而不是研究 continual
memory、迁移与遗忘。本项目采用自有的最小 benchmark-neutral candidate contract，
而不是耦合 AgentStream/Exgentic agent interface；后续可把 ante、pi、mini-swe-agent
等实现作为受控 H0 对照，但必须先适配同一个 contract 和资源预算。

### H. 方法优势、风险与迁移要求

优势包括：固定模型权重即可得到系统级改进；产物是可读代码；proposer 能利用失败
轨迹而不只有 scalar reward；prompt、memory、retrieval、tools 和 orchestration 可以
联合搜索。

主要风险是候选评测昂贵且有噪声、对 search/validation 过拟合、全量日志可能包含
敏感内容、任意候选代码需要严格 sandbox、指标可能不能代表真实目标，以及论文设置
与 cleaned release 默认值不完全一致。

迁移到新领域至少需要六部分：稳定候选接口、固定 evaluator、有区分度的 search
任务、可查询经验库、约束清楚的 proposer skill、支持 baseline/frontier/resume/freeze
的 outer orchestrator。正式长跑前应先做 3–5 轮低成本 smoke。

## Part II：本项目的 Continual Harness 实验协议

## 1. 实验对象

### 1.1 为什么选 BFCL 和 BrowseCompPlus

- BFCL 主要考查结构化 function calling、参数构造和多轮工具交互；
- BrowseCompPlus 主要考查开放域搜索、证据获取和答案整合；
- 二者都适合 agent/harness，但能力结构明显不同，比 `BFCL + Tau2` 更适合观察跨 benchmark 迁移。

### 1.2 完整 harness 的定义

实验中的 checkpoint 不是只有一份代码，而是：

```text
H = candidate.py + harness_store.json
```

- `candidate.py`：当前 harness 的执行策略；
- `harness_store.json`：memory、skills、session history 等持久状态。

历代代码、rollout、分数和 proposer 日志会单独保存为搜索历史，但不属于运行时 checkpoint。

### 1.3 Meta-Harness、Benchmark 与 AgentStream 的职责边界

- **Meta-Harness**：生成候选、独立运行候选、用 validation 更新 frontier，并选择当前 benchmark 的赢家；
- **AgentStream**：只提供 Sequential/Interleaved 等研究 setting 和任务排序规则的参考，不进入正式计分运行链路；
- **Benchmark-native solver backend**：只实现公开任务加载、tools、stateful environment
  和 grading artifact，不持有答案或 checker；
- **Private grader**：在 solver sandbox 销毁后，使用官方 checker/judge 与私有数据评分；
- **本项目的 Sequential controller**：实现固定任务顺序，让同一 candidate 的状态在 task 间持续更新，并把赢家完整 checkpoint 传给下一个 benchmark；
- **Base harness**：必须是可独立执行的完整 agent harness，负责 prompt、模型调用、工具循环和持久状态读写；
- **固定 evaluator**：由 solver controller 与 late verifier 两部分组成，搜索期间不得
  被 proposer 修改。

进化发生在 benchmark 内的 candidate iteration 边界，而不是每完成一个 task 就调用 proposer。task 只更新当前 candidate 的运行状态。

`candidate.py` 不实现或继承 AgentStream/Exgentic 的 `Agent`、`Action`、
`Observation` 接口。每个 backend 直接生成项目自有的
`ToolSpec/ToolCall/ToolResult`。backend 在任务结束后只导出 artifact，不调用 grader。
Exgentic 只用于少量公开 parity task 的参考校验，不是正式实验依赖。

### 1.4 当前代码协议

```text
candidate.py
  CandidateHarness(CandidateHarnessBase)
    start(task, context, tools, initial_results) -> HarnessStep
    react(tool_results)                          -> HarnessStep
    close(agent_visible_trajectory)              -> JSON state

solver runtime
  ModelClient + BenchmarkBackend + TaskEnvironment
    → state_after + grading_artifact

grader runtime（solver 销毁后才启动）
  PrivateGrader.grade(grading_artifact) → score
```

solver runtime 持有 base model key、公开任务环境和工具；grader runtime 持有私有
答案、grader model key（若需要）和 checker。candidate 只持有 `ModelClient`、公开
任务输入、工具 schema、工具结果及其 JSON 状态。controller 先调用 candidate
`close()`、导出 artifact、提交状态并销毁 solver sandbox，之后才创建 grader
sandbox，从进程与文件系统层面阻断分数、答案和 verifier 信息进入 memory。

## 2. 三个主 checkpoint

```text
H0：初始 base harness
 |
 | 在 BFCL 上进行 5 轮 Meta-Harness 进化
 v
H1：BFCL validation 选出的赢家（代码 + 状态）
 |
 | 将完整 H1 传入 BrowseCompPlus，再进行 5 轮进化
 v
H2：BrowseCompPlus validation 选出的赢家（代码 + 状态）
```

跨 benchmark 传递的是完整 `代码 + 状态`，不是只传代码，也不是把隐藏评测产生的状态传下去。

## 3. 数据固定划分

### 3.1 划分原则

在任何模型调用前完成以下操作：

1. 枚举两个 benchmark 的完整可用任务池；
2. 保存数据版本、任务 ID、排序规则和 dataset fingerprint；
3. 用固定 seed 生成不可变的 private split manifest；
4. 保存 manifest 的 hash 作为预注册记录；
5. proposer 只能看到 search/validation，不能读取 hidden/audit 的 ID、数据、grader 或结果。

当前按 BFCL `multi_turn_base` 200 条、BrowseCompPlus 830 条规划。正式执行前以实际适配器枚举结果为准；如果数量或 fingerprint 不一致，停止并重新 review，不能静默改变划分。

### 3.2 计划划分

| Benchmark | Search | Validation | Transfer/HDA hidden | Audit reserve | 合计 |
|---|---:|---:|---:|---:|---:|
| BFCL | 20 | 10 | 50 | 120 | 200 |
| BrowseCompPlus | 20 | 10 | 100 | 700 | 830 |

说明：

- `Search + Validation` 共 30 条，是该 benchmark 五轮进化反复使用的任务；
- hidden 集只用于 H0/H1/H2 的迁移矩阵和通过门检验后的正式 HDA；
- audit reserve 在主实验中保持未触碰，只有结果值得进一步确认时才另行批准启用；
- validation 可以向 proposer 提供分数和正常 rollout；hidden 和 audit 完全不可见。

### 3.3 Hidden 集的 stream 组织

评测单位是一条独立任务流，而不是把每个 task 当作独立样本：

- BFCL hidden：10 条 stream，每条 5 个 task；
- BrowseCompPlus hidden：10 条 stream，每条 10 个 task。

每条 stream 都从待评 checkpoint 的独立副本开始。stream 内部按固定顺序连续执行，memory/skills 可跨 task 更新；不同 stream 之间不共享评测中产生的状态。

这样既保留 AgentStream Sequential 的在线状态语义，又能得到 10 个成对的 stream-level 样本用于统计检验。

## 4. 每个 benchmark 内如何进化

每个 benchmark 严格采用 Meta-Harness 的候选搜索与 validation 选择逻辑。

### 4.1 Incoming baseline

- BFCL 的 incoming harness 是 H0；
- BrowseCompPlus 的 incoming harness 是 H1；
- incoming baseline 先运行相同的 20 search + 10 validation，成为 frontier 的正式成员。

### 4.2 五轮进化

每一轮：

1. proposer 读取当前 benchmark 已公开的 search/validation 轨迹、分数、候选代码和合法状态；
2. proposer 提出一个 candidate，并修改 `candidate.py`，必要时提供确定性状态迁移；
3. controller 校验接口、依赖、语法和状态迁移；
4. candidate 从该 benchmark 完全相同的 incoming checkpoint 开始；
5. 顺序运行固定的 20 个 search task，再运行固定的 10 个 validation task；
6. 保存逐 task 轨迹、状态、成本和 validation 分数；
7. 按预注册规则更新 validation frontier。

所有 candidate 相互隔离，不能继承上一个失败 candidate 的状态。candidate 内部的 30 条任务保持 sequential，前一 task 更新的状态可以被后一 task 读取。

### 4.3 赢家和 checkpoint

赢家由 validation 决定：

1. validation 平均分更高；
2. 平均分相同则成功任务数更多；
3. 再相同则平均 token/cost 更低；
4. 完全相同保留较早 candidate。

Baseline 允许获胜。隐藏评测不能用于选择赢家，也不能把其 rollout 或运行后状态写回 H0/H1/H2 主线。

### 4.4 Proposer 可以看到什么

为对齐 Meta-Harness，proposer 可以读取 search/validation 范围内的：

- 所有历代 candidate 代码和 proposer 日志；
- agent 正常可见的模型输出、工具调用、观察和完整 rollout；
- 每条任务的 reward、success/status 和 validation 聚合分数；
- token、model calls、步骤数、成本和运行错误；
- candidate 的合法持久状态以及当前 validation frontier；
- 之前 benchmark 的公开搜索历史。

proposer 不得读取：

- hidden/audit 的任务 ID、任务内容、分数、轨迹和 verifier 输出；
- grader/verifier 源码、solution 文件或 hidden metadata；
- API key 等环境秘密；
- 六格迁移矩阵、显著性检验和 HDA 私有结果。

如果某项信息本来就是 agent 在正常任务交互中看到的，它属于合法 rollout，不额外删除；但 evaluator 的内部判分信息不因此变为可见。

### 4.5 Frontier 与搜索历史

每个 benchmark 单独维护 validation frontier，baseline 和五轮 candidate 都是正式成员。所有候选的代码、输入/输出状态、逐任务轨迹和分数永久留档，即使候选没有晋升也不能覆盖或删除。

进入下一个 benchmark 的只有赢家完整 checkpoint；公开搜索历史可供后续 proposer 检索，但不会作为运行时 memory 自动注入 harness。hidden/audit 产物始终留在 proposer workspace 之外。

## 5. 六格迁移评测

得到 H0、H1、H2 后，在两个固定 hidden 集上评测以下六格：

| Checkpoint | BFCL hidden `E1` | Browse hidden `E2` |
|---|---:|---:|
| H0 | `S1(H0)` | `S2(H0)` |
| H1 | `S1(H1)` | `S2(H1)` |
| H2 | `S1(H2)` | `S2(H2)` |

由此计算：

```text
BFCL 内学习增益       = S1(H1) - S1(H0)
BFCL -> Browse 正迁移 = S2(H1) - S2(H0)
Browse 内学习增益     = S2(H2) - S2(H1)
反向迁移              = S1(H2) - S1(H1)
遗忘量                 = S1(H1) - S1(H2) = -反向迁移
```

同一个差值两端必须使用相同 stream、task 顺序、seed、模型参数、工具配置和评分器。每个 cell 都从 checkpoint 副本开始，运行后的状态只作为分析产物保存，不进入后续进化。

## 6. 显著性门：先证明有变化，再做 HDA

对上述四个差值分别进行 paired bootstrap：

1. 每条 stream 得到一对分数，例如 `(B_i, E_i)`；
2. 计算每条 stream 的配对差值 `d_i = E_i - B_i`；
3. 对 10 个 `d_i` 有放回重采样，默认 10,000 次；
4. 每次计算重采样后的平均差值；
5. 用 bootstrap 分布的 2.5% 和 97.5% 分位数形成 95% CI。

判断规则：

- CI 包含 0：记为没有足够证据，不对该差值画 HDA 的 T/O/G 归因条；
- CI 不包含 0：该差值通过门检验，可以进入 HDA 候选列表；
- 同时报告 signed delta、CI 和原始 10 对 stream 分数，不能只报显著/不显著。

这里的 10 条 stream 适合低成本筛选，但统计功效有限。因此结果表述为探索性证据；如果要做论文级强结论，再从 audit reserve 扩大 stream 数并重复预注册检验。

## 7. 低成本 HDA 漏斗

HDA 不对每个 iteration、每个 checkpoint 和两个全集全面运行。只对**通过显著性门且最关键的 1 个差值**做完整归因，第二个差值需要额外 review 后再启用。

优先级：

1. `BFCL -> Browse` 正迁移：`B=H0, E=H1, eval=E2`；
2. BFCL 遗忘/反向迁移：`B=H1, E=H2, eval=E1`；
3. benchmark 内学习增益只作为次级候选。

### 7.1 Compute matching：构造 Bcc

先用保存的轨迹计算 B 和 E 的推理 compute。口径在看结果前固定，例如 model tokens、model calls 和允许的 retry 分类。

- 如果 `compute(E) <= compute(B)`，按 HDA 规则令 `Bcc = B`，因此 `T = 0`，不额外重跑 Bcc；
- 如果 `compute(E) > compute(B)`，构造只增加等量 compute、但不引入 E 新策略的 `Bcc`。

这是 HDA 用户审批点 1：执行 Bcc 评测前，先提交实现、compute 匹配误差和 exact diff 给用户批准。

### 7.2 Overfitting neutralization：构造 Eneutral

检查 B 到 E 的代码和状态变化，识别答案绕过、硬编码知识、数据集捷径、
答案泄漏或 benchmark 特化等 overfitting 成分。额外模型调用、retry 和采样属于
test-time scaling，计入 T，不能混入 O。

- 若存在 overfitting，只移除这部分，保留 E 的其余策略，形成 `Eneutral`；
- 若没有可识别 overfitting，可建议 `Eneutral = E`、`O = 0`，但仍需提供证据和 exact diff。

这是 HDA 用户审批点 2：用户批准 Eneutral 后才运行评测，不能由实验代码自动决定。

### 7.3 正式分解

所有变体在同一固定 eval stream 集上运行：

```text
Δ = S(E)        - S(B)
T = S(Bcc)      - S(B)
O = S(E)        - S(Eneutral)
G = S(Eneutral) - S(Bcc)
```

- T、O、G 保留正负号；
- 份额只用 `|T| + |O| + |G|` 作分母，不能除以 Δ；
- O 是 detectable overfitting 贡献的下界，G 是 generalizable improvement 的上界；
- 如果显著性门未通过，不构造 Bcc/Eneutral，也不输出误导性的归因条形图。

## 8. 成本预算

以下单位是 task execution，不含基础设施失败后的无效重试：

### 8.1 固定主成本

```text
两 benchmark 的进化：
2 × (1 baseline + 5 candidates) × 30 tasks = 360

六格迁移矩阵：
3 checkpoints × (50 BFCL + 100 Browse) = 450

主实验固定合计 = 810 task executions
```

### 8.2 HDA 增量成本

B 和 E 已在六格矩阵中评过，只需考虑 Bcc 和 Eneutral：

- 若二者都可复用 B/E：额外 0，总计 810；
- 在 BFCL hidden 上完整增加两个变体：最多 `2 × 50 = 100`，总计 910；
- 在 Browse hidden 上完整增加两个变体：最多 `2 × 100 = 200`，总计 1010。

默认只做一个正式 HDA pair，因此预算上限暂定 1010。若两个迁移方向都做完整 HDA，最多再增加 300，总计 1110，必须另行批准。

Audit reserve 不计入当前预算，也不会自动运行。

## 9. Task-level retry 与断点恢复

上一版按整个 benchmark block 设置超时，会因为一条慢请求浪费之前已完成的
task。当前实现将最多 10 个连续 task 放进同一个 OpenSandbox solver worker，以复用
BrowseCompPlus 检索索引等重型只读环境；solver worker 内仍按 task 保存状态并执行重试。

### 9.1 原子 task 执行

每个 task 执行前保存：

```text
candidate hash
state_before_task hash
task ID
attempt ID
runtime snapshot ID
```

solver 成功后先写入 trajectory、usage、最小 grading artifact 和
`state_after_task`，随后才推进 worker 内的 stream cursor。模型或 benchmark 的临时错误只重试当前 task，并从完全相同
的 `state_before_task` 开始。worker 级故障会从该 10-task chunk 的入口 checkpoint
重试，而不是重跑整个 benchmark。

当前尚未完成 controller 进程重启后的精确 task cursor 恢复；现有 `--resume`
会安全地从当前 benchmark 的 incoming checkpoint 重跑该 benchmark。这是正式
大规模计分前仍需补齐的可靠性项。

### 9.2 Retry 规则

只重试基础设施错误：

- transport failure、429、服务端 5xx；
- 模型请求 timeout；
- sandbox/runtime 临时故障。

不重试合法的任务失败或 score=0。建议默认：

- 单次模型请求 timeout：300 秒；
- 每个模型请求最多 2 次额外基础设施 retry；
- 每个 task 最多 3 个 attempt。

retry 必须从完全相同的 `state_before_task` 重启，失败 attempt 的部分状态全部丢弃。基础设施 retry 单独记账，不算作 harness 的 compute；若 retry 是 candidate 策略主动触发，则必须计入 HDA compute。

grader retry 与 task retry 完全分开：solver sandbox 销毁且 `state_after` 提交后，
grader sandbox 才接收 artifact。grader timeout/5xx 最多重试三次，但绝不重跑
solver，也不因合法 score=0 重试。

## 10. OpenSandbox 隔离

- 每个 benchmark 使用两个已预构建、身份固定的 runtime snapshot：solver 与 grader；
- 每个至多 10-task 的 chunk 从 solver snapshot 创建新 sandbox；
- 上传 candidate 和 chunk 的 `state_before`，candidate module 在每个 task/attempt
  重新加载，跨 task 的正式持久状态仍只有 JSON checkpoint；
- solver 成功后取回轨迹、最小 grading artifact 和 chunk 的 `state_after`，先提交状态；
- solver sandbox 销毁后，才从 grader snapshot 创建独立 sandbox 并注入 artifact；
- BFCL solver snapshot 删除 `possible_answer`；BrowseCompPlus solver 清单只有
  `query_id/query`，答案与 evidence 只存在于 grader snapshot；
- proposer 运行在独立 workspace，不能挂载 hidden manifest、hidden 数据、grader、私有分数和模型密钥；
- hidden 评测在 checkpoint 副本上运行，其输出状态绝不写回主线。

镜像/snapshot 只固定依赖和运行环境，不保存实验密钥，也不替代 harness checkpoint。

## 11. 产物与报告

```text
run/
├── experiment.json
├── public_split_commitment.json
├── private_manifests/
├── checkpoints/
│   ├── H0/
│   ├── H1/
│   └── H2/
├── evolution/
│   ├── bfcl/
│   └── browsecompplus/
├── transfer_matrix/
├── significance/
├── hda/
│   ├── approval_1_bcc/
│   ├── approval_2_eneutral/
│   └── paired_scores/
└── report/
    └── agentstream_transfer_hda.html
```

最终 HTML 为自包含文件，至少包含：

- 数据 fingerprint、split commitment 和实验配置；
- H0/H1/H2 代码与状态 hash；
- 五轮进化曲线和 validation frontier；
- 六格迁移矩阵；
- 四个 delta 的 paired bootstrap CI；
- 被批准 HDA pair 的 exact diff、B/Bcc/Eneutral/E 配对分数和 T/O/G；
- token、model calls、wall time、infra retry 和失败记录；
- 所有限制、未通过显著性门的结果和未使用 audit reserve 的事实。

## 12. 实施顺序与停止点

### Phase A：先补执行可靠性

1. `[部分完成]` task-level state/attempt retry 已实现；controller 进程级 cursor 和
   幂等 resume 待补；
2. `[完成]` OpenSandbox 使用有界 task chunk，solver worker 故障从 chunk 入口重试；
3. `[完成]` deterministic candidate error 与可重试 model/backend error 已区分；
4. `[完成]` solver/grader 双 snapshot、late artifact injection、grader failure
   不重跑 solver，以及源码/答案隔离已有单元测试覆盖；真实 BFCL Harbor smoke
   已验证正常 solver→state commit→separate verifier 顺序，kill 注入仍待专项测试。

### Phase B：低成本 smoke

1. `[BFCL 已完成，BrowseCompPlus 待运行]` 各取极少量 search/validation task；
2. 只验证 H0 -> H1 -> H2 传递、逐 task resume 和隐藏目录隔离；
3. 不把 smoke 分数作为正式实验结果。

### Phase C：正式五轮进化

1. 锁定完整数据 fingerprint 和 split manifest；
2. 运行 BFCL 五轮，保存 H1；
3. 将完整 H1 传给 BrowseCompPlus，运行五轮，保存 H2；
4. 进化阶段不接触 hidden/audit。

### Phase D：迁移矩阵与门检验

1. `[已实现，待正式运行]` 运行六格 hidden 评测；
2. `[已实现，待正式运行]` 计算四个 paired delta 和 bootstrap CI；
3. `[已实现]` 未通过的 delta 立即停止归因；
4. `[已实现]` 按优先级生成 HDA gate manifest、exact code/state diff 和 HTML，
   停在审批点 1。

### Phase E：最多一个正式 HDA pair

1. 做静态 diff 和轨迹分析；
2. 提交 Bcc 方案，等待用户审批点 1；
3. 评测获批 Bcc；
4. 提交 Eneutral exact diff，等待用户审批点 2；
5. 评测获批 Eneutral，计算 T/O/G 并生成 HTML。

任何审批点都不能由“继续跑”隐式跳过。

## 13. 本次 review 需要确认的参数

| 项目 | 当前建议 |
|---|---|
| Benchmark | BFCL + BrowseCompPlus |
| 顺序 | BFCL -> BrowseCompPlus |
| 每 benchmark 进化预算 | 5 轮 × 1 candidate |
| 进化任务 | 20 search + 10 validation |
| Hidden 规模 | BFCL 10×5；Browse 10×10 |
| Base/Proposer model | Claude-Opus-4.6-hq |
| 显著性方法 | stream-level paired bootstrap，10,000 次，95% CI |
| 正式 HDA 数量 | 默认最多 1 个显著且关键的 pair |
| Audit reserve | 默认完全不运行 |
| 主预算 | 810；含一个完整 HDA 时最多约 1010 task executions |

## 14. 实现验收条件

- BFCL 和 BrowseCompPlus 的完整任务池、版本和划分均可复现；
- search、validation、hidden、audit 无重叠，且 proposer 在文件系统层面无法访问后两者；
- 没有 per-task Claude Code evolution；
- 每个 candidate 都从相同 incoming checkpoint 开始，candidate 之间不串状态；
- candidate 内 task 顺序固定，代码与状态均能跨 task 持续更新；
- validation 决定赢家，baseline 可以保留，hidden/audit 不参与选择；
- H1 的代码和状态一起传入 BrowseCompPlus；
- H0/H1/H2 的六格评测使用 checkpoint 副本，评测后状态不写回主线；
- task-level 中断恢复不会重复计分，并从精确的 `state_before_task` 重试；
- score=0 不会被误判为基础设施故障而重试；
- 所有 candidate、状态 hash、原始轨迹、成本和失败 attempt 都完整保存；
- HDA 未通过显著性门时停止，Bcc 和 Eneutral 分别经过两个明确审批点；
- HTML 能由落盘产物离线重建，并清楚标记探索性结论和未使用的 audit reserve。
