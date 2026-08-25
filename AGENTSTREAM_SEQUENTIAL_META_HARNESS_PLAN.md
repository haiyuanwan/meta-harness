# AgentStream Sequential + Meta-Harness 实验计划

## 1. 实验目标

本实验把两部分组合起来：

- **Meta-Harness** 负责在每个 benchmark 内生成候选、独立评测候选、维护 validation frontier，并选择赢家；
- **AgentStream Sequential** 负责任务流顺序、benchmark 内的连续状态更新，以及赢家跨 benchmark 传递。

本文中的“完整 harness”始终指：

```text
candidate.py + harness_store.json
```

前者是可执行逻辑，后者是 memory、skills、session history 等持久状态。磁盘上的历代代码、分数、轨迹和 proposer 日志统称为“搜索历史”，它们不属于运行中的 harness。

## 2. 固定实验条件

- Base model：`anthropic/Claude-Opus-4.8-C`
- Proposer：Claude Code，model `Claude-Opus-4.8-C`
- AgentStream mode：`sequential`
- Task selection seed：`42`
- Task ordering seed：正式复现默认使用 AgentStream 官方脚本的 `44`
- 正式规模：每个 benchmark 50 条任务
- 正式 benchmark 顺序由 AgentStream `get_unified_task_order()` 决定；对六个官方 benchmark，顺序为：

```text
appworld -> bfcl -> browsecompplus -> hle -> swebench -> tau2
```

Base model、benchmark 工具面和 evaluator 在搜索期间保持固定。

## 3. 每个 benchmark 的任务划分

AgentStream 先按官方逻辑选择并排序 50 条任务，再按连续位置划分：

| Split | 数量 | 用途 |
|---|---:|---|
| train/search | 30 | 让完整 harness 顺序执行并积累状态，向 proposer 提供搜索经验 |
| validation | 10 | 为候选计分、更新 frontier、选择赢家 |
| test | 10 | 搜索冻结后仅运行一次，报告未参与当前 benchmark 选择的结果 |

划分必须写入 `split_manifest.json`。三个 split 不得重叠；恢复运行时必须复用原 manifest，不能重新抽样。

小规模实验按相同比例显式传入 split 数量，不使用隐式四舍五入。

## 4. Benchmark 内的 Meta-Harness 循环

### 4.1 Incoming baseline

当前 benchmark 首先接收上一个 benchmark 的赢家完整 harness。第一个 benchmark 使用 generation-zero harness。

控制器克隆这份完整 harness，并依次运行 train/search 和 validation，得到：

- baseline 的 validation 分数；
- train/validation 的逐任务结果与完整 rollout；
- 运行后的持久状态；
- token、步骤数、成本和错误信息。

Baseline 是 frontier 的正式成员，后续候选没有提升时必须允许 baseline 继续获胜。

### 4.2 Evolution iteration

每一轮执行：

1. Claude Code 检索当前及之前 benchmark 的搜索历史；
2. 形成可证伪的改进假设；
3. 写出新 `candidate.py`，必要时提供确定性的状态迁移；
4. 控制器进行语法、接口、依赖和 checkpoint 校验；
5. 每个合法候选从当前 benchmark 完全相同的 incoming 持久状态开始；
6. 候选顺序运行完整 train/search 和 validation；
7. 控制器保存候选代码、输入/输出状态、分数、原始轨迹和 proposer 日志；
8. 根据 validation 指标更新 frontier。

代码能够 import 或通过 contract 只表示候选可以接受评测，不能据此自动晋升。

### 4.3 候选状态隔离

候选之间不能串状态。每个候选都必须从同一份 benchmark incoming checkpoint 克隆开始。候选内部则保持 AgentStream Sequential：前一条任务更新的 memory/skills 可以被后一条任务读取。

Proposer 可以改变状态结构及迁移逻辑，但不能向持久状态手工写入某道任务的隐藏答案。

## 5. 搜索信息权限

为对齐 Meta-Harness，proposer 可以读取 train/validation 的：

- 所有历代候选代码；
- agent-visible 原始轨迹、模型输出、工具调用与观察；
- 每条任务的 reward、success/status；
- validation 聚合分数；
- token、步骤数、成本与运行错误；
- 候选的合法持久状态；
- frontier 和历代 proposer 日志。

不向 proposer 开放：

- grader/verifier 源码；
- solution 文件或 hidden task metadata；
- API key 等环境秘密；
- test 分数、test verifier 输出和 test 私有评测目录。

如果某项信息本来就是 agent 在正常任务交互中看到的，它属于合法 rollout，不额外删除。Agentic benchmark 没有显式标准答案时，不额外制造或暴露答案。

## 6. Frontier 与赢家

每个 benchmark 单独维护 validation frontier，同时把所有历史永久留在全局搜索目录。

部署赢家采用预注册规则：

1. validation 平均分更高；
2. 平均分相同时，成功任务数更多；
3. 再相同时，平均 token/cost 更低；
4. 完全相同时保留较早候选。

同时保存 score-cost Pareto frontier，供分析使用。当前 benchmark 的选择不得使用 test 分数，也不得为了选择而重放已结束 benchmark。

## 7. Test 与跨 benchmark 传递

搜索预算结束后，恢复 validation 赢家运行结束时的完整 harness，冻结代码，并顺序运行一次 test split。

Test 阶段：

- 不调用 Claude Code；
- 不生成候选；
- 不更新 validation frontier；
- test 结果只写入私有指标目录；
- harness 仍可根据 agent-visible rollout 正常更新持久状态。

传给下一个 benchmark 的是：

```text
赢家 candidate.py
+
赢家跑完 test 后的 harness_store.json
```

搜索历史继续保留，供后续 proposer 检索；test 私有反馈不进入 proposer workspace。

## 8. 对照组与报告指标

先实现一个不运行 Meta-Harness 外循环的 Sequential control，验证：

- task set/order 与 AgentStream 官方一致；
- benchmark 内及 benchmark 间状态连续；
- 相同 base model、工具和 evaluator 配置可运行。

Meta-Harness 实验分别报告：

- baseline validation score；
- 每轮 candidate validation score；
- validation frontier；
- 每个 benchmark 的一次性 test score；
- Sequential 累计 test score；
- 跨 benchmark 传递前后的代码与状态 hash；
- proposer 与 solver 的 token、成本和 wall time。

Search/validation 分数不得标记为 test 成绩。

## 9. 输出结构

```text
run/
├── experiment.json
├── progress.json
├── current/
│   ├── candidate.py
│   └── harness_store.json
├── global_history/
│   ├── evolution_summary.jsonl
│   └── candidates/
└── benchmarks/
    ├── 000_appworld/
    │   ├── split_manifest.json
    │   ├── incoming/
    │   ├── baseline/
    │   ├── iterations/
    │   ├── frontier.json
    │   ├── winner/
    │   ├── private_test/
    │   └── outgoing/
    └── ...
```

## 10. Resume 与失败恢复

`progress.json` 至少记录 benchmark、phase、iteration、candidate、split、task index，以及 incoming/frontier harness hash。

- 已完成候选不重复评测；
- 未完成候选从它自己的输入 checkpoint 重新运行；
- 无效或崩溃候选不污染 frontier；
- proposer 失败时保留当前 frontier；
- test 不完整时从受信 checkpoint 恢复；
- 只有写出完整 outgoing checkpoint 后，benchmark 才算完成。

## 11. 验证阶段

### Smoke

- `bfcl,tau2`
- 每个 benchmark：2 train + 1 validation + 1 test
- 1 个 evolution iteration
- 每轮 1 个 candidate

### Pilot

- `bfcl,tau2`
- 每个 benchmark：6 train + 2 validation + 2 test
- 2 个 evolution iterations
- 每轮 1 个 candidate

### 正式实验

- 六个官方 benchmark
- 每个 benchmark：30 train + 10 validation + 10 test
- 初始建议：5 iterations × 1 candidate

正式建议规模约为：

```text
6 × [(1 baseline + 5 candidates) × 40 train/validation + 10 test]
= 1500 次 task evaluation
```

正式运行前根据 pilot 的成本、稳定性和候选差异锁定搜索预算。

## 12. 实现验收条件

- 没有 per-task Claude Code evolution；
- task ordering 与 AgentStream 一致；
- split 固定且无泄漏；
- 完整 harness 包含代码和持久状态；
- 所有候选从相同 incoming harness 克隆；
- validation 决定赢家，baseline 可保留；
- test 不可被 proposer 读取且不参与选择；
- 赢家代码和最新状态跨 benchmark 传递；
- 所有候选、分数和原始搜索轨迹完整保存；
- 中断恢复不会重复计分或串状态。
