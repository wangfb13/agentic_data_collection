# 状态图 v3 修改方案：通用 skill 构建框架

> 仓库：`6f5090:/home/dm/DM_das/bottle_grasp_experiment`
> 基线：`bottle_scan/state_graph/`（v2），输出 `outputs/state_graph/20261112_symbolic_v2/`
> v3 输出：`outputs/state_graph/<日期>_symbolic_v3/`。v1/v2 产物、原始 session、旧标签保持只读。

## 0. 目标与原则

v3 是一个**与具体 skill 无关**的框架：任务、动作、观测、成功条件、可扰动变量都由 skill 规格声明，框架代码不出现"a 轴""瓶子""ForcePolicy"之类具体概念。

当前的白瓶 a 轴接触位置恢复（下游 ForcePolicy）只是**第一个实例**。以后换成其他 skill（例如调整夹持力、调整高度、调整接触姿态，或者下游换成主策略的某个阶段），只需要写一份新的规格和对应的成功判定器（evaluator）。

```
Skill 规格  = 任务 + 动作空间 + 观测 + 后置条件(汇点) + 可扰动变量 + 安全包络 + 验收标准
任务        = (物体类别, 动作空间)；部署时观测不到的物体属性（如轻/重）不拆任务
节点        = skill 的一次观测窗口（由规格声明的观测协议定义，本实例为一次稳定双侧轻触）
距离        = max(各观测模态按重复噪声归一化的距离)，模态由规格声明
边          = 实测转移 | 动力学推断 f(o, u) → o′ | 相似 | 汇点
汇点        = 规格声明的后置条件 Success[post_condition_id]，代价 −log p̂
模型        = V（到后置条件的代价）+ π（恢复/操作策略）+ f（观测动力学）
框架代码中不写死任何具体轴、物体、控制器
```

### 0.1 框架层 vs 实例层

| 层 | 内容 | 换 skill 时 |
|---|---|---|
| **框架层**（`bottle_scan/state_graph/` 中的通用模块） | 节点表结构、门控、噪声与距离、四种边、汇点与 R、V/π/f 训练、聚类、事实、操作、导演协议、验证 | **不改** |
| **实例层**（`skills/<skill_id>/`） | `spec.json`、观测提取器、动作适配器、后置条件判定器、采集执行器适配 | **新写** |

本次修改需要同时完成两件事：①把 v2 中写死的内容抽到实例层；②按 v3 设计修改框架层。

---

## 1. Skill 规格：`skills/<skill_id>/spec.json`

规格是框架唯一的任务入口。当前实例 `skills/contact_position_a/spec.json`：

```json
{
  "schema": "skill_spec_v3",
  "skill_id": "contact_position_a",
  "task": {
    "object_category": "white_bottle",
    "hidden_object_properties": ["mass_class"],
    "action_space": {"id": "tcp_translation_a", "dim": 1, "names": ["delta_a_mm"],
                     "max_step": [15.0], "frame": "tcp"}
  },
  "observation": {
    "window": {"extractor": "skills.contact_position_a.observe:probe_window",
               "requires": ["bilateral_contact"]},
    "modalities": {
      "tac":   {"extractor": "contact_geometry", "encoder_features": false},
      "force": {"extractor": "wrench_zeroed"},
      "vis":   {"extractor": "wrist_dinov2", "weight": 0.5}
    },
    "record_only": ["gripper_opening", "closure_floor", "protocol_id", "time_period",
                    "sensor_version", "unloaded_reference_id", "approach_pose", "mass_class"]
  },
  "post_conditions": [
    {"id": "handoff_forcepolicy_floor0330",
     "description": "交给现有 ForcePolicy 后能夹到目标力并完成提起保持",
     "evaluator": "skills.contact_position_a.post:forcepolicy_l2",
     "consumer": {"controller": "existing_forcepolicy", "version": "<model sha>",
                  "params": {"closure_floor": 0.330, "target_force_sdk": 5.0}},
     "p_prior": [1, 1], "p_min": 0.05, "ready_p": 0.8},
    {"id": "handoff_e1e2_squeeze_floor0336",
     "evaluator": "skills.contact_position_a.post:e1e2_squeeze_l2",
     "consumer": {"controller": "e1e2_squeeze", "params": {"closure_floor": 0.336}},
     "p_prior": [1, 1], "p_min": 0.05, "ready_p": 0.8}
  ],
  "active_post_condition": "handoff_forcepolicy_floor0330",
  "perturbation_variables": {
    "a_offset_mm":    {"kind": "continuous", "envelope": [-25, 25], "executor": "trial.delta_mm"},
    "l_offset_mm":    {"kind": "continuous", "envelope": [-10, 10], "executor": "trial.perturbation.l_mm", "active": false},
    "approach_pose":  {"kind": "categorical", "levels": ["pose_0", "pose_yaw30"], "active": false},
    "closure_floor":  {"kind": "protocol", "levels": [0.336, 0.330], "affects": "post_condition"},
    "mass_class":     {"kind": "object_property", "levels": ["light", "heavy"], "observable_at_deploy": false}
  },
  "safety_envelope": {"max_action": [15.0], "force_max_sdk": 15},
  "acceptance": { "...": "沿用现有开发 B / 独立 B 指标" },
  "budget": {"max_new_trials": 160}
}
```

字段说明：

- `hidden_object_properties`：部署时观测不到、但会影响结果的物体属性。**这类属性不拆任务、不做门控**，只记录。它们在图里表现为"同一状态的结果有随机性"，交给后置条件的 p̂ 和 Agent 处理（见第 8、9 节）。
- `post_conditions`：skill 要达到的状态，即汇点。可以有多个，`active_post_condition` 指定训练和部署用哪一个。它不一定是"交给 ForcePolicy"，也可以是"进入主策略某阶段的前提"、"接触力达到某区间"等，由 `evaluator` 判定。
- `perturbation_variables`：Agent 可以操作的变量，分四类：`continuous`（偏移量）、`categorical`（姿态等）、`protocol`（影响后置条件的参数）、`object_property`（物体属性，只能分块采集）。规则演化（扩范围、加密、加维度）就是在这些变量上进行。
- 观测的 `extractor`、后置条件的 `evaluator` 都是实例层函数，按名称动态加载。

---

## 2. 实例层接口：`skills/<skill_id>/`

```
skills/contact_position_a/
  spec.json
  observe.py   # probe_window(trial) -> 观测窗口列表 [{window_index, frames, t_ns, is_terminal}]
  post.py      # forcepolicy_l2(trial, window) -> {"evaluated": bool, "success": bool, "level": "L2", "evidence": [...]}
               # e1e2_squeeze_l2(...)
  action.py    # executed_action(trial, window) -> {"u": [delta_a_mm], "receipt": ...}
  executor.py  # 把框架生成的采集计划翻译成现有采集器的 trial 配置
```

框架只通过这四个接口与具体 skill 交互。对应的通用协议定义在 `bottle_scan/state_graph/skill_interface.py`（新增），包含类型说明和一个最小的假实例用于测试。

v2 `extract_nodes.py` 中与白瓶/E1E2/RSI 有关的解析逻辑（读取 summary.csv、meta.json、SQUEEZE 状态、执行回执等）**整体移到** `skills/contact_position_a/observe.py`、`post.py`、`action.py`。框架的 `extract_nodes.py` 只负责：遍历 session → 调用实例层接口 → 计算通用观测模态 → 写节点表。

---

## 3. 节点：`extract_nodes.py`

节点表通用字段：

| 类别 | 字段 | 说明 |
|---|---|---|
| 标识 | `node_id, skill_id, session, trial_id, window_index, phase, t_ns, group_id` | phase：`A`（开环扰动）/ `B_pre` / `B_post` |
| 任务 | `task_id = f"{object_category}|{action_space.id}"` | 唯一的门控字段 |
| 前提 | `window_valid`（规格中 `observation.window.requires` 全部满足） | 不满足不建节点 |
| 观测 | `obs[<modality>]`：按规格中声明的模态提取的向量 | 参与距离 |
| 动作 | `u_executed`（B 转移中实际执行的动作向量）、`u_receipt` | 用于实测边 |
| 记录属性 | `attrs[<name>]`：规格中 `record_only` 和 `perturbation_variables` 的实际值 | 不参与距离；用于隐变量检验、漂移检测、分析 |
| 规划坐标 | `plan_coords[<var>]`：扰动变量的**实测值**（如实测 TCP 相对 T0 的 a 偏移） | 只用于 Agent 的采集规划和分析，不进入模型输入 |
| 终止 | `is_terminal`：该窗口之后控制权是否交给了后置条件的消费者 | 由实例层给出 |
| 后置条件结果 | `post[<post_condition_id>] = {evaluated, success, level, evidence}` | 只对终止窗口填写；没有评估的后置条件为 `evaluated: false` |

`downstream_success`、`tcp_offset_a` 等 v2 字段保留为兼容别名，标注 deprecated；新代码只读 `post[...]` 和 `plan_coords[...]`。

**当前实例的具体内容**：任务 `white_bottle|tcp_translation_a`（轻、重瓶同一任务，`mass_class` 为记录属性）；`plan_coords.a_offset_mm` = 实测 TCP 相对 T0 在 a 上的投影；`post.handoff_e1e2_squeeze_floor0336` 由历史 E1/E2 和 skill-loop 数据给出；`post.handoff_forcepolicy_floor0330` 需要新数据（第 11 节）。

---

## 4. 门控与距离：`graph.py`、`noise.py`

### 4.1 门控

```python
def gate(a, b, config=None):
    return a.get('window_valid') is True and b.get('window_valid') is True \
        and a.get('task_id') == b.get('task_id')
```

删除 `GATE_FIELDS` 及相关分支。`build_graph` 开头检查：所有节点 `task_id` 相同；每个模态的提取器版本（包括编码器版本）在整张图内一致。

### 4.2 距离

- `MODALITIES` 不再写死，从规格的 `observation.modalities` 读取；每个模态的权重从规格读取（视觉默认 0.5）。
- `observation_vectors` 改为通用：直接取 `obs[<modality>]`。原来的接触几何计算移到通用的模态提取器 `modalities/contact_geometry.py`，作为可复用组件。
- `repeat_group = (session, task_id, 当前激活的扰动变量的计划值, 分块属性)`，这里的分块属性就是 `object_property` 类变量（如 mass_class）。重复只在真正相同的条件下估计噪声。
- 保持：2Σ、Bonferroni（M = 启用的模态数）、只用训练折拟合、Ledoit-Wolf 收缩。
- 新增 `drift_report`：按 session 比较零点（所有扰动变量为参考值的试验）节点的观测分布，产生 `drift(session)` 事实，不自动剔除。

---

## 5. 边

### 5.1 实测转移边

- 同一试验内的 `B_pre → B_post`（以及 B_post → 下一个 B_post）；同一 `task_id`；有 `u_receipt`；窗口递增。
- 边保存 `u`（动作向量）、前后观测。
- 代价：`step_cost + λ·风险`。风险项（力超限、失触等）由规格中的 `safety_envelope` 定义的通用指标计算。

### 5.2 删除运动学边

删除 kinematic 分支。`provenance.same_reference` 保留，仅用于分析。

### 5.3 观测动力学模型 f：新增 `dynamics.py`

```
输入：o（规格中 dynamics.inputs 指定的模态，默认 tac + force）、u（动作向量，维度来自 action_space）
输出：Δo 的低维部分（默认 tac 的接触几何 + force 的法向/切向分量）、p_off（转移偏离可预测范围的概率）
模型：集成（默认 5 个小 MLP），输出均值和标准差
训练：所有实测转移边，按试验组留出
```

- f 是通用的"在这个动作空间里，执行动作后观测怎么变"。不读取 `plan_coords`、`attrs`、T0。
- `p_off` 的标签：实测终点与 f 的均值预测的距离 > 1（按第 4 节的距离）。它表示"这次转移发生了模型解释不了的变化"，在当前实例中多半是瓶子被推动；换 skill 后可能是别的原因，框架不需要知道是什么。

### 5.4 推断边

```
for 每个节点 v, 每个候选动作 u ∈ 规格声明的动作网格（action_space + safety_envelope）:
    ô, σ = f(o_v, u)
    if ‖σ‖_归一化 > max_uncertainty: continue
    w = argmin_w d(ô, o_w)，距离计算中把 σ² 加到噪声协方差上
    if d ≤ 1: 建边 v → w，cost = step_cost + λ_u·‖σ‖ − log(1 − p_off)，保存 u、σ
```

- 同一对节点只保留代价最小的 u；已有实测边的节点对不加推断边。
- 不再一票否决；在留出的实测转移上报告推断边的终点准确率和 σ 的校准情况。

### 5.5 相似边

`gate` ∧ `d ≤ 1` ∧ 互为 K 近邻（K 来自配置，默认 10），代价 0。

### 5.6 汇点：新增 `sink.py`

对每个后置条件 c 分别计算：

```
T_c = { t : t.is_terminal ∧ t.post[c].evaluated }
对每个节点 v：
    N(v) = { t ∈ T_c : d(v, t) ≤ 1 } ∪ ({v} 若 v ∈ T_c)
    s, n = N(v) 中成功数、总数
    p̂_c(v) = (s + α) / (n + α + β)              # (α, β) = p_prior
    若 n ≥ 1 且 p̂_c(v) ≥ p_min：建边 v → SINK[c]，cost = −log p̂_c(v)
```

- 每个后置条件有一套独立的 R：`R[c][node]`。
- 历史数据中只评估过某一个后置条件的节点，只会连到那一个汇点。
- 因为轻、重瓶同属一个任务，同一触觉状态下轻瓶成功、重瓶失败，会表现为 p̂ 处于中间值。这是**真实的部署不确定性**，不是数据错误：恢复目标会被推向轻、重瓶都能成功的区域；如果这样的区域不存在，Agent 会通过 `stochastic_outcome` + `hidden_variable_candidate(mass_class)` 得到证据，再决定是否需要一个能感知该属性的动作（见第 9 节）。

---

## 6. R、可达性、标签：`graph.py`、`labels.py`

- `reverse_costs(node_ids, edges, sink_id)`：对指定后置条件计算 R。
- 可达性：

| 状态 | 条件 |
|---|---|
| `ready` | 节点自身连到汇点，且 `−log p̂ ≤ −log ready_p` |
| `recoverable` | 非 ready，且 R 有限 |
| `unreachable` | R = ∞，且有"执行过动作后仍未满足后置条件"的实测证据（≥1 个试验组） |
| `unsupported` | R = ∞，且没有上述证据 |

- 标签：
  - `exit_label`（β）= ready
  - `invocation_label`（I）= recoverable → 1；ready → 0；其余为空
  - `value_target` = R
  - `action_target` = 最短路径上**各条边的动作向量之和**（多步动作也成立，不依赖任何参考位姿）
  - 每条实测转移的优势 `A = R(u) − R(v) − cost(u→v)`
- 符号命名改为通用：`HandoffReady` → `Ready(post_condition)`；`NeedShift(a, sign)` → `NeedAction(u_direction)`，方向由动作空间各分量的符号组成。

---

## 7. 模型：`skill.py`（拆分为 `value.py`、`policy.py`、`dynamics.py`）

| 模型 | 输入 | 输出 | 训练 |
|---|---|---|---|
| V | 规格声明的观测模态 | 预测 R 及其不确定度 | 有限 R 的节点，按试验组留出 |
| π | 同上 | 动作向量 u（维度来自 action_space，按 safety_envelope 截断） | 实测转移，权重 `exp(η·A)`；可加入推断边样本，权重再乘 `(1 − p_off)·exp(−‖σ‖)` |
| f | 观测 + u | Δo、p_off | 见 5.3 |

- I、β 由 V 派生（ready / recoverable / Unknown）；另训练独立的二分类头作为对照。
- 可选规划：`u* = argmin_{u ∈ π(o) 附近} V(f(o,u)) + step_cost`；π 和规划结果不一致时记录 `disagreement`。
- 运行时（`SymbolicRecoverySkill`）改为通用：观测 → V 判断 → π（或规划）→ 动作适配器执行 → 重新观测 → 循环，直到 ready 或达到规格中的最大步数。交接时输出 `{post_condition, p̂, V, cluster, graph_version, latest_observation}`。
- 所有模型的输入都通过白名单检查：`plan_coords`、`attrs`、`post`、R 不得进入。

---

## 8. 聚类与符号：`clusters.py`

- 在观测空间（规格声明的模态）中聚类，不使用 `plan_coords`。
- 符号由簇内 V / p̂ 决定：

| 簇内情况 | 符号 |
|---|---|
| p̂ 中位数 ≥ ready_p | `Ready(c)` |
| R 有限，π 给出的动作方向一致 | `NeedAction(dir)` |
| R = ∞ 且全部为 unreachable | `Unreachable` |
| R = ∞，否则 | `Unsupported` |
| p̂ ∈ [0.2, 0.8] 且终止样本数 ≥ 4 | `Stochastic(c)` |

---

## 9. 事实：`facts.py`

所有谓词都以**规格中的变量名**作为参数，不写死 a 轴：

| 谓词 | 定义 |
|---|---|
| `no_failure(rule, c)` | 当前任务、当前规则下，所有终止节点对后置条件 c 都成功 |
| `failure_present(var, side, c)` | 沿扰动变量 var 的某一侧存在失败的终止节点（按 `plan_coords[var]`） |
| `boundary_gap(var, [x1,x2], c)` | 沿 var，同一任务内相邻的成功与失败位置之间距离 > 2 × 名义步长且中间无节点（跨 session 合并） |
| `stochastic_outcome(cluster, c)` | 簇内 p̂ ∈ [0.2, 0.8] 且终止样本 ≥ 4 |
| `hidden_variable_candidate(cluster, var, c)` | 在 `stochastic_outcome` 的簇内，按记录属性 var 分组的成功率差异显著（Fisher 精确检验，Bonferroni 校正后 p < 0.05） |
| `unsupported(v)` / `unreachable(v)` | 见第 6 节 |
| `inferred_edge_unverified(n)` | 最短路径上推断边的占比 |
| `dynamics_uncertain(region)` | f 的集成标准差在某簇或某段 `plan_coords` 内超过训练集 90% 分位 |
| `high_disagreement(region)` | V 的集成标准差超过训练集 90% 分位 |
| `coverage(var)` | 每个扰动变量已覆盖的取值或范围，以及每种取值的终止样本数 |
| `coverage_ok(head)` | 沿用现有门槛（V、π、f 各自的最少样本） |
| `drift(session)` | 见 4.2 |
| `model_executable` / `dev_B_pass` / `indep_B_pass` / `label_conflict` | 沿用 |

删除：`object_split`（轻重瓶不再拆分）、`conflict`（改为 `stochastic_outcome`）、`edge_unverified`、`edge_refuted`（运动学边已删除）。

---

## 10. 操作：`operators.py`

操作作用在规格声明的扰动变量上，变量名作为参数：

| 操作 | 前提 | 作用 |
|---|---|---|
| `expand_range(var, side)` | var 为 continuous 且激活 ∧ `no_failure(当前规则)` | 在安全包络内每侧扩展 ≤ 规格声明的最大步长 |
| `densify(var, interval)` | `boundary_gap` ∨ `dynamics_uncertain` ∨ `high_disagreement` 落在该区间 | 区间内至少 3 个新位置 |
| `activate_variable(var)` | var 在规格中声明但未激活，且满足以下之一：① 有 ≥3 个试验组的 `unreachable`（当前动作修不好）；② `coverage(var)` 只有一种取值，且 var 是 categorical（如接近姿态，用于检验泛化） | 新变量先用 3 个水平（含参考）或 2 个类别 |
| `probe_hidden_variable(var)` | `hidden_variable_candidate(·, var, ·)` ∧ var 为 protocol 类 | 只改变 var，成对采集 |
| `block_by_property(var)` | `hidden_variable_candidate(·, var, ·)` ∧ var 为 object_property 类 | 按该属性分块采集同一状态，量化它对结果的影响（属性仍不进入门控） |
| `repeat_state(cluster)` | `stochastic_outcome` ∧ 无 `hidden_variable_candidate` | 在该状态附近重复，判断是否为纯随机 |
| `request_capability(desc)` | 存在显著的 object_property 类 `hidden_variable_candidate`，且没有任一状态对所有属性取值都 ready | **不执行采集**，生成一份说明交给人：当前观测无法区分该属性，需要新的感知动作或新的观测模态 |
| `verify_edges` | `inferred_edge_unverified` 占比 > 0.5 | 采集最短路径上不确定度最大的推断边的起点 B |
| `train` / `validate_B` / `freeze` / `complete` / `review_labels` / `pause` | 沿用 | `validate_B` 需覆盖 `Ready` 与 `NeedAction` 两类簇 |

- `add_axis` 并入 `activate_variable`；`split_by_object` 删除。
- `request_capability` 体现的是框架的边界：Agent 发现"靠现有观测和动作解决不了"时，明确提出需要什么，而不是在现有变量里反复采集。

---

## 11. 导演：`director_bridge.py`

- 系统提示改为通用版本：不出现 a 轴、瓶子、ForcePolicy 等具体名称；具体信息全部来自上下文中的规格（变量名、动作空间、后置条件描述）。
- 新增说明：后置条件可能有多个；`stochastic_outcome` 是数据而非错误；物体属性不可观测时不能用作门控；`request_capability` 的使用条件。
- 假设匹配放宽：`support_when/refute_when` 支持按参数位置部分匹配、数值指标给区间：
  ```json
  {"pred": "hidden_variable_candidate", "args_match": {"1": "mass_class"}}
  {"pred": "boundary_gap", "args_match": {"0": "a_offset_mm"}, "metrics": {"gap_mm": [0, 2.5]}}
  ```

---

## 12. 当前实例需要的新数据

框架改完后，当前实例（`contact_position_a`）需要一个新 campaign：

1. 后置条件 `handoff_forcepolicy_floor0330`：A 的每次轻触直接交给 ForcePolicy，记录 L1（力就绪）、L2（提起保持无掉落），走主策略时登记 L3。
2. B 实验闭环：修正前后都轻触，最后一步交给 ForcePolicy。
3. 轻、重瓶混合采集（同一任务），按 `mass_class` 记录，按块平衡。
4. 扩大 a 偏移范围直到出现失败。
5. 至少两种接近姿态，用于检验 f 和 V 的跨姿态泛化。
6. 同一盖板下 0.330 / 0.336 成对：两个后置条件各自评估。

scheduler 侧：恢复闭环，在 `events.jsonl` 中写入修正前后的轻触窗口和交接结果，使部署 episode 也能抽成节点。

---

## 13. 验证（`report.py`）

| 编号 | 内容 |
|---|---|
| V1 | 零点重复一致率 |
| V2 | 逐模态单独评估：每个模态用岭回归探针预测 `plan_coords`、后置条件结果（AUC），按试验组留出 |
| V3 | p̂ 校准（可靠性图） |
| V4 | 多后置条件对比：同一节点在不同后置条件下的 p̂；`hidden_variable_candidate` 是否指向 closure_floor |
| V5 | 隐藏物体属性：`mass_class` 是否造成 `stochastic_outcome`；是否存在两种属性都 ready 的状态区域 |
| V6 | f 的留出预测误差、σ 校准、p_off 的 AUC |
| V7 | 图标签与旧标签对照 |
| V8 | V 的留出误差；π 在留出转移上的 R 下降率；有/无推断边时 unsupported 的比例 |
| V9 | 导演回放 |
| V10 | 跨姿态泛化：用一种姿态训练，在另一种姿态的留出数据上评估 |
| **V11** | **框架通用性**：用一个合成的第二 skill（`tests/fixtures/skill_toy/`，例如二维动作、不同后置条件）跑通完整流程，证明框架代码中没有写死当前实例 |

---

## 14. 测试

- `test_skill_interface.py`：用玩具 skill 跑通抽取 → 建图 → 训练 → 事实 → 操作；框架模块中 grep 不到 `tcp_offset_a`、`white_`、`forcepolicy` 等实例相关字符串（只允许出现在 `skills/` 下）。
- `test_gate_v3.py`：不同协议、时期、mass_class 的节点可以相似；不同 task_id 不可以。
- `test_sink.py`：Beta 平滑、p_min、多后置条件分别计算 R、未评估的后置条件不连汇点。
- `test_dynamics.py`：f 输入白名单；推断边代价随 σ 单调；已有实测边时不加推断边；多维动作空间可用。
- `test_labels_v3.py`：`action_target` 等于路径动作之和（含多维）。
- `test_facts_ops_v3.py`：`hidden_variable_candidate` 只在显著时出现；protocol 类变量触发 `probe_hidden_variable`，object_property 类变量触发 `block_by_property` / `request_capability`。
- 泄露检查：`plan_coords`、`attrs`、`post`、R 不进入 V、π、f。

---

## 15. 实施顺序

| 步骤 | 内容 | 机器人 |
|---|---|---|
| 1 | 定义 `skill_interface.py` 与 `spec.json` 结构；把 v2 中的实例逻辑移到 `skills/contact_position_a/`；建玩具 skill | 否 |
| 2 | 节点表改为通用结构；门控、距离、漂移检测 | 否 |
| 3 | `sink.py`、多后置条件 R、可达性与标签 | 否 |
| 4 | `dynamics.py`、推断边，删除运动学边 | 否 |
| 5 | V、π、f 训练与通用运行时 | 否 |
| 6 | 聚类、事实、操作、导演改为通用版本 | 否 |
| 7 | 用历史数据跑 V1–V9、V11；scheduler 恢复闭环并记录交接结果 | 否 |
| 8 | 按第 12 节采集当前实例的新 campaign；跑 V10；训练、冻结、独立 B、主任务对比 | **是，需操作员授权** |

约束沿用旧指导第 12 节：不改旧数据、不泄露标签字段、评估只用封存的独立 B 和主任务结果、不合格如实报告。
