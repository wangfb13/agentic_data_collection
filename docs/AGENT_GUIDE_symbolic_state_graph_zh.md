# 实施指导：触觉状态图 + 符号推理驱动的 skill 构建

> 适用仓库：`6f5090:/home/dm/DM_das/bottle_grasp_experiment`
> 目标：在现有 skill-loop（A 采集 → 三部分训练 → B 验证 → Agent 决策）之上，加入**触觉状态图**和**符号推理层**。用图给出以下游结果为准的标签和恢复目标，用图上自动提取的符号事实驱动 Agent 更新采集规则。
> 实施原则：**先离线、后真机**。第 1–4 阶段只读已有数据，不连接机器人，不修改任何已有 session、campaign、标签或冻结配置。

---

## 0. 背景与总体逻辑

1. 主策略 StarVLA 只用 UMI 数据训练，部署时存在三类本体感知误差：UMI 轨迹记录误差、夹爪编码器误差、UMI 与真机的本体差异。这些误差会导致接触位置偏移、夹持状态不可靠。
2. 触觉 skill 在接触时接管控制，由三部分组成：
   - **调用条件 I**：当前接触是否需要由本 skill 修正、能否修正；
   - **恢复模型 π**：输出修正量 Δa（将来可扩展 Δl）；
   - **退出条件 β**：修正后的状态能否交给下游（ForcePolicy → StarVLA）。
3. 本方案的核心：
   - 把每个接触状态建成图节点，从下游成功结果反向搜索，得到每个节点到成功的代价 R(v)；
   - 用 R(v) 自动生成 I、π、β 的标签和目标，不再使用"回到 T0"或人工看图打的标签；
   - 程序从图中自动计算符号谓词（事实）；
   - LLM Agent 在事实之上提出假设，并从带前提条件的操作库中选择下一步；本地校验器检查前提是否成立。

分层与职责：

| 层 | 负责 | 由谁产生 |
|---|---|---|
| 状态图 | 节点、边、R(v)、聚类 | 程序，从数据确定性计算 |
| 符号事实（谓词） | 当前证据的符号描述 | 程序，从图确定性计算，**LLM 不得编写** |
| 假设 | 对失败原因、区域边界的可证伪陈述 | LLM 提出，程序记录状态 |
| 操作 | 下一步做什么（扩范围、加密、加轴、训练、验证……） | LLM 选择，程序校验前提和参数 |
| skill 模型 | I / π / β | 训练代码，标签来自图 |

---

## 1. 输入数据

| 来源 | 路径（outputs/ 下） | 说明 |
|---|---|---|
| 白瓶 A（闭合下限 0.336） | `white_a_current_x5_20261110_fast` | 22 次，负偏移失败多 |
| 白瓶 A（闭合下限 0.330） | `white_a_director_live_0330_20261110/batches/001/A` | 22 次，全部成功 |
| 白瓶开发 B | `white_b_current_x5_dev_20261109` | 6 次 |
| 绿瓶 E1 A | `e1e2_A_formal01` | 99 次，含失败区 |
| 绿瓶 E2 B | `e1e2_B_formal01` | 33 次，含多轮修正转移 |
| RSI 采集 | `rsi_contact01/batches/*` | 178 次，按协议兼容性选用 |

每个试验目录下有：`streams.h5`（`dd_left/right`、`dd_encoded_*`、`gel_keyframe_*`、六维力、`gripper_normalized`、腕部图像等）、`meta.json`、`summary.csv` 中的状态与标签。

**不能用作时间边的数据**：部署侧 `contact_scheduler_*_ood_debug_*/outputs/ood_debug_plus5`。该版本修正后没有再次轻触，缺少修正后的观测。

共享编码器：
- 触觉：`v1_analysis:/mnt/shared_data/zxd/260611_tac_encoder_vit-b-16/vit-b-tac2_s/output/checkpoint-99.pth`（MAE ViT-B/16，输入为单指 depth + deformation）。按第 3.2 节的方式使用，**不要用 CLS token**。
- 视觉：冻结的通用预训练模型（如 DINOv2 ViT-S/B），只编码腕部图像。

---

## 2. 节点定义

**一个节点 = 一个接触观测窗口**：一次 PROBE 稳定窗口（A 中每次试验一个；B 中每轮修正前后各一个）。

### 2.1 字段表（`nodes.parquet`，每行一个节点）

| 类别 | 字段 | 计算方式 | 部署时可得 | 用途 |
|---|---|---|---|---|
| 标识 | `node_id, session, trial_id, window_index, phase(A/B_pre/B_post), t_ns` | 元数据 | — | 追溯 |
| **门控** | `object_id`（white_light / white_heavy / green_*） | 试验元数据 | 是 | 必须一致 |
| | `protocol_id`（闭合下限、轻触力、目标力的哈希） | 配置快照 | 是 | 必须一致 |
| | `bilateral_contact` | 双指法向力 ≥ 接触阈值 | 是 | 必须为真才建节点 |
| | `gripper_state`（open / closing / near_floor） | 开度与闭合下限之差 | 是 | 必须一致 |
| | `edge_class`（none / root / tip，按手指） | 接触掩码到指根、指尖边缘的距离 < 阈值 | 是 | 必须一致 |
| | `force_bin` | 法向力分档（按轻触力带） | 是 | 必须一致 |
| | `encoder_version` | 编码器哈希 | 是 | 必须一致 |
| **观测** | `z_tac_left, z_tac_right`（768 维） | 第 3.2 节 | 是 | d_tac |
| | `sensor_contact_{left,right}`：深度加权质心 (u, v) mm、面积 mm²、到根/尖边距离 mm | 从 DD 直接计算 | 是 | d_pos |
| | `sensor_contact_mean, sensor_contact_diff` | 左右均值、差值 | 是 | d_pos |
| | `wrench_{left,right}`（6 维，扣零后，传感器坐标系） | 窗口内中值 | 是 | d_force |
| | `gripper_opening` | 窗口内中值 | 是 | d_open |
| | `z_vis`（腕部图像特征） | 视觉编码器，取窗口中间帧 | 是 | d_vis（低权重） |
| **标签/分析** | `tcp_offset_a, tcp_offset_l`（mm，实测 TCP 相对 T0） | session_meta + 位姿流 | **否** | 运动学边、校验 |
| | `object_contact_h, object_contact_theta` | 实测 TCP + 传感器接触点 + 固定物体位姿 | **否** | 校验、任务头目标 |
| | `planned_delta_mm` | 计划偏移 | 否 | 仅分析，**禁止作为特征** |
| **结果** | `downstream_success` | SQUEEZE 达到目标力 ∧ LIFT ∧ HOLD 无掉落 ∧ 人工标签为正常 | 否 | 成功汇点 |
| | `downstream_status` | hold_complete / force_not_reached / no_contact / … | 否 | 诊断 |
| | `R`, `cluster_id`, `symbol` | 第 4、5 节 | 否 | 标签 |

**数据泄露规则**：标签/分析类和结果类字段，只能用于建边、训练目标和评估，**不得进入相似度计算、任务头输入或部署时的判定**。

### 2.2 传感器接触位置（必须实现）

```
mask = depth > depth_thresh（使用已有 dd_threshold，并扣除空载参考）
w    = depth * mask
(u,v) = Σ w·(x,y) / Σ w        → 像素坐标换算成 mm（按有效胶面尺寸标定）
area = Σ mask × 像素面积
edge_dist_root / edge_dist_tip = 接触掩码到对应边的最小距离
```
质心计算复用现有 `window_descriptors.py` / `sensing.py`，补上深度加权和 mm 换算。

### 2.3 物体上的接触位置（离线，只用作标签）

同一 session 内瓶子固定，物体位姿 = 参考 T0 下的瓶轴。将接触点（TCP + 手指几何 + 传感器上的接触位置）变换到物体坐标系，得到沿瓶轴的高度 h（mm）和绕瓶轴的角度 θ。只在同一物体、同一 session 内比较。

---

## 3. 观测特征

### 3.1 噪声估计（所有距离都依赖它）

- 重复组：同一 session、同一物体、同一计划偏移的试验（包括零点监测）。
- 每种模态 m 用重复组内样本的协方差估计 Σ_m（样本少时用 Ledoit-Wolf 收缩）。高维特征先在训练集上做 PCA，降到 16–32 维，再估计协方差。
- 保存到 `noise_model.json`，记录每个模态的维度、样本数和来源组。

### 3.2 触觉编码器使用方式

- 两根手指各自输入：`dd_encoded` 的 3 通道，归一化方式与训练时一致（ch0 / 3000；ch1、ch2 先减 25000 再除以 10000）。**先统计你这批数据的通道分布**，如有截断，记录比例并报告。
- 取 patch token 的均值作为特征（不用 CLS）。推理时关闭 mask（mask_ratio = 0）。
- 窗口内取 5 帧特征的中值。
- 不做任何翻转增强。

### 3.3 编码器的后续训练（可选，第 6 阶段）

继续 MAE 预训练，加入本仓库约 40 万帧 DD，并加一个**六维力回归**辅助头：
- 小 MLP 接在池化特征之后的投影层上；
- 回归目标为扣零后、传感器坐标系下、按标准差归一化的力；
- 力损失权重 0.1–0.5，用线性探针选择；
- **不加入**任务结果、R、相对位移这些监督，以保持跨任务可用。

---

## 4. 状态图

### 4.1 距离

```
gate(s,s') 任一不一致 → d = ∞
d_m = sqrt((x−x')ᵀ Σ_m⁻¹ (x−x'))       m ∈ {tac, pos, force, open, vis}
d_meas = max(d_tac/τ_tac, d_pos/τ_pos, d_force/τ_force, d_open/τ_open, w_vis·d_vis/τ_vis)
τ_m = χ²(df_m) 的 95% 分位的平方根；w_vis 默认 0.5
```
`d_meas ≤ 1` 表示"在测量噪声范围内不可区分"。

### 4.2 边

| 边类型 | 建立条件 | 代价 | 状态 |
|---|---|---|---|
| `observed`（实测转移） | B 中同一试验内，修正前节点 → 修正后节点，带执行回执 | 1 + λ_f·力风险 + λ_c·失触风险 | 已证实 |
| `kinematic`（运动学） | 同一 session、同一物体、同一协议的 A 节点 u、v，假设瓶子不动、沿 a 平移 tcp_offset 之差即可从 u 到达 v；平移量 ≤ 动作上限 | 1 + 同上 | 待证实，代价 × κ（默认 1.5） |
| `similar`（相似） | 门控通过、d_meas ≤ 1、在 d_meas 下互为 K 近邻（K = 10） | 0（等价合并，不计步数） | — |
| `sink` | `downstream_success = true` 的节点 → 虚拟成功汇点 | 0 | — |

运动学边的校验：当 B 中出现与某条运动学边对应的实测转移时（起点相似、Δa 相近），比较实际到达的节点：
- 与运动学边预测的终点 d_meas ≤ 1：标记 `confirmed`，代价恢复为 1；
- 否则标记 `refuted`（可能是瓶子被推动或打滑），删除这条边并记录。

### 4.3 R(v)

从成功汇点在反向图上做 Dijkstra，得到每个节点的 R(v)。到不了成功的节点 R = ∞。相似边代价为 0，即相似节点视为同一状态。

### 4.4 由图生成的 skill 标签

| 目标 | 规则 |
|---|---|
| 退出条件 β 的正例 | R(v) = 0 |
| β 的负例 | R(v) > 0 |
| 调用条件 I 的正例 | 0 < R(v) < ∞ |
| I 的负例（无需调用） | R(v) = 0 |
| 不适用（交给其他 skill） | R(v) = ∞ 且 bilateral_contact = true |
| 恢复目标 | 沿最短路径到达的 R = 0 节点中，**到最近失败节点（同物体、同协议）的 tcp_offset 距离最大**的那个；Δa_target = 目标节点 tcp_offset_a − 当前节点 tcp_offset_a |
| B 转移的权重 | w = clip(1 + β_w·tanh(γ·(R_before − R_after)), 0, 1.5)；R 不下降的转移权重为 0 |

按物体分别计算，**不得跨物体合并恢复目标**。

---

## 5. 聚类与符号落地

1. 在 `similar` 边构成的子图上取连通分量，再在每个分量内按 R 和 tcp_offset_a 用 HDBSCAN 细分（`min_cluster_size = 3`）。
2. 每个簇按其中节点的 R 分布赋予符号：

| 簇内 R 分布 | 符号 |
|---|---|
| 全部为 0 | `HandoffReady(obj, proto)` |
| 全部在 (0, ∞)，恢复目标方向一致 | `NeedShift(axis, sign, obj)` |
| 全部为 ∞ | `Unrecoverable(axis, obj)` |
| 混杂（同时有 0 和 > 0，或同时有有限值和 ∞） | `Ambiguous(cluster)` → 产生 `conflict` 事实 |

3. 部署时的符号判定：计算新观测到各簇的 d_meas；落在某簇内（≤ 1）就取该簇的符号；不在任何簇内，或多个簇冲突，就输出 `Unknown`，由调用条件拒绝。

---

## 6. 符号事实（谓词）

由程序在每次图更新后重新计算，保存到 `facts.json`。每条事实都带 `evidence`（节点 ID 列表）和数值依据。

| 谓词 | 计算方式 |
|---|---|
| `no_failure(rule_id)` | 当前规则范围内所有节点 R = 0 |
| `failure_present(axis, side, obj)` | 该侧存在 R > 0 或 downstream_success = false 的节点 |
| `boundary_gap(axis, [x1,x2], obj)` | 沿该轴，相邻成功节点与失败节点的 tcp_offset 间距 > 2 × 名义分辨率，且区间内没有节点 |
| `object_split(obj1, obj2, axis)` | 两个物体 R = 0 区间的中心差 > 噪声的 2 倍，或区间交集占比 < 50% |
| `conflict(v1, v2, attr?)` | d_meas ≤ 1 但下游结果相反；如果两者某个门控外的属性不同（如协议、时间段），在 attr 中给出 |
| `unreachable(v, axis)` | bilateral_contact ∧ R = ∞ ∧ 沿 axis 的所有运动学边都无法连到成功 |
| `edge_unverified(n)` | 待证实的运动学边数 n，及其在路径中的占比 |
| `edge_refuted(e)` | 被推翻的运动学边 |
| `high_disagreement(axis, x)` | 恢复模型集成在 x 处的分歧（Energy Score 形式）高于训练集 90% 分位 |
| `coverage_ok(head)` | I / β 各类别满足最少训练 4、校准 2、验证 2 个不同试验组；π 满足 ≥ 20 条合格样本 |
| `drift(session)` | 零点重复节点之间 d_meas > 1 的比例 > 20% |
| `model_executable(model_id)` | 三部分训练通过现有门槛 |
| `dev_B_pass / indep_B_pass` | 沿用现有验收指标 |

---

## 7. 操作库（采集规则更新与流程推进）

### 7.1 采集规则的两层

- **安全包络**（写在 skill 规格里，不可更改）：每个轴的最大物理范围（a ±25 mm，l ±10 mm）、允许引入的轴、动作上限、力限值。
- **当前规则**（Agent 更新）：`{rule_id, active_axes, range_mm[axis], nominal_step_mm[axis], objects, reps_policy, parent_rule_id, reason, evidence_refs}`。每个版本都保存在 `rules/rule_XXXX.json`；节点记录它是在哪个 `rule_id` 下采集的。

### 7.2 操作定义

| 操作 | 前提（全部为真） | 参数约束 | 效果 |
|---|---|---|---|
| `expand_range(axis, side)` | `no_failure(current)` ∨ ¬`failure_present(axis, side, *)` | 每侧 ≤ 5 mm，不超安全包络 | 新规则 + collect_A 计划 |
| `densify(axis, [x1,x2], obj)` | `boundary_gap(axis,[x1,x2],obj)` ∨ `high_disagreement(axis, x∈[x1,x2])` | 区间内 ≥ 3 个新位置 | 同上 |
| `split_by_object(obj1, obj2)` | `object_split(obj1, obj2, axis)` | — | 两个物体分别计算恢复目标和验收 |
| `probe_hidden_variable(var, levels)` | `conflict(v1, v2, attr)` 且 var 是规格中允许变化的协议变量，或可记录的属性 | 只改变 var，其余保持一致，同一偏移下成对采集 | 对照批次；var 加入节点属性和门控 |
| `add_axis(axis)` | `unreachable(v, current_axis)` 的节点数 ≥ 3 ∧ axis 已在规格中声明 | 首次只用 3 个水平，并保留原轴的参考点 | 新规则 |
| `verify_edges` | `edge_unverified(n)` 且其占比 > 50% | 选择覆盖最多待证实边的 B 起点 | validate_B 计划 |
| `train` | 图或标签自上次训练后发生了变化 ∧ `coverage_ok(I) ∧ coverage_ok(β) ∧ coverage_ok(π)` | — | 三部分同时重训 |
| `validate_B` | `model_executable` | 偏移位置必须覆盖 `NeedShift` 和 `HandoffReady` 簇 | 开发 B |
| `freeze` | `dev_B_pass` | — | 独立 B（封存） |
| `complete` | `indep_B_pass` | — | 导出 |
| `review_labels` | 人工标签与 downstream_success 矛盾 | 列出试验 | 等待人工 |
| `pause` | 总可用（兜底） | 必须说明缺少哪条事实 | 停止 |

零点监测、轻重瓶平衡、预算等现有规则继续由原有校验器执行。

---

## 8. LLM 导演协议（修改现有 director）

### 8.1 输入

```json
{
  "spec": { "skill": "...", "envelope": {...}, "fixed_protocol": {...}, "budget_left": ... },
  "current_rule": { ... },
  "facts": [ {"id":"F12","pred":"boundary_gap","args":["a",[-12,-8],"white_heavy"],"evidence":[...]} ],
  "hypotheses": [ {"id":"H1","claim":"...","status":"open|supported|refuted","evidence":[...]} ],
  "applicable_operators": [ {"op":"densify","premises_satisfied_by":["F12"]} , ... ],
  "history": [ 最近 5 次决定及其结果摘要 ]
}
```
`applicable_operators` 由程序预先计算，只列前提已经成立的操作。

### 8.2 输出（只允许一个 JSON 对象）

```json
{
  "hypotheses_update": [
    {"id":"H3","claim":"heavy 的成功区比 light 偏后 2±1 mm","new":true},
    {"id":"H1","status_proposal":"supported","evidence":["F12","F15"]}
  ],
  "operator": "densify",
  "args": {"axis":"a","interval":[-12,-8],"object":"white_heavy"},
  "premises": ["F12"],
  "discriminates": ["H3"],
  "plan": {"cells":[{"offset_mm":-12,"reps":2}, ...]},
  "expected": "θ_heavy ∈ [-11,-9]；若在 [-9,-8] 内则 H3 被否定",
  "reason": "中文简述"
}
```

### 8.3 校验规则（程序执行）

1. `operator` 必须在 `applicable_operators` 中；`premises` 引用的事实 ID 必须存在，且在当前图上为真。
2. `args` 和 `plan` 满足第 7.2 节的参数约束，以及现有的范围、预算、零点、轻重平衡规则。
3. 新假设必须是可证伪的：`expected` 中要给出能否定它的观测结果。
4. 假设状态由程序根据事实更新；LLM 只能"提议"状态变化，程序在下一次图更新后确认或拒绝。
5. 不通过时返回具体的失败原因，最多修订 2 次；仍不通过则暂停。
6. LLM 不能编写事实、修改节点或标签、修改安全包络、验收标准或固定协议。
7. 当 LLM 提出的假设引用了图中尚不存在的变量（例如 `closure_floor`）时，程序允许把它作为 `probe_hidden_variable` 的 var，前提是它在规格中是可记录或可变化的量。这是 LLM 不可替代的作用，需要单独记录到 `decisions/`。

### 8.4 留档

每次决定保存：输入上下文的哈希、事实快照、LLM 原始输出、校验结果、执行后的图版本。保持现有 `decisions/` 的格式，新增字段。

---

## 9. 部署接口（修改 scheduler）

1. 恢复为**闭环**：修正 → 重新轻触 → β 判定 → 未满足且轮数 < 3 时再修正；不再使用 single-open 版本作为正式结果。
2. β 通过时输出符号状态：
   `{"symbol":"HandoffReady","object":"white_heavy","protocol":"...","cluster_id":"...","R":0,"graph_version":"...","latest_contact":{...}}`
   将其写入交接事件，交给 ForcePolicy 和主策略。
3. I 判定为 `Unrecoverable` 或 `Unknown` 时：不执行动作，输出符号状态，交给其他 skill 或中止。

---

## 10. 离线验收（第 1–4 阶段，必须全部完成后才允许上真机）

| 编号 | 检验 | 合格标准 |
|---|---|---|
| V1 | 重复一致性：零点重复节点之间 d_meas ≤ 1 的比例 | ≥ 90%（偏低则检查漂移和噪声模型） |
| V2 | 位置可辨性：在 d_meas 下用 kNN 预测 tcp_offset_a（按试验组留出） | MAE < 1.5 mm，且优于只用 20 维旧特征 |
| V3 | 结果一致性：kNN 邻居的 downstream_success 与自身一致的比例 | 不一致样本集中在 `boundary_gap` / `conflict` 附近 |
| V4 | 历史复现：0.336 / 0.330 两批数据合并建图（暂时去掉 protocol 门控） | 必须产生 `conflict`，并且 `probe_hidden_variable(closure_floor)` 出现在可用操作中 |
| V5 | 物体差异：白瓶轻、重分别计算 R = 0 区间 | 输出两者的中心差；报告是否与"重瓶需后移约 2 mm"的现场观察一致（无论是否一致都如实报告） |
| V6 | 绿瓶 E1/E2：用 A 建图，再用 E2 的实测转移校验运动学边 | 报告 confirmed / refuted 比例 |
| V7 | 标签对比：图生成的 I/β 标签 vs 现有人工 / 弱标签 | 给出混淆矩阵和不一致样本列表，不自动覆盖原标签 |
| V8 | 编码器消融：{20 维旧特征, 传感器位置 + 力 + 开度, + 触觉编码器, + 视觉} | 每种组合跑 V1–V3，给出表格 |
| V9 | 回放导演：在历史数据上按时间顺序重放事实，记录导演选择的操作 | 不调用真机；如果调用外部 LLM，需要 `--allow-agent` |

---

## 11. 实施阶段与产出

| 阶段 | 内容 | 产出 | 机器人 |
|---|---|---|---|
| 1 | 节点抽取：A/B 窗口 → `nodes.parquet`；传感器接触位置；物体接触位置；下游结果 | `outputs/state_graph/<ver>/nodes.parquet`、抽取报告 | 否 |
| 2 | 特征与噪声：触觉 / 视觉编码；噪声模型 | `features/*.npy`、`noise_model.json` | 否 |
| 3 | 建图、R、标签、聚类、符号 | `graph.json`、`labels.parquet`、`clusters.json`、`facts.json`、V1–V8 报告 | 否 |
| 4 | 操作库 + 导演协议修改 + 回放 | `bottle_scan/state_graph/*.py`、`symbolic_ops.py`、V9 报告 | 否 |
| 5 | 三部分用图标签重训；scheduler 改为闭环并输出符号状态 | 新模型（新 model_id），旧模型保留 | 否（离线测试） |
| 6 | 编码器继续预训练 + 力监督（可选） | 新编码器版本，V8 复测 | 否（GPU） |
| 7 | 真机：新 campaign，按导演和操作库运行 | 新 `outputs/` 目录 | **是，需操作员现场确认** |

建议的模块划分：

```
bottle_scan/state_graph/
  extract_nodes.py      # 阶段 1
  contact_geometry.py   # 传感器上 / 物体上的接触位置
  features.py           # 编码器调用（离线批处理 + 部署单帧）
  noise.py              # 噪声模型
  graph.py              # 门控、距离、边、R、运动学边校验
  labels.py             # I/β/π 标签与恢复目标
  clusters.py           # 聚类与符号落地
  facts.py              # 谓词
  operators.py          # 操作库与前提检查
  director_bridge.py    # 导演的输入输出与校验
  report.py             # V1–V9 报告
tests/test_state_graph_*.py
```

---

## 12. 必须遵守的约束

1. **不改旧数据**：所有新产出写到新目录；标签视图另存，引用原始记录的哈希。
2. **不泄露**：标签/分析类字段（tcp_offset、object_contact、planned_delta）和结果类字段，不得进入特征、距离或部署判定。单元测试中要检查这一点。
3. **协议不混用**：protocol_id 不同的节点不连相似边。V4 例外，仅用于诊断，报告中需注明。
4. **评估分离**：图标签由开发数据生成；结论只能来自封存的独立 B 和主任务实验，不能来自建图所用的数据。
5. **符号来源可追溯**：每条事实、每个簇符号都能追溯到节点 ID；每个 Agent 决定都能追溯到事实 ID。
6. **真机门槛**：第 7 阶段必须由操作员显式授权；沿用现有的运动确认、扣零、命令所有权和安全出口，不得放宽。
7. **如实报告**：任何检验不合格就照实记录并暂停，不调阈值去凑合格；不能把单次成功写成成功率。

---

## 13. 完成判据

- 离线：V1–V9 全部给出结果，V1、V2、V4 合格；报告中列出所有不合格项及原因。
- 方法：至少一次真机 campaign 中，Agent 的规则更新完全经由"事实 → 操作"完成，并且至少出现一次 `expand_range`、`densify` 或 `add_axis`。
- skill：独立 B 通过；部署时重瓶无需人工补偿位置（报告自然入口下的次数和比例）。
- 论文所需对照：自然入口下基线 vs 加 skill；同预算下均匀网格 / 纯 LLM / LLM + 图 + 符号；单次修正 vs 闭环修正。
