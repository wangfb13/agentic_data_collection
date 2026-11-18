# 状态图 H8–H11 实施和离线验收

基线 `8040be6`。本次在 6f5090 的开发工作区实施。历史 P3 的 19 个已完成试次、试次结果摘要和原独立 B 计划均保留。新目录保存了独立的配置、模型和第一轮计划，实际新发起的物理尝试为 0。

## P3 第一轮

新目录为 `outputs/state_graph/P3_manual_lift_v5H811_20261118_release/`。来源为 `outputs/state_graph/campaign_white_rh_P3_manual_lift_v5/`。复验时重新读取全部 19 个试次，逐条核对结果摘要、恢复的结局及导入审计，与新快照一致。新图有 75 个窗口，冻结候选模型为 `be1a7951742dab95d10d564b`。

| 当前证据 | 值 | 对计划的影响 |
|---|---:|---|
| 合格单轴动作边 | r 为 1，h 为 4 | r 少于最少 3 条的要求，生成 `axis_undersampled(r_offset_mm)` |
| ForcePolicy Ready / Recoverable | 0 / 0 | 当前候选未满足开发 B 的 Ready 门槛 |
| 开发 B 计划起点中有支持的比例 | 0 / 24 | 当前候选也未达到 50% 的起点支持门槛 |
| 已使用的新预算 | 0 / 80 | 24 次为独立 B 预留，当前有 56 次可用于轮次 |

固定优先级导演在 `batch_1_727d83c2` 选择 `collect_transitions`。第 1、2 次分别用轻瓶、重瓶沿 r 正向 5 mm，第 3、4 次用轻瓶、重瓶沿 r 负向 5 mm。四次 h 分量均为 0。导演记录了 h 正方向的现场偏好和 r 的实测证据缺口。上述 B 接前检查是对当前候选的离线评估；进入实际开发 B 阶段时将用当轮训练后的候选重算。

## 修改及默认值

新增参数在 `bottle_scan/state_graph/profiles/h8_h11.json`，逐次实际值见相应新目录中的 `spec.json`。

| 修订 | 本次设置与行为 | 主要实现 |
|---|---|---|
| H8 预算 | `loop.budget_check=true`、`min_rounds=2`、`exploration_per_round=4`。P3 接受 `--extra-trials`。初始化按实际展开的开发 B 与独立 B 计划计算最低新预算，同时检查存在的物理派发上限。运行中也为后续最少轮数保留容量。预算保证有机会跑这些轮次；达到实验停止条件时仍可结束。 | `p3_loop.py`、`rounds.py` |
| H9 开发 B | `loop.dev_gate=true`、`dev_min_ready=1`、`dev_min_support_fraction=0.5`。每个计划起点按物体分组，在可信实测坐标中找最近节点。同距时须全部为 Ready/Recoverable 才算支持。跳过原因和候选模型写入轮报告与事实；采集建议使用证据不足的起始区域和相应的物体条件。 | `development_gate.py`、`rounds.py`、`operators.py`、`v5_operators.py`、`campaign.py` |
| H9 途中停止 | `acceptance.futility_stop=true`、`early_success_stop=false`。用最乐观的假设逐项检查主要比率：剩余所有试次都进入该指标的分母且全部成功。即便如此 Wilson 下界仍不可能通过时，取消尚待执行的请求，保留已花费次数和技术重试成本。可选提前通过要求主要指标的有效分母足够，且既有验收整体通过。独立 B 使用原冻结计划。 | `development_gate.py`、`campaign.py` |
| H10 Ready | `sinks.cost_based_threshold=true`，默认误放代价 8、漏放代价 1、`max_false_release_rate=null`。按训练组留出最小化加权错判数，平局取高门槛；可另行声明误放率上限。成本比是本次预设决策偏好，并非实测损失。报告对每版扫描 0.3–0.9 的候选门槛，按精确相同误放数比较，无对应操作点时留空。 | `ready_kernel.py`、`ready_comparison.py`、`compare_state_graph_versions.py` |
| H11 轴覆盖及先验 | `facts.axis_balance=true`。沿用 `facts.min_axis_trials=3` 与 H2 的实测边判据。补采操作要求同一缺样轴、正负两方向均覆盖；其他轴的分量为零。开启先验时记录冲突与决定，固定优先级模式优先选择可执行的缺样轴方案。 | `axis_resolution.py`、`v5_facts.py`、`v5_operators.py`、`v5_director.py`、`director_bridge.py` |
| H11 导入 | `import.consumer_audit=true`。导入回执按来源试次和下游记录已导入、未执行、明确跳过、执行但无法匹配及原因；不以计划顺序充当执行证据。遇到无法匹配的记录会警告，并将试次 ID、下游和原因写入新 `state.json`。 | `consumer_audit.py`、`p3_import.py`、`p3_loop.py`、`campaign.py` |

上一版预算 60 的新建入口已离线实测，返回最低 80，即 24 + 2 × (24 + 4)。原有已初始化的目录沿用自身预算与配置。在新目录第一次给出预算后，续跑可省略该参数，另给不一致的数值会被提示。

### 源试次下游审计

| 下游 | 已导入 | 未执行 | 明确跳过 | 执行但无法匹配 |
|---|---:|---:|---:|---:|
| ForcePolicy | 7 | 12 | 0 | 0 |
| 固定力 | 5 | 12 | 2 | 0 |

两下游未执行的 12 次均为同一组起点，11 次模型拒绝、1 次接触丢失。另有 2 次固定力排在第二，第一下游执行后记录为物体或接触状态变化，因而跳过第二下游。这些次数不能计成下游提起失败。

## 白瓶、绿瓶的 Ready 操作点

主表使用历史 E1/E2 0.336 夹紧下游。所有版本外层共用相同的五折试次组。H10 在各训练折选定部署门槛，外层只评分。表中各格为正确放行 / 错误放行。

| 数据 | v4 | 原 v5 | H1–H7 | H8–H11 |
|---|---:|---:|---:|---:|
| 白瓶 | 2 / 0 | 0 / 0 | 2 / 0 | 2 / 0 |
| 绿瓶 | 234 / 3 | 42 / 2 | 201 / 3 | 227 / 3 |

在绿瓶统一扫描门槛且恰好误放 3 次时，v4 的最好操作点为 234 次正确放行，H1–H7 和 H8–H11 都为 227 次。H10 的变化来自训练折选点；两版的固定门槛操作曲线一致。原 v5 在本次候选门槛上没有恰好误放 3 次的点。白瓶的自选点无变化。历史表仅评价特定控制器的终端放行。

共同误放按各版训练折自选工作点求交集。白瓶 0 组。绿瓶共同误放两组，均来自 `e1e2_A_formal01`，放置编号缺失，以下坐标来自旧试次实测 TCP 相对本会话 T0 的偏移。

| 试次 | 质量 | a (mm) | l (mm) |
|---|---|---:|---:|
| 00017 | 轻 | −10.33 | +1.05 |
| 00067 | 重 | −10.23 | +1.57 |

以有相应属性记录的留出失败组作为对照，用单侧 Fisher 精确检验，物体内对检验的属性和水平做 Bonferroni 校正，显著性 0.05。属性包括质量、会话、放置、时间段及有记录的坐标箱（2.5 mm）。本次没有产生显著的 `hidden_variable_in_ready` 事实。完整操作曲线、共同错误原始记录及计数见 `docs/experiments/state_graph_v5_H811_comparison.md` 和 `outputs/state_graph/v5_history_comparison_H811_release/`。

## 离线复核

```bash
cd /home/dm/DM_das/bottle_grasp_experiment
bash scripts/start_p3_forcepolicy_loop.sh \
  --source "$PWD/outputs/state_graph/campaign_white_rh_P3_manual_lift_v5" \
  --out "$PWD/outputs/state_graph/P3_manual_lift_v5H811_20261118_release" \
  --overrides configs/p3_v5_manual_lift.json --director-mode fixed_priority --check

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/compare_state_graph_versions.py \
  --out "outputs/state_graph/H811_recheck_$(date +%Y%m%d_%H%M%S)" \
  --report /tmp/H811_recheck.md
```

独立复查 P3 候选的接前判据保存在 `outputs/diagnostics/state_graph_H811_release_20261118/P3_development_screen_before_collection.json`。上面两条命令均为离线读数或重建。

## 测试

依 H8、H9、H10、H11 顺序的全量合成/回归快照分别有 1636、1638、1640、1642 项通过，每个快照有 7 项跳过。最终实仓全量回归 1648 项通过、5 项跳过、1 条原有测试依赖告警。各次另有 673 个子测试通过。合成用例覆盖两轮预算、零能力跳过 B、最乐观情形仍不达标时及时停止、未用次数和历史派发不变、代价比及可选上限、曲线和属性浓集、轴覆盖事实、双向动作校验、导入四类结局与告警。全套证据保存在 `outputs/diagnostics/state_graph_H811_release_20261118/`。