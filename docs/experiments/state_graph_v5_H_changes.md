# 状态图 v5-H 实施与验收

2026-11-18，在 6f5090 的 `/home/dm/DM_das/bottle_grasp_experiment` 基于 `ea3557b` 实施。历史白瓶的 Ready 已恢复。绿瓶留出正确放行大幅回升，尚差 10 次才达到预定历史验收线。当前 P3 的 ForcePolicy 仍无可放行的 Ready 节点。此次均为离线结果。

## 修改、参数与验收

新值在 `bottle_scan/state_graph/profiles/h_revision.json`。H1、H2、H7、H3、H4、H5 的开关在新建 v5 配置中默认开启，关闭相应开关走原算法。历史 v5 对照用 `configure(..., revision_h=False)` 复现旧配置。H6 在新版实机入口生效，有单独的显式退化选项。各次运行的有效合并参数写入该次的 `spec.json`。

| 项 | 实现与可调参数 | 验收证据 |
|---|---|---|
| H1 Ready | `sinks.kernel_ready`。每个终端独立试次组贡献一个高斯核权重。带宽候选 0.25、0.5、1、2，按组留出对数似然选带宽。后验取 5% 下分位，候选门槛为 0.3 到 0.9，间隔 0.1。以实际失败组为分母，组内校准误放上限 0.15，在可行门槛中优先保留正确放行。无可支持门槛时产生 `ready_threshold_unattainable`。 | 合成成功区、失败区、组隔离和不可达门槛均有测试。建图、留出评分与运行时接入同一套训练校准。每节点保存有效成功数、有效试次数与后验下界。所有留出预测可逐条核对。 |
| H2 轴尺度 | `graph.observable_axes_only`。动作距离中位数至少为重复距离中位数的 1.5 倍。接纳单轴动作的实测正交残差上限同时满足 1 mm 与主位移的 15%。沿用原有最少 3 条边、最小幅度档相对容差 0.2。无可用尺度时回退到归一化距离 1。 | 图和谓词复用 `axis_observable`，分清证据不足、可观测但尚无尺度、低于噪声。合成不可观测轴不会压缩其他轴的半径。 |
| H7 放置连续性 | `geometry.automatic_reference_check`。在参考 TCP 做新的轻触，以冻结的噪声模型与 B1 的相似半径核对；参考位姿复核容差 2 mm。无法比较或超半径时要求重新见证。可在无毫米标签条件下采集。 | 核验窗口、距离、模型、决定和操作员回答分开记录；轮报告分开统计坐标覆盖、核验通过和重新见证。已有历史试次没有新核验，不能据此补认放置连续。 |
| H3 度量头 | `observation.metric_per_axis`。r、h 分别拟合、分别按组留出。各轴均用 1.5 mm MAE 上限、0.25 mm 最低动作量、相对空模型至少 5% 改善。继承 4 折、最少 3 组、岭系数 0.01、位移/重复项权重 1/0.5。 | 合成 r 有信号而 h 无关时只采用 r。尺度与度量距离只用通过的轴。P3 两轴训练证据都不足，未启用度量头。 |
| H4 能力边界 | `facts.rate_based_frontier`。从零起的每个连续箱都须达到成功率下界 0.8。箱宽复用 C5 分辨率，P3 基础阈值 0.5 mm、噪声倍率 2。95% Wilson 下界、按试次组自助重采样 200 次、种子 0。比较两角色要求边界区间重叠和相邻箱内各自满足 5 次。物理终止要求边界箱和外邻箱跨轮累计至少 5 次失败，物理类占比至少 0.9。 | 保留偶然的最远成功；空箱终止已验证范围的延伸。单点成功、单轮几次物理失败都不足以终止。离线合成用例覆盖对应情况。 |
| H5 B 验收 | `acceptance.lower_bound_gates`。默认主要指标为 `recovery_success_rate`，95% Wilson 下界须达到 0.8，其余门槛保持点估计。点估计通过但主要下界不足会记录 `development_point_pass_only`。 | 10/12 无法通过。在全部成功的理想情况下，主要指标分母也至少需 16。新 P3 的开发与独立 B 各预设 24 次，原计划重复 2 遍（预期 20 次需要恢复、4 次直接就绪）。能否达到所需实际分母取决于独立的起始状态证据。 |
| H6 标定检查 | 启用 `auto_l2` 需重量与载荷标定，申请 `calibrated_rh` 需有效双指指面标定。实机入口缺项会列出原因；`--allow-degraded` 才显式放行。离线和退化运行都会记录缺项。 | 手工标签 P3 用 `auto_l2=false`、`tac_version=legacy_pixel`，这两类标定均不需要。合成测试覆盖缺项阻止、手工配置、显式退化和记录。未在本次接通机器人。 |

主要实现位于 `bottle_scan/state_graph/` 下的 `ready_kernel.py`、`axis_resolution.py`、`axis_metric.py`、`frontier_rates.py`、`rate_gates.py` 和 `calibrations.py`，已接入现有建图、事实、操作、轮次、技能推理和报告。放置核验与 P3 入口在 `skills/contact_position_a/reference_check.py`、`p3_prepare.py`、`p3_trial.py`、`p3_trial_v5.py` 和 `p3_loop.py`。导入与结局匹配在 `p3_import.py` 和 `ah_transition_v5.py`。已结束的试次和原冻结 B 计划沿用其原始记录。

## 历史白瓶、绿瓶

五折按试次组拆分。三版使用同一批划分和同一下游 `handoff_e1e2_squeeze_floor0336`。表中“放行”是留出试次的 Ready 门预测；图节点数来自全量建图。两者分母不同。

| 物体 | 留出窗口（实际成功 / 失败） | v4 正确 / 错误放行 | v5 正确 / 错误放行 | v5-H 正确 / 错误放行 | v5-H 全图 Ready / Recoverable |
|---|---:|---:|---:|---:|---:|
| 白瓶 | 11 / 12 | 2 / 0 | 0 / 0 | 2 / 0 | 8 / 0 |
| 绿瓶 | 235 / 17 | 234 / 3 | 42 / 2 | 201 / 3 | 237 / 71 |

绿瓶预设正确放行容差为 v4 的 10%，即至少 211 次，误放最多 3 次。本次两项分别为 201 和 3，整体验收未通过。外层留出误放率为 3/17，约 17.6%；0.15 是训练折内校准约束。逐折检查表明，5 个训练折都选了带宽 0.5。其中 3 折选门槛 0.7，留出漏放 32 个实际成功、误放 0 个；另外 2 折选门槛 0.4，漏放 2 个实际成功、误放 3 个。这给出了覆盖率缺口所在，也说明直接全局降低门槛会面临什么取舍。未使用留出真值回调门槛。白瓶全图选择带宽 0.5、门槛 0.3，绿瓶分别为 0.5、0.8。绿瓶 71 个 Recoverable 中 57 个由自身实测转移支持，14 个由邻域借用支持。逐折结果、簇符号、参数及所用尺度见同目录 `state_graph_v5_comparison.md` 和 `outputs/state_graph/v5_history_comparison_H/`。这组数据无法评估当前 P3 的 ForcePolicy 提起成功率。

## 白瓶 P3 快照与第一轮

新目录为 `outputs/state_graph/P3_manual_lift_v5H_release_20261118/`，来源为 `outputs/state_graph/campaign_white_rh_P3_manual_lift_v5/`，快照时原状态为 `deferred_review`。导入回执保存原始试次与结果摘要。仅采用已提交且可用于开发的记录；独立 B 和未完成试次均排除。

- 当前来源有 19 个已提交试次，4 个直接基线、12 个开发 B、3 个探索。14 个有可信初始毫米坐标，覆盖 73.7%。新图含 75 个窗口（含历史种子），试次覆盖率以 19 个试次计算。
- 按原记录的消费者身份、版本及执行顺序恢复了 7 个 ForcePolicy 结局。原试次文件保留原样。新图出现 7 条实测边、496 条相似边，无可放行终点。ForcePolicy 和固定力两种下游各有 0 个 Ready、0 个 Recoverable、75 个 Unsupported。
- ForcePolicy 有 7 个可评估终端组，选择带宽 0.25；固定力有 5 个，选择带宽 2。两者都没有找到有证据支持的放行门槛。不能从这 75 个窗口推断策略已经具备恢复能力。
- r 有 1 条合格单轴动作，证据不足。h 有 4 条，动作距离中位数 2.408，重复中位数 0.366（30 对）。h 达到可观测条件，但最小幅度档不足 3 条，尚未取得尺度；几何比较使用回退半径。两轴的监督度量头均未通过数据量要求。
- 候选模型 `0d4c27fe33310fae80b7285f`。固定优先级导演在 `development_class_missing(forcepolicy_lift)` 和 `phase_plan(forcepolicy_lift)` 下选择 `collect_outcomes`。首批 `batch_1_56ee1a5e` 只安排 2 个直接基线，轻、重白瓶各一次，闭合下限 0.330，无位置纠正，目标为 ForcePolicy 提起。已生成请求，实机执行次数为 0。
- 可核查 `import_receipt.json`、`revisions/bootstrap/graph.json`、`revisions/bootstrap/noise_model.json`、`rounds/round_0001/plan.json`、`state.json`。未采集新的放置数据，H7 核验率和尚未执行的 P3 独立 B 成功率均不可估计。

## 离线复现

从 6f5090 仓库根目录运行；使用新的输出名生成另一份快照。后续来源目录如果增加试次，新的快照会不同。已交付的 `import_receipt.json` 和逐试次摘要确定本次数据。

```bash
cd /home/dm/DM_das/bottle_grasp_experiment
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/compare_state_graph_versions.py \
  --out "outputs/state_graph/v5_history_H_check_$(date +%Y%m%d_%H%M%S)" \
  --report /tmp/state_graph_v5_history_check.md

bash scripts/start_p3_forcepolicy_loop.sh \
  --source "$PWD/outputs/state_graph/campaign_white_rh_P3_manual_lift_v5" \
  --overrides configs/p3_v5_manual_lift.json \
  --director-mode fixed_priority \
  --out "$PWD/outputs/state_graph/P3_manual_lift_v5H_check_$(date +%Y%m%d_%H%M%S)"
```

## 测试记录

依照 H1 → H2 → H7 → H3 → H4 → H5 → H6 的累积快照，各自运行全套 pytest，退出码均为 0。下表为各快照的最终复验记录。

| 快照 | 通过 | 跳过 |
|---|---:|---:|
| H1 | 1625 | 7 |
| H2 | 1626 | 7 |
| H7 | 1629 | 7 |
| H3 | 1630 | 7 |
| H4 | 1631 | 7 |
| H5 | 1632 | 7 |
| H6 | 1633 | 7 |
| 最终 6f5090 实仓 | 1637 | 5 |

各次均另通过了 673 个子测试。最终全量测试有 1 条 warning。另在有 Torch 的容器使用 CPU 跑了 31 项相关测试，全部通过。逐项新增合成用例在 `tests/test_graph_h_*.py`，覆盖核门槛、不可观测轴、放置核验、单轴度量、连续能力边界、B 验收、标定入口与历史试次导入。离线测试不能测量真机提起成功率或实际放回漂移。

每项对应的完整日志和退出码存于 `outputs/diagnostics/state_graph_v5H_release_20261118/`。
