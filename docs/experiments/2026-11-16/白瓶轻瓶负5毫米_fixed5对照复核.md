# 轻白瓶 −5 mm 纠正对照复核

## 实际运行的控制器

Batch 3–5 来自 6f5090 上的 `outputs/state_graph/campaign_white_a_v4_fixed5_history`。名义起点为 −5 mm，轻白瓶，侧抓。请求、结果及 episode 事件均指向 `fixed_squeeze_5sdk_lift_floor0330`。成功条件为双指平均法向力在 5.0 ± 0.5 SDK 范围内稳定 0.5 秒，再提起 40 mm，保持 2 秒且未检测到掉落。闭合下限为 0.330。

| 批次和试次 | 首触实测 a / mm | 实验动作 | 实测动作回执 / mm | 与本批绑定起点的距离 | fixed5 端点 |
|---|---:|---|---:|---:|---|
| 3-0001 | −5.76 | 无 | — | 1.25 | 未达力，未提起 |
| 3-0002 | −4.99 | +10 mm | +10.59 | 1.33 | 达力、提起、保持 |
| 5-0001 | 无窗口 | 计划无动作 | — | 未知 | 定位阶段疑似接触停止，无端点评估 |
| 5-0002 | −5.29 | 无 | — | 0.71 | 未达力，未提起 |
| 5-0003 | −6.07 | +10 mm | +10.22 | 0.49 | 达力、提起、保持 |
| 5-0004 | −5.46 | +10 mm | +10.36 | 0.59 | 达力、提起、保持；之后发生高度走廊故障 |

距离按各批采集前的匹配规则计算，≤1 为匹配。Batch 3 两次均未匹配当时的绑定节点。Batch 5 的三个可评估起点均匹配该批绑定节点；三个起点的触觉及实测位置仍有差别。回执使用当时 V4 的口径，单列于实测首触位置之外。Batch 4 的另一个动作方向是 −5 mm，未纳入 +10 mm 对照。无动作试次后来在 fixed5 抱持力上未达标，不等同于已独立量出其接触偏移错误。

## 现有模型的离线验证

- 当前纯触觉／力模型使用 `campaign_white_a_v4_forcepolicy` 的 ForcePolicy L1 交接标签。48 个支持窗口中，可评估的终端结果为 18 次成功、1 次失败。每个窗口的相似邻域覆盖 17–19 个有结果的试次组，平滑后的就绪概率为 0.895–0.905，全部高于阈值 0.8。以上 fixed5 对照窗口的节点 ID 均不在该模型的训练支持集中。对 Batch 5 三个有效首触和两个纠正后窗口，当前图模型均输出 Ready。无法用 fixed5 端点判定 ForcePolicy 会不会交接成功。
- 读取历史 fixed5 的第 6 版模型，在未用于其训练的 Batch 5 上回放。无动作且未达力的 0002 首触被判 Ready；0003、0004 的首触均为 Unknown，实际按计划纠正后重新观测的窗口均为 Ready。这两次 +10 mm 为探索计划动作。
- 加入 Batch 5 后冻结的历史 fixed5 第 7 版进入开发 B。Batch 6 的 −5 mm 轻瓶 0003，真实事件记录了策略请求 +6.11 mm、执行回执 +6.43 mm、重触判 Ready、fixed5 实际提起保持成功。相同名义条件的 0001 为 Unknown、无消费者端点；0005 首触判 Ready、实际未达力。三个首触的离线回放再现了历史模型决定。
- 另外保留上述两个版本各自的训练窗口、分组、fixed5 目标和 0.8 阈值，仅保留触觉几何与六维力，重新拟合噪声、动力学、图及策略。第 6 版用 78 个窗口，整批留出 Batch 5；第 7 版用 83 个窗口，整批留出 Batch 6。每个留出试次的组均未出现在对应训练集中。第 6 版对 Batch 5 的 0002、0003、0004 首触分别输出 Unknown、Ready、Ready，均未发动作。第 7 版对 Batch 6 的 0003 输出 +8.85 mm，对 0001 输出 Unknown，对 0005 仍输出 Ready。+8.85 mm 是离线预测；该试次的真机端点只对应当时实际执行的动作。

## 证据位置

以下路径相对于 6f5090 上的 `/home/dm/DM_das/bottle_grasp_experiment`。

- Batch 3 和 5 的来源匹配为 `outputs/state_graph/campaign_white_a_v4_fixed5_history/reviews/batch_3_4566554c.json` 和 `reviews/batch_5_f4ad8ab2.json`（同一 campaign）。
- 原始端点与先后顺序可在该 campaign 的 `trials/batch_5_f4ad8ab2_000*/episodes/episode_*/events.jsonl` 中查看。重点事件为 `state_graph_window`、`state_graph_consumer_result` 和 `fault`。
- Batch 6 的自主动作保存在同一 campaign 的 `trials/batch_6_91578e00_0003/episodes/episode_*/events.jsonl`，决策事件为 `recovery_skill_decision`，模型保存在 `revisions/0007/models/bundle.joblib`。
- 当前 ForcePolicy 图的逐节点概率保存在 `outputs/local_contact_a_v1/graph/graph.json` 的 `probabilities` 内，目标为 `forcepolicy_handoff_l1_floor0330`。

本次复核未启动机器人或新批次。触觉／力的 fixed5 对比模型仅在内存中训练，部署模型和 campaign 状态均未修改。
