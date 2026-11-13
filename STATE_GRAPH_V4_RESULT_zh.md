# V4 代码接通与离线验证结果

2026-11-13，6f5090：`/home/dm/DM_das/bottle_grasp_experiment`。

本次完成当前白瓶 a 轴实例的程序闭环。没有启动机器人、没有发运动命令、没有运行真机 A/B、没有发送新的外部 Agent 请求，也没有实施消融实验。合成闭环测试使用显式测试 director 和合成消费者，不能作为采集效率或实际恢复成功率的证据。

## 已实现

1. 相似边退出恢复路径；只有具有实测支持的 Ready 节点连零代价汇点。直接交接概率/风险与恢复代价分别保存。
2. 调用 I、退出 β、V/p、按实测动作训练的 π 同步更新。实际推理使用分类结果及对应状态支持；策略权重均值归一化。路径保留有类型动作及顺序。
3. 新 campaign 控制器接上分析、事实、Agent、计划、采集、结果回收、训练、开发 B、冻结、独立 B 与达标停止。
4. B 结果按实际使用的模型归属；小批开发 B 累积时固定候选模型。失败指标回到 Agent；冻结保存已测试模型及独立 B 计划。
5. 中断恢复使用试验台账与实际结果文件；不会自动重放执行状态不明的动作。未知的物理结果不写成成功或失败标签。
6. V4 的实际 backend 加载新图模型适配器，仍复用原运动/接触执行器和 ForcePolicy 交接。新图入口允许原物理配置范围内的 2.5 mm 等补采点；旧入口行为保持。
7. 在线和离线复用同一观测特征；新 session 的重复试验按 campaign/T0/传感器配置正确归组。可选触觉编码器可进入动力学输入，压缩仅在训练集拟合。
8. 假设记录提出时的证据，仅用后续新证据更新；区域标识保留延续关系。诊断补充留出重复噪声检查、动力学物理误差基线、概率和价值基线。

## 测试

| 范围 | 结果 |
|---|---|
| 仓库全量回归 | 1,442 passed、4 skipped、668 subtests passed；387.35 秒 |
| 后续最终状态图相关回归 | 40 passed、4 skipped |
| 最后计划字段/闭环相关复测 | 18 passed |
| Torch 环境下触觉编码器测试 | 8 passed，覆盖主环境跳过的 Torch 用例 |
| 现有 V4 scheduler/backend 假硬件回归 | 82 passed |
| 新图模型接实际 V4 类的假硬件验证 | 2 passed：实际纠正回执→新窗口→退出/保留交接控制权；图入口小偏移与旧入口兼容 |
| 当前检查点加载与前向 | DINOv2 输出 384 维；触觉批量输出 10×768，均有限；合成数组、CPU，仅测试代码与权重加载 |
| 白瓶历史诊断 | 按完整试验组完成三折验证，噪声/PCA 仅训练集拟合 |
| 新启动脚本默认模式 | `awaiting_director`，0 次试验、0 次动作，不启动硬件 |

全量回归完成后，B 小批累积和独立计划冻结等收尾修改又执行了对应模块回归；上述数字分别列出，没有冒称为同一次完整测试。现有全量测试中的一条警告来自认证单元测试的短测试密钥。

## 历史数据重建

只读导入原来的 440 个节点（绿瓶 389、白瓶 51），新结果写入：

`outputs/state_graph/v4_implementation_20261113/historical_rebuild/`

- 所有后置条件中，“Recoverable 却给零修正”的节点数均为 **0**；Ready 的非零恢复代价数也为 **0**。
- 绿瓶旧 SQUEEZE 0.336 条件：358 Ready、28 Recoverable、3 Unsupported；仍有 114 条实测转移。
- 白瓶旧 SQUEEZE 0.336 条件：3 Ready、48 Unsupported，没有实测转移。过去由相似边造成的伪恢复标签不再保留。
- 白瓶旧 SQUEEZE 0.330 条件：39 Ready、12 Unsupported。
- 此次导入的数据中，当前 ForcePolicy 后置条件仍没有有效实测终态。因此没有把旧 SQUEEZE 结果改写为 ForcePolicy 成功，当前白瓶新图模型也没有被冒称为已训练验收。

下一轮需要实际记录：首次接触→a 轴动作回执→重新接触→同一个 ForcePolicy 的达力与提起保持结果，并覆盖直接可交接和确需恢复的情况。新的 Agent 闭环会根据这些缺口生成计划；不是固定要求重做原来的 22 次。

## 文件与现场入口

- 使用说明：`STATE_GRAPH_V4_GUIDE_zh.md`
- 一键入口：`scripts/start_state_graph_loop_v4.sh`
- 已准备但未启动的 campaign：`outputs/state_graph/campaign_white_a_v4`
- V4 安装器：`skills/contact_position_a/install_runtime.py`
- 被修改的 V4 外部文件：`runtime/bottle_scan/contact_scheduler.py` 和 `contact_skill_backend.py`，位于原 `contact_scheduler_20261104_experiment_b_v4_single_open_20261110` 目录。原文件备份在本仓库 `outputs/state_graph/v4_install_backups/`。

当前真实动作适配器覆盖 a 轴平移/重新轻触/ForcePolicy 交接。其他任务的微提、二维放置、擦拭动作仍需各自的实例执行器；框架不会把未实现的能力当作已经能够执行。
