# 白瓶放置参考、r/h 动作与双下游验证

## 两个运行入口

单独采集轻触和实测转移，在仓库根目录执行 `bash scripts/start_state_graph_ah.sh`。默认创建独立的 `outputs/state_graph/campaign_white_rh_P3_transitions_v5`。已有目录作为只读数据来源。新目录启用水平 r、基座 h、放置参考和多步转移。该入口的硬件能力只包括轻触和移动。离线启动不需要 `--real`。在原始 a/h 方案上续跑使用 `--graph-version 4`。

带下游检验、开发 B 与独立 B 的入口为 `bash scripts/start_p3_forcepolicy_loop.sh --check`。现场准备完毕后，操作者可在 6f5090 的 ROS 容器中运行 `bash scripts/start_p3_forcepolicy_loop.sh --allow-agent --real`。在允许动作前核对打印的试次、瓶重、计划与路线上下边界。无真实执行记录时，应报告现场未验证。

## 坐标与标定

1. 趁瓶子直立、右爪空载，将手放到事先确认的标准轻触位置。第一次采集输入 `REFERENCE` 保存这一放置的独立双侧轻触见证。几何参考记录在本轮目录的 `placement_references/<组名>/`。瓶子被移动、更换或重新放置时，对“同一放置”回答 n，重新见证。固定夹法的身份与版本始终来自原触觉参考。
2. r 指向当前 TCP a 在水平面上的投影，h 指向基座 +Z。沿倾斜的 a 移动所得的旧回执会同时贡献 r 和 h。迁移保留旧计划坐标与回执，并按原测量基转换。缺少原测量基时，单独标记缺失。对于没有准备坐标的历史试次，仍可保存实际转移。
3. 每步只沿一轴，最多三步。每步记动作时实测位姿、动作后重触位姿和净位移。初始和累计位置均受各自包络约束，r 默认区间 −10 到 +10 mm，h 默认区间 −3 到 +45 mm。手的 a 几乎竖直时水平投影无法稳定定义，拒绝该方向。
4. 若要启用毫米接触几何，提供双指标定、安装方向、轴和本次安装的传感器版本。无标定的新 P3 会在冻结的配置中明确记录有效像素基线和不可用原因。传感器换位或更换盖板后应更新相应资产版本。

## 双下游与自动 L2

`fixed_force_lift` 沿用 SDK 法向力调节器，5.0 ± 0.5 SDK，稳定 0.5 s 后提起。`forcepolicy_lift` 使用身份可核验的 ForcePolicy 和原控制器。两者使用同一提起保持检验。条件及阈值集中在 `bottle_scan/state_graph/profiles/geometry_consumer.json`，现场参数应通过新实例的配置覆盖并冻结。不要修改已经产生验证结果的配置。

- L2 读取提起和保持时双侧接触、TCP 到位、两指切向负载增量、保持时两指质心的最大漂移。默认提起 40 mm，保持 2 s，双指合计负载误差上限 0.25 N，每指质心相对保持开始的漂移上限 2 px。
- 自动称重证据需要 `consumer.weight_newtons` 中对应轻瓶与重瓶实测重量，以及 `consumer.load_calibration.calibration_id`。推荐提供 `finger_forces_to_tcp_newtons`，形状 [2,3,3]，两个线性映射将两指的零点修正 SDK 力投影到 TCP 的牛顿力。程序按当前 TCP 姿态投影到基座竖直，去掉法向通道。仅标定了一个固定姿态时，可以提供 `upward_tangent_n_per_sdk`（[2,3]）、`calibration_rotation_base_tcp` 和 `orientation_tolerance_deg`。超出标定姿态会记录 L2 未知。单位标定与瓶重未提供时，双侧接触和 TCP 高度本身不足以给出 L2 成功。
- 默认按请求种子对约 10% 的结果抽查。自动结论与人工审计不符时同时保留两项，生成 label_conflict。回桌时实测卸载与高度不足以确认支撑，会请现场确认 TABLE 才松开。
- 固定力下游若开度持续落在下限容差内且法向力低于目标，记 `closure_floor_reached`。保持中接触丢失、质心滑移分别记 `contact_lost`、`slip_during_hold`。达到时间上限、仍未达力但未满足下限饱和证据，则记 `target_force_not_reached_other`。
- 同一次恢复后，两种消费者各从一次新鲜轻触开始，顺序由试次种子决定。配对须同时满足冻结噪声模型、图的相似半径，以及位置/朝向各自容差。无法判断时保留未配对标记；物体被推移、掉落或接触中断时记录跳过第二次的原因。最终看 `terminal_outcomes`、`consumer_order`、`pairing` 与每个节点各自的 post。
- 专家来源 `scripted_coordinate` 读取有见证的实测位置，选择绝对偏移较大的轴，在单步和包络许可内向标准位置修正；每步重新轻触。遥操作备选由指定专家现场输入单轴指令，经同一执行器校验和实测回执。记录来源、动作序列以及真正被测消费者的结局。模型估计或设定值无法充当该专家的毫米坐标。

合成回归测试在 6f5090 仓库运行 `.venv/bin/python -m pytest -q tests/test_v5_geometry_consumers.py`，包括固定候选按目标切换、两步转移、终端配对、滑移、接触丢失、饱和与审计冲突。
