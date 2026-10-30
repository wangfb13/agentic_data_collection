# 第一版离线验收（2026-10-26）

本文为首日历史记录。当前公开 ROS 接入、右 X5 DD 服务切换、在线诊断与 176 项测试结果见 [2026-10-27 验收记录](PUBLIC_ROS_VALIDATION_20261027.md)；实际操作按 [完整指南](ADVANCED.md)。

执行位置：`6f5090:/home/dm/DM_das/bottle_grasp_experiment`。使用项目独立 `.venv`；未启动机器人/夹爪/传感器程序，未修改原驱动、力控、policy 或系统安全参数。

## 已生成的结果

| 产物 | 结果 |
|---|---|
| `outputs/dry_contact/` | 完整合成计划 738 条：694 可行、44 被几何/路径检查拒绝 |
| `outputs/dry_visual/` | 合成视觉候选 400 个；此演示几何下全部可行 |
| `outputs/acceptance_demo/` | 8 次接触试验，覆盖 normal/no_contact/stall/force_timeout/dropped/slip；轻/重区块沿用固定参考 |
| `outputs/acceptance_demo/analysis/` | 14 条记录汇总（8 接触＋6 视觉），22 条描述子、1751 条位姿误差；分析 warnings 为空 |
| `outputs/acceptance_demo/acceptance.json` | 关键结果断言、两路参考不变、视觉无闭合指令、标签正确的验收记录 |
| `outputs/commissioning_demo/` | 三点 CLI 单步模拟通过；越盖点标记 no_bilateral_contact，未进入 LIFT；随后视觉 CLI 3 点通过 |
| `outputs/validation/resume.log` | 同一三点计划续跑没有重复完成记录 |
| `outputs/validation/unit_tests.log` | 66 项单元与回归测试通过，包含几何、单位、原始存储、模板、状态机、安全异常、磁盘故障与 CLI 门禁 |
| `requirements.lock.txt` | 5090 实际隔离依赖版本；`pip check` 通过 |

这些是**合成软件验收数据**，不是机器人实验结果。完整演示计划估计接触采集约 25.1 小时、338.0 GiB，视觉约 12.0 分钟、2.69 GiB；这些数字使用示例几何/速度/帧率和 JPEG 大小，排除人工等待和复位，不能替代实测参数后的估算。HDF5 估计未假定压缩收益。

## 关键安全覆盖

- 所有 mm/m 换算集中在一个函数；T0 右乘平移/旋转、基座竖直提放已测试。
- 目标/路径/桌面/工作空间、圆柱和旋转长方体、连续夹爪闭合宽度扫掠与携带瓶体检查均有回归；不会把目标截断到可行边界。
- 实机 TODO、未验证硬件能力、模拟/真实混用、未完成三点验证、过期数据、无效原始 DD、计划变化会被拒绝。
- LIFT 前设置持久恢复锁；空中异常停止机械臂并保持夹爪；开夹/启动下一点被禁止。
- 提起前过力或触觉异常时独立紧急释放；失去夹爪反馈不能继续后撤；后撤失败终止序列。
- 固定两腕模板必须两路均通过；超阈、低置信、歧义、缺图、纹理不足均不能通过。复位和换瓶不修改初始参考。
- 原始 DD 的 dtype、shape、独立时间戳和源序号保存；HDF5 未提交尾部不被分析读取。
- 人工 x 和安全剔除项保留数据，但不污染接触统计分母；缺少物理标定时不伪造 mm/N/CoP。

## 明天还需要完成

按 `PLACEHOLDERS.md` 记录实际位姿、几何、轴、传感器与参数。`hardware_bridge_template.py` 仍明确拒绝实机；停止、真实直线限速、mm/counts、N 标定、完整 raw SDK 数据流与新鲜双腕图像接口需现场验证后绑定。

第一版仅支持 `fixed` 力目标实验。`forcepolicy` 尚未接入，采集启动前即拒绝该模式；不会加载学习模型。
