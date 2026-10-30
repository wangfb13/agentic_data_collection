# 发布者检查修正：系统关节输出误报

`doctor` 是只读诊断，不需要先运行另一个发布者。正式实验会在连接后发布 `/wbc_goal_pose` 和实验侧夹爪目标。先前“拒绝”来自错误地把系统关节输出和外部控制输入一并当作竞争控制源。

## 只读证据

[实际 ROS 图和节点接口快照](../outputs/ownership_fix_20261027/live_graph_before.json) 显示：

| 方向 | 话题 | 节点关系 |
|---|---|---|
| TCP 输入 | `/wbc_goal_pose` | `/daimon_hw_interface_node` 订阅 |
| 关节输入 | `/joint_control` | `/daimon_hw_interface_node` 订阅 |
| 夹爪输入 | `/left_gripper_command`、`/right_gripper_command` | `/daimon_hw_interface_node` 订阅 |
| 系统输出 | `/joint_commands` | `/daimon_hw_interface_node` 发布，`/wbc_state_publisher` 订阅 |
| 状态输出 | `/joint_states`、`/left_gripper_state`、`/right_gripper_state` | `/daimon_hw_interface_node` 发布 |

硬件节点不订阅 `/joint_commands`。此次证据来自 master 的 getSystemState、lookupNode 和节点的 getPublications/getSubscriptions，只读，无服务触发或命令发布。没有读取或修改机器人主机底层运控源码，也没有验证其动力学行为。

## 修正范围

公开 ROS 配置显式设置 `hardware.public_ros.command_ownership.infrastructure_output_policy: daimon_public_ros_v1`。每次检查仍读取同一份实时 ROS 图；只有上表全部必要关系、硬件节点不向这些命令输入发布、`/joint_commands` 唯一消费者为 `/wbc_state_publisher` 同时满足，才把精确的 `/joint_commands` + `/daimon_hw_interface_node` 标记为已核对系统输出。

原始发布者列表保留，所有未知发布者仍拒绝；系统节点若出现在真正命令输入的发布者列表中也拒绝。关系缺失、出现未知消费者、未知策略值、查询失败，均不得产生放行。新增 `/joint_control` 为必查输入，不删旧话题。通用模板、模拟配置和省略策略时仍为 strict。

这是纠正已经观测到的话题方向与节点角色，不是允许多个上层控制器竞争。底层节点应保持运行。ROS 图检查仍只是当时的注册关系，不是身份认证或分布式控制权锁。

## 验证与使用

新增 11 项回归，发布者模块 30 项测试通过。针对实际快照的离线重放已通过：[结果](../outputs/ownership_fix_20261027/replayed_live_check.json)、[终端输出](../outputs/ownership_fix_20261027/replayed_live_check.txt)。完整本机测试 **404 项全部通过**：[日志](../outputs/ownership_fix_20261027/unit_tests.log)。三点 dry-run 仍为 3 候选、2 接受、1 几何拒绝：[输出](../outputs/ownership_fix_20261027/dry_contact.log)。完整模拟完成 8 个接触案例、6 个视觉点及分析：[acceptance](../outputs/ownership_fix_20261027/smoke/acceptance.json)、[日志](../outputs/ownership_fix_20261027/smoke.log)。5090 容器中发布者与配置专项 38 项通过：[日志](../outputs/ownership_fix_20261027/remote_tests.log)。

同步后在原容器终端重新运行：

```bash
bash run_on_5090.sh doctor --config config.ros.yaml
```

它应把这一项显示为“已核对系统输出”；没有外部输入发布者且角色匹配时，发布者检查通过。其它传感器连通性、参数 TODO、零点、物理安全检查及真实验收仍分别检查。本修正未执行机器人或夹爪运动。

配置和源码绑定已改变；旧 session 不自动继承新验收，不能手改旧快照或通过标志绕过绑定。现场正式试验应使用当前配置准备的新 session，按指南取得真实验收。

## 部署后的实际只读复查

修复已同步至 5090，修改前备份位于 `outputs/v2_deploy_backup/20261027_195706`。用原容器启动器运行 doctor，实际 `checks.command_ownership.allowed=true`，该系统输出已正确显示，四个真正控制输入当前没有外部发布者。[实际报告](../outputs/ownership_fix_20261027/doctor_after.json)、[终端日志](../outputs/ownership_fix_20261027/doctor_after.log)。

**整份 doctor 尚未通过**：`dd: no valid processed DD for both fingers`，相机联合源检查也报告 `dd_left, dd_right` 缺失/过期。发布者误报已解决，但当前 DD 连通性另需处理；没有将这一传感器问题忽略或伪装为准备就绪。本次没有重启 X5 或底层服务，没有执行机器人、夹爪运动。

后续更新：两指 DD 缺失已定位并恢复，完整 doctor 现已通过，详见 [DD 服务恢复记录](DD_RECOVERY_20261027.md)。前文的失败状态为当时的诊断证据，保留以便追溯。
