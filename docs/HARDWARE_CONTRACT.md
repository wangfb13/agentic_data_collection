# 严格硬件接口合同（保留模板）

本文件保留最初 **strict** 模板的硬件能力要求。2026-10-27 操作员明确选择公开 ROS 命令和现有处理后 DD，现通过 `config.ros.yaml` 的 `public_ros` profile 接入 `bridge_ros_public.py`；操作方法见 [完整操作指南](ADVANCED.md)。该 profile 采用软件插值、软件位置保持和软件限力，不声明满足下文的原生停止、硬件看门狗、原始 DD 或全链路无丢帧要求。缺失电流保留 NaN，开度需实测映射、速度为接收位置差分；来源信息随数据保存。严格模板仍拒绝用这份软件桥替代其原能力合同。

`hardware_bridge_template.py` 的能力位全部为 false。这里不是新驱动，也不把旧接口的名字直接套成已验证的能力。补齐实机接口后，将 `hardware.bridge_module` 指向你核验过的模块；`create(cfg)` 只能建立无运动副作用的连接。现有力控制器和策略代码保持原样，由薄适配层调用。

静态审计见 `interfaces_arm.md`、`interfaces_force.md`、`interfaces_sensors.md`。目前不能确认真实停止、笛卡尔线段/限速、物理开度、N 标定和未处理 SDK DD 的完整流。这些接口不验证，模板就保持不可实机启动。

## 位置与动作

- 所有实验位姿为 `base_link` 下 TCP 的 4×4 矩阵，平移 mm，旋转正交矩阵。唯一 m/mm 边界函数为 `geometry.interface_units`。ROS quaternion 的 xyzw 顺序已从代码确认；TCP 外参和 m 单位需实物/接口核验。
- `move_linear(T_mm, speed_mm_s, angular_speed_deg_s)`：**非阻塞**。执行必须是检查过的直线平移和最短旋转插值；保证速度/角速度上限、受控停止与实时跟踪。仅发布一个目标点不满足合同。不得修改既有控制器/碰撞参数。
- `motion_done()`：非阻塞返回到位状态；运行器另检查实测位置/旋转误差。
- `stop_arm()`：核验过的机械臂停止，不能用“再发一次当前位姿”替代，不能包含夹爪释放。控制通信失联时需要独立 watchdog 停臂并保持夹爪。
- `command_open(width_mm, speed_mm_s)` / `command_close(lower_width_mm, speed_mm_s)`：非阻塞、物理单位，遵守 w_min/F_max。`set_gripper_min_width(width_mm)`：在闭合/力控内周期强制本次几何闭合下界；仅20 Hz Python监督不足以实现这一能力。
- `set_force_target(per_finger_N)`：调用现有已验证控制器；明确这里是每指目标 N。适配旧 mean 或 sum 控制器时必须统一聚合/符号/偏置/单位，不直接混用数值。任何一指超过 F_max 都要在底层制动闭合。
- `exit_force_mode()`：停止力控回路继续下发，接管位置控制且不能意外松开；`hold_gripper()`：停止开合并保持当前状态，空中异常使用。原实现仅停止发新目标不等于电机已停止，必须验证。
- `close()`：无动作清理；不能松开夹爪或把末次目标改成默认值。

## 状态与数据

`simulated=False`；`monotonic_ns()` 直接返回采集主机 `time.monotonic_ns()`；`set_stage(name)` 只修改数据标签；`set_trial(item, scenario='normal')` 仅记录试验元数据，实机不得注入模拟故障。

`snapshot()` 非阻塞、线程安全，返回最新完整健康状态：

```text
timestamp_ns: host monotonic int
tcp: (4,4) mm
width_mm, velocity_mm_s, current_A: calibrated numeric
normal_N: (2,) left/right, calibrated compressive normal force
wrench: (2,6) Fx,Fy,Fz,Mx,My,Mz, calibrated force/explicit torque units
dd: [left_raw_DD, right_raw_DD], exact configured native shape/dtype
force_target_N, controller_state, fault
```

若任何必要源（TCP、夹爪、两指力、两指DD）超过 `recording.source_max_age_s`，`fault` 必须报告失鲜；不能只给最近一次整体打包时间来掩盖某一路断流。记录独立源帧号/设备时间语义。

独立采集线程持续收集每个源帧；`drain_samples()` 非阻塞返回 `backend.Sample` 列表。每帧在接收/获取时打本机单调时间，保留设备时间和 source_sequence，不能在控制周期重复“最近帧”伪装新数据，不能用零场/降采样/阈值化替代 DD。任何队列溢出、帧序号缺失、磁盘写失败均报告 fault，不能静默漏帧。

必须流名：`dd_left`, `dd_right`, `wrench_left`, `wrench_right`, `normal_force_N` (shape 2), `tcp_measured`, `tcp_command`, `gripper` ([mm, mm/s, A]), `gripper_command`, `force_target`, `controller_state`, `camera/left_wrist`, `camera/right_wrist`。需要保留未标定原始六维力时另加 `wrench_raw_left/right`，不得只留下单位换算后的值。SDK RGB/marker 可作为额外原样流保留。

`images()` 返回恰好两路 `{left_wrist: RGB uint8, right_wrist: RGB uint8}`；必须等待请求之后两路各自产生的新帧，不能返回运动前缓存。`verified_fresh_image_capture` 能力必须实测验证。记录原生拍摄/接收时间和序号，等待上限与 stale_timeout 一致；缺任一路应抛错或返回缺项使比较不通过。源曝光时间未知时只写接收时间，不声称硬同步。

两路现有腕流来自不同 X5；必须确认都能在 T_check 稳定看到同一瓶身。ZMQ PUSH/PULL 多客户端会分流，现有 latest-only 流不能作为完整原始流保证，不能直接额外启动一个 PULL 消费者抢走已有客户端的帧。

## 操作约束

初次实机必须在操作者守护下单步运行三点 commissioning；没有任何脚本可自动启动机器人、修改系统安全参数或重新定义 TCP。Python 碰撞模型只覆盖配置的夹爪和三段瓶体，整臂和其他环境仍由既有实机碰撞系统保护。

所有实机动作入口先验证 TODO、配置、dry-run、安全检查和能力合同。第一版只完成模拟闭环和适配合同，硬件初始化/停止/力控生命周期需要明天在现场核验后绑定，不能通过将能力位随意改为 true 来替代验证。
