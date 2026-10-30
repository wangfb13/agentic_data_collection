# 公开 ROS 接入与现场只读验收（2026-10-27）

当前操作入口见 [完整操作指南](ADVANCED.md)。本次按操作员指示使用公开 ROS 位姿/夹爪命令和现有 X5 数据；操作员明确接受已有处理后的 DD，并授权把右 X5 触觉服务替换成现有能同时发送 DD 的服务。

## 部署与在线检查

| 项目 | 实际结果 |
|---|---|
| 5090 项目 | `/home/dm/DM_das/bottle_grasp_experiment` |
| Lenovo 副本 | `/home/lenovo/wfb/bottle_grasp_experiment`，包含完整指南、源码和测试 |
| 实机环境 | 已运行的 `daimon_ros1_das`；项目 `.venv_robot`，Python 3.12.13；复用 starvla 环境，只在项目环境补装 h5py 3.16.0 |
| ROS | `http://192.168.9.77:11311`；5090 容器 `ROS_IP=192.168.9.57` |
| 位姿 | 两臂 `/wbc_state_pose`、实时 TF 可读；base_link，ROS m/xyzw，项目转换到 mm/矩阵 |
| TCP | 实时 `arm_{side}_7_link → {side}_gripper_base_link`，不使用猜测的固定工具偏移 |
| 夹爪 | 左右 `/..._gripper_state` 均可读；物理 mm 映射仍须实测，不能由 normalized=1 推断毫米 |
| 腕图 | 左/右 X5 的 5552；两路实际 JPEG 解码均为 1920×1080 RGB；诊断成功 |
| 六维力 | 右 X5 39020，两指序列号 `X2620410318`、`X2620410347` 均持续收到 |
| DD | 右 X5 39030，两指均收到 `uint16[96,128,3]`，约 27–28 Hz；解码 float32 与收到的 uint16 分别保存 |
| 总诊断 | `outputs/live_diagnostics/latest.json`：`read_only_checks_passed=true`，无连接错误 |
| 位姿登记入口 | 成功生成 `outputs/live_diagnostics/connectivity_pose.json`；标签明确为连接测试，**不是 T0**，没有填入实测配置 |

右 X5 原触觉 owner 为 `demo_all_sensors_broadcast_server_with_wrench.py`。停止该 owner 及其子进程、确认端口释放后，启用已安装的 `starvla-tactile-dd.service`。使用原有源码，未修改算法、机器人控制器或策略；腕图原进程保持运行。39010 原图、39020 六维力、39030 DD 的两指六条流均检查通过。新服务的会话标识、参考哈希及 force fid 对应信息可读；SDK 源帧号存在缺口，未声明原生、无丢帧或硬同步。

右 X5 备份目录：`/home/cftc/261022_tactile_dd/backup/bottle_20261027_before_switch`。切换清单与验证见 `outputs/x5_service_switch_20261027/`，不含登录凭据。需要恢复旧触觉服务时，在右 X5 停止 `starvla-tactile-dd.service`、确认触觉端口释放，再进入 `/home/cftc/server/daimon_tactile_codec-broadcast_x5` 用 `/usr/bin/python3.10 demo_all_sensors_broadcast_server_with_wrench.py` 启动原服务；恢复后没有 DD。不要同时启动两个 SDK owner。

## 软件验收

- 最终部署版本在 `.venv_robot` 中 **176 项测试全部通过**：几何、原状态机、公共 ROS 假 I/O、单臂掩码、软件插值与反馈到位、旧目标取消、开度映射、旧力控纯函数复用、空中锁、提起前独立应急释放、数据队列、时间戳、图像与 DD 来源、准备交互和分析。
- 完全模拟端到端验收位于 `outputs/public_ros_offline_acceptance_20261027/`：8 个接触场景、6 个视觉点；固定参考哈希保持、视觉无闭合分支；分析 14 条记录、22 条描述子、1751 条位姿误差，warnings 为空。
- 操作指南的 27 个 shell 命令块通过 `bash -n`，内嵌 Python 通过语法解析。未执行其中的实机运动命令。
- 源码部署通过 Python 读取明文字节传输并逐文件核对 SHA256，旧文件备份在 5090 `outputs/deployment_backups/`。

本次实际连接仅进行了只读 ROS/TF、公开传感器接收和已授权的 X5 触觉服务切换。**没有发送机械臂目标、开合夹爪、调用暂停/恢复/回零服务或修改控制器参数。** 动作路径的测试使用假 ROS I/O，首次物理运行仍由操作员按指南单步启动。

## 能力范围和剩余实测项

`public_ros` profile 明确使用上层软件插值、公开位置命令和当前实测位姿保持，不宣称硬件急停、硬件速度/力限制或进程断开后的独立保护。严格模板保留，不以 public profile 冒充其原生能力保证。

夹爪电流没有可证实的公开输入，保存 NaN 及 availability=false；速度是开度反馈在主机接收时间上的差分估计。DD 是现有缩小/量化流，保留编码载荷和解码结果、源 fid/会话/参考信息；相机源未提供曝光时间或源序号，记录接收时间，不声称硬同步。

`config.ros.yaml` 的真实 T0、自由空间位姿、工作空间、几何、轴向、夹爪 mm 映射、SDK 力到 N/力矩换算、像素间距和实验限值仍待实测。软件空载扣零不能代替单位换算。诊断连接通过不绕过这些 TODO，也不自动把读取的当前位置当成理想抓取点。
