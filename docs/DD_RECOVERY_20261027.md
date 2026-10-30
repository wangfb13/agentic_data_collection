# 右 X5 DD 通道恢复记录（2026-10-27）

完整只读 doctor 已恢复通过：`read_only_checks_passed=true`，`errors=[]`。没有执行机械臂或夹爪运动。

## 原因与处理

右 X5 `192.168.9.11` 的旧触觉进程 `demo_all_sensors_broadcast_server_with_wrench.py` 运行中，仅监听 39010/39020；已有 `starvla-tactile-dd.service` 为 inactive/disabled，没有 39030 监听。这解释了六维力正常、两指 DD 均为 0 的报错。只读 [服务与进程清单](../outputs/dd_recovery_20261027/inventory_result.json) 保留了现场证据。

核对当前没有外部命令输入发布者、两夹爪状态均为张开后，按精确 PID、启动时间和命令身份停止旧触觉 owner 及其子进程。确认相关端口释放后启动已安装的 DD 服务。腕部相机进程保留；触觉源码、编码器和服务单元 SHA 均保持不变。没有修改 SDK 算法、机器人底层运控或实验安全门禁。

右 X5 备份目录：`/home/cftc/261022_tactile_dd/backup/bottle_dd_recovery_20261027_7c4eae5c`。恢复后服务 PID=81940，active/running，39010、39020、39030 均由同一 owner 监听。见 [切换结果](../outputs/dd_recovery_20261027/recovery_result.json)。

## 原命令实际复查

使用现有容器和统一启动器执行：

```bash
bash run_on_5090.sh doctor --config config.ros.yaml
```

| 检查 | 实际结果 |
|---|---|
| 完整诊断 | 退出码 0，read_only_checks_passed=true，errors=[] |
| 命令发布者 | 检查通过，底层系统输出保留 |
| 双臂/夹爪状态 | 实时反馈可读，双夹爪状态新鲜 |
| 六维力 | 两指均 ready |
| DD | 5.000 s 内两指各 142 帧，约 28.4 帧/s 的接收计数 |
| DD 表示 | uint16[96,128,3]，解码 float32；处理后 DD，native_dd=false |
| DD 帧对应 | depth/deformation fid 对应检查通过 |
| 两腕图像 | 左右均 RGB uint8[1080,1920,3] |

证据：[完整 JSON](../outputs/dd_recovery_20261027/doctor_after.json)、[终端日志](../outputs/dd_recovery_20261027/doctor_after.log)。该结果证明当前连通性，不代替物理标定、停止测试、力控和 commissioning 的现场验收。

## 后续启动

带 DD 的服务目前为手动启动。不要与旧触觉程序同时启动；下次设备启动应选 `systemctl --user start starvla-tactile-dd.service` 作为触觉入口。若旧程序已被其它启动流程拉起，先在其原终端正常结束，确认端口释放。保持两指胶面空载再启动 DD，服务启动会重取固定参考。没有修改未知的机器人总启动脚本或自动启用开机时的胶面参考重置。详细顺序见 [指南 1.1](ADVANCED.md#11-如果六维力正常但两指-dd-没有数据)。

服务重启后创建新 session 并重新 prepare；不自动更新已有 session 的参考、零点或验收标志。
