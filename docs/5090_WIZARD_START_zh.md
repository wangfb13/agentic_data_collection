# 5090 实机向导启动

```bash
cd /home/dm/DM_das/bottle_grasp_experiment
bash run_on_5090.sh setup --real
```

首次按终端向导完成尺寸、开度、几何、示教及真实停止/力控/开度验收，生成 `station_profile.yaml`。所有动作由你在现场输入 MOVE 启动，数据检查及 PASS 不能跳过。

之后日常：

```bash
bash run_on_5090.sh run --mode contact
```

视觉模式改为 `--mode visual`，断点续跑加 `--session 之前的目录`。同一启动器兼容 5090 宿主机和现有 ROS 容器，无需在容器里安装 Docker。

只读接通检查：`bash run_on_5090.sh setup --check-only`，不会创建控制后端。明确加 `--simulate` 才是模拟；`--rehearsal` 在实机被拒绝。不能拿旧的模拟 profile 或验收记录替代实机设置。

[日常一页指南](../OPERATOR_GUIDE_zh.md) · [首次 setup](FIRST_SETUP.md) · [故障恢复](RECOVERY.md) · [高级工具](ADVANCED.md)
